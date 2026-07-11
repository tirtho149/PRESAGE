#!/usr/bin/env python3
"""Verify BOTH link-liveness AND quote-fidelity for every source in the PRESAGE KB.

For every {url, quote} node anywhere in the KB (canonical fields incl. nested
`summary`/`diagnostic_features`, AND every delta `web_support` entry) this:
  1. fetches the URL (browser headers, GET w/ HEAD fallback, retries), and
  2. checks that the stored `quote` actually appears on the fetched page.

Quote matching is elision-aware: a quote is split on ' ... ' fragments and each
fragment is matched against the normalized page text (exact substring first,
then a difflib longest-match ratio). Node verdicts:
    quote_found     - every fragment present on the live page
    quote_partial   - some (but not all) fragments / high fuzzy ratio
    quote_not_found - page is live but the quote is not on it
    url_blocked     - 403/429 publisher bot-block (page presumed live, quote unchecked)
    url_dead        - 404/410/DNS/refused (genuinely broken)
    url_unknown     - timeout/SSL/other transient (undecided)

Outputs:
    artifacts/quote_check.json   - per-URL: {status, code, text_len, quotes:[{path,crop,disease,
                                             quote_status,ratio,fragments_found,fragments_total}]}
    artifacts/quote_check.csv    - one row per quote node (flat, for inspection)
    artifacts/page_cache/*.txt   - extracted page text cache (so re-runs skip refetch)

Usage:
    python scripts/verify_kb_quotes.py                 # verify everything
    python scripts/verify_kb_quotes.py --use-cache     # reuse page_cache text, only fetch new
    python scripts/verify_kb_quotes.py --problems      # after a run, print only failing nodes
    python scripts/verify_kb_quotes.py --limit 50      # smoke test on first 50 urls
"""
import json, glob, csv, sys, os, re, html, hashlib, collections, argparse, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "artifacts/pathome_kb")
CACHE_JSON = os.path.join(ROOT, "artifacts/quote_check.json")
CSVOUT = os.path.join(ROOT, "artifacts/quote_check.csv")
PAGE_DIR = os.path.join(ROOT, "artifacts/page_cache")

BROWSER = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# thresholds
FRAG_SUBSTR = True          # exact normalized-substring counts as found
FRAG_RATIO = 0.90           # difflib ratio on a fragment to count as fuzzy-found
MIN_FRAG_CHARS = 12         # ignore tiny fragments when scoring quote_status


# ---------------------------------------------------------------- collection
def walk(o, path, crop, disease, out):
    """Collect every dict node carrying BOTH a url and a quote."""
    if isinstance(o, dict):
        if o.get("url") and o.get("quote"):
            out.append({"path": path, "crop": crop, "disease": disease,
                        "url": o["url"], "quote": o["quote"]})
        # track disease name as we descend
        dname = o.get("disease_name", disease)
        for k, v in o.items():
            walk(v, f"{path}/{k}", crop, dname, out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", crop, disease, out)


def collect_nodes():
    nodes = []
    for f in sorted(glob.glob(f"{KB}/*/final_registry.json")):
        crop = os.path.basename(os.path.dirname(f))
        walk(json.load(open(f)), crop, crop, "", nodes)
    return nodes


# ---------------------------------------------------------------- fetch
def cache_path(url):
    return os.path.join(PAGE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".txt")


TAG = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.S)
STRIP = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def extract_text(htmltext):
    t = TAG.sub(" ", htmltext)
    t = STRIP.sub(" ", t)
    t = html.unescape(t)
    return WS.sub(" ", t).strip()


def fetch(url, use_cache):
    """Return (status, code, text). status in alive/blocked/dead/unknown."""
    cp = cache_path(url)
    if use_cache and os.path.exists(cp):
        return "alive", 200, open(cp, encoding="utf-8", errors="ignore").read()
    last_code = 0
    for method in ("get", "head"):
        for attempt in range(2):
            try:
                r = requests.request(method, url, headers=BROWSER, timeout=25,
                                     allow_redirects=True)
                last_code = r.status_code
                if r.status_code < 400:
                    text = ""
                    if method == "get":
                        ct = r.headers.get("content-type", "")
                        if "html" in ct or "text" in ct or not ct:
                            text = extract_text(r.text)
                            os.makedirs(PAGE_DIR, exist_ok=True)
                            open(cp, "w", encoding="utf-8").write(text)
                    return "alive", r.status_code, text
                if r.status_code in (403, 429):
                    return "blocked", r.status_code, ""
                if r.status_code in (404, 410):
                    return "dead", r.status_code, ""
                # other 4xx/5xx: retry once then give up as unknown
            except requests.exceptions.SSLError:
                return "unknown", -1, ""
            except (requests.exceptions.ConnectionError,) as e:
                msg = str(e).lower()
                if "nodename" in msg or "name or service" in msg or "refused" in msg:
                    return "dead", -2, ""
                time.sleep(1)
            except requests.exceptions.RequestException:
                time.sleep(1)
    return "unknown", last_code, ""


# ---------------------------------------------------------------- quote match
def norm(s):
    s = html.unescape(s or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return WS.sub(" ", s).strip()


def fragments(quote):
    # split on ellipsis variants used by the extractors
    parts = re.split(r"\s*\.\.\.\s*|\s*…\s*|\s*\[\.\.\.\]\s*", quote)
    return [p for p in (f.strip() for f in parts) if p]


def match_quote(quote, page_norm):
    """Return (status, ratio, found, total) for a quote vs normalized page text."""
    frags = fragments(quote)
    scored = [f for f in frags if len(norm(f)) >= MIN_FRAG_CHARS] or frags
    total = len(scored)
    found = 0
    ratios = []
    for f in scored:
        nf = norm(f)
        if not nf:
            total -= 1
            continue
        if nf in page_norm:
            found += 1
            ratios.append(1.0)
            continue
        # fuzzy: best local alignment ratio
        sm = SequenceMatcher(None, nf, page_norm, autojunk=False)
        m = sm.find_longest_match(0, len(nf), 0, len(page_norm))
        r = m.size / max(1, len(nf))
        ratios.append(r)
        if r >= FRAG_RATIO:
            found += 1
    total = max(total, 1)
    avg = sum(ratios) / len(ratios) if ratios else 0.0
    if found == total:
        return "quote_found", avg, found, total
    if found > 0 or avg >= 0.6:
        return "quote_partial", avg, found, total
    return "quote_not_found", avg, found, total


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--problems", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    nodes = collect_nodes()
    by_url = collections.defaultdict(list)
    for n in nodes:
        by_url[n["url"]].append(n)
    urls = list(by_url)
    if args.limit:
        urls = urls[:args.limit]
    print(f"{len(nodes)} quote nodes across {len(by_url)} unique URLs "
          f"({len(urls)} to fetch)")

    if args.problems:
        return print_problems()

    results = {}
    done = 0
    t0 = time.time()

    def work(u):
        status, code, text = fetch(u, args.use_cache)
        return u, status, code, text

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, u) for u in urls]
        for fut in as_completed(futs):
            u, status, code, text = fut.result()
            page_norm = norm(text) if text else ""
            qentries = []
            for n in by_url[u]:
                if status == "alive" and page_norm:
                    qs, ratio, found, total = match_quote(n["quote"], page_norm)
                elif status == "alive" and not page_norm:
                    qs, ratio, found, total = "url_unfetched_text", 0.0, 0, 0
                elif status == "blocked":
                    qs, ratio, found, total = "url_blocked", 0.0, 0, 0
                elif status == "dead":
                    qs, ratio, found, total = "url_dead", 0.0, 0, 0
                else:
                    qs, ratio, found, total = "url_unknown", 0.0, 0, 0
                qentries.append({"path": n["path"], "crop": n["crop"],
                                 "disease": n["disease"], "quote": n["quote"],
                                 "quote_status": qs, "ratio": round(ratio, 3),
                                 "frags_found": found, "frags_total": total})
            results[u] = {"status": status, "code": code,
                          "text_len": len(text), "quotes": qentries}
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(urls)}  ({time.time()-t0:.0f}s)")

    json.dump(results, open(CACHE_JSON, "w"), indent=1)
    write_csv(results)
    summarize(results)


def write_csv(results):
    with open(CSVOUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "disease", "path", "url", "url_status", "http_code",
                    "quote_status", "ratio", "frags_found", "frags_total", "quote"])
        for u, r in results.items():
            for q in r["quotes"]:
                w.writerow([q["crop"], q["disease"], q["path"], u, r["status"],
                            r["code"], q["quote_status"], q["ratio"],
                            q["frags_found"], q["frags_total"], q["quote"][:300]])


def summarize(results):
    url_c = collections.Counter(r["status"] for r in results.values())
    q_c = collections.Counter(q["quote_status"]
                              for r in results.values() for q in r["quotes"])
    nnodes = sum(len(r["quotes"]) for r in results.values())
    print("\n=== URL liveness (unique urls) ===")
    for k, n in url_c.most_common():
        print(f"  {k:16s} {n}")
    print(f"\n=== Quote fidelity ({nnodes} nodes) ===")
    for k, n in q_c.most_common():
        print(f"  {k:20s} {n}")
    bad = q_c["quote_not_found"] + q_c["url_dead"] + q_c["url_unknown"]
    print(f"\nNeeds repair (quote_not_found + url_dead + url_unknown): {bad}")
    print(f"Report: {CACHE_JSON}  |  {CSVOUT}")


def print_problems():
    results = json.load(open(CACHE_JSON))
    bad_status = {"quote_not_found", "url_dead", "url_unknown"}
    n = 0
    for u, r in results.items():
        probs = [q for q in r["quotes"] if q["quote_status"] in bad_status]
        if not probs:
            continue
        print(f"\n[{r['status']} {r['code']}] {u}")
        for q in probs:
            print(f"    {q['crop']}/{q['disease']} :: {q['quote_status']} "
                  f"(r={q['ratio']}) {q['path']}")
            print(f"      quote: {q['quote'][:160]}")
            n += 1
    print(f"\n{n} problem nodes")


if __name__ == "__main__":
    main()
