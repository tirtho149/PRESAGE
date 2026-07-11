#!/usr/bin/env python3
"""Verify liveness of every URL stored in the PRESAGE final registry.

Classifies each unique URL as:
  alive        - 2xx / 3xx reachable
  blocked      - 403 / 429 (bot-blocked by publisher; almost always live in a browser)
  dead         - 404 / 410 / DNS failure / connection refused (genuinely broken)
  unknown      - timeout / SSL / other transient error (could not decide)

Outputs:
  artifacts/link_check.json  - {url: {status, code, count, kinds}} full result cache
  artifacts/link_check.csv   - flat table for inspection
and prints a summary. Re-run is cheap: pass --use-cache to skip already-checked URLs.

Usage:
  python scripts/verify_kb_links.py                 # check all
  python scripts/verify_kb_links.py --use-cache     # only check new/unknown URLs
  python scripts/verify_kb_links.py --dead-only      # print the genuinely-dead list
"""
import json, glob, csv, sys, collections, os
import requests
from concurrent.futures import ThreadPoolExecutor

KB = "artifacts/pathome_kb"
CACHE = "artifacts/link_check.json"
CSVOUT = "artifacts/link_check.csv"

BROWSER = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def collect_urls():
    """Return {url: {'count': n, 'kinds': set()}} across the whole KB."""
    urls = collections.defaultdict(lambda: {"count": 0, "kinds": set()})
    for f in glob.glob(f"{KB}/*/final_registry.json"):
        d = json.load(open(f))
        for dz in d.get("diseases", []):
            for k in ["pathogen_scientific_name", "type_of_disease",
                      "affected_parts", "visual_symptoms", "treatments"]:
                v = dz.get(k)
                if isinstance(v, dict) and v.get("url"):
                    urls[v["url"]]["count"] += 1
                    urls[v["url"]]["kinds"].add("canonical_source")
            for st in (dz.get("regional_observations") or {}).values():
                for dl in st.get("deltas", []):
                    for w in (dl.get("web_support") or []):
                        if w.get("url"):
                            urls[w["url"]]["count"] += 1
                            urls[w["url"]]["kinds"].add("delta_web_support")
    return urls


def classify(url):
    """Return (status, code). Try GET (browser headers); HEAD as a light fallback."""
    for method in ("get", "head"):
        try:
            r = getattr(requests, method)(
                url, headers=BROWSER, timeout=15, allow_redirects=True,
                stream=(method == "get"))
            code = r.status_code
            if code < 400:
                return "alive", code
            if code in (403, 429):
                return "blocked", code          # publisher bot-block; treat as likely-live
            if code in (404, 410):
                return "dead", code
            if method == "head":
                return "unknown", code
        except requests.exceptions.SSLError:
            return "unknown", "SSLError"
        except (requests.exceptions.ConnectionError,) as e:
            # DNS / refused == genuinely unreachable
            msg = str(e)
            if "NameResolutionError" in msg or "Name or service not known" in msg:
                return "dead", "DNSError"
            return "unknown", "ConnectionError"
        except requests.exceptions.Timeout:
            return "unknown", "Timeout"
        except Exception as e:
            return "unknown", type(e).__name__
    return "unknown", "no-response"


def main():
    use_cache = "--use-cache" in sys.argv
    dead_only = "--dead-only" in sys.argv

    urls = collect_urls()
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE))

    if dead_only:
        dead = {u: c for u, c in cache.items() if c.get("status") == "dead"}
        print(f"{len(dead)} genuinely-dead URLs:")
        for u, c in sorted(dead.items()):
            print(f"  [{c['code']}] x{c['count']}  {u}")
        return

    todo = [u for u in urls
            if not (use_cache and u in cache and cache[u].get("status") not in (None, "unknown"))]
    print(f"{len(urls)} unique URLs | checking {len(todo)} "
          f"({'skipping cached alive/blocked/dead' if use_cache else 'full re-check'})")

    results = {}
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, (u, (status, code)) in enumerate(zip(todo, ex.map(classify, todo)), 1):
            results[u] = {"status": status, "code": code}
            if i % 200 == 0:
                print(f"  ...{i}/{len(todo)}")

    # merge
    for u, meta in urls.items():
        entry = results.get(u) or cache.get(u) or {"status": "unknown", "code": None}
        entry["count"] = meta["count"]
        entry["kinds"] = sorted(meta["kinds"])
        cache[u] = entry

    json.dump(cache, open(CACHE, "w"), indent=1)
    with open(CSVOUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["url", "status", "code", "occurrences", "kinds"])
        for u, c in sorted(cache.items()):
            w.writerow([u, c["status"], c["code"], c["count"], ";".join(c.get("kinds", []))])

    by_status = collections.Counter(c["status"] for c in cache.values())
    occ = collections.Counter()
    for c in cache.values():
        occ[c["status"]] += c["count"]
    n = len(cache)
    print("\n==== LINK VERIFICATION (unique URLs) ====")
    for st in ["alive", "blocked", "dead", "unknown"]:
        print(f"  {st:8} {by_status.get(st,0):5}  ({100*by_status.get(st,0)/n:4.1f}%)   "
              f"occurrences={occ.get(st,0)}")
    print(f"  TOTAL    {n:5}")
    print(f"\nwrote {CACHE} and {CSVOUT}")
    print("genuinely-dead list: python scripts/verify_kb_links.py --dead-only")


if __name__ == "__main__":
    main()
