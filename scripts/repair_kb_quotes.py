#!/usr/bin/env python3
"""Repair unverifiable quotes / dead links in the PRESAGE KB.

Driven by artifacts/quote_check.json (post render_recheck). For every problem node
(quote_not_found / url_dead / url_unknown / render_error) it produces a fix:

  MODE requote  - the cited page is LIVE and we have its text. Ask Claude to return a
                  VERBATIM sentence from THAT page supporting the claim (keep the URL,
                  swap in a real quote). If the page genuinely doesn't support the
                  claim, Claude says NOT_SUPPORTED -> node escalates to websearch.
  MODE websearch- page is dead/unreachable OR page doesn't support the claim. Ask Claude
                  (+WebSearch) for an ALTERNATIVE url that supports the claim and a
                  verbatim quote from it. The proposed (url,quote) is then re-verified
                  DETERMINISTICALLY (fetch/render + match) before being accepted.

Steps (resumable, idempotent):
  plan   -> artifacts/repair_plan.json     (group problem nodes by URL, assign mode)
  run    -> artifacts/repair_patches.json  (Claude proposes fixes; caches per URL/claim)
  verify -> re-check proposed (url,quote) pairs deterministically; mark accept/reject
  apply  -> backup KB, write accepted patches into final_registry.json, regen Excel

Usage:
  python scripts/repair_kb_quotes.py plan
  python scripts/repair_kb_quotes.py run   [--crop Soybean] [--limit N] [--mode requote|websearch]
  python scripts/repair_kb_quotes.py verify
  python scripts/repair_kb_quotes.py apply [--dry-run]
"""
import json, os, sys, glob, argparse, subprocess, tempfile, hashlib, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_kb_quotes import (norm, match_quote, fetch, extract_text, cache_path,
                              PAGE_DIR, KB)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QC = os.path.join(ROOT, "artifacts/quote_check.json")
PLAN = os.path.join(ROOT, "artifacts/repair_plan.json")
# Per-crop patch files so a parallel slurm array (one task per crop) never races
# on a shared file. verify/apply merge across all crop files.
PATCHES_DIR = os.path.join(ROOT, "artifacts/repair_patches")
CLAUDE = "/work/mech-ai-scratch/tirtho/npm-global/bin/claude"

# Repair targets: claim genuinely NOT on the cited page, or page unjudgeable
# (PDF/blocked/render-error -> no extractable text). Set by rescore_quotes.py.
# 'supported'/'weak' are left alone (link is fine, quote merely reworded).
PROBLEM_CLAIM = {"unsupported", "no_text"}
MIN_TEXT = 800  # a cached page shorter than this is treated as "no usable text"


# ------------------------------------------------------------------ helpers
def load_qc():
    return json.load(open(QC))


def best_cached_text(url):
    """Prefer the rendered text; fall back to the requests text."""
    rp = os.path.join(PAGE_DIR, "render_" + hashlib.sha1(url.encode()).hexdigest() + ".txt")
    if os.path.exists(rp):
        t = open(rp, encoding="utf-8", errors="ignore").read()
        if len(t) >= MIN_TEXT:
            return t
    cp = cache_path(url)
    if os.path.exists(cp):
        t = open(cp, encoding="utf-8", errors="ignore").read()
        if len(t) >= MIN_TEXT:
            return t
    return ""


def claude_json(prompt, system, allow_websearch, max_turns=20, timeout=1200):
    """Invoke the headless claude CLI, return parsed 'result' text or None."""
    env = {k: v for k, v in os.environ.items()
           if not any(k.startswith(p) for p in
                      ("CLAUDE", "CURSOR", "MCP_CONNECTION", "VSCODE", "ELECTRON"))}
    env.pop("ANTHROPIC_API_KEY", None)
    cmd = [CLAUDE, "-p", "Follow the instructions provided via stdin.",
           "--output-format", "json", "--max-turns", str(max_turns),
           "--append-system-prompt", system]
    if allow_websearch:
        cmd += ["--allowedTools", "WebSearch"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as pf:
        pf.write(prompt); pp = pf.name
    try:
        with open(pp) as fin:
            r = subprocess.run(cmd, stdin=fin, capture_output=True, text=True,
                               env=env, cwd=ROOT, timeout=timeout)
        if r.returncode != 0:
            return None
        obj = json.loads(r.stdout)
        return obj.get("result", "")
    except Exception:
        return None
    finally:
        os.unlink(pp)


def parse_json_block(text):
    """Pull the first JSON object/array out of a claude text reply."""
    if not text:
        return None
    s = text.find("{"); a = text.find("[")
    start = min([x for x in (s, a) if x >= 0], default=-1)
    if start < 0:
        return None
    depth = 0; opener = text[start]; closer = "}" if opener == "{" else "]"
    for i in range(start, len(text)):
        if text[i] == opener: depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    return None
    return None


# ------------------------------------------------------------------ plan
def cmd_plan(args):
    qc = load_qc()
    by_url = collections.defaultdict(list)
    for url, d in qc.items():
        for q in d["quotes"]:
            if q.get("claim_status") in PROBLEM_CLAIM:
                by_url[url].append(q)
    plan = []
    n_requote = n_web = 0
    for url, quotes in by_url.items():
        txt = best_cached_text(url)
        dead = qc[url]["status"] in ("dead", "render_error", "unknown")
        mode = "requote" if (txt and not dead) else "websearch"
        if mode == "requote": n_requote += len(quotes)
        else: n_web += len(quotes)
        plan.append({"url": url, "mode": mode, "has_text": bool(txt),
                     "url_status": qc[url]["status"], "n_quotes": len(quotes),
                     "quotes": quotes})
    json.dump(plan, open(PLAN, "w"), indent=1)
    urls_req = sum(1 for p in plan if p["mode"] == "requote")
    urls_web = sum(1 for p in plan if p["mode"] == "websearch")
    print(f"problem nodes: {n_requote+n_web}  (requote {n_requote} / websearch {n_web})")
    print(f"URLs: requote {urls_req}  websearch {urls_web}  -> {PLAN}")


# ------------------------------------------------------------------ run
REQUOTE_SYS = (
    "You verify agricultural plant-disease knowledge. You are given the full TEXT of a "
    "specific web page and a list of CLAIMS that were supposedly sourced from it. For "
    "each claim, find a short VERBATIM sentence (copied exactly, word for word) from the "
    "page text that directly supports the claim. If the page text does not support the "
    "claim, return null for that claim. Output ONLY a JSON array of objects "
    '{"id": <int>, "quote": <verbatim string or null>}. No commentary.')

WEBSEARCH_SYS = (
    "You find authoritative sources for agricultural plant-disease facts. For each claim "
    "you are given, use WebSearch to find ONE reputable page (university extension, APS, "
    "peer-reviewed, gov) that supports it, and copy a short VERBATIM sentence from that "
    "page. Prefer stable, non-paywalled URLs. Output ONLY a JSON array of objects "
    '{"id": <int>, "url": <string or null>, "quote": <verbatim string or null>}.')


def crop_patch_file(crop):
    os.makedirs(PATCHES_DIR, exist_ok=True)
    safe = crop.replace("/", "_").replace(" ", "_")
    return os.path.join(PATCHES_DIR, safe + ".json")


def load_patches(crop=None):
    """crop given -> that crop's file; else merge every crop file."""
    if crop is not None:
        f = crop_patch_file(crop)
        return json.load(open(f)) if os.path.exists(f) else {}
    merged = {}
    for f in glob.glob(f"{PATCHES_DIR}/*.json"):
        merged.update(json.load(open(f)))
    return merged


def save_patches(p, crop):
    json.dump(p, open(crop_patch_file(crop), "w"), indent=1)


def node_key(url, q):
    return hashlib.sha1((url + "||" + q["path"] + "||" + q["quote"]).encode()).hexdigest()


def cmd_run(args):
    if not args.crop:
        sys.exit("run requires --crop (one crop per task; parallel-safe per-crop files)")
    plan = json.load(open(PLAN))
    patches = load_patches(args.crop)
    # restrict plan to nodes of THIS crop (a URL may be shared across crops)
    plan = [dict(p, quotes=[q for q in p["quotes"] if q["crop"] == args.crop])
            for p in plan]
    plan = [p for p in plan if p["quotes"]]

    # websearch pass also picks up requote-misses escalated in an earlier pass.
    # An escalation has no successful proposal yet (no new_quote) and hasn't been
    # websearched (_ws_done). Do NOT gate on `verified` — a prior verify run may
    # have stamped verified=False on these, which must not block re-processing.
    escalated = []
    if args.mode in (None, "websearch"):
        for k, pt in patches.items():
            if pt.get("escalate") and not pt.get("new_quote") and not pt.get("_ws_done"):
                escalated.append(pt)

    if args.mode:
        plan = [p for p in plan if p["mode"] == args.mode]
    if args.limit:
        plan = plan[:args.limit]
    print(f"processing {len(plan)} URLs ({args.mode or 'all modes'}) "
          f"+ {len(escalated)} escalated claims  (workers={args.workers})")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock = threading.Lock()

    def do_url(p):
        url = p["url"]
        todo = [q for q in p["quotes"] if node_key(url, q) not in patches]
        if not todo:
            return url, []
        if p["mode"] == "requote":
            txt = best_cached_text(url)
            if not txt:
                return url, []
            claims = "\n".join(f'{j}. CLAIM: {q.get("disease","")} — '
                               f'{(q.get("path","")).split("/")[-1]}: '
                               f'"{q["quote"][:200]}"' for j, q in enumerate(todo))
            prompt = (f"PAGE URL: {url}\n\nPAGE TEXT:\n{txt[:16000]}\n\n"
                      f"CLAIMS (find a verbatim supporting sentence for each):\n{claims}")
            res = parse_json_block(claude_json(prompt, REQUOTE_SYS, False))
        else:
            claims = "\n".join(f'{j}. {q.get("crop","")} / {q.get("disease","")}: '
                               f'"{q["quote"][:200]}"' for j, q in enumerate(todo))
            prompt = ("Find a supporting source + verbatim quote for each claim below.\n"
                      + claims)
            res = parse_json_block(claude_json(prompt, WEBSEARCH_SYS, True, max_turns=30))
        out = []
        if isinstance(res, list):
            for item in res:
                j = item.get("id")
                if not isinstance(j, int) or j >= len(todo):
                    continue
                q = todo[j]
                nq = item.get("quote")
                escalate = (p["mode"] == "requote" and not nq)
                out.append((node_key(url, q), {
                    "crop": q["crop"], "disease": q.get("disease", ""),
                    "path": q["path"], "old_url": url, "old_quote": q["quote"],
                    "mode": p["mode"],
                    "new_url": item.get("url", url if p["mode"] == "requote" and nq else None),
                    "new_quote": nq, "verified": None, "escalate": escalate}))
        return url, out

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_url, p): p for p in plan}
        for fut in as_completed(futs):
            url, out = fut.result()
            with lock:
                for k, v in out:
                    patches[k] = v
                done += 1
                if done % 5 == 0 or out:
                    save_patches(patches, args.crop)
            print(f"  [{done}/{len(plan)}] {url} -> {len(out)} claims")

    # second pass: websearch for requote-misses escalated above (or in a prior run)
    def do_escalate(e):
        prompt = ("Find a supporting source + verbatim quote for the claim below.\n"
                  f'0. {e.get("crop","")} / {e.get("disease","")}: "{e["old_quote"][:200]}"')
        res = parse_json_block(claude_json(prompt, WEBSEARCH_SYS, True, max_turns=30))
        e["_ws_done"] = True
        if isinstance(res, list) and res:
            it = res[0]
            e["new_url"] = it.get("url")
            e["new_quote"] = it.get("quote")
            e["mode"] = "websearch_escalated"
        return e
    if escalated:
        ndone = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for _ in as_completed([ex.submit(do_escalate, e) for e in escalated]):
                ndone += 1
                with lock:
                    if ndone % 5 == 0:
                        save_patches(patches, args.crop)
        save_patches(patches, args.crop)
        print(f"  websearch-escalated {len(escalated)} requote-misses")


# ------------------------------------------------------------------ verify
def _verify_one(p):
    nu, nq = p.get("new_url"), p.get("new_quote")
    if not nu or not nq:
        p["verified"] = False; p["reason"] = "no proposal"; return
    txt = best_cached_text(nu)
    if not txt:
        _s, _c, txt = fetch(nu, False)
    pn = norm(txt) if txt else ""
    if not pn:
        p["verified"] = False; p["reason"] = "page unfetchable"; return
    qs, ratio, f, t = match_quote(nq, pn)
    p["verified"] = qs in ("quote_found",)
    p["verify_status"] = qs; p["verify_ratio"] = round(ratio, 3)


def cmd_verify(args):
    files = ([crop_patch_file(args.crop)] if args.crop
             else sorted(glob.glob(f"{PATCHES_DIR}/*.json")))
    total = ok = 0
    for f in files:
        patches = json.load(open(f))
        for p in patches.values():
            if p.get("verified") is None:
                _verify_one(p)
        json.dump(patches, open(f, "w"), indent=1)
        total += len(patches)
        ok += sum(1 for p in patches.values() if p.get("verified"))
    print(f"verified proposals: {ok}/{total} accepted across {len(files)} crop files")


# ------------------------------------------------------------------ apply
def resolve_node(reg, path):
    """Walk a '/'-and-[i] path (crop-rooted) to the containing dict."""
    # path looks like Crop/diseases[0]/visual_symptoms/summary or .../web_support[2]
    toks, buf = [], ""
    for ch in path:
        if ch == "/":
            if buf: toks.append(buf); buf = ""
        elif ch == "[":
            if buf: toks.append(buf); buf = ""
            buf = "["
        else:
            buf += ch
    if buf: toks.append(buf)
    node = reg
    for tk in toks[1:]:  # skip crop root token
        if tk.startswith("["):
            node = node[int(tk[1:-1])]
        else:
            node = node[tk]
    return node


def _parent_delta_path(path):
    """For a '.../deltas[N]/web_support[M]' path return the enclosing delta path
    ('.../deltas[N]'); for any other (canonical) path return None."""
    import re
    m = re.match(r"^(.*/deltas\[\d+\])/web_support\[\d+\]$", path)
    return m.group(1) if m else None


def cmd_apply(args):
    patches = load_patches()
    accepted = {k: p for k, p in patches.items() if p.get("verified")}
    # Problem nodes we could NOT verify a fix for. Per user policy we do NOT keep
    # the dead link: erase the offending web_support entry and, if that leaves the
    # delta with no web support at all, relabel it 'field_observation' (image-
    # grounded but not web-corroborated) — mirrors validate_kb._relabel_field_observations.
    unfixable = {k: p for k, p in patches.items() if not p.get("verified")}
    print(f"{len(accepted)} verified repairs to apply; "
          f"{len(unfixable)} unfixable dead links to erase")
    if not args.dry_run:
        import tarfile
        bak = os.path.join(ROOT, "artifacts/pathome_kb_backup_pre_quoterepair.tgz")
        if not os.path.exists(bak):
            with tarfile.open(bak, "w:gz") as t:
                t.add(KB, arcname="pathome_kb")
            print("backup ->", bak)
    crops = sorted({p["crop"] for p in patches.values()})
    applied = erased = relabeled = canon_stripped = 0
    for crop in crops:
        f = os.path.join(KB, crop, "final_registry.json")
        reg = json.load(open(f))
        # 1) apply verified repairs in place (keep url or swap, replace quote)
        for p in accepted.values():
            if p["crop"] != crop:
                continue
            try:
                node = resolve_node(reg, p["path"])
                if not isinstance(node, dict):
                    continue
                node["_repaired_from"] = {"prev_url": node.get("url"), "prev_quote": node.get("quote")}
                if p.get("new_url"):
                    node["url"] = p["new_url"]
                node["quote"] = p["new_quote"]
                applied += 1
            except Exception as e:
                print(f"  apply fail {crop} {p['path']}: {e}")
        # 2) erase unfixable dead links (run AFTER repairs so a repaired entry,
        #    now carrying a new quote/url, never matches the old dead pair)
        touched = {}  # id(delta) -> delta, for the empty-web relabel pass
        for p in unfixable.values():
            if p["crop"] != crop:
                continue
            try:
                dpath = _parent_delta_path(p["path"])
                if dpath is not None:  # regional delta web_support entry
                    delta = resolve_node(reg, dpath)
                    ws = delta.get("web_support")
                    if isinstance(ws, list):
                        old = (p.get("old_url"), p.get("old_quote"))
                        kept = [w for w in ws if not (isinstance(w, dict)
                                and (w.get("url"), w.get("quote")) == old)]
                        erased += len(ws) - len(kept)
                        delta["web_support"] = kept
                        touched[id(delta)] = delta
                else:                 # canonical single {value,url,quote} node
                    node = resolve_node(reg, p["path"])
                    if isinstance(node, dict) and (node.get("url") or node.get("quote")):
                        node["_unsupported_citation_removed"] = {
                            "prev_url": node.get("url"), "prev_quote": node.get("quote")}
                        node.pop("url", None)
                        node.pop("quote", None)
                        node["citation_status"] = "field_observation"
                        canon_stripped += 1
            except Exception as e:
                print(f"  erase fail {crop} {p['path']}: {e}")
        # 3) any delta left with NO web support -> field_observation
        for delta in touched.values():
            if not delta.get("web_support"):
                status = (delta.get("verification_status") or "").lower()
                if status != "field_observation":
                    delta["_prior_status"] = status
                    delta["verification_status"] = "field_observation"
                    relabeled += 1
        if not args.dry_run:
            json.dump(reg, open(f, "w"), indent=1, ensure_ascii=False)
    print(f"applied {applied} repairs; erased {erased} dead web_support entries; "
          f"relabeled {relabeled} delta(s) -> field_observation; "
          f"stripped {canon_stripped} canonical citation(s)"
          + (" (dry-run, nothing written)" if args.dry_run else ""))


# ------------------------------------------------------------------ cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--crop"); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--mode", choices=["requote", "websearch"])
    r.add_argument("--workers", type=int, default=6)
    v = sub.add_parser("verify"); v.add_argument("--crop")
    a = sub.add_parser("apply"); a.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "verify": cmd_verify,
     "apply": cmd_apply}[args.cmd](args)
