#!/usr/bin/env python3
"""Second-pass CLAIM-support scoring over already-cached page text (no refetch).

The strict verbatim matcher in verify_kb_quotes.py flags any quote that isn't a
near-exact contiguous substring as quote_not_found. That over-counts: many quotes
are lightly reworded or stitch together non-adjacent sentences, yet the page clearly
SUPPORTS the claim. This pass adds a sentence-level token-overlap verdict so we can
tell apart:
    supported    - the page contains the claim (verbatim OR clear paraphrase)
    weak         - partially present
    unsupported  - the page does not contain the claim (real problem -> repair)

Adds `claim_status` + `claim_score` to every node in artifacts/quote_check.json and
prints a combined breakdown. Uses cached page text only (render_*.txt preferred).
"""
import json, os, re, hashlib, collections, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_kb_quotes import norm, PAGE_DIR, CACHE_JSON, write_csv

STOP = set("the a an and or of to in on for with is are be as at by from that this it "
           "may can will which these those such into other more most some are was were "
           "has have had not but also than then when where their its they them".split())


def content_tokens(s):
    return {w for w in norm(s).split() if len(w) > 3 and w not in STOP}


def sentences(text):
    return [s for s in re.split(r"(?<=[.;:])\s+|\n+", text) if len(s) > 25]


def best_cached_text(url, status):
    for pre in ("render_", ""):
        p = os.path.join(PAGE_DIR, pre + hashlib.sha1(url.encode()).hexdigest() + ".txt")
        if os.path.exists(p):
            t = open(p, encoding="utf-8", errors="ignore").read()
            if len(t) > 500:
                return t
    return ""


def claim_score(quote, page_sent_tokens):
    """Fraction of quote-sentences whose content words are >=0.7 covered by some
    single page sentence (captures rewording + concatenation)."""
    qs = sentences(quote) or [quote]
    supported = 0
    scored = 0
    for s in qs:
        qt = content_tokens(s)
        if len(qt) < 3:
            continue
        scored += 1
        best = 0.0
        for pt in page_sent_tokens:
            if not pt:
                continue
            ov = len(qt & pt) / len(qt)
            if ov > best:
                best = ov
                if best >= 0.95:
                    break
        if best >= 0.7:
            supported += 1
    if scored == 0:
        return 1.0, 0
    return supported / scored, scored


def main():
    r = json.load(open(CACHE_JSON))
    # cache page token-sets per url
    n = 0
    for url, d in r.items():
        text = best_cached_text(url, d.get("status"))
        psent = [content_tokens(s) for s in sentences(text)] if text else []
        have_text = bool(psent)
        for q in d["quotes"]:
            if not have_text:
                q["claim_status"] = "no_text"
                q["claim_score"] = 0.0
                continue
            sc, ns = claim_score(q["quote"], psent)
            q["claim_score"] = round(sc, 3)
            q["claim_status"] = ("supported" if sc >= 0.6 else
                                 "weak" if sc >= 0.3 else "unsupported")
        n += 1
        if n % 300 == 0:
            print(f"  {n}/{len(r)}")
    json.dump(r, open(CACHE_JSON, "w"), indent=1)
    write_csv(r)

    # combined breakdown
    comb = collections.Counter()
    for d in r.values():
        for q in d["quotes"]:
            comb[(q["quote_status"] in ("quote_found", "quote_partial"),
                  q.get("claim_status"))] += 1
    print("\n=== verbatim x claim-support ===")
    verb = collections.Counter(); claim = collections.Counter()
    for d in r.values():
        for q in d["quotes"]:
            verb["verbatim" if q["quote_status"] in ("quote_found", "quote_partial")
                 else "not_verbatim"] += 1
            claim[q.get("claim_status")] += 1
    print("verbatim:", dict(verb))
    print("claim   :", dict(claim))
    # the genuine repair set: NOT verbatim AND claim not supported
    genuine = sum(1 for d in r.values() for q in d["quotes"]
                  if q["quote_status"] not in ("quote_found", "quote_partial")
                  and q.get("claim_status") in ("unsupported",))
    no_text = sum(1 for d in r.values() for q in d["quotes"]
                  if q.get("claim_status") == "no_text")
    paraphrase = sum(1 for d in r.values() for q in d["quotes"]
                     if q["quote_status"] not in ("quote_found", "quote_partial")
                     and q.get("claim_status") == "supported")
    print(f"\nGENUINELY UNSUPPORTED (not verbatim & claim unsupported): {genuine}")
    print(f"paraphrase-but-supported (link fine, quote reworded):     {paraphrase}")
    print(f"no usable page text (PDF/blocked/render_error):           {no_text}")


if __name__ == "__main__":
    main()
