# PRESAGE KB — Link + Quote Audit & Repair

**Goal (from user):** every source URL in the KB (canonical sources **and** every
delta `web_support` entry — i.e. everything that lands in the Excel export) must be
(1) actually fetchable, and (2) its stored `quote` must genuinely appear on that page.
Where a link is dead or the quote can't be verified, find a working alternative link.
All checks done with reproducible Python scripts.

Scope: 76 crops · **2,031 unique URLs** · **9,127 quote-bearing nodes**
(1,942 canonical + 7,185 delta `web_support`).

---

## Pipeline (3 phases)

### Phase A — deterministic fetch + quote match  `scripts/verify_kb_quotes.py`
- Recursively walks every `final_registry.json`, collecting every dict node that has
  **both** `url` and `quote` (covers canonical fields incl. nested `summary` /
  `diagnostic_features`, and every delta `web_support[]`).
- Fetches each unique URL with browser headers (`requests`, GET → HEAD fallback, retries).
- Extracts page text (strip script/style/tags, unescape, collapse whitespace).
- **Quote match is elision-aware**: splits the quote on ` ... ` fragments; each fragment
  must appear as a normalized substring (or ≥0.90 difflib ratio) on the page.
- Verdict per node: `quote_found` / `quote_partial` / `quote_not_found` /
  `url_blocked` (403/429) / `url_dead` (404/410/DNS) / `url_unknown` (timeout/SSL).
- Output: `artifacts/quote_check.json`, `artifacts/quote_check.csv`,
  page-text cache `artifacts/page_cache/*.txt`.

### Phase B — JS-render re-check  `scripts/render_recheck.py`
- **Why:** many cited domains are JavaScript SPAs (cropprotectionnetwork.org,
  canr.msu.edu, ncbi/pmc, parts of ncsu). Plain `requests` sees only nav chrome, so
  their real quotes were **falsely** flagged `quote_not_found`. Verified: CPN pages that
  scored 0 found under `requests` score 7–9 `quote_found` once rendered.
- Re-fetches every **ambiguous** URL (1,745 of them) with headless Chromium
  (Playwright), extracts rendered `body` text, re-matches quotes, updates verdicts in
  place (adds `alive_rendered`, `render_error`). Rendered text cached as
  `page_cache/render_<sha1>.txt`.
- This gives the **true** dead-link / unverifiable-quote counts (the pre-render numbers
  are inflated by SPA false-negatives — do NOT trust them).
- Playwright + chromium-headless-shell installed under
  `/work/mech-ai-scratch/tirtho/.pw_browsers` (set `PLAYWRIGHT_BROWSERS_PATH`).

### Phase C — repair  `scripts/repair_kb_quotes.py`  *(to be built after Phase B numbers)*
For each node still failing after render:
- **Case A — page live & correct, quote is paraphrased (expected majority):**
  re-extract a *verbatim* supporting sentence from the **same** page (page is fine, only
  the quote string drifted). Batched one `claude -p` call per URL over all its bad quotes.
- **Case B — url dead / unreachable:** `claude -p --allowedTools WebSearch` finds an
  **alternative** URL that supports the claim, returns a verbatim quote; re-verified
  deterministically before accepting.
- **Case C — page live but does NOT support the claim:** escalate to WebSearch for an
  alternative source, else flag/downgrade.
- Backup KB first, idempotent, resumable per-crop (same pattern as the reverify array
  job). Regenerate `artifacts/PRESAGE_final_registry.xlsx` afterward.
- `claude` invocation pattern (headless, no API key), matching `scripts/_probe_verifier.py`:
  `claude -p "..." --output-format json --allowedTools WebSearch --append-system-prompt ... --max-turns 30`,
  env stripped of CLAUDE*/VSCODE*/ANTHROPIC_API_KEY.

---

## Findings so far

Link **liveness** is good — only ~5–7 genuinely dead URLs, ~300 publisher bot-blocks
(live in a browser). The real issue is **quote fidelity**: a large share of stored quotes
are LLM paraphrases/near-quotes, not verbatim page text. Phase B is quantifying the true
size after removing SPA false-negatives.

Concrete confirmed mismatch (survives full JS render): Soybean › Cercospora leaf blight
delta cites cropprotectionnetwork.org with quote *"…small, reddish-purple, angular to
irregular lesions…limited by the leaf vein"* — the real page says *"light purple, pinpoint
spots to larger, irregularly shaped patches."* → genuine paraphrase, needs a real quote.

---

## Status
- [x] Phase A built + run over full KB (2,031 URLs / 9,127 nodes).
- [~] Phase B render re-check **running** (headless Chromium, 1,745 URLs).
- [ ] Phase B true numbers → decide repair strategy (re-quote same page vs find alt link).
- [ ] Phase C repair + re-verify + Excel regen.

## Artifacts / scripts
- `scripts/verify_kb_quotes.py` · `scripts/render_recheck.py` · `scripts/repair_kb_quotes.py` (pending)
- `artifacts/quote_check.{json,csv}` — per-node report (open the CSV to inspect)
- `artifacts/page_cache/` — cached page text (`*.txt` requests, `render_*.txt` rendered)
- backups: `artifacts/pathome_kb_backup_pre_linkprune.tgz` (and repair will add its own)
