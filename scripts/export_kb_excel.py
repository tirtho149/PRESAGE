#!/usr/bin/env python3
"""Export the PRESAGE final registry to a single complete Excel workbook.

Sheets:
  1. Summary        — crop/disease/delta counts + verification-status breakdown
  2. Canonical_KB   — initial/canonical KB, one row per (crop, disease), with source URLs
  3. Regional_Deltas— per-delta regional observations w/ verification_status + web-cited support
"""
import json, glob, os, collections
import pandas as pd

KB = "artifacts/pathome_kb"
OUT = "artifacts/PRESAGE_final_registry.xlsx"
LINKCHECK = "artifacts/link_check.json"   # produced by scripts/verify_kb_links.py

# load link verification (url -> status) if available; else everything = "unchecked"
_lc = json.load(open(LINKCHECK)) if os.path.exists(LINKCHECK) else {}
def link_status(url):
    if not url:
        return ""
    return (_lc.get(url) or {}).get("status", "unchecked")
def worst_status(urls):
    """Worst link status across a set of URLs (dead > unknown > blocked > alive)."""
    rank = {"dead": 3, "unknown": 2, "blocked": 1, "alive": 0, "unchecked": 0, "": 0}
    if not urls:
        return ""
    st = max((link_status(u) for u in urls), key=lambda s: rank.get(s, 0))
    return st


def s(x):
    """Render a value/list/dict field into a readable string."""
    if x is None:
        return ""
    if isinstance(x, dict):
        # canonical fields look like {"value":..., "url":..., "quote":...} or {"summary": {...}}
        if "value" in x:
            return join_list(x["value"])
        if "summary" in x and isinstance(x["summary"], dict):
            return join_list(x["summary"].get("value"))
        return json.dumps(x, ensure_ascii=False)
    return join_list(x)


def join_list(v):
    if isinstance(v, list):
        return "; ".join(str(i) for i in v)
    return "" if v is None else str(v)


def url_of(x):
    if isinstance(x, dict):
        return x.get("url", "")
    return ""


def web_urls(ws):
    if not ws:
        return "", ""
    urls = " | ".join(w.get("url", "") for w in ws if w.get("url"))
    quotes = " || ".join(w.get("quote", "") for w in ws if w.get("quote"))
    return urls, quotes


canon_rows, delta_rows = [], []
status_counts = {}
crops = 0

for f in sorted(glob.glob(f"{KB}/*/final_registry.json")):
    d = json.load(open(f))
    crop = d.get("crop", os.path.basename(os.path.dirname(f)))
    crops += 1
    for dz in d.get("diseases", []):
        canon_rows.append({
            "crop": crop,
            "disease": dz.get("disease_name", ""),
            "pathogen": s(dz.get("pathogen_scientific_name")),
            "type": s(dz.get("type_of_disease")),
            "affected_parts": s(dz.get("affected_parts")),
            "visual_symptoms": s(dz.get("visual_symptoms")),
            "treatments": s(dz.get("treatments")),
            "confidence": dz.get("confidence", ""),
            "num_sources": dz.get("num_sources", ""),
            "pathogen_source_url": url_of(dz.get("pathogen_scientific_name")),
            "pathogen_link_status": link_status(url_of(dz.get("pathogen_scientific_name"))),
            "type_source_url": url_of(dz.get("type_of_disease")),
            "type_link_status": link_status(url_of(dz.get("type_of_disease"))),
            "n_conflicts": len(dz.get("conflicts") or []),
            "n_regional_deltas": sum(len(st.get("deltas", []))
                                     for st in (dz.get("regional_observations") or {}).values()),
        })
        for state, st in (dz.get("regional_observations") or {}).items():
            for dl in st.get("deltas", []):
                stt = dl.get("verification_status", "")
                status_counts[stt] = status_counts.get(stt, 0) + 1
                urls, quotes = web_urls(dl.get("web_support"))
                _url_list = [w.get("url") for w in (dl.get("web_support") or []) if w.get("url")]
                _dead = [u for u in _url_list if link_status(u) == "dead"]
                delta_rows.append({
                    "crop": crop,
                    "disease": dz.get("disease_name", ""),
                    "state": st.get("state", state),
                    "field": dl.get("field", ""),
                    "verification_status": stt,
                    "web_cited": "yes" if dl.get("web_support") else "no",
                    "canonical_says": dl.get("canonical_says", ""),
                    "image_shows": dl.get("image_shows", ""),
                    "image_quote": dl.get("image_quote", ""),
                    "image_id": dl.get("image_id", ""),
                    "reasoning": dl.get("reasoning", ""),
                    "web_link_status": worst_status(_url_list),
                    "dead_urls": " | ".join(_dead),
                    "web_support_urls": urls,
                    "web_support_quotes": quotes,
                    "swarm_support": json.dumps(dl.get("swarm_support"), ensure_ascii=False)
                                     if dl.get("swarm_support") else "",
                    "prior_status": dl.get("_prior_status", ""),
                })

canon = pd.DataFrame(canon_rows)
deltas = pd.DataFrame(delta_rows)

web_cited = int((deltas["web_cited"] == "yes").sum()) if len(deltas) else 0
summary_rows = [
    ("Crops", crops),
    ("Diseases (canonical KB entries)", len(canon)),
    ("Regional deltas (total)", len(deltas)),
    ("Regional deltas — web-cited", web_cited),
    ("Regional deltas — not web-cited", len(deltas) - web_cited),
    ("", ""),
    ("--- Delta verification_status breakdown ---", ""),
]
for k in sorted(status_counts, key=lambda x: -status_counts[x]):
    summary_rows.append((k, status_counts[k]))

# link verification rollup (covers BOTH canonical KB + delta KB urls)
if _lc:
    lc_status = collections.Counter(v.get("status", "unchecked") for v in _lc.values())
    summary_rows += [
        ("", ""),
        ("--- Link verification (unique URLs, both KBs) ---", ""),
        ("URLs alive (2xx/3xx)", lc_status.get("alive", 0)),
        ("URLs blocked (403/429, bot-blocked but live)", lc_status.get("blocked", 0)),
        ("URLs DEAD (404/410/DNS)", lc_status.get("dead", 0)),
        ("URLs unknown (timeout/SSL)", lc_status.get("unknown", 0)),
        ("URLs total unique", sum(lc_status.values())),
    ]
summary = pd.DataFrame(summary_rows, columns=["metric", "count"])

# dead-links sheet
dead_rows = [{"url": u, "code": v.get("code"), "occurrences": v.get("count"),
              "kinds": ";".join(v.get("kinds", []))}
             for u, v in sorted(_lc.items()) if v.get("status") == "dead"]
dead_df = pd.DataFrame(dead_rows) if dead_rows else pd.DataFrame(
    columns=["url", "code", "occurrences", "kinds"])

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    summary.to_excel(xw, sheet_name="Summary", index=False)
    canon.to_excel(xw, sheet_name="Canonical_KB", index=False)
    deltas.to_excel(xw, sheet_name="Regional_Deltas", index=False)
    dead_df.to_excel(xw, sheet_name="Dead_Links", index=False)
    # widen columns a little for readability
    for name, df in [("Summary", summary), ("Canonical_KB", canon),
                     ("Regional_Deltas", deltas), ("Dead_Links", dead_df)]:
        ws = xw.sheets[name]
        for i, col in enumerate(df.columns, 1):
            width = min(60, max(12, int(df[col].astype(str).str.len().quantile(0.9)) + 2)) if len(df) else 20
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width

print(f"WROTE {OUT}")
print(f"  crops={crops}  diseases={len(canon)}  deltas={len(deltas)}  web_cited={web_cited}")
print(f"  status={status_counts}")
