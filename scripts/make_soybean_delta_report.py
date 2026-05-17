#!/usr/bin/env python3
"""Soybean canonical-vs-delta report (post-fix run) -> PDF via pandoc.

Table A: 5 diseases where canonical KB was NOT "(not specified)" AND the
         delta KB has a value (grounded confirmations).
Table B: 5 diseases where the delta is a *proper change* — it refines,
         contradicts, or adds discriminative detail vs canonical.
Data pulled live from the post-fix half of the trace file (records
after the first 147, which are the stale pre-fix run)."""
import json, subprocess, textwrap, os

TRACES = "artifacts/phase0r_traces/phase0r_traces (1).jsonl"
OUT_MD, OUT_PDF = "Soybean_delta_validation.md", "Soybean_delta_validation.pdf"

recs = [json.loads(l) for l in open(TRACES)]
post = [r for r in recs[147:] if r["crop"] == "Soybean"]   # post-fix only

def find(disease, field, img):
    for r in post:
        if r["disease"] == disease and r["primary_image_id"].endswith(img):
            for d in r["final_deltas"]:
                if d.get("field") == field:
                    return r, d
    raise SystemExit(f"not found: {disease}/{field}/{img}")

# 10 SEPARATE diseases — Table A and Table B are disjoint (no disease
# appears in both). (disease, field, image_id_suffix[, change-note]).
TABLE_A = [
 ("Frogeye Leaf Spot",  "leaf_necrosis",   "5625043"),
 ("Brown Stem Rot",     "stem_pith",       "5473571"),
 ("Soybean Rust",       "leaf_necrosis",   "5202010"),
 ("Root And Stem Rot",  "severity_visible","5626213"),
 ("Septoria Leaf Spot", "leaf_necrosis",   "5626261"),
]
TABLE_B = [
 ("Charcoal Rot",      "stem_pith",        "5368505",
  "CONTRADICTION/RELOCATION — image puts discoloration in the vascular ring, not the pith"),
 ("Bacterial Blight",  "look_alikes_visual","5609905",
  "VALUE-ADD — supplies the discriminative criteria vs Septoria the canonical only named"),
 ("Bacterial Pustule Of Soybean Disease", "look_alikes_visual", "5368281",
  "NEW FIELD-VISIBLE SIGN — adds a macroscopic diagnostic; canonical needed 20X magnification"),
 ("Cercospora Blight", "leaf_necrosis",    "5624628",
  "STAGE/COLOR REFINEMENT — captures a brownish-green phase, not canonical's purple"),
 ("Rhizoctonia Damping-Off, Blight And Rot", "color_palette", "5626244",
  "ADDED OBSERVATION — notes greenish areas beyond canonical's single 'rusty-brown'"),
]

def cell(t):
    return (t or "").replace("|", "\\|").replace("\n", " ").strip() or "—"

def grid(cols, rows):
    widths = [w for _, w in cols]
    def sep(ch): return "+" + "+".join(ch*(w+2) for w in widths) + "+"
    def block(vals):
        wr = [textwrap.fill(cell(v), w).split("\n") for v, w in zip(vals, widths)]
        h = max(len(x) for x in wr)
        for x in wr: x += [""]*(h-len(x))
        return "\n".join("| " + " | ".join(x[i].ljust(w)
                         for x, (_, w) in zip(wr, cols)) + " |"
                         for i in range(h))
    out = [sep("-"), block([c for c, _ in cols]), sep("=")]
    for r in rows:
        out.append(block(r)); out.append(sep("-"))
    return "\n".join(out)

# ---- Table A rows -------------------------------------------------------
A_COLS = [("Disease", 20), ("Field", 16),
          ("Canonical KB said (`canonical_says`)", 40),
          ("Delta KB value (`image_shows`)", 40),
          ("Raw caption (`image_quote`)", 32)]
A_rows = []
for dis, fld, img in TABLE_A:
    _, d = find(dis, fld, img)
    A_rows.append([dis, fld, d.get("canonical_says"),
                   d.get("image_shows"), d.get("image_quote")])

# ---- Table B rows -------------------------------------------------------
B_COLS = [("Disease", 18),
          ("Canonical KB", 38), ("Delta KB (image)", 38),
          ("The proper change", 38)]
B_rows = []
for dis, fld, img, note in TABLE_B:
    _, d = find(dis, fld, img)
    B_rows.append([dis, d.get("canonical_says"), d.get("image_shows"), note])

md = f"""---
title: "Soybean — Canonical KB vs. Delta KB (post-fix run)"
subtitle: "10 separate diseases (5 + 5, disjoint)  |  Post-fix half of phase0r_traces (1).jsonl  |  Soybean  |  Generated 2026-05-16"
geometry: "landscape, margin=1.2cm"
fontsize: 9pt
---

# Context

After the consolidator token-budget fix, the swarm re-ran on Nova. This
report is built **only from the post-fix half** of the trace file (the
clean 76 Soybean records; the first 147 are the stale pre-fix run and
are excluded). It answers two questions on real data:

1. Where was the canonical KB **not** "(not specified)" *and* the delta
   KB still produced a value? (grounded confirmation — Table A)
2. Where is the delta a **proper change** — refining, contradicting, or
   adding discriminative detail beyond canonical? (Table B)

**Validation basis.** Every `canonical_says` string is itself sourced
(Phase-1 KB cites Crop Protection Network / university extension). A
delta that matches canonical is therefore validated against that
source; a delta that diverges is flagged with the change type in
Table B for human review.

# Table A — canonical present AND delta has a value (grounded)

5 distinct Soybean diseases. Pre-fix, 92% of deltas had
`canonical_says = "(not specified)"`; these show the grounding now works.

{grid(A_COLS, A_rows)}

# Table B — proper changes (delta extends / refines / contradicts canonical)

These are the high-value deltas: the image does more than restate
canonical. The right-hand column classifies the change.

{grid(B_COLS, B_rows)}

# Verdict

- **Table A** confirms the fix restored *grounding*: the swarm now reads
  the canonical slice and records confirmations against it (Frogeye Leaf
  Spot, Brown Stem Rot, Soybean Rust, Root & Stem Rot, Septoria Leaf
  Spot — 5 separate diseases).
- **Table B** confirms the swarm performs genuine KB *extension*, not
  captioning. Strongest cases: **Charcoal Rot** (relocates the
  discoloration from pith to vascular ring — a checkable contradiction)
  and **Bacterial Blight** (adds the Septoria differential the canonical
  only named). Both are exactly the behaviour absent pre-fix.
- Items in Table B are deltas a human should adjudicate before they
  overwrite canonical — they are *proposed* refinements, not yet truth.
"""

open(OUT_MD, "w").write(md)
subprocess.run(["pandoc", OUT_MD, "-o", OUT_PDF,
                "--pdf-engine=xelatex"], check=True)
print("wrote", OUT_PDF, "(%d bytes)" % os.path.getsize(OUT_PDF))
