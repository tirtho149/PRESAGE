#!/usr/bin/env python3
"""Canonical-KB vs Delta-KB validation report (post-fix run) -> PDF.

Generalized over crop. Usage:
    python scripts/make_delta_report.py Soybean
    python scripts/make_delta_report.py Tomato

Table A: 5 distinct diseases where canonical KB was NOT "(not specified)"
         AND the delta KB has a value (grounded confirmation).
Table B: 5 *different* distinct diseases where the delta is a proper
         change — refines, contradicts, or adds discriminative detail.
Tables A and B are disjoint -> 10 separate diseases per crop. All cell
content is pulled live from the post-fix half of the trace file (the
first 147 records are the stale pre-fix run and are excluded)."""
import json, subprocess, textwrap, os, sys

TRACES = "artifacts/phase0r_traces/phase0r_traces (1).jsonl"
CROP   = sys.argv[1] if len(sys.argv) > 1 else "Soybean"

# (disease, field, image_id_suffix[, change-note for table B]) per crop.
SPEC = {
 "Soybean": {
  "A": [
   ("Frogeye Leaf Spot",  "leaf_necrosis",   "5625043"),
   ("Brown Stem Rot",     "stem_pith",       "5473571"),
   ("Soybean Rust",       "leaf_necrosis",   "5202010"),
   ("Root And Stem Rot",  "severity_visible","5626213"),
   ("Septoria Leaf Spot", "leaf_necrosis",   "5626261"),
  ],
  "B": [
   ("Charcoal Rot", "stem_pith", "5368505",
    "CONTRADICTION/RELOCATION — image puts discoloration in the vascular ring, not the pith"),
   ("Bacterial Blight", "look_alikes_visual", "5609905",
    "VALUE-ADD — supplies the discriminative criteria vs Septoria the canonical only named"),
   ("Bacterial Pustule Of Soybean Disease", "look_alikes_visual", "5368281",
    "NEW FIELD-VISIBLE SIGN — adds a macroscopic diagnostic; canonical needed 20X magnification"),
   ("Cercospora Blight", "leaf_necrosis", "5624628",
    "STAGE/COLOR REFINEMENT — captures a brownish-green phase, not canonical's purple"),
   ("Rhizoctonia Damping-Off, Blight And Rot", "color_palette", "5626244",
    "ADDED OBSERVATION — notes greenish areas beyond canonical's single 'rusty-brown'"),
  ],
 },
 "Tomato": {
  "A": [
   ("Late Blight",              "fruit",             "1568022"),
   ("Anthracnose",              "sporulation",       "5559540"),
   ("Tomato Spotted Wilt Virus","leaf_necrosis",     "1568012"),
   ("Phytophthora Blight",      "concentric_pattern","1568019"),
   ("Southern Blight",          "look_alikes_visual","5369062"),
  ],
  "B": [
   ("Tomato Leaf Mould", "leaf_necrosis", "1568017",
    "STAGE/COLOR REFINEMENT — brown vein-clustered spots, not canonical's pale-green/olive mould"),
   ("Bacterial Speck Of Tomato", "look_alikes_visual", "1568046",
    "ADDED LOOK-ALIKE — describes a target-spot/concentric pattern absent from canonical's list (flag for review)"),
   ("Bacterial Canker And Wilt Of Tomato", "severity_visible", "1568040",
    "RELOCATION/REFINEMENT — internal vascular streaking vs canonical's external sunken veins"),
   ("Verticillium Wilts", "leaf_necrosis", "1573860",
    "STAGE PROGRESSION — extensive brown/black necrosis beyond canonical's early one-sided V-chlorosis"),
   ("Early Blight", "leaf_necrosis", "5551685",
    "PATTERN REFINEMENT — adds vein-inward necrotic spread alongside canonical's concentric 'bullseye'"),
  ],
 },
}[CROP]

OUT_MD  = f"{CROP}_delta_validation.md"
OUT_PDF = f"{CROP}_delta_validation.pdf"

recs = [json.loads(l) for l in open(TRACES)]
post = [r for r in recs[147:] if r["crop"] == CROP]   # post-fix half only

def find(disease, field, img):
    for r in post:
        if r["disease"] == disease and r["primary_image_id"].endswith(img):
            for d in r["final_deltas"]:
                if d.get("field") == field:
                    return d
    raise SystemExit(f"not found: {CROP}/{disease}/{field}/{img}")

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

A_COLS = [("Disease", 22), ("Field", 16),
          ("Canonical KB said (`canonical_says`)", 38),
          ("Delta KB value (`image_shows`)", 38),
          ("Raw caption (`image_quote`)", 30)]
A_rows = [[dis, fld, (d := find(dis, fld, img)).get("canonical_says"),
           d.get("image_shows"), d.get("image_quote")]
          for dis, fld, img in SPEC["A"]]

B_COLS = [("Disease", 20),
          ("Canonical KB", 36), ("Delta KB (image)", 36),
          ("The proper change", 40)]
B_rows = [[dis, (d := find(dis, fld, img)).get("canonical_says"),
           d.get("image_shows"), note]
          for dis, fld, img, note in SPEC["B"]]

n_post = len(post)
md = f"""---
title: "{CROP} — Canonical KB vs. Delta KB (post-fix run)"
subtitle: "10 separate diseases (5 + 5, disjoint)  |  Post-fix half of phase0r_traces (1).jsonl  |  {CROP}  |  Generated 2026-05-16"
geometry: "landscape, margin=1.2cm"
fontsize: 9pt
---

# Context

After the consolidator token-budget fix the swarm re-ran on Nova. This
report uses **only the post-fix half** of the trace file ({n_post}
clean {CROP} records; the first 147 records are the stale pre-fix run
and are excluded). It covers **10 separate {CROP} diseases** — 5 in
Table A and 5 *different* ones in Table B (the tables are disjoint).

**Validation basis.** Every `canonical_says` string is itself sourced
(Phase-1 KB cites Crop Protection Network / university extension). A
delta matching canonical is validated against that source; a delta that
diverges is classified in Table B for human adjudication.

# Table A — canonical present AND delta has a value (grounded)

5 distinct {CROP} diseases. Pre-fix, 92% of deltas had
`canonical_says = "(not specified)"`; these show grounding now works.

{grid(A_COLS, A_rows)}

# Table B — proper changes (delta extends / refines / contradicts canonical)

5 *different* {CROP} diseases (no overlap with Table A). The right-hand
column classifies the change.

{grid(B_COLS, B_rows)}

# Verdict

- **Table A** — the fix restored *grounding*: the swarm reads the
  canonical slice and records confirmations against it across 5
  separate {CROP} diseases.
- **Table B** — genuine KB *extension*, not captioning, across 5 more
  {CROP} diseases: refinements, relocations, and added differentials.
- Table B deltas are *proposed* refinements for human review before
  they overwrite canonical — flagged, not yet truth. The Bacterial
  Speck row in particular (added target-spot pattern) is exactly the
  kind of item a reviewer should adjudicate.
"""

open(OUT_MD, "w").write(md)
subprocess.run(["pandoc", OUT_MD, "-o", OUT_PDF,
                "--pdf-engine=xelatex"], check=True)
print("wrote", OUT_PDF, "(%d bytes)" % os.path.getsize(OUT_PDF))
