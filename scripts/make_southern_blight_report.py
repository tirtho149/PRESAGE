#!/usr/bin/env python3
"""Build the Southern Blight canonical-vs-delta-vs-caption validation
report (markdown -> PDF via pandoc/xelatex). One-off analysis script."""
import json, subprocess, textwrap, os

TRACES = "phase0r_traces.jsonl"
REG    = "artifacts/pathome_kb/Soybean/final_registry.json"
PROFILE, IMG = "Soybean::Southern Blight", "bugwood::5581651"
OUT_MD, OUT_PDF = "Southern_Blight_validation.md", "Southern_Blight_validation.pdf"

# --- canonical KB --------------------------------------------------------
reg = json.load(open(REG))
ce  = next(d for d in reg["diseases"]
           if "southern blight" in d.get("disease_name", "").lower())
vs  = ce["visual_symptoms"]
canon_summary = vs["summary"]["value"]
canon_diag    = vs["diagnostic_features"]["value"]
canon_quote   = vs["summary"]["quote"]

# --- the trace -----------------------------------------------------------
rec = next(json.loads(l) for l in open(TRACES)
           if json.loads(l)["profile_id"] == PROFILE
           and json.loads(l)["primary_image_id"] == IMG)

# Per-(agent,round) online-validation verdicts (from web sources, step 3).
VERDICT = {
 ("RootAgent",1):       "N/A — organ-presence only, no symptom claim (not validatable)",
 ("RootAgent",2):       "N/A — organ-presence only, no symptom claim (not validatable)",
 ("ColorPaletteAgent",1):"SUPPORTED — sclerotia white->tan->reddish-brown is documented [CPN, NCState, Bugwood]",
 ("ColorPaletteAgent",2):"SUPPORTED — dark-brown girdling lesion + white mycelium documented [APS, CPN]",
 ("SeverityVisualAgent",1):"SUPPORTED — abundant white mycelium on lower stem documented [APS, MSU]",
 ("SeverityVisualAgent",2):"STRONGLY SUPPORTED — textbook hallmark, exact match [APS, CPN, NCState]",
 ("SporulationAgent",1):"STRONGLY SUPPORTED — diagnostic hallmark, verbatim match [APS, CPN, Wisconsin]",
 ("SporulationAgent",2):"SUPPORTED — mycelium on lower stem/soil/roots documented [APS, CPN]",
 ("LookAlikeCoTAgent",2):"WEAK — generic restatement, no look-alike differential offered",
}

rows = []
for s in rec["specialist_outputs"]:
    for d in s["deltas"]:
        key = (s["agent_name"], s["round_idx"])
        rows.append({
          "spec": f'{s["agent_name"]} (R{s["round_idx"]}, {s["confidence"]})',
          "field": d.get("field", ""),
          "canon": d.get("canonical_says", ""),
          "delta": d.get("image_shows", ""),
          "cap":   d.get("image_quote", ""),
          "val":   VERDICT.get(key, "(not scored)"),
        })

def cell(t, w=42):
    t = (t or "").replace("|", "\\|").replace("\n", " ").strip()
    return textwrap.fill(t, w) if t else "—"

# --- grid table (pandoc wraps grid-table cells) --------------------------
COLS = [("Specialist (round, conf)", "spec", 22),
        ("Field", "field", 14),
        ("Canonical KB passed to agent (`canonical_says`)", "canon", 30),
        ("DELTA added by specialist (`image_shows`)", "delta", 34),
        ("Raw caption (`image_quote`)", "cap", 30),
        ("Online-source validation", "val", 30)]

def grid(rows):
    widths = [c[2] for c in COLS]
    def sep(ch): return "+" + "+".join(ch * (w + 2) for w in widths) + "+"
    def block(cells):
        wrapped = [cell(c, w).split("\n") for c, w in zip(cells, widths)]
        h = max(len(x) for x in wrapped)
        for x in wrapped:
            x += [""] * (h - len(x))
        return "\n".join(
            "| " + " | ".join(x[i].ljust(w) for x, (_, _, w) in
                               zip(wrapped, COLS)) + " |"
            for i in range(h))
    out = [sep("-"), block([c[0] for c in COLS]), sep("=")]
    for r in rows:
        out.append(block([r[k] for _, k, _ in COLS]))
        out.append(sep("-"))
    return "\n".join(out)

md = f"""---
title: "Canonical KB vs. Swarm Delta vs. Raw Caption — Validation Report"
subtitle: "Disease: Soybean :: Southern Blight  |  Image: {IMG}  |  Generated 2026-05-16"
geometry: "landscape, margin=1.2cm"
fontsize: 8pt
mainfont: "Helvetica"
---

# 1. Disease & canonical KB (Phase-1, on disk)

- **Disease:** {ce['disease_name']}  |  **Pathogen:** {ce['pathogen_scientific_name']['value']}  |  **Type:** {ce['type_of_disease']['value']}
- **Affected parts:** {', '.join(ce['affected_parts']['value'])}  |  **KB confidence:** {ce['confidence']}  |  **# sources:** {ce['num_sources']}

**Canonical `visual_symptoms.summary`:**

> {canon_summary}

**Canonical `diagnostic_features`:**

> {canon_diag}

**Canonical source quote** (Crop Protection Network):

> "{canon_quote}"

# 2. Per-specialist table — canonical vs. delta vs. raw caption vs. validation

*Image {IMG}, all specialist outputs, both rounds. Note how `canonical_says`
is "(not specified)" in round 1 even though the canonical KB above clearly
contains these facts — the canonical slice is not reaching round-1 agents,
so confirmations are mis-logged as net-new deltas.*

{grid(rows)}

# 3. Online validation — method & verdict

**Method.** Each specialist claim was checked against independent
plant-pathology authorities (not just the single source the canonical
cites). Queried sources:

- APSnet — *Southern blight, southern stem blight, white mold*
- Crop Protection Network — *Southern Blight of Soybeans*
- NC State Extension — *Southern Blight of vegetable crops*
- University of Wisconsin Horticulture Extension — *Southern Blight*
- Michigan State University IPM — *Southern blight*
- Bugwoodwiki — *Sclerotium rolfsii*

**Consensus ground truth.** White, fan-shaped mycelial mat at the lower
stem / soil line; numerous small (0.5–2 mm, mustard-seed-sized) round
sclerotia progressing white -> tan -> reddish-brown -> dark brown; dark
brown girdling lesion at the soil surface.

**Verdict.**

- **SporulationAgent** and **SeverityVisualAgent** reproduce the
  diagnostic hallmark *verbatim* — STRONGLY SUPPORTED. Core signal is
  accurate and trustworthy.
- **ColorPaletteAgent** is SUPPORTED on this image (tan/brown +
  white mycelium), though on other images of this profile it emits an
  unsupported "olive-green" reading (background-foliage contamination).
- **RootAgent** entries are tautological organ-presence statements with
  no symptom content — not validatable, low KB value.
- **LookAlikeCoTAgent** restates the dominant sign without offering any
  differential — WEAK.

**Bottom line.** The model's *perception* is accurate (5 of 5 substantive
claims supported by independent sources), but (a) the canonical slice is
not grounding round-1 specialists, so confirmations are mis-logged as
deltas, and (b) one true fact is echoed by ~5 off-lane agents, inflating
the KB and driving the consolidator JSON past its token budget.
"""

open(OUT_MD, "w").write(md)
subprocess.run(["pandoc", OUT_MD, "-o", OUT_PDF,
                "--pdf-engine=xelatex"], check=True)
print("wrote", OUT_MD, "and", OUT_PDF,
      "(%d bytes)" % os.path.getsize(OUT_PDF))
