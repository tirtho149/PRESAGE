---
title: "Canonical KB vs. Swarm Delta vs. Raw Caption — Validation Report"
subtitle: "Disease: Soybean :: Southern Blight  |  Image: bugwood::5581651  |  Generated 2026-05-16"
geometry: "landscape, margin=1.2cm"
fontsize: 8pt
mainfont: "Helvetica"
---

# 1. Disease & canonical KB (Phase-1, on disk)

- **Disease:** Southern Blight  |  **Pathogen:** Agroathelia rolfsii  |  **Type:** Fungal
- **Affected parts:** Foliar, Stem, Root, Whole plant  |  **KB confidence:** high  |  **# sources:** 2

**Canonical `visual_symptoms.summary`:**

> Seedling infection causes pre- or post-emergence damping-off; later in the season entire plants yellow and wilt with leaves turning brown and remaining attached, accompanied by a dark brown girdling lesion at the soil surface and white fanlike mycelium with small round sclerotia on the lower stem.

**Canonical `diagnostic_features`:**

> Conspicuous white, fanlike mats of fungal mycelium at the base of the stem with numerous mustard-seed-sized sclerotia that progress from yellow-tan to reddish-brown to dark brown at maturity.

**Canonical source quote** (Crop Protection Network):

> "Seedling infection results in pre- or post-emergence damping-off. Later in the season, entire plants may become yellow and wilt, with leaves turning brown and often remaining attached to the plant. A dark brown lesion that girdles the stem occurs at the soil surface. This lesion is generally accompanied by the development of conspicuous white, fanlike mats of fungal mycelium that form on the base of the stem, on leaf residue, and on the soil surface around infected plants. Numerous, small round fungal bodies that are about the size of mustard seeds (called sclerotia) form on these fungal mats and on the lower stem."

# 2. Per-specialist table — canonical vs. delta vs. raw caption vs. validation

*Image bugwood::5581651, all specialist outputs, both rounds. Note how `canonical_says`
is "(not specified)" in round 1 even though the canonical KB above clearly
contains these facts — the canonical slice is not reaching round-1 agents,
so confirmations are mis-logged as net-new deltas.*

+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| Specialist (round,     | Field          | Canonical KB passed to agent   | DELTA added by specialist          | Raw caption (`image_quote`)    | Online-source validation       |
| conf)                  |                | (`canonical_says`)             | (`image_shows`)                    |                                |                                |
+========================+================+================================+====================================+================================+================================+
| RootAgent (R1, high)   | root_visible   | (not specified)                | The image shows the roots of a     | The image displays a close-up  | N/A — organ-presence only, no  |
|                        |                |                                | plant.                             | view of the roots of a plant.  | symptom claim (not             |
|                        |                |                                |                                    |                                | validatable)                   |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| ColorPaletteAgent (R1, | color_palette  | (not specified)                | The dominant color is tan,         | The roots appear tan and       | SUPPORTED — sclerotia          |
| high)                  |                |                                | representing the affected root     | covered with a fanlike         | white->tan->reddish-brown is   |
|                        |                |                                | tissue, with a secondary color of  | structure of rust-orange,      | documented [CPN, NCState,      |
|                        |                |                                | rust-orange representing the       | indicating sclerotia.          | Bugwood]                       |
|                        |                |                                | sclerotia.                         |                                |                                |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| SeverityVisualAgent    | severity_visib | (not specified)                | The image shows extensive mycelial | The plant has a tangled mass   | SUPPORTED — abundant white     |
| (R1, high)             | le             |                                | growth covering much of the stem.  | of white, fan-like mycelium.   | mycelium on lower stem         |
|                        |                |                                |                                    |                                | documented [APS, MSU]          |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| SporulationAgent (R1,  | sporulation    | (not specified)                | White, fanlike mycelium with small | white fanlike mycelium with    | STRONGLY SUPPORTED —           |
| high)                  |                |                                | round sclerotia on the lower stem. | small round sclerotia on the   | diagnostic hallmark, verbatim  |
|                        |                |                                |                                    | lower stem                     | match [APS, CPN, Wisconsin]    |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| RootAgent (R2, high)   | root_visible   | (not specified)                | The image displays a plant's root  | The photo shows a plant's root | N/A — organ-presence only, no  |
|                        |                |                                | system.                            | system with visible roots.     | symptom claim (not             |
|                        |                |                                |                                    |                                | validatable)                   |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| ColorPaletteAgent (R2, | color_palette  | dark brown girdling lesion at  | The affected roots show a mix of   | The roots exhibit a complex    | SUPPORTED — dark-brown         |
| high)                  |                | the soil surface               | brown and tan colors.              | pattern of brown and tan hues. | girdling lesion + white        |
|                        |                |                                |                                    |                                | mycelium documented [APS, CPN] |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| ColorPaletteAgent (R2, | color_palette  | (not specified)                | The mycelium appears white and     | The mycelium is predominantly  | SUPPORTED — dark-brown         |
| high)                  |                |                                | fanlike with a hint of off-white.  | white with some off-white      | girdling lesion + white        |
|                        |                |                                |                                    | tones.                         | mycelium documented [APS, CPN] |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| SeverityVisualAgent    | severity_visib | White, fanlike mycelium with   | The image shows extensive white,   | The mycelium covers a large    | STRONGLY SUPPORTED — textbook  |
| (R2, high)             | le             | small round sclerotia on the   | fanlike mycelium with numerous     | portion of the visible root    | hallmark, exact match [APS,    |
|                        |                | lower stem.                    | small round sclerotia on the lower | structure.                     | CPN, NCState]                  |
|                        |                |                                | stem.                              |                                |                                |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| LookAlikeCoTAgent (R2, | look_alikes_vi | (not specified)                | The photo shows extensive mycelial | The image shows extensive      | WEAK — generic restatement, no |
| high)                  | sual           |                                | growth covering much of the stem.  | mycelial growth covering much  | look-alike differential        |
|                        |                |                                |                                    | of the stem.                   | offered                        |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+
| SporulationAgent (R2,  | sporulation    | (not specified)                | Mycelium is visible on the roots,  | The image shows extensive      | SUPPORTED — mycelium on lower  |
| high)                  |                |                                | extending upwards towards the      | mycelial growth covering much  | stem/soil/roots documented     |
|                        |                |                                | stem.                              | of the stem.                   | [APS, CPN]                     |
+------------------------+----------------+--------------------------------+------------------------------------+--------------------------------+--------------------------------+

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
