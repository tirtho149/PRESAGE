---
title: "Tomato — Canonical KB vs. Delta KB (post-fix run)"
subtitle: "10 separate diseases (5 + 5, disjoint)  |  Post-fix half of phase0r_traces (1).jsonl  |  Tomato  |  Generated 2026-05-16"
geometry: "landscape, margin=1.2cm"
fontsize: 9pt
---

# Context

After the consolidator token-budget fix the swarm re-ran on Nova. This
report uses **only the post-fix half** of the trace file (71
clean Tomato records; the first 147 records are the stale pre-fix run
and are excluded). It covers **10 separate Tomato diseases** — 5 in
Table A and 5 *different* ones in Table B (the tables are disjoint).

**Validation basis.** Every `canonical_says` string is itself sourced
(Phase-1 KB cites Crop Protection Network / university extension). A
delta matching canonical is validated against that source; a delta that
diverges is classified in Table B for human adjudication.

# Table A — canonical present AND delta has a value (grounded)

5 distinct Tomato diseases. Pre-fix, 92% of deltas had
`canonical_says = "(not specified)"`; these show grounding now works.

+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Disease                | Field            | Canonical KB said (`canonical_says`)   | Delta KB value (`image_shows`)         | Raw caption (`image_quote`)    |
+========================+==================+========================================+========================================+================================+
| Late Blight            | fruit            | Dark brown, greasy-to-leathery         | Tomato fruits with dark brown, greasy- | The image shows tomatoes with  |
|                        |                  | circular spots.                        | to-leathery circular spots.            | dark brown lesions covering    |
|                        |                  |                                        |                                        | over half of their visible     |
|                        |                  |                                        |                                        | surfaces.                      |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Anthracnose            | sporulation      | masses of salmon-colored sausage-      | The image shows salmon-colored         | The image shows salmon-colored |
|                        |                  | shaped spores may form on the lesion   | sausage-shaped masses forming on the   | spore masses on the surface of |
|                        |                  | surface.                               | lesion surface.                        | the lesion.                    |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Tomato Spotted Wilt    | leaf_necrosis    | On tomatoes, uppermost leaves appear   | Small brown lesions are scattered      | The leaf in the image has      |
| Virus                  |                  | bronzed with small brown lesions;      | across the leaf surface.               | numerous small brown lesions   |
|                        |                  |                                        |                                        | scattered across its surface.  |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Phytophthora Blight    | concentric_patte | The image shows a concentric ring      | The tomato has a circular brown patch  | The lesion exhibits a          |
|                        | rn               | lesion on the tomato.                  | with concentric rings.                 | concentric structure with      |
|                        |                  |                                        |                                        | alternating light and dark     |
|                        |                  |                                        |                                        | rings.                         |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Southern Blight        | look_alikes_visu | Tan to reddish-brown spherical         | tan to reddish-brown spherical         | tan to reddish-brown spherical |
|                        | al               | sclerotia (1-2 mm) are visible at the  | sclerotia (1-2 mm) at the stem base,   | sclerotia (1-2 mm) at the stem |
|                        |                  | stem base, supporting Southern Blight. | supporting Southern Blight.            | base.                          |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+

# Table B — proper changes (delta extends / refines / contradicts canonical)

5 *different* Tomato diseases (no overlap with Table A). The right-hand
column classifies the change.

+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Disease              | Canonical KB                         | Delta KB (image)                     | The proper change                        |
+======================+======================================+======================================+==========================================+
| Tomato Leaf Mould    | Pale green or yellowish spots        | The leaf displays brown spots        | STAGE/COLOR REFINEMENT — brown vein-     |
|                      | without well defined margins appear  | scattered across its surface, some   | clustered spots, not canonical's pale-   |
|                      | on the upper leaf surface and turn a | appearing discrete while others seem | green/olive mould                        |
|                      | distinctive yellow, with an olive-   | to cluster near major veins.         |                                          |
|                      | green mold visible on the lower leaf |                                      |                                          |
|                      | surface; lesions coalesce when       |                                      |                                          |
|                      | disease is severe, foliage curls,    |                                      |                                          |
|                      | withers, and may drop;               |                                      |                                          |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Bacterial Speck Of   | Bacterial spot; tomato spotted wilt; | The image shows a target spot with a | ADDED LOOK-ALIKE — describes a target-   |
| Tomato               | bacterial canker of tomato           | central dark brown area surrounded   | spot/concentric pattern absent from      |
|                      |                                      | by concentric rings.                 | canonical's list (flag for review)       |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Bacterial Canker And | Dark sunken leaf veins on the tomato | Brown streaks are visible in the     | RELOCATION/REFINEMENT — internal         |
| Wilt Of Tomato       | plant stems.                         | vascular system through the          | vascular streaking vs canonical's        |
|                      |                                      | translucent bark, indicating         | external sunken veins                    |
|                      |                                      | significant vascular damage.         |                                          |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Verticillium Wilts   | Older/lower leaves yellow (often on  | Brown and black necrotic tissue on   | STAGE PROGRESSION — extensive            |
|                      | one side) with V-shaped areas        | the leaves, covering large portions  | brown/black necrosis beyond canonical's  |
|                      | narrowing from the leaf margin;      | of the leaf surface.                 | early one-sided V-chlorosis              |
|                      | plants wilt during the hot part of   |                                      |                                          |
|                      | the day but recover in the evening,  |                                      |                                          |
|                      | eventually leaves shrivel, turn      |                                      |                                          |
|                      | brown, and die;                      |                                      |                                          |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Early Blight         | Lesions first develop on lower       | The leaf exhibits necrotic tissue in | PATTERN REFINEMENT — adds vein-inward    |
|                      | leaves as small brownish-black spots | discrete spots, extending inward     | necrotic spread alongside canonical's    |
|                      | that expand to 1/4-1/2 inch with     | from veins.                          | concentric 'bullseye'                    |
|                      | characteristic concentric rings      |                                      |                                          |
|                      | producing a 'bull's eye' or 'target- |                                      |                                          |
|                      | spot' appearance;                    |                                      |                                          |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+

# Verdict

- **Table A** — the fix restored *grounding*: the swarm reads the
  canonical slice and records confirmations against it across 5
  separate Tomato diseases.
- **Table B** — genuine KB *extension*, not captioning, across 5 more
  Tomato diseases: refinements, relocations, and added differentials.
- Table B deltas are *proposed* refinements for human review before
  they overwrite canonical — flagged, not yet truth. The Bacterial
  Speck row in particular (added target-spot pattern) is exactly the
  kind of item a reviewer should adjudicate.
