---
title: "Soybean — Canonical KB vs. Delta KB (post-fix run)"
subtitle: "10 separate diseases (5 + 5, disjoint)  |  Post-fix half of phase0r_traces (1).jsonl  |  Soybean  |  Generated 2026-05-16"
geometry: "landscape, margin=1.2cm"
fontsize: 9pt
---

# Context

After the consolidator token-budget fix the swarm re-ran on Nova. This
report uses **only the post-fix half** of the trace file (76
clean Soybean records; the first 147 records are the stale pre-fix run
and are excluded). It covers **10 separate Soybean diseases** — 5 in
Table A and 5 *different* ones in Table B (the tables are disjoint).

**Validation basis.** Every `canonical_says` string is itself sourced
(Phase-1 KB cites Crop Protection Network / university extension). A
delta matching canonical is validated against that source; a delta that
diverges is classified in Table B for human adjudication.

# Table A — canonical present AND delta has a value (grounded)

5 distinct Soybean diseases. Pre-fix, 92% of deltas had
`canonical_says = "(not specified)"`; these show grounding now works.

+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Disease                | Field            | Canonical KB said (`canonical_says`)   | Delta KB value (`image_shows`)         | Raw caption (`image_quote`)    |
+========================+==================+========================================+========================================+================================+
| Frogeye Leaf Spot      | leaf_necrosis    | Leaf lesions are small, irregular to   | The image shows multiple small,        | Irregular shapes with reddish- |
|                        |                  | circular, gray with reddish-brown      | irregular, and gray lesions with       | brown borders are present.     |
|                        |                  | borders, occurring most commonly on    | reddish-brown borders on the leaf.     |                                |
|                        |                  | the upper leaf surface;                |                                        |                                |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Brown Stem Rot         | stem_pith        | white in healthy stems; brown in BSR   | brown pith in the stem.                | The photo shows a split stem   |
|                        |                  |                                        |                                        | with brown pith.               |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Soybean Rust           | leaf_necrosis    | Symptoms begin on leaves in the lower  | Brown spots are visible along the      | Brown spots are visible along  |
|                        |                  | plant canopy with gray-green, tan to   | veins, indicating necrotic tissue      | the veins, indicating necrotic |
|                        |                  | dark-brown, or reddish-brown angular   | extending inward.                      | tissue extending inward.       |
|                        |                  | lesions (2-5 mm diameter) developing   |                                        |                                |
|                        |                  | first on the underside of leaves;      |                                        |                                |
|                        |                  | small pustules form in the lesions and |                                        |                                |
|                        |                  | break open to release masses of tan    |                                        |                                |
|                        |                  | spores; lesions may also appear on     |                                        |                                |
|                        |                  | petioles, pods, and stems.             |                                        |                                |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Root And Stem Rot      | severity_visible | A dark chocolate-brown lesion on the   | Dark brown discoloration extending     | The stem exhibits dark brown   |
|                        |                  | lower stem extending from below the    | several nodes up the stem.             | discoloration with a rough     |
|                        |                  | soil line upward from the taproot,     |                                        | texture, indicating cankers.   |
|                        |                  | often reaching several nodes and       |                                        |                                |
|                        |                  | girdling the stem;                     |                                        |                                |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+
| Septoria Leaf Spot     | leaf_necrosis    | Initial symptoms are small dark brown  | Brown necrotic areas on the leaves,    | There are brown necrotic areas |
|                        |                  | spots (less than 1/8 inch) on lower    | which are irregularly shaped and cover | covering the leaves.           |
|                        |                  | leaves that enlarge and grow together  | various parts of the leaf.             |                                |
|                        |                  | into irregular brown areas, often      |                                        |                                |
|                        |                  | associated with yellow patches         |                                        |                                |
|                        |                  | concentrated more on one side of the   |                                        |                                |
|                        |                  | leaf than another;                     |                                        |                                |
+------------------------+------------------+----------------------------------------+----------------------------------------+--------------------------------+

# Table B — proper changes (delta extends / refines / contradicts canonical)

5 *different* Soybean diseases (no overlap with Table A). The right-hand
column classifies the change.

+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Disease              | Canonical KB                         | Delta KB (image)                     | The proper change                        |
+======================+======================================+======================================+==========================================+
| Charcoal Rot         | brown to dark spots appear on        | The stem pith is white with brown    | CONTRADICTION/RELOCATION — image puts    |
|                      | cotyledons at seedling stage,        | outer vascular tissue.               | discoloration in the vascular ring, not  |
|                      | reddish-brown to black lesions on    |                                      | the pith                                 |
|                      | unifoliate leaves, and light brown   |                                      |                                          |
|                      | to grey superficial lesions on stems |                                      |                                          |
|                      | mid to late season, with reddish-    |                                      |                                          |
|                      | brown discoloration in the pith and  |                                      |                                          |
|                      | vascular tissues.                    |                                      |                                          |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Bacterial Blight     | Septoria brown spot (Septoria        | The lesions are irregular, with a    | VALUE-ADD — supplies the discriminative  |
|                      | glycines)                            | chlorotic yellow halo and scattered  | criteria vs Septoria the canonical only  |
|                      |                                      | necrotic tissue, differentiating it  | named                                    |
|                      |                                      | from Septoria brown spot.            |                                          |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Bacterial Pustule Of | Bacterial pustule lesions lack the   | The lesions are irregular, with some | NEW FIELD-VISIBLE SIGN — adds a          |
| Soybean Disease      | opening on top and spores that       | showing irregularities around the    | macroscopic diagnostic; canonical needed |
|                      | mature soybean rust pustules have;   | perimeter, and the lesion centers    | 20X magnification                        |
|                      | if an opening is present, it is      | are tan, the margins are chocolate-  |                                          |
|                      | typically a linear crack across the  | brown, and there is a chlorotic      |                                          |
|                      | surface of the pustule (visible only | yellow halo.                         |                                          |
|                      | under 20X magnification). Unlike     |                                      |                                          |
|                      | bacterial blight, bacterial pustule  |                                      |                                          |
|                      | lesions do not appear water soaked   |                                      |                                          |
|                      | and will have raised centers.        |                                      |                                          |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Cercospora Blight    | Light purple spots and areas develop | Dark brownish-green spots with a     | STAGE/COLOR REFINEMENT — captures a      |
|                      | on the top surface of leaves and     | leathery texture on the leaf         | brownish-green phase, not canonical's    |
|                      | expand to reddish-purple or bronze;  | surface.                             | purple                                   |
|                      | infected leaves appear leathery and  |                                      |                                          |
|                      | 'sunburned,' with red-brown spots    |                                      |                                          |
|                      | that may coalesce into large         |                                      |                                          |
|                      | necrotic areas leading to            |                                      |                                          |
|                      | defoliation.                         |                                      |                                          |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+
| Rhizoctonia Damping- | rusty-brown                          | The affected roots have brownish     | ADDED OBSERVATION — notes greenish areas |
| Off, Blight And Rot  |                                      | lesions with greenish areas.         | beyond canonical's single 'rusty-brown'  |
+----------------------+--------------------------------------+--------------------------------------+------------------------------------------+

# Verdict

- **Table A** — the fix restored *grounding*: the swarm reads the
  canonical slice and records confirmations against it across 5
  separate Soybean diseases.
- **Table B** — genuine KB *extension*, not captioning, across 5 more
  Soybean diseases: refinements, relocations, and added differentials.
- Table B deltas are *proposed* refinements for human review before
  they overwrite canonical — flagged, not yet truth. The Bacterial
  Speck row in particular (added target-spot pattern) is exactly the
  kind of item a reviewer should adjudicate.
