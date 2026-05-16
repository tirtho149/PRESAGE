#!/usr/bin/env python
"""
smoke/run_one_image_swarm.py
============================
Dedicated one-image smoke for the Phase 0R swarm.

Picks ONE real Soybean (disease, state, cached-image) tuple using the
*exact* resolution the production pipeline uses
(`build_state_image_map` + `_resolve_cached_image`), then runs the
**full** swarm on it via `plantswarm.delta_pipeline.run_for_state`:

    OrganDetectionAgent -> route to that organ's deep specialists
    -> round 1 -> blackboard -> round 2 (stigmergy)
    -> VisualDiagnosisAgent consolidator -> K-of-N agreement
    -> conservative merge

It prints the chosen tuple, the per-pass detected organ + active
agent count, raw vs agreed delta counts, and the final merged deltas,
then asserts the output is well-formed. Verifier is OFF by default
(Claude is not available on a GPU node; the swarm is what we're
testing).

Env knobs (all optional; defaults give a faithful but quick smoke):
  CROP                 Soybean         crop to smoke
  SMOKE_DISEASE        (auto)          force a disease name, else first
                                       Soybean tuple with a cached image
  VLLM_N_RUNS          3               stochastic passes (K-of-N)
  VLLM_AGREEMENT_MIN   2               K
  VLLM_SWARM_ROUNDS    2               2 = full swarm (round-2 blackboard)
  SWARM_GRANULARITY    routed          routed | grouped | specialists
  VLLM_MAX_NEW_TOKENS  512             generation cap
  PATHOME_USE_VERIFIER 0               kept OFF here
  PATHOME_USABLE_CSV   BugWood_Diseases_usable.csv

Exit 0 = swarm ran end-to-end and produced well-formed output
         (deltas may legitimately be 0 if the image adds nothing).
Exit 2 = setup problem (no registry / no cached image / bad tuple).
Exit 3 = swarm ran but output is malformed (a real failure).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Default: do not call the Claude web verifier in a GPU-node smoke.
os.environ.setdefault("PATHOME_USE_VERIFIER", "0")

CROP = os.environ.get("CROP", "Soybean")
CSV = os.environ.get("PATHOME_USABLE_CSV", "BugWood_Diseases_usable.csv")
N_RUNS = int(os.environ.get("VLLM_N_RUNS", "3"))
K = int(os.environ.get("VLLM_AGREEMENT_MIN", "2"))


def _fail(code: int, msg: str) -> "None":
    print(f"\n[SMOKE FAIL] {msg}")
    sys.exit(code)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)

    reg_path = repo / "artifacts" / "pathome_kb" / CROP / "final_registry.json"
    if not reg_path.is_file():
        _fail(2, f"no {reg_path} — run Phase 0/1 (canonical KB) first.")
    if not (repo / CSV).is_file():
        _fail(2, f"no {CSV} — run Step 0 (filter) first.")

    from pathome_kb.regional_observation import (
        build_state_image_map, _resolve_cached_image,
    )
    from plantswarm.delta_pipeline import (
        run_for_state, existing_deltas_for_state,
    )

    reg = json.loads(reg_path.read_text())
    by_disease = {
        (d.get("disease_name") or "").strip(): d
        for d in reg.get("diseases", []) or []
        if (d.get("disease_name") or "").strip()
    }
    if not by_disease:
        _fail(2, f"{reg_path} has no diseases.")

    force = os.environ.get("SMOKE_DISEASE", "").strip()

    # Resolve a REAL (crop, disease, state, cached image) tuple exactly
    # like run_regional_observation does.
    smap = build_state_image_map(CSV)
    chosen = None
    for (c, disease, state), image_ids in smap.items():
        if c != CROP:
            continue
        if force and disease != force:
            continue
        if disease not in by_disease:
            continue
        for img_id in image_ids:
            p = _resolve_cached_image(img_id)
            if p:
                chosen = (disease, state, p, img_id, list(image_ids))
                break
        if chosen:
            break

    if not chosen:
        _fail(2, f"no {CROP} tuple with a cached image found "
                 f"(cache empty? run scripts/ensure_state_image_cache.py).")

    disease, state, img_path, img_id, image_ids = chosen
    drec = by_disease[disease]
    existing = existing_deltas_for_state(drec, state)

    print("=" * 64)
    print(f"ONE-IMAGE SWARM SMOKE — {CROP}")
    print("=" * 64)
    print(f"  disease : {disease}")
    print(f"  state   : {state}")
    print(f"  image   : {img_path}  (id={img_id})")
    print(f"  granularity={os.environ.get('SWARM_GRANULARITY','routed')} "
          f"N={N_RUNS} K={K} rounds={os.environ.get('VLLM_SWARM_ROUNDS','2')} "
          f"verifier={os.environ.get('PATHOME_USE_VERIFIER')}")
    print(f"  existing deltas for this state: {len(existing)}")
    print("-" * 64)

    t0 = time.time()
    rec = run_for_state(
        crop=CROP, disease=disease, state=state,
        canonical_record=drec, image_path=Path(img_path),
        primary_image_id=img_id, existing_deltas=existing,
        n_runs=N_RUNS, agreement_min=K,
    )
    dt = time.time() - t0

    sm = rec.get("__swarm_meta__", {}) or {}
    deltas = rec.get("deltas", []) or []
    print(f"\n=== swarm finished in {dt:.0f}s ===")
    print(f"  granularity         : {sm.get('granularity')}")
    print(f"  detected organ/pass : {sm.get('detected_organ_per_pass')}")
    print(f"  active agents/pass  : {sm.get('n_active_agents_per_pass')}")
    print(f"  raw deltas/pass     : {sm.get('n_raw_per_pass')}")
    print(f"  after K-of-N        : {sm.get('n_after_agreement')}")
    print(f"  merge counts        : {sm.get('merge')}")
    print(f"  FINAL merged deltas : {len(deltas)}")
    for d in deltas[:12]:
        print(f"    - [{d.get('field')}] {str(d.get('image_shows',''))[:100]}")

    # ---- assertions: swarm ran and output is well-formed -------------
    problems = []
    if "granularity" not in sm:
        problems.append("no __swarm_meta__.granularity (run_for_state contract broke)")
    if sm.get("n_raw_per_pass") is None or len(sm.get("n_raw_per_pass") or []) != N_RUNS:
        problems.append(f"expected {N_RUNS} passes in n_raw_per_pass, got "
                        f"{sm.get('n_raw_per_pass')}")
    for d in deltas:
        if not isinstance(d, dict) or not d.get("field") or not d.get("image_shows"):
            problems.append(f"malformed delta: {d!r}")
            break

    if problems:
        for p in problems:
            print(f"  [bad] {p}")
        _fail(3, "swarm ran but output is malformed.")

    print("\n[SMOKE PASS] full swarm ran end-to-end on one image; "
          f"output well-formed ({len(deltas)} deltas; "
          "0 is acceptable if the image adds nothing).")
    sys.exit(0)


if __name__ == "__main__":
    main()
