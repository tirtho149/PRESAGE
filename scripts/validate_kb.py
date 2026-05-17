"""
scripts/validate_kb.py
======================
Standalone Claude+WebSearch validator over a Phase 0R-produced KB.

Walks every ``artifacts/pathome_kb/<Crop>/final_registry.json``,
finds deltas with ``verification_status == "unverified"`` (or missing),
groups them by (crop, disease, state), and calls
``pathome_kb.verifier.verify_candidates`` to assign each delta a
real verification status + web_support citations. Updates the
registries in place.

Why this is a SEPARATE step from Phase 0R
-----------------------------------------
Nova has the GPU + vLLM-serving Qwen2.5-VL but does not have the
authenticated ``claude`` CLI. LOCAL has Claude but not the GPU.
Splitting the Phase 0R pipeline gives each step the right tool:

    NOVA   step 2  : Qwen swarm runs WITHOUT verifier (writes
                      verification_status="unverified")
    LOCAL  step 3  : Claude verifier runs over the unverified
                      deltas, fills in verification_status +
                      web_support

The verifier API and the dedup / merge logic are unchanged — this
script is just an iterator that drives the existing
``verify_candidates`` function over an already-merged KB.

Knobs (env vars)
----------------
    CROPS              comma-separated crop allowlist (default: all crops
                       present under artifacts/pathome_kb/)
    KB_ROOT            default artifacts/pathome_kb
    DRY_RUN            set 1 to print plan without calling Claude
    MAX_TUPLES         cap on (crop, disease, state) tuples to verify
                       (useful for cost-bounded smoke runs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathome_kb.verifier import verify_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--kb-root", default="artifacts/pathome_kb",
                   help="root containing <Crop>/final_registry.json files")
    p.add_argument("--crops", default="",
                   help="comma-separated crop allowlist (default: all)")
    p.add_argument("--max-tuples", type=int, default=0,
                   help="cap on (crop, disease, state) tuples to verify "
                        "(0 = no cap)")
    p.add_argument("--timeout", type=int, default=600,
                   help="per-tuple verifier timeout (seconds)")
    p.add_argument("--max-turns", type=int, default=30,
                   help="claude -p --max-turns budget")
    p.add_argument("--dry-run", action="store_true",
                   help="print plan; don't call Claude")
    return p.parse_args()


def _flatten_canonical(record: Dict[str, Any]) -> Dict[str, Any]:
    """Same shape as plantswarm.delta_pipeline.flatten_canonical — kept
    inline here to avoid a hard dep on plantswarm.delta_pipeline
    (which transitively imports vLLM client / requests)."""
    def _v(field: Any) -> Any:
        if not isinstance(field, dict):
            return field
        return field.get("value")

    visual = record.get("visual_symptoms") or {}
    return {
        "summary":                  _v(visual.get("summary"))               or "",
        "diagnostic_features":      _v(visual.get("diagnostic_features"))   or [],
        "look_alikes":              _v(visual.get("look_alikes"))           or [],
        "affected_parts":           _v(record.get("affected_parts"))        or [],
        "treatments":               _v(record.get("treatments"))            or [],
        "pathogen_scientific_name": _v(record.get("pathogen_scientific_name")) or "",
        "type_of_disease":          _v(record.get("type_of_disease"))       or "",
    }


def _gather_unverified_tuples(
    registry: Dict[str, Any],
    crop: str,
) -> List[Dict[str, Any]]:
    """Yield one record per (disease, state) that still has unverified
    deltas. Each record carries the slice of deltas to verify plus the
    full existing-KB and canonical context the verifier needs."""
    tuples: List[Dict[str, Any]] = []
    for disease_record in registry.get("diseases") or []:
        disease = disease_record.get("disease_name") or ""
        canonical = _flatten_canonical(disease_record)
        ro = disease_record.get("regional_observations") or {}
        for state, state_block in ro.items():
            if not isinstance(state_block, dict):
                continue
            deltas = state_block.get("deltas") or []
            # Split into already-verified (existing context) vs unverified
            # candidates (the ones we need to verify now).
            existing: List[Dict[str, Any]] = []
            candidates: List[Dict[str, Any]] = []
            for d in deltas:
                if not isinstance(d, dict):
                    continue
                status = (d.get("verification_status") or "").lower()
                if status and status not in ("unverified", ""):
                    existing.append(d)
                else:
                    candidates.append(d)
            if not candidates:
                continue
            tuples.append({
                "crop":         crop,
                "disease":      disease,
                "state":        state,
                "canonical":    canonical,
                "existing":     existing,
                "candidates":   candidates,
                "image_id":     (state_block.get("image_ids") or [""])[0],
                "_ref_disease": disease_record,    # mutate-back handle
                "_ref_state":   state_block,
            })
    return tuples


def _apply_verifier_result(
    state_block: Dict[str, Any],
    verifier_result: Dict[str, Any],
) -> str:
    """Fold the verifier result back into this state's delta block.

    Returns the tuple outcome: ``"ok"`` (a real verdict was applied) or
    ``"failed"`` (verifier could not produce a verdict — candidates are
    preserved as ``unverified``, NEVER silently dropped).

    On success: unverified candidates are replaced by the accepted
    (verified / weakly_supported / provisional / novel_plausible) ones;
    contradictory delta *bodies* are persisted under
    ``__verifier_meta__.dropped_contradictory`` (makes the true rejection
    rate recoverable, not just a count)."""
    existing: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for d in state_block.get("deltas") or []:
        status = (d.get("verification_status") or "").lower()
        if status and status not in ("unverified", ""):
            existing.append(d)
        else:
            candidates.append(d)

    if verifier_result.get("_verifier_failed"):
        # Preserve every candidate as unverified — do not drop, do not
        # mislabel as verified. The driver counts this tuple as failed.
        preserved = verifier_result.get("_preserved_unverified") or candidates
        state_block["deltas"] = existing + preserved
        state_block["__verifier_meta__"] = {
            "failed":          True,
            "failure_reason":  verifier_result.get("_failure_reason", ""),
            "preserved_count": len(preserved),
        }
        return "failed"

    accepted = verifier_result.get("accepted") or []
    contradictory = verifier_result.get("contradictory") or []
    state_block["deltas"] = existing + accepted
    state_block["__verifier_meta__"] = {
        "failed":              False,
        "verified_count":      len(verifier_result.get("verified") or []),
        "provisional_count":   len(verifier_result.get("provisional") or []),
        "contradictory_count": len(contradictory),
        "duplicates_count":    len(verifier_result.get("duplicates_of_existing") or []),
        "dropped_contradictory": contradictory,
    }
    return "ok"


def main() -> None:
    args = parse_args()
    kb_root = Path(args.kb_root)
    if not kb_root.is_dir():
        raise SystemExit(f"KB root not found: {kb_root}")

    crop_filter: List[str] = []
    raw_crops = os.environ.get("CROPS") or args.crops
    if raw_crops:
        if raw_crops == "smoke":
            crop_filter = ["Soybean", "Tomato"]
        elif raw_crops != "all":
            crop_filter = [c.strip() for c in raw_crops.split(",") if c.strip()]
    max_tuples = int(os.environ.get("MAX_TUPLES") or args.max_tuples or 0)
    dry_run = bool(os.environ.get("DRY_RUN") or args.dry_run)

    # Discover registries.
    registries: List[Path] = []
    for crop_dir in sorted(kb_root.iterdir()):
        if not crop_dir.is_dir():
            continue
        if crop_filter and crop_dir.name not in crop_filter:
            continue
        reg = crop_dir / "final_registry.json"
        if reg.is_file():
            registries.append(reg)
    if not registries:
        raise SystemExit(
            f"no final_registry.json files under {kb_root} "
            f"(crop_filter={crop_filter})"
        )

    print(f"=== validate_kb ===")
    print(f"  kb_root       : {kb_root}")
    print(f"  registries    : {len(registries)}")
    print(f"  crops         : {[r.parent.name for r in registries]}")
    if max_tuples:
        print(f"  max-tuples    : {max_tuples}")
    if dry_run:
        print(f"  DRY-RUN       : on")

    # Collect tuples to verify across all registries.
    all_tuples: List[Dict[str, Any]] = []
    by_reg: Dict[Path, List[Dict[str, Any]]] = {}
    for reg in registries:
        try:
            data = json.loads(reg.read_text())
        except Exception as e:
            print(f"  [skip] {reg}: {type(e).__name__}: {e}")
            continue
        crop = data.get("crop") or reg.parent.name
        tuples = _gather_unverified_tuples(data, crop)
        by_reg[reg] = (data, tuples)
        for t in tuples:
            t["_ref_registry"] = data
            t["_ref_path"] = reg
            all_tuples.append(t)

    print(f"  tuples w/ unverified deltas: {len(all_tuples)}")
    if max_tuples and len(all_tuples) > max_tuples:
        print(f"  capping to first {max_tuples}")
        all_tuples = all_tuples[:max_tuples]

    if not all_tuples:
        print("  nothing to verify. Exiting cleanly.")
        return

    if dry_run:
        print()
        print("  --- plan (dry-run) ---")
        for t in all_tuples[:50]:
            print(f"    {t['crop']}/{t['disease']}/{t['state']}: "
                  f"{len(t['candidates'])} candidate deltas")
        if len(all_tuples) > 50:
            print(f"    ... +{len(all_tuples) - 50} more")
        return

    # Run the verifier per tuple.
    print()
    print("  --- verifying ---")
    n_ok = n_failed = n_error = 0
    for i, t in enumerate(all_tuples, 1):
        print(f"  [{i}/{len(all_tuples)}] {t['crop']}/{t['disease']}/{t['state']} "
              f"({len(t['candidates'])} candidates)")
        try:
            result = verify_candidates(
                crop=t["crop"], disease=t["disease"], state=t["state"],
                canonical=t["canonical"],
                existing_kb_deltas=t["existing"],
                candidates=t["candidates"],
                primary_image_id=t["image_id"],
                timeout_secs=args.timeout,
                max_turns=args.max_turns,
            )
        except Exception as e:
            # state_block untouched → candidates remain unverified (preserved).
            print(f"      [ERROR] {type(e).__name__}: {e}; leaving unverified")
            n_error += 1
            continue
        if _apply_verifier_result(t["_ref_state"], result) == "failed":
            n_failed += 1
        else:
            n_ok += 1

    # Persist updated registries.
    print()
    print("  --- writing back ---")
    written: set = set()
    for t in all_tuples:
        path = t["_ref_path"]
        if path in written:
            continue
        data = t["_ref_registry"]
        path.write_text(json.dumps(data, indent=2))
        written.add(path)
        print(f"    wrote {path}")
    print()
    print(f"  validate_kb: tuples={len(all_tuples)} "
          f"verified_ok={n_ok} failed={n_failed} error={n_error} "
          f"across {len(written)} registry files.")

    degraded = n_failed + n_error
    if degraded > 0:
        allow = bool(os.environ.get("ALLOW_UNVERIFIED"))
        print()
        print("  " + "=" * 66)
        print(f"  !! {degraded} tuple(s) could NOT be verified "
              f"(failed={n_failed} error={n_error}).")
        print("  !! Their deltas were PRESERVED as 'unverified' (not dropped),")
        print("  !! but this KB is NOT fully verified.")
        if allow:
            print("  !! ALLOW_UNVERIFIED set — continuing anyway (exit 0).")
            print("  " + "=" * 66)
        else:
            print("  !! Refusing to report success. Fix Claude auth / rate")
            print("  !! limit and re-run, or set ALLOW_UNVERIFIED=1 to accept")
            print("  !! a partially-verified KB.")
            print("  " + "=" * 66)
            raise SystemExit(3)


if __name__ == "__main__":
    main()
