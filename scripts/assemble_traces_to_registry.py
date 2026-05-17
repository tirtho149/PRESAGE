#!/usr/bin/env python3
"""Salvage assembler: phase0r trace deltas -> final_registry.json.

The Nova STEP 2 post-fix run produced phase0r_traces (1).jsonl but never
merged its deltas into artifacts/pathome_kb/<Crop>/final_registry.json
(regional_observations) nor pushed them, so STEP 3 (validate_kb.py) sees
zero candidates. This one-off bridges that gap WITHOUT another GPU run:
it reads the post-fix half of the trace file, groups final_deltas by
(crop, disease, state), runs the SAME conservative merge the live
pipeline uses (plantswarm.delta_pipeline._merge_with_existing), and
writes regional_observations blocks in the exact schema validate_kb.py /
pathome_kb.verifier consume (verification_status="unverified").

Only the post-fix half is used: records[147:] of the trace file are the
clean post-fix run (the first 147 are the byte-identical stale pre-fix
run, verified separately). Idempotent: rewrites regional_observations
for the (disease,state) tuples present in the traces; leaves canonical
fields and untouched diseases intact.

Usage:  python scripts/assemble_traces_to_registry.py [--dry-run]
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from plantswarm.delta_pipeline import _merge_with_existing

TRACES   = "artifacts/phase0r_traces/phase0r_traces (1).jsonl"
KB_ROOT  = Path("artifacts/pathome_kb")
CROPS    = ("Soybean", "Tomato")
PREFIX_STALE = 147   # records[:147] = stale pre-fix run, excluded
SIM_TAU  = float(os.environ.get("VLLM_SIM_THRESHOLD", "0.4"))


def load_post_fix_records():
    recs = [json.loads(l) for l in open(TRACES)]
    post = recs[PREFIX_STALE:]
    assert post, "no post-fix records (file shorter than 147 lines?)"
    return [r for r in post if r.get("crop") in CROPS]


def collect(records):
    """(crop, disease, state) -> {deltas:[...], image_ids:set}."""
    groups: dict = defaultdict(lambda: {"deltas": [], "image_ids": set()})
    for r in records:
        key = (r["crop"], r["disease"], r["state"])
        img = str(r.get("primary_image_id") or "")
        groups[key]["image_ids"].add(img)
        for d in r.get("final_deltas") or []:
            shows = (d.get("image_shows") or "").strip()
            if not shows:
                continue
            groups[key]["deltas"].append({
                "field":          str(d.get("field") or "other"),
                "canonical_says": str(d.get("canonical_says") or "(not specified)"),
                "image_shows":    shows,
                "image_quote":    (d.get("image_quote") or "").strip(),
                "image_id":       img,
            })
    return groups


def merge_group(deltas):
    """Same conservative merge the live pipeline uses; tags unverified."""
    merged, counts = _merge_with_existing(
        existing=[], new=deltas, similarity_threshold=SIM_TAU)
    for m in merged:
        m.setdefault("verification_status", "unverified")
        m.setdefault("web_support", [])
        m.setdefault("canonical_says", "(not specified)")
        m.setdefault("image_quote", "")
        m.setdefault("image_id", "")
    return merged, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan, do not write registries")
    ap.add_argument("--fill-failed-only", action="store_true",
                    help="non-destructive: keep blocks that were "
                         "successfully verified (__verifier_meta__ with "
                         "any nonzero count); only (re)write blocks that "
                         "are missing or were wiped by a failed verify "
                         "(meta absent or all-zero). Use to recover from "
                         "a rate-limited STEP 3 without losing good work.")
    args = ap.parse_args()

    records = load_post_fix_records()
    groups = collect(records)
    print(f"post-fix records: {len(records)}  |  (crop,disease,state) "
          f"tuples: {len(groups)}  |  sim_tau={SIM_TAU}")

    # crop -> disease -> state -> block
    by_crop: dict = defaultdict(lambda: defaultdict(dict))
    tot_in = tot_out = 0
    for (crop, disease, state), g in sorted(groups.items()):
        merged, c = merge_group(g["deltas"])
        tot_in += len(g["deltas"])
        tot_out += len(merged)
        by_crop[crop][disease][state] = {
            "deltas":    merged,
            "image_ids": sorted(x for x in g["image_ids"] if x),
        }

    print(f"raw deltas in: {tot_in}  ->  merged/clustered out: {tot_out}")

    for crop in CROPS:
        reg_path = KB_ROOT / crop / "final_registry.json"
        reg = json.loads(reg_path.read_text())
        diseases = reg["diseases"]
        name_idx = {d["disease_name"]: d for d in diseases}
        n_blocks = n_deltas = n_kept = 0
        for disease, states in by_crop.get(crop, {}).items():
            rec = name_idx.get(disease)
            if rec is None:
                print(f"  [WARN] {crop}: '{disease}' not in registry — skipped")
                continue
            ro = rec.setdefault("regional_observations", {})
            for state, block in states.items():
                if args.fill_failed_only:
                    old = ro.get(state)
                    if isinstance(old, dict):
                        m = old.get("__verifier_meta__") or {}
                        verified_sum = sum(m.get(k, 0) for k in (
                            "verified_count", "provisional_count",
                            "contradictory_count", "duplicates_count"))
                        if verified_sum > 0:
                            n_kept += 1
                            continue   # genuinely verified — preserve
                ro[state] = block
                n_blocks += 1
                n_deltas += len(block["deltas"])
        # top-level count = total (disease,state) regional blocks written
        total_blocks = sum(
            len(d.get("regional_observations") or {}) for d in diseases)
        reg["regional_observations_count"] = total_blocks
        print(f"  {crop}: wrote/restored {n_blocks} blocks ({n_deltas} "
              f"deltas), kept {n_kept} verified blocks; registry "
              f"regional_observations_count={total_blocks}")
        if args.dry_run:
            print(f"  [dry-run] not writing {reg_path}")
            continue
        reg_path.write_text(
            json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
        print(f"  wrote {reg_path}")

    print("done." + ("  (dry-run — nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
