#!/usr/bin/env python3
"""Assemble phase0r_traces_apple.jsonl deltas into Apple final_registry.json.

Same conservative merge as scripts/assemble_traces_to_registry.py
(which is hardcoded for Soybean+Tomato with a stale-prefix offset).
This one is the Apple-only variant: reads phase0r_traces_apple.jsonl,
filters to crop=="Apple" (the file contains records for other crops
too), groups final_deltas by (disease, state), runs the same
plantswarm.delta_pipeline._merge_with_existing, and writes
regional_observations blocks with verification_status="unverified"
into artifacts/pathome_kb/Apple/final_registry.json. STEP 3
(scripts/validate_kb.py --crops Apple) then verifies them.

Usage:  python scripts/assemble_apple_traces.py [--dry-run]
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plantswarm.delta_pipeline import _merge_with_existing

TRACES   = "artifacts/phase0r_traces/phase0r_traces_apple.jsonl"
KB_ROOT  = Path("artifacts/pathome_kb")
CROP     = "Apple"
SIM_TAU  = float(os.environ.get("VLLM_SIM_THRESHOLD", "0.4"))


def load_apple_records():
    with open(TRACES, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    return [r for r in recs if r.get("crop") == CROP]


def collect(records):
    """(disease, state) -> {deltas:[...], image_ids:set}."""
    groups: dict = defaultdict(lambda: {"deltas": [], "image_ids": set()})
    for r in records:
        key = (r["disease"], r["state"])
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
    merged, _counts = _merge_with_existing(
        existing=[], new=deltas, similarity_threshold=SIM_TAU)
    for m in merged:
        m.setdefault("verification_status", "unverified")
        m.setdefault("web_support", [])
        m.setdefault("canonical_says", "(not specified)")
        m.setdefault("image_quote", "")
        m.setdefault("image_id", "")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = load_apple_records()
    groups = collect(records)
    print(f"apple records: {len(records)}  |  (disease,state) tuples: "
          f"{len(groups)}  |  sim_tau={SIM_TAU}")

    by_disease: dict = defaultdict(dict)
    tot_in = tot_out = 0
    for (disease, state), g in sorted(groups.items()):
        merged = merge_group(g["deltas"])
        tot_in += len(g["deltas"])
        tot_out += len(merged)
        by_disease[disease][state] = {
            "deltas":    merged,
            "image_ids": sorted(x for x in g["image_ids"] if x),
        }
    print(f"raw deltas in: {tot_in}  ->  merged/clustered out: {tot_out}")

    reg_path = KB_ROOT / CROP / "final_registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    diseases = reg["diseases"]
    name_idx = {d["disease_name"]: d for d in diseases}

    n_blocks = n_deltas = 0
    for disease, states in by_disease.items():
        rec = name_idx.get(disease)
        if rec is None:
            print(f"  [WARN] '{disease}' not in Apple registry — skipped")
            continue
        ro = rec.setdefault("regional_observations", {})
        for state, block in states.items():
            ro[state] = block
            n_blocks += 1
            n_deltas += len(block["deltas"])

    total_blocks = sum(
        len(d.get("regional_observations") or {}) for d in diseases)
    reg["regional_observations_count"] = total_blocks
    print(f"  wrote {n_blocks} blocks ({n_deltas} deltas);  "
          f"registry regional_observations_count={total_blocks}")

    if args.dry_run:
        print(f"  [dry-run] not writing {reg_path}")
        return
    reg_path.write_text(
        json.dumps(reg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {reg_path}")
    print("done.")


if __name__ == "__main__":
    main()
