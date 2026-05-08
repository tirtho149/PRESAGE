"""
scripts/enhance_pathome_from_traces.py
======================================
Phase 2 of the Pathome pipeline: enhance a Claude-seeded PathomeDB with
aggregated PlantSwarm trace observations.

Pipeline position:

    (Phase 0) seed_pathome_with_claude.py   →  artifacts/pathome_seed/symptoms_seed.json
    (Phase 1) build_pathome.py              →  artifacts/pathome_v1_seed/   (seed + auto-derived geo + refs)
    (Phase 1) run_pathome_traces.py         →  results/bugwood/traces/plantswarm_traces.jsonl
    (Phase 2) THIS SCRIPT                   →  artifacts/pathome_v1_enhanced/

What it does
------------
Reads the trace JSONL, groups records by ground-truth (crop, disease) drawn
from each trace's ``bugwood_meta``, and computes per-class aggregates:

    n_traces             — number of completed routing traces for this class
    avg_path_length      — mean handoff count
    backtrack_rate       — fraction of traces with at least one backtrack
    high_confidence_rate — fraction of traces that early-terminated (κ=H)
    confusion_targets    — disease the swarm misroutes to → count
                           (final_predictions["T3"] when ≠ ground-truth)

Stored on each ``SymptomProfile.swarm_observations``. The Claude-seeded
visual block is left untouched — enhancement is additive. The before/after
comparison script (``compare_pathome_versions.py``) reads both DB versions.

Usage
-----
    python scripts/enhance_pathome_from_traces.py \
        --seed-db    artifacts/pathome_v1_seed/ \
        --traces     results/bugwood/traces/plantswarm_traces.jsonl \
        --out        artifacts/pathome_v1_enhanced/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathome import PathomeDB  # noqa: E402
from pathome.symptoms import SwarmObservations  # noqa: E402


# ---------------------------------------------------------------------------
# Trace ingestion
# ---------------------------------------------------------------------------

def iter_trace_records(path: Path) -> Iterator[dict]:
    """Yield one dict per JSONL line. Skips malformed lines silently."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def aggregate_by_class(traces: Iterator[dict]) -> Dict[Tuple[str, str], dict]:
    """Group traces by (crop, disease) and compute per-class aggregates.

    Each value is a dict ready to be turned into a ``SwarmObservations``.
    """
    buckets: Dict[Tuple[str, str], dict] = defaultdict(lambda: {
        "n_traces": 0,
        "path_length_sum": 0,
        "backtrack_count": 0,
        "early_term_count": 0,
        "confusion_targets": defaultdict(int),
    })

    skipped_no_meta = 0
    for t in traces:
        meta = t.get("bugwood_meta") or {}
        crop = meta.get("crop_species")
        disease = meta.get("disease_name")
        if not crop or not disease:
            skipped_no_meta += 1
            continue
        b = buckets[(crop, disease)]
        b["n_traces"] += 1
        b["path_length_sum"] += int(t.get("path_length") or 0)
        if int(t.get("backtrack_count") or 0) > 0:
            b["backtrack_count"] += 1
        if t.get("early_terminated"):
            b["early_term_count"] += 1
        preds = t.get("final_predictions") or {}
        pred_t3 = (preds.get("T3") or "").strip()
        if pred_t3 and pred_t3 != disease:
            b["confusion_targets"][pred_t3] += 1

    if skipped_no_meta:
        print(f"  [warn] {skipped_no_meta} traces missing bugwood_meta — skipped")
    return buckets


def to_swarm_obs(bucket: dict) -> SwarmObservations:
    n = max(bucket["n_traces"], 1)
    return SwarmObservations(
        n_traces=bucket["n_traces"],
        avg_path_length=bucket["path_length_sum"] / n,
        backtrack_rate=bucket["backtrack_count"] / n,
        high_confidence_rate=bucket["early_term_count"] / n,
        confusion_targets=dict(
            sorted(bucket["confusion_targets"].items(), key=lambda kv: -kv[1])
        ),
        common_lesion_terms={},   # path stores agent names only; texts not persisted
        common_signs={},
        last_updated=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# Build summary
# ---------------------------------------------------------------------------

def write_summary(
    db_in_dir: Path, db_out_dir: Path, traces_path: Path,
    profiles_updated: int, total_traces_used: int,
) -> None:
    summary = {
        "source_db": str(db_in_dir),
        "traces_file": str(traces_path),
        "enhanced_db": str(db_out_dir),
        "profiles_updated": profiles_updated,
        "total_traces_used": total_traces_used,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(db_out_dir / "enhancement_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seed-db", required=True,
                   help="path to the Claude-seeded PathomeDB directory")
    p.add_argument("--traces", required=True,
                   help="path to plantswarm_traces.jsonl")
    p.add_argument("--out", required=True,
                   help="output directory for the enhanced PathomeDB")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_dir = Path(args.seed_db)
    traces_path = Path(args.traces)
    out_dir = Path(args.out)

    if not seed_dir.is_dir():
        raise SystemExit(f"seed-db not a directory: {seed_dir}")
    if not traces_path.is_file():
        raise SystemExit(f"traces file not found: {traces_path}")

    print(f"loading seed PathomeDB from {seed_dir}")
    db = PathomeDB.load(str(seed_dir))
    print(f"  symptoms : {len(db.symptoms)} profiles")
    print(f"  refs     : {len(db.refs)}")

    print(f"\naggregating traces from {traces_path}")
    buckets = aggregate_by_class(iter_trace_records(traces_path))
    total_traces = sum(b["n_traces"] for b in buckets.values())
    print(f"  buckets: {len(buckets)} (crop, disease) classes")
    print(f"  total traces aggregated: {total_traces}")

    print("\nupdating profiles...")
    n_updated = 0
    n_unmatched = 0
    for (crop, disease), bucket in buckets.items():
        prof = db.symptoms.get(crop, disease)
        if prof is None:
            # The class appeared in traces but isn't in the seed DB. Create
            # a fresh profile so we still capture the swarm signal.
            prof = db.symptoms.get_or_create(crop, disease)
            n_unmatched += 1
        prof.swarm_observations = to_swarm_obs(bucket)
        n_updated += 1

    print(f"  profiles updated: {n_updated}")
    if n_unmatched:
        print(f"  profiles created from traces alone: {n_unmatched} "
              "(class was in traces but missing in seed DB)")

    db.version = f"{db.version}+swarm"
    out_dir.mkdir(parents=True, exist_ok=True)
    db.save(str(out_dir))
    write_summary(seed_dir, out_dir, traces_path, n_updated, total_traces)
    print(f"\nenhanced PathomeDB saved to {out_dir}")


if __name__ == "__main__":
    main()
