"""
scripts/build_pathome.py
========================
Build PathomeDB v1 from a Bugwood-curated tree.

Reads ``configs/bugwood_pathome.yaml``, walks the 7-trace and 3-reference
splits, and serialises all five layers under ``pathome.out_dir``.

Usage:
    python scripts/build_pathome.py --config configs/bugwood_pathome.yaml
    python scripts/build_pathome.py --config configs/bugwood_pathome.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from data.bugwood_loader import BugwoodLoader
from pathome import PathomeDB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bugwood_pathome.yaml")
    p.add_argument("--out_dir", default=None,
                   help="override pathome.out_dir from config")
    p.add_argument("--dry-run", action="store_true",
                   help="print stats but do not write the database")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = args.out_dir or cfg["pathome"]["out_dir"]
    print(f"Building PathomeDB → {out_dir}")
    print(f"Source: {cfg['data']['bugwood_root']}")

    print("\nLoading Bugwood records...")
    trace_loader = BugwoodLoader(cfg["data"], split="trace")
    ref_loader = BugwoodLoader(cfg["data"], split="reference")

    trace_records = list(trace_loader)
    ref_records = list(ref_loader)
    print(f"  trace split: {len(trace_records)} images")
    print(f"  reference split: {len(ref_records)} images")

    if not trace_records:
        raise SystemExit("No Bugwood images found — check bugwood_root in config.")

    # Class breakdown
    classes = {}
    for r in trace_records:
        key = (r.crop_species, r.disease_name)
        classes.setdefault(key, 0)
        classes[key] += 1
    print(f"  classes: {len(classes)}; mean trace images/class = "
          f"{len(trace_records) / max(len(classes), 1):.1f}")

    # GPS coverage check (paper §5.1: GPS required)
    with_gps = sum(1 for r in trace_records if r.lat is not None and r.lon is not None)
    print(f"  GPS coverage: {with_gps}/{len(trace_records)} trace images "
          f"({100*with_gps/max(len(trace_records),1):.1f}%)")

    print("\nBuilding PathomeDB...")
    db = PathomeDB.build_from_bugwood(
        trace_records=trace_records,
        reference_records=ref_records,
        layer1_path=cfg["pathome"].get("layer1_path"),
        layer2_path=cfg["pathome"].get("layer2_path"),
        version=cfg["pathome"].get("version", "v1.0"),
    )

    print(f"  Layer 1 — {len(db.layer1)} pathogen pathways")
    print(f"  Layer 2 — {len(db.layer2)} cross-crop manifestations")
    region_cells = sum(1 for v in db.layer3._region_totals.values() if v > 0)
    print(f"  Layer 3 — {region_cells} populated AEZ-month cells")
    print(f"  Layer 4 — decision graph rooted at "
          f"{db.layer4.root().node_id if db.layer4.root() else 'EMPTY'}")
    print(f"  Layer 5 — {len(db.layer5)} reference images")

    if args.dry_run:
        print("\n[dry-run] Skipping save.")
        return

    db.save(out_dir)
    print(f"\nPathomeDB v{db.version} saved to {out_dir}")

    # Manifest snapshot for traceability
    summary = {
        "version": db.version,
        "trace_records": len(trace_records),
        "reference_records": len(ref_records),
        "classes": len(classes),
        "gps_coverage": with_gps,
        "layer3_cells": region_cells,
        "layer5_size": len(db.layer5),
    }
    with open(os.path.join(out_dir, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
