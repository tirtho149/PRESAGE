#!/usr/bin/env python3
"""Smoke-test TFDS Plant Village → DataFrame (no VLM)."""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-examples", type=int, default=8)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    from data.tfds_plant_village import build_plant_village_dataframe

    df = build_plant_village_dataframe(
        max_examples=args.max_examples,
        data_dir=args.data_dir,
    )
    print(df[["id", "disease_name", "crop_species", "benchmark"]].head())
    print(f"\nRows: {len(df)}  |  image_bytes dtype: {type(df['image_bytes'].iloc[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
