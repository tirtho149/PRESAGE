#!/usr/bin/env python3
"""Smoke-test PlantDoc GitHub tree → DataFrame (no VLM)."""

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
    p.add_argument(
        "repo_root",
        help="Path to cloned PlantDoc-Dataset repo (contains train/ and test/)",
    )
    p.add_argument("--split", choices=("train", "test"), default="train")
    args = p.parse_args()

    from data.plantdoc_github import build_plantdoc_dataframe

    df = build_plantdoc_dataframe(args.repo_root, split=args.split)
    print(df[["id", "disease_name", "crop_species", "benchmark"]].head(12))
    print(f"\nRows: {len(df)}  splits: {args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
