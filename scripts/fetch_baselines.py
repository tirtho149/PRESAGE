"""
scripts/fetch_baselines.py
==========================
Pre-download off-shelf CLIP-style baselines for PathomeOOD evaluation.
Running this once on the GPU host warms the open_clip / HF hub cache so
subsequent eval jobs don't pay the per-job download cost.

Each entry is exactly the (model, pretrained) tuple that the eval
scripts will pass to ``open_clip.create_model_and_transforms``.

To add or remove baselines, edit BASELINES below — the aggregator
(scripts/aggregate_pathomeood_tables.py) iterates the same set.

NOTE: imageomics/biocap is intentionally NOT in this list — PathomeOOD's
research claim is that we train our own model from neutral (openai-CLIP)
init using KB-grounded supervision on Bugwood, so comparing against a
warm-started variant of the same model would be tautological.

Usage:
    python scripts/fetch_baselines.py [--skip-if-cached]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# (run_id, model_name_for_open_clip, pretrained_tag, paper_table_columns)
BASELINES = [
    ("clip_vitb16",   "ViT-B-16",                         "openai", "T1,T2,T17,T18,T19,T20"),
    ("siglip_vitb16", "hf-hub:timm/ViT-B-16-SigLIP-256",  "",       "T1,T2,T17,T18,T19,T20"),
    ("fgclip",        "hf-hub:qihoo360/fg-clip-base",     "",       "T1,T2"),
    ("biotrove",      "hf-hub:BGLab/BioTrove-CLIP",       "",       "T1,T2,T17,T19"),
    ("bioclip",       "hf-hub:imageomics/bioclip",        "",       "T1,T2,T17,T18,T19,T20"),
    ("bioclip2",      "hf-hub:imageomics/bioclip-2",      "",       "T1,T2,T18,T19"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--skip-if-cached", action="store_true",
                   help="don't re-download if the model is already cached")
    p.add_argument("--list", action="store_true",
                   help="just list baselines without downloading")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=== fetch_baselines ===")
    for run_id, name, pretrained, tbls in BASELINES:
        print(f"  {run_id:14s}  model={name:50s}  pretrained={pretrained or '(default)':10s}  tables={tbls}")

    if args.list:
        return

    # Lazy import — open_clip is only available on the GPU host.
    repo_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(repo_root / "train_and_eval"))
    import open_clip

    for run_id, name, pretrained, _tbls in BASELINES:
        print(f"\n--- {run_id}: {name} ---")
        try:
            open_clip.create_model_and_transforms(name, pretrained=pretrained or None)
            print(f"  OK cached")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
