"""
scripts/setup_plantdoc.py
=========================
One-shot downloader for PlantDoc — the plant-disease classification
benchmark used in BioCAP paper Table 19.

PlantDoc is hosted as a public GitHub repository
(https://github.com/pratikkayal/PlantDoc-Dataset) with folder-per-class
images in ``train/`` and ``test/`` subdirs. We clone it once into
``data/eval/PlantDoc/`` so that ``scripts/evaluate_biocap.py
--plantdoc-root data/eval/PlantDoc/test`` works.

Usage:
    python scripts/setup_plantdoc.py [--dest data/eval/PlantDoc]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PLANTDOC_REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dest", default="data/eval/PlantDoc",
                   help="target directory (will be created)")
    p.add_argument("--repo", default=PLANTDOC_REPO,
                   help="git URL to clone (override only if upstream moves)")
    p.add_argument("--force", action="store_true",
                   help="re-clone even if the dest already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest)
    if dest.exists():
        if args.force:
            print(f"  --force given; removing existing {dest}")
            shutil.rmtree(dest)
        else:
            print(f"  PlantDoc already at {dest} (use --force to re-clone)")
            train = dest / "train"
            test  = dest / "test"
            for d in (train, test):
                if d.is_dir():
                    n_classes = sum(1 for _ in d.iterdir() if _.is_dir())
                    print(f"    {d}  ({n_classes} class dirs)")
            return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {args.repo} -> {dest}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", args.repo, str(dest)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"git clone failed: {e}")

    train = dest / "train"
    test  = dest / "test"
    for d in (train, test):
        if d.is_dir():
            n_classes = sum(1 for _ in d.iterdir() if _.is_dir())
            print(f"    {d}  ({n_classes} class dirs)")
        else:
            print(f"    {d}  (MISSING — upstream may have changed layout)")

    print()
    print("  Use with:")
    print(f"    python scripts/evaluate_biocap.py --plantdoc-root {test} ...")


if __name__ == "__main__":
    main()
