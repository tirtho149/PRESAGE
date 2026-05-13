"""
scripts/build_pathomeood_captions.py
================================
Build per-image (taxon, caption) text rows for PathomeOOD training.

For each Bugwood row in ``BugWood_Diseases_usable.csv`` whose
(NormCrop, NormDisease) has a KB profile in
``artifacts/pathome_kb/<crop>/final_registry.json``, emit one record
under the chosen caption strategy. Output is a Parquet table consumed
by ``scripts/build_pathomeood_shards.py``.

Strategies (see plantswarm/captioning.STRATEGIES):
    label_only          Table 3 row "None" baseline
    summary_only        canonical summary line only
    canonical_full      canonical summary + diagnostic + look-alikes + parts
    canonical_deltas_1  canonical_full + top-1 regional delta  (Table 6)
    canonical_deltas_3  canonical_full + top-3 deltas (main, Tables 3 & 6)
    canonical_deltas_5  canonical_full + top-5 deltas (Table 6)
    canonical_deltas_7  canonical_full + top-7 deltas (Table 6)

Delta strategies HARD-FAIL when any matched profile has no
``regional_observations`` populated — Phase 0R must run first.

Usage:
    python scripts/build_pathomeood_captions.py \\
        --strategy canonical_deltas_3 \\
        [--crop Tomato] \\
        [--csv BugWood_Diseases_usable.csv] \\
        [--kb-root artifacts/pathome_kb] \\
        [--cache-dir .bugwood_cache] \\
        [--holdout-state CA] \\
        [--val-frac 0.1] \\
        [--seed 42] \\
        [--out data/bugwood_captions/<crop>_<strategy>.parquet]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plantswarm.captioning import (
    STRATEGIES, DELTA_STRATEGIES,
    caption_for_row, load_kb_profiles, taxon_text, assert_deltas_populated,
)


_IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--strategy", required=True, choices=STRATEGIES,
                   help="Caption strategy (Table 3 / Table 6 row)")
    p.add_argument("--crop", default=None,
                   help="Restrict to one crop (default = all crops with KB)")
    p.add_argument("--csv", default="BugWood_Diseases_usable.csv")
    p.add_argument("--kb-root", default="artifacts/pathome_kb")
    p.add_argument("--cache-dir", default=".bugwood_cache",
                   help="Image cache dir (need not exist at caption-build time)")
    p.add_argument("--holdout-state", default=None,
                   help="State to mark as 'holdout' (Table 2 retrieval bench)")
    p.add_argument("--val-frac", type=float, default=0.10,
                   help="Train/val split fraction (stratified by disease)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Output parquet. Default: data/bugwood_captions/<crop>_<strategy>.parquet")
    return p.parse_args()


def _resolve_image_path(image_id: str, cache_dir: Path) -> str:
    """Return the expected path to the cached image. We do NOT require
    the cache to exist at caption-build time — only the shard builder
    needs the bytes."""
    for ext in _IMAGE_EXTS:
        candidate = cache_dir / f"{image_id}.{ext}"
        if candidate.is_file():
            return str(candidate)
    # Default expected path; shard builder will check existence.
    return str(cache_dir / f"{image_id}.jpg")


def _stratified_split(
    rows: List[Dict[str, str]],
    val_frac: float,
    seed: int,
) -> List[str]:
    """Return a parallel list of 'train' / 'val' tags, stratified by
    (crop, disease). Deterministic given seed."""
    by_class: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_class[(r["crop"], r["disease"])].append(i)
    rng = random.Random(seed)
    tags = ["train"] * len(rows)
    for indices in by_class.values():
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_frac))) if len(indices) > 1 else 0
        for j in indices[:n_val]:
            tags[j] = "val"
    return tags


def main() -> None:
    args = parse_args()

    # 1. Load KB profiles ----------------------------------------------------
    crop_filter = [args.crop] if args.crop else None
    profiles = load_kb_profiles(args.kb_root, crop_filter=crop_filter)
    if not profiles:
        raise SystemExit(
            f"No KB profiles loaded from {args.kb_root} "
            f"(crop_filter={crop_filter}). Did Phase 0 run?"
        )
    crops_with_kb = sorted({c for c, _ in profiles})
    print(f"=== build_pathomeood_captions ===")
    print(f"  strategy       : {args.strategy}")
    print(f"  KB profiles    : {len(profiles)} across crops {crops_with_kb}")
    print(f"  CSV            : {args.csv}")
    print(f"  cache_dir      : {args.cache_dir}")
    print(f"  holdout-state  : {args.holdout_state or '(none)'}")

    # 2. Guard: delta strategies require populated regional_observations ----
    assert_deltas_populated(profiles, strategies=[args.strategy])

    # 3. Walk CSV ------------------------------------------------------------
    cache_dir = Path(args.cache_dir)
    rows: List[Dict[str, str]] = []
    fallback_pairs: Counter = Counter()
    n_kb: int = 0
    n_fallback: int = 0
    skipped_no_state: int = 0

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            crop    = (r.get("NormCrop")    or "").strip()
            disease = (r.get("NormDisease") or "").strip()
            if not crop or not disease:
                continue
            if args.crop and crop != args.crop:
                continue
            image_id = (r.get("Image Number") or "").strip()
            if not image_id:
                continue
            state = (r.get("Location (State)") or "").strip() or None
            if state is None:
                skipped_no_state += 1
            try:
                caption, used_kb = caption_for_row(
                    crop=crop, disease=disease, state=state,
                    profiles=profiles, strategy=args.strategy,
                )
            except ValueError as e:
                # Per-profile delta absence on a KB-covered class.
                raise SystemExit(f"caption build failed for {image_id}: {e}")
            if used_kb:
                n_kb += 1
            else:
                n_fallback += 1
                fallback_pairs[(crop, disease)] += 1
            split = "holdout" if (args.holdout_state and state == args.holdout_state) else None
            rows.append({
                "image_id":     image_id,
                "image_path":   _resolve_image_path(image_id, cache_dir),
                "crop":         crop,
                "disease":      disease,
                "state":        state or "",
                "taxon_text":   taxon_text(crop, disease),
                "caption_text": caption,
                "used_kb":      "1" if used_kb else "0",
                "split":        split or "",  # filled in step 4
            })

    if not rows:
        raise SystemExit("No matched rows produced — CSV may be filtered out.")

    # 4. Train/val split (within non-holdout rows) --------------------------
    eligible_idx = [i for i, r in enumerate(rows) if r["split"] != "holdout"]
    sub = [rows[i] for i in eligible_idx]
    tags = _stratified_split(sub, val_frac=args.val_frac, seed=args.seed)
    for j, i in enumerate(eligible_idx):
        rows[i]["split"] = tags[j]

    # 5. Stats + write -------------------------------------------------------
    split_counts = Counter(r["split"] for r in rows)
    per_class = Counter((r["crop"], r["disease"]) for r in rows)
    print(f"  rows kept           : {len(rows)}")
    print(f"  split               : {dict(split_counts)}")
    print(f"  classes             : {len(per_class)}")
    print(f"  with KB caption     : {n_kb} rows")
    print(f"  with fallback cap   : {n_fallback} rows "
          f"across {len(fallback_pairs)} (crop,disease) pairs")
    if fallback_pairs:
        print("  top fallback pairs (no KB profile):")
        for (c, d), n in fallback_pairs.most_common(8):
            print(f"      - {c}/{d}: {n}")
    if skipped_no_state:
        print(f"  rows w/ no state    : {skipped_no_state}")

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        crop_tag = args.crop or "all"
        out_path = Path("data/bugwood_captions") / f"{crop_tag}_{args.strategy}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write parquet (pyarrow) or fall back to TSV if pyarrow missing.
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path)
        print(f"  wrote          : {out_path}  ({len(rows)} rows)")
    except ImportError:
        tsv_path = out_path.with_suffix(".tsv")
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            cols = list(rows[0].keys())
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  pyarrow missing; wrote TSV instead: {tsv_path} ({len(rows)} rows)")

    # Sample previews
    print()
    print("  --- sample rows ---")
    for r in rows[:3]:
        print(f"    [{r['split']:7s}] {r['crop']}/{r['disease']}  ({r['state']})  id={r['image_id']}")
        print(f"        taxon  : {r['taxon_text']}")
        print(f"        caption: {r['caption_text'][:160]}...")


if __name__ == "__main__":
    main()
