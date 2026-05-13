"""
scripts/build_pathomeood_shards.py
==============================
Package per-image (image bytes, taxon text, caption text) tuples into
WebDataset-compatible .tar shards for BioCAP training.

Input: the parquet/TSV produced by ``scripts/build_pathomeood_captions.py``.
Output layout:
    data/wds_shards/<crop>_<strategy>/train/shard-000000.tar
    data/wds_shards/<crop>_<strategy>/val/shard-000000.tar
    data/wds_shards/<crop>_<strategy>/holdout/shard-000000.tar   (if any)

Each sample in a shard:
    <key>.jpg            image bytes (whatever extension the cache holds)
    <key>.taxon.txt      short label-side string
    <key>.caption.txt    long descriptive caption

The shard format is plain tar (stdlib ``tarfile``); BioCAP's
``open_clip_train/data.py`` reads via ``webdataset.tarfile_to_samples``,
which expects exactly this layout.

Usage:
    python scripts/build_pathomeood_shards.py \\
        --captions data/bugwood_captions/Tomato_canonical_deltas_3.parquet \\
        --out-dir  data/wds_shards/Tomato_canonical_deltas_3 \\
        [--shard-size-mb 256]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--captions", required=True,
                   help="parquet or TSV from scripts/build_pathomeood_captions.py")
    p.add_argument("--out-dir", required=True,
                   help="root dir for shards (one subdir per split)")
    p.add_argument("--shard-size-mb", type=int, default=256,
                   help="target shard size in MB before rolling over")
    p.add_argument("--max-shards", type=int, default=None,
                   help="cap on shards per split (smoke-test knob)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap on total samples (smoke-test knob)")
    return p.parse_args()


def _read_captions(path: Path) -> List[Dict[str, str]]:
    """Read either parquet or TSV (auto-detect by suffix)."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as e:
            raise SystemExit(
                f"pyarrow required to read {path}. "
                f"Install with: pip install pyarrow"
            ) from e
        table = pq.read_table(path)
        return table.to_pylist()
    if suf in (".tsv", ".csv"):
        rows: List[Dict[str, str]] = []
        delim = "\t" if suf == ".tsv" else ","
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delim)
            for r in reader:
                rows.append(dict(r))
        return rows
    raise SystemExit(f"unsupported captions format: {path}")


def _write_shard(
    out_path: Path,
    samples: Iterable[Dict[str, bytes]],
) -> int:
    """Write one tar shard. Returns number of samples written."""
    n = 0
    with tarfile.open(out_path, mode="w") as tf:
        for s in samples:
            key = s["__key__"].decode() if isinstance(s["__key__"], bytes) else s["__key__"]
            for ext, payload in s.items():
                if ext == "__key__":
                    continue
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            n += 1
    return n


def _build_samples(
    rows: List[Dict[str, str]],
    max_samples: int | None,
) -> Iterable[Dict[str, bytes]]:
    """Yield one shard-ready dict per row. Skips rows whose image is
    missing on disk (reports a count at the end)."""
    skipped_missing = 0
    skipped_total = 0
    emitted = 0
    for r in rows:
        if max_samples is not None and emitted >= max_samples:
            break
        img_path = Path(r["image_path"])
        if not img_path.is_file():
            # Try alternative extensions in the same dir.
            stem = img_path.with_suffix("")
            for ext in ("jpg", "jpeg", "png", "webp"):
                cand = stem.with_suffix("." + ext)
                if cand.is_file():
                    img_path = cand
                    break
            else:
                skipped_missing += 1
                continue
        ext = img_path.suffix.lstrip(".").lower() or "jpg"
        if ext == "jpeg":
            ext = "jpg"
        try:
            img_bytes = img_path.read_bytes()
        except OSError:
            skipped_total += 1
            continue
        yield {
            "__key__":     r["image_id"],
            ext:           img_bytes,
            "taxon.txt":   r["taxon_text"].encode("utf-8"),
            "caption.txt": r["caption_text"].encode("utf-8"),
        }
        emitted += 1
    if skipped_missing or skipped_total:
        print(f"  WARNING: skipped {skipped_missing} missing images, "
              f"{skipped_total} read errors", file=sys.stderr)


def main() -> None:
    args = parse_args()

    rows = _read_captions(Path(args.captions))
    if not rows:
        raise SystemExit(f"no rows in {args.captions}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    by_split: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_split.setdefault(r["split"] or "train", []).append(r)

    target_bytes = args.shard_size_mb * 1024 * 1024
    print(f"=== build_pathomeood_shards ===")
    print(f"  captions  : {args.captions}  ({len(rows)} rows)")
    print(f"  out_dir   : {out_root}")
    print(f"  splits    : {dict((k, len(v)) for k, v in by_split.items())}")
    print(f"  target    : {args.shard_size_mb} MB / shard")

    total_samples = 0
    for split, srows in sorted(by_split.items()):
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        idx = 0
        buf: List[Dict[str, bytes]] = []
        buf_bytes = 0
        max_sam = args.max_samples
        for s in _build_samples(srows, max_samples=max_sam):
            buf.append(s)
            buf_bytes += sum(len(v) for k, v in s.items() if k != "__key__")
            if buf_bytes >= target_bytes:
                shard_path = split_dir / f"shard-{idx:06d}.tar"
                n = _write_shard(shard_path, buf)
                print(f"    {shard_path}  ({n} samples, {buf_bytes/1e6:.1f} MB)")
                total_samples += n
                idx += 1
                buf = []
                buf_bytes = 0
                if args.max_shards is not None and idx >= args.max_shards:
                    break
        if buf:  # flush
            shard_path = split_dir / f"shard-{idx:06d}.tar"
            n = _write_shard(shard_path, buf)
            print(f"    {shard_path}  ({n} samples, {buf_bytes/1e6:.1f} MB)")
            total_samples += n
            idx += 1
        print(f"  split={split:7s} : {idx} shards written")

    print(f"  TOTAL     : {total_samples} samples packaged")


if __name__ == "__main__":
    main()
