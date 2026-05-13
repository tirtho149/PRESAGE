"""
tests/test_biocap_shards.py
===========================
Smoke tests for ``scripts/build_biocap_shards.py``.

Builds a tiny TSV pointing at synthetic JPEGs in a tmp dir, runs the
shard packager, and asserts the resulting tar contains exactly the
expected `<key>.jpg + <key>.taxon.txt + <key>.caption.txt` triples
that BioCAP's data.py expects to find.
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from PIL import Image


def _write_synthetic_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "olive").save(path, format="JPEG")


def _write_captions_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def smoke_inputs(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    img_paths = []
    rows = []
    for i, (crop, disease, split) in enumerate([
        ("Tomato", "Early Blight",     "train"),
        ("Tomato", "Late Blight",      "train"),
        ("Tomato", "Bacterial Speck",  "val"),
    ]):
        image_id = f"smoke_{i:03d}"
        img_path = cache_dir / f"{image_id}.jpg"
        _write_synthetic_image(img_path)
        img_paths.append(img_path)
        rows.append({
            "image_id":     image_id,
            "image_path":   str(img_path),
            "crop":         crop,
            "disease":      disease,
            "state":        "TX",
            "taxon_text":   f"{crop} {disease}",
            "caption_text": f"A field photograph of {crop} affected by {disease}.",
            "split":        split,
        })
    captions_tsv = tmp_path / "captions.tsv"
    _write_captions_tsv(captions_tsv, rows)
    out_dir = tmp_path / "shards"
    return {
        "captions_tsv": captions_tsv,
        "out_dir":      out_dir,
        "row_count":    len(rows),
    }


def test_shard_builder_produces_expected_tar_layout(smoke_inputs):
    script = Path(__file__).parent.parent / "scripts" / "build_biocap_shards.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--captions",      str(smoke_inputs["captions_tsv"]),
         "--out-dir",       str(smoke_inputs["out_dir"]),
         "--shard-size-mb", "1"],  # small to force rollovers but our rows are tiny
        capture_output=True, text=True, check=True,
    )
    assert "TOTAL" in result.stdout, result.stdout

    train_shards = sorted((smoke_inputs["out_dir"] / "train").glob("shard-*.tar"))
    val_shards   = sorted((smoke_inputs["out_dir"] / "val").glob("shard-*.tar"))
    assert len(train_shards) >= 1
    assert len(val_shards)   >= 1

    # First train shard must contain a triple for the first sample.
    with tarfile.open(train_shards[0]) as tf:
        names = sorted(tf.getnames())
    keys = sorted({n.rsplit(".", 1)[0].rsplit(".", 1)[0] for n in names})
    # Each sample yields three files: <key>.jpg, <key>.taxon.txt, <key>.caption.txt
    assert any(name.endswith(".taxon.txt")   for name in names)
    assert any(name.endswith(".caption.txt") for name in names)
    assert any(name.endswith(".jpg")         for name in names)


def test_shard_builder_skips_missing_images(tmp_path: Path):
    """Rows whose image_path is missing should be silently dropped with
    a warning, not crash the whole shard build."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    img_id = "present"
    _write_synthetic_image(cache_dir / f"{img_id}.jpg")
    rows = [
        {
            "image_id": "present", "image_path": str(cache_dir / "present.jpg"),
            "crop": "Tomato", "disease": "Early Blight", "state": "TX",
            "taxon_text": "Tomato Early Blight",
            "caption_text": "x", "split": "train",
        },
        {
            "image_id": "missing", "image_path": str(cache_dir / "missing.jpg"),
            "crop": "Tomato", "disease": "Late Blight", "state": "TX",
            "taxon_text": "Tomato Late Blight",
            "caption_text": "y", "split": "train",
        },
    ]
    tsv = tmp_path / "captions.tsv"
    _write_captions_tsv(tsv, rows)
    out_dir = tmp_path / "shards"
    script = Path(__file__).parent.parent / "scripts" / "build_biocap_shards.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--captions",      str(tsv),
         "--out-dir",       str(out_dir),
         "--shard-size-mb", "1"],
        capture_output=True, text=True, check=True,
    )
    # exactly one sample makes it into a shard
    assert "1 samples packaged" in result.stdout
    assert "skipped 1 missing images" in (result.stderr + result.stdout)
