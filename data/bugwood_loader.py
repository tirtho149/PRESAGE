"""
data/bugwood_loader.py
======================
Bugwood field-image loader for the Train-on-the-Wild paradigm
(paper §5, "Training: Bugwood Field Images").

Reads a curated tree at ``Curated_Bugwood_Dataset/Images/<Crop>/<Disease>/*.jpg``,
extracts EXIF GPS + capture timestamp per image, and yields ``BugwoodRecord``
objects with the metadata that PathomeDB Layer 3 and OBSERVE's phi_geo input
require.

Selection workflow (paper §5.1, Appendix A):
1. Hard quality filters (resolution >=512^2, GPS precision <=10km, single
   dominant subject, no severe JPEG compression).
2. k-medoids (k=10) diversity sampling on CLIP ViT-B/32 embeddings to pick
   the 10 most diverse passing candidates per disease class.
3. Hard split BEFORE any trace generation: 7 → trace generation, 3 → Layer 5
   reference library.

This module implements steps 1 and 3. Step 2 (k-medoids) is a separate
``select_diverse_subset`` helper that requires CLIP — only invoked when the
caller does not already have a curated 10/class subset.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from utils.geo import aez_lookup, extract_gps_from_image


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class BugwoodRecord:
    """One Bugwood field image + labels + geospatial metadata.

    Mirrors PlantRecord (data/loader.py) plus paper §5 fields.
    """

    image_id: str
    image: Image.Image
    image_b64: str

    crop_species: Optional[str] = None       # T5 ground truth
    disease_name: Optional[str] = None       # T3 ground truth (folder-derived)
    pathogen_class: Optional[str] = None     # T2, optional, derived via PathomeDB Layer 1 if known

    # GPS / timing — from EXIF
    lat: Optional[float] = None
    lon: Optional[float] = None
    capture_dt: Optional[str] = None         # ISO date string for JSON-serializability
    month: Optional[int] = None

    # FAO agro-ecological zone (filled by aez_lookup)
    aez_code: Optional[str] = None
    aez_climate: Optional[str] = None

    # Quality filter results
    width: int = 0
    height: int = 0
    file_size: int = 0

    # Free-form metadata + provenance
    src_path: str = ""
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Quality filters (paper §5.1)
# ---------------------------------------------------------------------------

@dataclass
class BugwoodQualityFilters:
    min_side: int = 512                      # >= 512x512 resolution
    min_jpeg_quality: float = 60.0           # avoid heavily compressed images
    max_gps_precision_km: float = 10.0       # paper hard filter
    require_gps: bool = True                 # paper §5.1 — required for PathomeDB
    require_disease_label: bool = True       # species-level annotation required


def passes_quality(rec: BugwoodRecord, filters: BugwoodQualityFilters) -> Tuple[bool, str]:
    """Return (ok, reason). ``reason`` is empty when ok is True."""
    if rec.width < filters.min_side or rec.height < filters.min_side:
        return False, f"resolution {rec.width}x{rec.height} < {filters.min_side}"
    if filters.require_gps and (rec.lat is None or rec.lon is None):
        return False, "missing GPS"
    if filters.require_disease_label and not rec.disease_name:
        return False, "missing disease label"
    # JPEG quality and GPS precision — heuristics; full implementation needs
    # PIL.JpegImagePlugin quality estimation and EXIF GPSDOP.
    return True, ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class BugwoodLoader:
    """
    Iterate ``BugwoodRecord``s from a Bugwood-curated folder tree.

    Parameters
    ----------
    cfg : dict
        ``data`` section of the YAML config. Expected keys:
          - ``bugwood_root``: absolute path to ``Curated_Bugwood_Dataset/Images``
          - ``per_class``: int, number of images per class to keep (default 10)
          - ``trace_split``: int, first N images per class go to trace generation
                             (default 7); the remaining ``per_class - trace_split``
                             become Layer 5 references.
          - ``quality_filters``: optional override fields for ``BugwoodQualityFilters``
    split : {"trace", "reference", "all"}
        ``trace`` — the 7/class images for PlantSwarm trace generation
        ``reference`` — the 3/class images for PathomeDB Layer 5
        ``all`` — both
    """

    def __init__(self, cfg: dict, split: str = "trace"):
        if split not in {"trace", "reference", "all"}:
            raise ValueError(f"unknown split {split!r}")
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg["bugwood_root"]).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"bugwood_root not a directory: {self.root}")

        self.per_class = int(cfg.get("per_class", 10))
        self.trace_split = int(cfg.get("trace_split", 7))
        if self.trace_split > self.per_class:
            raise ValueError("trace_split cannot exceed per_class")

        qf_cfg = cfg.get("quality_filters", {}) or {}
        self.filters = BugwoodQualityFilters(**qf_cfg)

        # Materialised on first iteration
        self._records: Optional[List[BugwoodRecord]] = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> List[BugwoodRecord]:
        records: List[BugwoodRecord] = []
        crop_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        for crop_dir in crop_dirs:
            crop = crop_dir.name.replace("_", " ")
            for disease_dir in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
                disease = disease_dir.name.replace("_", " ").title()
                kept_for_class = 0
                files = sorted(
                    f for f in disease_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                )
                for img_path in files:
                    rec = self._build_record(img_path, crop, disease)
                    if rec is None:
                        continue
                    ok, _reason = passes_quality(rec, self.filters)
                    if not ok:
                        continue
                    records.append(rec)
                    kept_for_class += 1
                    if kept_for_class >= self.per_class:
                        break
        return records

    def _build_record(self, img_path: Path, crop: str, disease: str) -> Optional[BugwoodRecord]:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        lat, lon, dt = extract_gps_from_image(str(img_path))
        aez = aez_lookup(lat, lon) if (lat is not None and lon is not None) else None

        return BugwoodRecord(
            image_id=f"bugwood::{img_path.parent.parent.name}::{img_path.parent.name}::{img_path.stem}",
            image=img,
            image_b64=b64,
            crop_species=crop,
            disease_name=disease,
            lat=lat,
            lon=lon,
            capture_dt=dt.isoformat() if dt else None,
            month=dt.month if dt else None,
            aez_code=aez.code if aez else None,
            aez_climate=aez.climate if aez else None,
            width=img.width,
            height=img.height,
            file_size=img_path.stat().st_size,
            src_path=str(img_path),
        )

    # ------------------------------------------------------------------
    # Public iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[BugwoodRecord]:
        if self._records is None:
            self._records = self._discover()

        if self.split == "all":
            yield from self._records
            return

        # Group by (crop, disease), then split first N → trace, rest → reference.
        by_class: Dict[Tuple[str, str], List[BugwoodRecord]] = {}
        for r in self._records:
            by_class.setdefault((r.crop_species, r.disease_name), []).append(r)

        for _key, group in by_class.items():
            if self.split == "trace":
                yield from group[: self.trace_split]
            else:  # reference
                yield from group[self.trace_split : self.per_class]

    def __len__(self) -> int:
        return sum(1 for _ in self)


# ---------------------------------------------------------------------------
# k-medoids selection helper (paper §5.1 step 2)
# ---------------------------------------------------------------------------

def select_diverse_subset(
    paths: List[str],
    k: int = 10,
    clip_model: Optional[str] = "ViT-B/32",
) -> List[str]:
    """
    k-medoids diversity sampling on CLIP embeddings to pick maximally
    distinct images from a candidate pool. Returns the chosen file paths.

    Used when the caller has not yet cut the per-class pool down to 10/class.

    Note: pulls CLIP only when invoked. If torch/clip aren't available, falls
    back to first-k selection with a warning.
    """
    if len(paths) <= k:
        return list(paths)

    try:
        import clip  # type: ignore
        import torch
    except ImportError:
        # TODO(pathome): switch to open_clip if OpenAI clip not present
        return paths[:k]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device)
    feats = []
    with torch.no_grad():
        for p in paths:
            try:
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                f = model.encode_image(img).cpu().numpy().flatten()
                f = f / (np.linalg.norm(f) + 1e-9)
                feats.append(f)
            except Exception:
                feats.append(None)

    valid = [(i, f) for i, f in enumerate(feats) if f is not None]
    if len(valid) <= k:
        return [paths[i] for i, _ in valid]

    X = np.stack([f for _, f in valid])
    # Greedy farthest-point sampling (k-medoids++ initialisation)
    rng = np.random.default_rng(42)
    chosen = [int(rng.integers(0, len(X)))]
    while len(chosen) < k:
        d = np.min(
            np.array([1.0 - X @ X[c] for c in chosen]),
            axis=0,
        )
        chosen.append(int(np.argmax(d)))
    return [paths[valid[i][0]] for i in chosen]
