"""
data/bugwood_loader.py
======================
Bugwood field-image loader for the Train-on-the-Wild paradigm
(paper §5, "Training: Bugwood Field Images").

Source of truth is ``BugWood_Diseases.csv`` (the Bugwood IPMNet export).
Each row carries an ``Image URL`` plus host (crop), subject (disease) and
US-state location, but no per-image GPS or capture date — geolocation is
therefore at state-centroid resolution and the ``month`` axis is treated
as "unknown" (sentinel 0). PathomeDB Layer 3 is consequently a
state-resolution histogram rather than the AEZ-month grid the paper §6.3
sketches against ideal EXIF data; the rest of the pipeline is unaffected.

Selection workflow (paper §5.1, Appendix A — adapted for CSV input):
1. Crop normalisation via the same exact-match map used by
   ``DataLoader.load_BugwoodMerged`` (subset embedded below to avoid
   importing the 6.7k-line DataLoader module).
2. Disease normalisation: strip the parenthetical scientific suffix from
   ``Subject Display Name`` to yield the common-name disease label.
3. Hard quality filters (state present, normalised crop + disease present,
   reachable URL when ``download.enabled``).
4. Per-class capping (default 10) and 7/3 trace/reference split.
5. k-medoids diversity sampling on CLIP embeddings (``select_diverse_subset``)
   is still available when callers want to pick the per-class subset
   directly from the candidate pool.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from utils.geo import aez_lookup, extract_gps_from_image, state_to_latlon


# ---------------------------------------------------------------------------
# Crop / disease normalisation
# (lifted from DataLoader.BUGWOOD_EXACT_CROP_MAP — kept in sync by hand)
# ---------------------------------------------------------------------------

BUGWOOD_EXACT_CROP_MAP: Dict[str, str] = {
    "alfalfa": "Alfalfa", "apple": "Apple", "banana": "Banana",
    "bananas": "Banana", "basil": "Basil", "bean": "Bean",
    "bell pepper": "Bell Pepper", "bell_pepper": "Bell Pepper",
    "blueberry": "Blueberry", "broccoli": "Broccoli", "cabbage": "Cabbage",
    "carrot": "Carrot", "cashew": "Cashew", "cassava": "Cassava",
    "cauliflower": "Cauliflower", "celery": "Celery", "cherry": "Cherry",
    "chickpea": "Chickpea", "citrus": "Citrus", "coconut": "Coconut",
    "coconut palm": "Coconut", "coffee": "Coffee", "corn": "Corn",
    "common corn": "Corn", "field corn": "Corn", "sweet corn": "Corn",
    "cotton": "Cotton", "cucumber": "Cucumber", "durian": "Durian",
    "eggplant": "Eggplant", "garden tomato": "Tomato", "garlic": "Garlic",
    "ginger": "Ginger", "grape": "Grape", "grapevine": "Grape",
    "wine grape": "Grape", "lettuce": "Lettuce", "mango": "Mango",
    "maple": "Maple", "melon": "Melon", "watermelon": "Watermelon",
    "muskmelon": "Melon", "cantaloupe": "Melon",
    "common hop": "Hops", "hop": "Hops",
    "oak": "Oak", "orange": "Orange", "orange haunglongbing": "Orange",
    "orange huanglongbing": "Orange", "peach": "Peach", "pear": "Pear",
    "pepper": "Pepper", "plum": "Plum", "potato": "Potato",
    "sweetpotato": "Sweet Potato", "sweet potato": "Sweet Potato",
    "pumpkin": "Pumpkin", "raspberry": "Raspberry", "rice": "Rice",
    "rose": "Rose", "rye": "Rye", "soybean": "Soybean", "squash": "Squash",
    "squashes (general)": "Squash", "winter squash": "Squash",
    "summer squash": "Squash", "strawberry": "Strawberry",
    "sugarcane": "Sugarcane", "tea": "Tea", "tobacco": "Tobacco",
    "tomato": "Tomato", "vanilla": "Vanilla", "wheat": "Wheat",
    "common wheat": "Wheat", "durum wheat": "Wheat", "spring wheat": "Wheat",
    "winter wheat": "Wheat", "zucchini": "Zucchini",
}

# Bugwood taxonomy entries that are not crop hosts and should be dropped
# rather than promoted to a crop folder. Mirrors DataLoader.BUGWOOD_NON_CROP_KEYS.
BUGWOOD_NON_CROP_KEYS = {
    "wood decay fungi", "wood decay fungus", "canker complex",
    "shelf fungi", "bark beetle", "powdery mildew", "downy mildew",
}


def _normalize_key(value: str) -> str:
    s = str(value or "").strip().lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Normalise lookup keys once so callers don't need to think about whether the
# key in the literal map happened to contain parens / punctuation that
# _normalize_key would strip (e.g. "squashes (general)" → "squashes general").
_CROP_LOOKUP: Dict[str, str] = {_normalize_key(k): v for k, v in BUGWOOD_EXACT_CROP_MAP.items()}
_NON_CROP_LOOKUP: set = {_normalize_key(k) for k in BUGWOOD_NON_CROP_KEYS}


def _map_crop(raw_crop: str) -> Optional[str]:
    """Canonicalize a Bugwood ``Host Name`` to a crop label, or None to drop."""
    key = _normalize_key(raw_crop)
    if not key or key in _NON_CROP_LOOKUP:
        return None
    if key in _CROP_LOOKUP:
        return _CROP_LOOKUP[key]
    # Fallback: title-case the raw host so unmapped-but-plausible entries
    # ("oak", "rose") still survive instead of being silently dropped.
    pretty = re.sub(r"\s+", " ", str(raw_crop or "").replace("_", " ")).strip().title()
    return pretty or None


_DISEASE_PAREN_RE = re.compile(r"\s*\(.*$")


def _clean_disease(raw_disease: str) -> str:
    """Strip the parenthetical scientific suffix from Subject Display Name.

    "Phytophthora blight (Phytophthora capsici Leonian)" -> "Phytophthora blight"
    """
    if not raw_disease:
        return ""
    base = _DISEASE_PAREN_RE.sub("", str(raw_disease)).strip()
    return base.title() if base else ""


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class BugwoodRecord:
    """One Bugwood field image + labels + geospatial metadata.

    Mirrors PlantRecord (data/loader.py) plus paper §5 fields. ``lat`` /
    ``lon`` / ``aez_*`` are state-centroid-resolution when sourced from
    the CSV; ``month`` is None (no capture date in the CSV export).
    """

    image_id: str
    image: Optional[Image.Image]
    image_b64: str

    crop_species: Optional[str] = None
    disease_name: Optional[str] = None
    pathogen_class: Optional[str] = None

    lat: Optional[float] = None
    lon: Optional[float] = None
    capture_dt: Optional[str] = None
    month: Optional[int] = None

    aez_code: Optional[str] = None
    aez_climate: Optional[str] = None

    width: int = 0
    height: int = 0
    file_size: int = 0

    src_path: str = ""
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

@dataclass
class BugwoodQualityFilters:
    min_side: int = 256                      # CSV thumbnails go down to 768x512; older default 512 was too strict
    require_state: bool = True               # Layer-3 needs a US state
    require_disease_label: bool = True       # paper §5.1


def _passes_quality(rec: BugwoodRecord, filters: BugwoodQualityFilters) -> Tuple[bool, str]:
    if filters.require_disease_label and not rec.disease_name:
        return False, "missing disease label"
    if filters.require_state and (rec.lat is None or rec.lon is None):
        return False, "missing state location"
    if rec.image is not None and (rec.width < filters.min_side or rec.height < filters.min_side):
        return False, f"resolution {rec.width}x{rec.height} < {filters.min_side}"
    return True, ""


# ---------------------------------------------------------------------------
# CSV-driven loader
# ---------------------------------------------------------------------------

class BugwoodLoader:
    """Iterate ``BugwoodRecord``s from ``BugWood_Diseases.csv``.

    Parameters
    ----------
    cfg : dict
        ``data`` section of the YAML config. Recognised keys:

        - ``csv_path`` (required): path to ``BugWood_Diseases.csv``.
        - ``image_cache_dir`` (default ``./.bugwood_cache``): where downloaded
          JPEGs land. Cache key is the Bugwood Image Number.
        - ``per_class``: max images per ``(crop, disease)`` after filtering
          (default 10).
        - ``trace_split``: first N per class go to traces; the remaining
          ``per_class - trace_split`` become Layer-5 references (default 7).
        - ``min_per_class``: drop classes whose post-filter pool is smaller
          than this (default 1; raise to 5+ for the tighter paper subset).
        - ``download``: dict with ``enabled`` (bool, default True),
          ``timeout`` (s, default 20), ``retries`` (int, default 2). When
          ``enabled`` is False, records are yielded with ``image=None`` and
          ``image_b64=""`` for fast metadata-only iteration.
        - ``quality_filters``: overrides for ``BugwoodQualityFilters``.

        Legacy folder-tree mode is honoured for backwards compatibility:
        if ``csv_path`` is missing but ``bugwood_root`` is set and points at
        an existing directory, the loader falls back to the original
        per-folder discovery path that uses EXIF GPS.

    split : {"trace", "val", "reference", "all"}
        - "trace"     — first ``trace_split`` images per class (training pool
                        for Phase 2 stochastic Qwen-VL trace generation).
        - "val"       — held-out Bugwood images for in-domain evaluation;
                        guaranteed image-disjoint from "trace" by Image
                        Number slicing.
        - "reference" — back-compat alias for "val" (older configs).
        - "all"       — every kept image, no split semantics.
    """

    def __init__(self, cfg: dict, split: str = "trace"):
        # Back-compat: older configs called the held-out split "reference"
        # (it doubled as the PathomeDB Layer-5 exemplar pool). The split is
        # now exposed as "val" — same slice, new semantics.
        if split == "reference":
            split = "val"
        if split not in {"trace", "val", "all"}:
            raise ValueError(f"unknown split {split!r}")
        self.cfg = cfg
        self.split = split

        self.csv_path: Optional[Path] = None
        self.bugwood_root: Optional[Path] = None
        if cfg.get("csv_path"):
            self.csv_path = Path(cfg["csv_path"]).expanduser()
        elif cfg.get("bugwood_root"):
            root = Path(cfg["bugwood_root"]).expanduser()
            if root.is_dir():
                self.bugwood_root = root
        if self.csv_path is None and self.bugwood_root is None:
            raise FileNotFoundError(
                "BugwoodLoader requires data.csv_path (preferred) or "
                "data.bugwood_root pointing to an existing directory"
            )
        if self.csv_path is not None and not self.csv_path.is_file():
            raise FileNotFoundError(f"csv_path not found: {self.csv_path}")

        self.per_class = int(cfg.get("per_class", 10))
        self.trace_split = int(cfg.get("trace_split", 7))
        if self.trace_split > self.per_class:
            raise ValueError("trace_split cannot exceed per_class")
        self.min_per_class = int(cfg.get("min_per_class", 1))

        download_cfg = cfg.get("download") or {}
        self.download_enabled = bool(download_cfg.get("enabled", True))
        self.download_timeout = float(download_cfg.get("timeout", 20.0))
        self.download_retries = int(download_cfg.get("retries", 2))

        cache_dir = cfg.get("image_cache_dir", ".bugwood_cache")
        self.cache_dir = Path(cache_dir).expanduser()
        if self.download_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        qf_cfg = cfg.get("quality_filters", {}) or {}
        # Drop legacy-only keys silently so old configs still load.
        qf_cfg = {k: v for k, v in qf_cfg.items()
                  if k in {"min_side", "require_state", "require_disease_label"}}
        self.filters = BugwoodQualityFilters(**qf_cfg)

        self._records: Optional[List[BugwoodRecord]] = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> List[BugwoodRecord]:
        if self.csv_path is not None:
            return self._discover_from_csv()
        return self._discover_from_folder_tree()

    def _discover_from_csv(self) -> List[BugwoodRecord]:
        # First pass: read all rows, normalise, group by (crop, disease).
        groups: Dict[Tuple[str, str], List[dict]] = {}
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                crop = _map_crop(row.get("Host Name", ""))
                disease = _clean_disease(row.get("Subject Display Name", ""))
                if not crop or not disease:
                    continue
                state = (row.get("Location") or "").strip()
                if self.filters.require_state and not state:
                    continue
                groups.setdefault((crop, disease), []).append({
                    "image_number": (row.get("Image Number") or "").strip(),
                    "image_url": (row.get("Image URL") or "").strip(),
                    "state": state,
                    "host_raw": (row.get("Host Name") or "").strip(),
                    "subject_raw": (row.get("Subject Display Name") or "").strip(),
                    "scientific": (row.get("Scientific Name") or "").strip(),
                    "photographer": (row.get("Photographer") or "").strip(),
                    "organization": (row.get("Organization") or "").strip(),
                })

        # Second pass: cap per class, materialise records.
        records: List[BugwoodRecord] = []
        for (crop, disease), rows in groups.items():
            # Deduplicate by image_number — the raw Bugwood CSV occasionally
            # has the same (image, crop, disease) row twice (different
            # descriptor codes). Keeping both makes the deterministic
            # trace/val slice produce the same image_id in both halves.
            seen_numbers: set = set()
            unique_rows = []
            for row in rows:
                n = row["image_number"]
                if n and n not in seen_numbers:
                    seen_numbers.add(n)
                    unique_rows.append(row)
            rows = unique_rows
            if len(rows) < self.min_per_class:
                continue
            # Stable ordering: by Image Number ascending so the 7/3 split is
            # deterministic across runs.
            rows.sort(key=lambda r: int(r["image_number"]) if r["image_number"].isdigit() else 0)
            for row in rows[: self.per_class]:
                rec = self._build_record_from_csv_row(row, crop, disease)
                if rec is None:
                    continue
                ok, _reason = _passes_quality(rec, self.filters)
                if not ok:
                    continue
                records.append(rec)
        return records

    def _build_record_from_csv_row(
        self, row: dict, crop: str, disease: str,
    ) -> Optional[BugwoodRecord]:
        image_number = row["image_number"] or hashlib.md5(row["image_url"].encode()).hexdigest()[:10]
        lat, lon = state_to_latlon(row["state"])
        aez = aez_lookup(lat, lon) if (lat is not None and lon is not None) else None

        image: Optional[Image.Image] = None
        image_b64 = ""
        width = height = 0
        file_size = 0
        cached_path = ""

        if self.download_enabled and row["image_url"]:
            cached_path = self._fetch_image(row["image_url"], image_number)
            if cached_path and os.path.isfile(cached_path):
                try:
                    image = Image.open(cached_path).convert("RGB")
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG", quality=92)
                    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    width, height = image.width, image.height
                    file_size = os.path.getsize(cached_path)
                except Exception:
                    image = None

        return BugwoodRecord(
            image_id=f"bugwood::{image_number}",
            image=image,
            image_b64=image_b64,
            crop_species=crop,
            disease_name=disease,
            lat=lat,
            lon=lon,
            capture_dt=None,
            month=0,  # sentinel: CSV has no capture date; Layer 3 collapses time
            aez_code=aez.code if aez else None,
            aez_climate=aez.climate if aez else None,
            width=width,
            height=height,
            file_size=file_size,
            src_path=cached_path or row["image_url"],
            meta={
                "host_raw": row["host_raw"],
                "subject_raw": row["subject_raw"],
                "scientific": row["scientific"],
                "state": row["state"],
                "photographer": row["photographer"],
                "organization": row["organization"],
                "image_url": row["image_url"],
            },
        )

    def _fetch_image(self, url: str, image_number: str) -> str:
        """Download ``url`` into ``cache_dir`` and return the local path.

        Idempotent: returns immediately if a cached file exists. Empty string
        on failure so the caller can still emit a metadata-only record.
        """
        suffix = os.path.splitext(url.split("?", 1)[0])[1] or ".jpg"
        if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        dest = self.cache_dir / f"{image_number}{suffix}"
        if dest.exists() and dest.stat().st_size > 0:
            return str(dest)

        last_err: Optional[Exception] = None
        for attempt in range(self.download_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "PlantSwarm/Pathome (research)"},
                )
                with urllib.request.urlopen(req, timeout=self.download_timeout) as resp:
                    data = resp.read()
                if not data:
                    raise IOError("empty response")
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                return str(dest)
            except (urllib.error.URLError, IOError, TimeoutError) as e:
                last_err = e
                if attempt < self.download_retries:
                    time.sleep(0.5 * (attempt + 1))
        # Drop a sidecar so subsequent runs can see why we gave up.
        try:
            (self.cache_dir / f"{image_number}.failed").write_text(
                f"{url}\n{type(last_err).__name__}: {last_err}\n"
            )
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Legacy folder-tree path (only used when csv_path is unset)
    # ------------------------------------------------------------------

    def _discover_from_folder_tree(self) -> List[BugwoodRecord]:
        records: List[BugwoodRecord] = []
        assert self.bugwood_root is not None
        crop_dirs = sorted(p for p in self.bugwood_root.iterdir() if p.is_dir())
        for crop_dir in crop_dirs:
            crop = _map_crop(crop_dir.name) or crop_dir.name.replace("_", " ").title()
            for disease_dir in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
                disease = _clean_disease(disease_dir.name.replace("_", " "))
                kept = 0
                files = sorted(
                    f for f in disease_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                )
                for img_path in files:
                    rec = self._build_record_from_path(img_path, crop, disease)
                    if rec is None:
                        continue
                    ok, _reason = _passes_quality(rec, self.filters)
                    if not ok:
                        continue
                    records.append(rec)
                    kept += 1
                    if kept >= self.per_class:
                        break
        return records

    def _build_record_from_path(
        self, img_path: Path, crop: str, disease: str,
    ) -> Optional[BugwoodRecord]:
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

        by_class: Dict[Tuple[str, str], List[BugwoodRecord]] = {}
        for r in self._records:
            by_class.setdefault((r.crop_species, r.disease_name), []).append(r)

        for _key, group in by_class.items():
            if self.split == "trace":
                yield from group[: self.trace_split]
            else:  # "val" — held-out, image-disjoint from "trace"
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
    """k-medoids diversity sampling on CLIP embeddings to pick maximally
    distinct images from a candidate pool. Returns the chosen file paths.

    Used when the caller has not yet cut the per-class pool down to 10/class.

    Note: pulls CLIP only when invoked. If torch/clip aren't available, falls
    back to first-k selection.
    """
    if len(paths) <= k:
        return list(paths)

    try:
        import clip  # type: ignore
        import torch
    except ImportError:
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
    rng = np.random.default_rng(42)
    chosen = [int(rng.integers(0, len(X)))]
    while len(chosen) < k:
        d = np.min(
            np.array([1.0 - X @ X[c] for c in chosen]),
            axis=0,
        )
        chosen.append(int(np.argmax(d)))
    return [paths[valid[i][0]] for i in chosen]
