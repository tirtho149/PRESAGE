"""
data/tfds_plant_village.py
==========================
Load `plant_village` from TensorFlow Datasets (TFDS) into a pandas DataFrame
compatible with ``PlantDiagBenchLoader``.

Catalog: https://www.tensorflow.org/datasets/catalog/plant_village

Requires::
    pip install tensorflow tensorflow-datasets

Example::

    df = build_plant_village_dataframe(max_examples=1000, data_dir=\"~/tensorflow_datasets\")
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image


def _require_tfds():
    try:
        import tensorflow_datasets as tfds  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TensorFlow Datasets is required for data.tfds_name: plant_village.\n"
            "  pip install -r requirements-tfds.txt\n"
            "or: pip install tensorflow tensorflow-datasets\n"
            "(Run pip on its own line; do not paste shell comments after the command.)"
        ) from e


def _decode_tfds_filename(ex: Dict[str, Any]) -> str:
    raw = ex.get("image/filename")
    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw).decode("utf-8", errors="replace")
    return str(raw)


def _tfds_root(data_dir: Optional[str]) -> Path:
    return Path(data_dir).expanduser() if data_dir else Path.home() / "tensorflow_datasets"


def _clear_plant_village_incomplete_dirs(data_dir: Optional[str]) -> None:
    """Remove TFDS ``incomplete.*`` temp dirs left after a crash (they block re-prepare)."""
    root = _tfds_root(data_dir) / "plant_village"
    if not root.is_dir():
        return
    for p in root.glob("incomplete.*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def _label_to_crop_disease(class_name: str) -> tuple[str, str]:
    """TFDS class names often look like ``Apple___Apple_scab`` or ``Corn_(maize)___Common_rust_``."""
    s = str(class_name).strip()
    if "___" in s:
        crop, rest = s.split("___", 1)
        crop = crop.replace("_", " ").strip()
        disease = rest.replace("_", " ").strip()
        return crop, disease
    return "Unknown", s


def build_plant_village_dataframe(
    *,
    max_examples: Optional[int] = None,
    data_dir: Optional[str] = None,
    split: str = "train",
    seed: int = 42,
    image_col: str = "image_bytes",
    jpeg_quality: int = 95,
) -> pd.DataFrame:
    """
    Download (if needed) and iterate Plant Village from TFDS; return a DataFrame with:

    - ``id``: stable string id
    - ``image_bytes``: JPEG bytes (for ``PlantDiagBenchLoader`` with ``image_col: image_bytes``)
    - ``disease_name`` (T3): full TFDS class string (38-way)
    - ``crop_species`` (T5): crop parsed from class name
    - ``pathogen_class`` (T2): ``\"Unknown\"`` (not provided by TFDS)
    - ``symptom_type`` (T1), ``severity_class`` (T4): ``None``
    - ``benchmark``: ``\"plantvillage\"`` for LaTeX ``by_benchmark`` sync
    """
    _require_tfds()
    import tensorflow as tf

    tf.random.set_seed(seed)

    import tensorflow_datasets as tfds

    builder = tfds.builder("plant_village", data_dir=data_dir)
    # Stale ``incomplete.*`` dirs or an existing version folder can make TFDS raise
    # FileExistsError when renaming the temp dir onto ``<version>``. Clear incomplete
    # temps first; if prepare still hits FileExistsError, the version dir is usually
    # already usable—continue to ``as_dataset``. If loading fails, remove
    # ``~/tensorflow_datasets/plant_village`` and retry.
    _clear_plant_village_incomplete_dirs(data_dir)
    try:
        builder.download_and_prepare()
    except FileExistsError:
        pass
    ds = builder.as_dataset(split=split, as_supervised=False, shuffle_files=True)
    try:
        n_ex = int(builder.info.splits[split].num_examples)
    except (KeyError, TypeError, AttributeError):
        n_ex = 10_000
    ds = ds.shuffle(min(10_000, int(n_ex)), seed=seed)
    if max_examples is not None:
        ds = ds.take(int(max_examples))

    info = builder.info
    label_feature = info.features["label"]
    names = label_feature.names  # type: ignore[attr-defined]

    rows: list[Dict[str, Any]] = []
    for i, ex in enumerate(tfds.as_numpy(ds)):
        img_arr = ex["image"]
        if not isinstance(img_arr, np.ndarray):
            img_arr = np.asarray(img_arr)
        label_id = int(ex["label"])
        class_name = names[label_id]
        crop, disease = _label_to_crop_disease(class_name)

        pil = Image.fromarray(img_arr.astype("uint8"), mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=jpeg_quality)
        jpeg_bytes = buf.getvalue()

        rows.append(
            {
                "id": f"pv_{i:06d}",
                image_col: jpeg_bytes,
                "disease_name": class_name,
                "crop_species": crop,
                "pathogen_class": "Unknown",
                "symptom_type": None,
                "severity_class": None,
                "benchmark": "plantvillage",
                "tfds_label_id": label_id,
                "tfds_filename": _decode_tfds_filename(ex),
            }
        )

    return pd.DataFrame(rows)
