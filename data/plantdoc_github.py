"""
data/plantdoc_github.py
=======================
Index the **Cropped PlantDoc** dataset from the official GitHub tree:

    https://github.com/pratikkayal/PlantDoc-Dataset

Expected layout after ``git clone``::

    <repo>/
      train/<class_name>/*.jpg
      test/<class_name>/*.jpg

Each ``<class_name>`` folder is one disease (or healthy) label; folder depth is **1**
under ``train`` or ``test``, so labels map to **T3** (``disease_name``) by default.
A ``benchmark`` column is set to ``plantdoc`` for LaTeX ``by_benchmark`` metrics.

License: CC BY 4.0 (see repo ``LICENSE.txt``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from data.directory_index import build_directory_dataframe


def build_plantdoc_dataframe(
    repo_root: str | Path,
    *,
    split: str = "train",
    image_col: str = "image_path",
    id_col: str = "id",
    label_cols: Optional[Dict[str, str]] = None,
    infer_crop_for_t5: bool = True,
    benchmark: str = "plantdoc",
) -> pd.DataFrame:
    """
    Build a DataFrame from ``<repo_root>/<split>/`` using the same rules as
    :func:`data.directory_index.build_directory_dataframe`.

    Parameters
    ----------
    repo_root
        Path to the cloned repository root (the folder containing ``train/`` and ``test/``).
    split
        ``\"train\"`` or ``\"test\"``.
    infer_crop_for_t5
        If True, set ``crop_species`` (T5) when the class folder name contains ``___``
        or a leading token that looks like a crop name.
    """
    root = Path(repo_root).expanduser().resolve()
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"PlantDoc split not found: {split_dir}\n"
            "Clone: git clone https://github.com/pratikkayal/PlantDoc-Dataset.git"
        )

    lc = label_cols or {
        "T1": "symptom_type",
        "T2": "pathogen_class",
        "T3": "disease_name",
        "T4": "severity_class",
        "T5": "crop_species",
    }
    cfg: Dict[str, Any] = {
        "directory_root": str(split_dir),
        "directory_layout": {},
        "label_cols": lc,
        "id_col": id_col,
        "image_col": image_col,
        "image_extensions": None,
        "follow_symlinks": False,
    }

    df = build_directory_dataframe(cfg)
    df = df.copy()
    df["benchmark"] = benchmark

    t3_col = lc.get("T3", "disease_name")
    t5_col = lc.get("T5", "crop_species")

    if infer_crop_for_t5 and t3_col in df.columns and t5_col in df.columns:
        # Class folders normalize to spaces (e.g. ``Tomato_Late_Blight`` → multi-word T3).
        # Use first token as a coarse crop proxy when there are ≥2 tokens.
        for idx in df.index:
            parts = [p for p in str(df.at[idx, t3_col]).split() if p]
            if len(parts) >= 2:
                df.at[idx, t5_col] = parts[0]

    return df
