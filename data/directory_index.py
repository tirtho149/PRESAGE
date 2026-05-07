"""
data/directory_index.py
=======================
Build a tabular index from an image tree: labels inferred from folder names.

Typical layout (CyAg-style):
    Images/<crop_species>/<disease_name>/photo.jpg

Auto-mapping by depth (when ``segment_tasks`` is omitted):
    1 folder  -> T3 disease_name
    2 folders -> T5 crop_species, T3 disease_name
    3 folders -> T5, T2 pathogen_class, T3 disease_name
    4 folders -> T5, T2, T1 symptom_type, T3 disease_name
    5 folders -> T5, T2, T1, T4 severity_class, T3 disease_name

Override with ``data.directory_layout.segment_tasks`` (same length as folder depth).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

TASK_TO_COLUMN = {
    "T1": "symptom_type",
    "T2": "pathogen_class",
    "T3": "disease_name",
    "T4": "severity_class",
    "T5": "crop_species",
}


def normalize_folder_label(name: str) -> str:
    """Turn ``Cordana_Leaf_Spot`` into ``Cordana Leaf Spot`` for prompts."""
    s = str(name).strip()
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def default_segment_tasks(num_segments: int) -> List[str]:
    if num_segments <= 0:
        return []
    if num_segments == 1:
        return ["T3"]
    if num_segments == 2:
        return ["T5", "T3"]
    if num_segments == 3:
        return ["T5", "T2", "T3"]
    if num_segments == 4:
        return ["T5", "T2", "T1", "T3"]
    if num_segments >= 5:
        # Deeper trees: crop, pathogen, symptom, severity, disease (disease may absorb extra levels)
        return ["T5", "T2", "T1", "T4", "T3"]


def _iter_image_paths(root: Path, extensions: set[str], follow_symlinks: bool) -> List[Path]:
    out: List[Path] = []
    root = root.resolve()
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        for fn in filenames:
            suf = Path(fn).suffix.lower()
            if suf not in extensions:
                continue
            p = Path(dirpath) / fn
            if p.is_file():
                out.append(p.resolve())
    return sorted(out)


def _relative_dir_parts(path: Path, root: Path) -> List[str]:
    rel = path.relative_to(root)
    parts = list(rel.parts[:-1])
    return [normalize_folder_label(p) for p in parts]


def infer_layout_depth(paths: List[Path], root: Path) -> Tuple[int, List[Path]]:
    """Pick the most common number of parent directories; keep only matching files."""
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    depths = []
    for p in paths:
        parts = _relative_dir_parts(p, root)
        depths.append(len(parts))
    mode_depth, _count = Counter(depths).most_common(1)[0]
    kept = [p for p in paths if len(_relative_dir_parts(p, root)) == mode_depth]
    if not kept:
        kept = paths
        mode_depth = depths[0]
    if len(kept) < len(paths):
        import warnings

        warnings.warn(
            f"Directory layout: using depth {mode_depth} for {len(kept)}/{len(paths)} images "
            f"(dropped {len(paths) - len(kept)} with other depths). "
            f"Set data.directory_layout.segment_tasks to control mapping.",
            stacklevel=2,
        )
    return mode_depth, kept


def build_directory_dataframe(cfg: dict) -> pd.DataFrame:
    """
    Parameters
    ----------
    cfg : data section of YAML (directory_root, directory_layout, label_cols, ...).
    """
    root = Path(cfg["directory_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"directory_root is not a directory: {root}")

    ext_cfg = cfg.get("image_extensions")
    if ext_cfg:
        extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in ext_cfg}
    else:
        extensions = set(IMAGE_EXTENSIONS)

    follow = bool(cfg.get("follow_symlinks", False))
    all_paths = _iter_image_paths(root, extensions, follow)
    mode_depth, paths = infer_layout_depth(all_paths, root)

    layout = cfg.get("directory_layout") or {}
    segment_tasks: Optional[List[str]] = layout.get("segment_tasks")
    if segment_tasks:
        norm: List[str] = []
        for x in segment_tasks:
            s = str(x).strip().upper()
            if len(s) == 2 and s[0] == "T" and s[1] in "12345":
                norm.append(s)
            else:
                raise ValueError(
                    f"Invalid segment_tasks entry {x!r}; use T1, T2, T3, T4, or T5 "
                    f"(see data/directory_index.py)."
                )
        segment_tasks = norm
        if len(segment_tasks) != mode_depth:
            paths = [p for p in all_paths if len(_relative_dir_parts(p, root)) == len(segment_tasks)]
            if not paths:
                raise FileNotFoundError(
                    f"No images with exactly {len(segment_tasks)} folder levels under {root}"
                )
            mode_depth = len(segment_tasks)
    else:
        segment_tasks = default_segment_tasks(min(mode_depth, 5))
        if mode_depth > 5:
            # Five semantic slots; deeper folders are joined into disease (T3).
            pass
        elif len(segment_tasks) != mode_depth:
            raise ValueError(
                f"Internal layout error: depth {mode_depth} vs tasks {segment_tasks}. "
                f"Set directory_layout.segment_tasks explicitly."
            )

    lc = cfg.get("label_cols", {})
    id_col = cfg.get("id_col", "id")
    image_col = cfg.get("image_col", "image_path")

    rows = []
    for p in paths:
        dir_parts = _relative_dir_parts(p, root)
        rel = p.relative_to(root)
        image_id = str(rel.as_posix())

        row = {
            id_col: image_id,
            image_col: str(p),
            lc.get("T1", "symptom_type"): "Unknown",
            lc.get("T2", "pathogen_class"): "Unknown",
            lc.get("T3", "disease_name"): "Unknown",
            lc.get("T4", "severity_class"): "Unknown",
            lc.get("T5", "crop_species"): "Unknown",
        }

        if len(dir_parts) > len(segment_tasks) and segment_tasks and segment_tasks[-1] == "T3":
            fixed = list(dir_parts[: len(segment_tasks) - 1])
            tail = dir_parts[len(segment_tasks) - 1 :]
            fixed.append(normalize_folder_label(" / ".join(tail)))
            dir_parts = fixed

        for i, task in enumerate(segment_tasks):
            if i >= len(dir_parts):
                break
            col = lc.get(task, TASK_TO_COLUMN[task])
            row[col] = dir_parts[i]

        rows.append(row)

    return pd.DataFrame(rows)
