"""
data/plantwild_local.py
=======================
Load PlantWild dataset from local cloned directory.

Usage:
    from data.plantwild_local import build_plantwild_local_dataframe

    df = build_plantwild_local_dataframe(
        data_dir="data/PlantWild",
        max_examples=None,
        image_col="image",
        seed=42
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def build_plantwild_local_dataframe(
    data_dir: str | Path = "data/PlantWild",
    *,
    max_examples: Optional[int] = None,
    image_col: str = "image",
    seed: int = 42,
    benchmark_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build PlantWild DataFrame from local cloned directory.

    PlantWild dataset structure (from HuggingFace):
    - Image files organized by disease name
    - Filename format: <crop>___<disease>.jpg
    - Disease name parsed to extract crop (T5) and disease (T3)

    Args:
        data_dir: Path to cloned PlantWild directory
        max_examples: Limit number of examples (None = all)
        image_col: Column name for image paths
        seed: Random seed for sampling
        benchmark_col: Column name for benchmark label

    Returns:
        DataFrame with columns:
        - image_path: Full path to image file
        - T3 (disease name): Extracted from filename
        - T5 (crop species): Extracted from filename
        - benchmark: "plantwild" (if benchmark_col specified)
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"PlantWild dataset not found at {data_dir}\n"
            f"Download with: sbatch scripts/submit_setup_plantwild.sh"
        )

    logger.info(f"Loading PlantWild dataset from {data_dir}")

    # Collect all image files
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = Path(root) / file
                image_files.append(full_path)

    logger.info(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        raise ValueError(f"No image files found in {data_dir}")

    # Parse filenames to extract labels
    rows = []
    for img_path in image_files:
        filename = img_path.stem  # Remove extension

        # Try to parse disease name from parent directory or filename
        # PlantWild format: typically stored as subdirs or in filename
        # Example: tomato___early_blight.jpg
        if "___" in filename:
            crop_disease = filename.split("___")
            if len(crop_disease) == 2:
                crop = crop_disease[0].lower()
                disease = crop_disease[1].lower().replace("_", " ")
                rows.append({
                    image_col: str(img_path),
                    "T3": disease,  # Disease name
                    "T5": crop,     # Crop species
                })
        else:
            # Fallback: just use filename as disease
            rows.append({
                image_col: str(img_path),
                "T3": filename.lower(),
                "T5": "unknown",
            })

    df = pd.DataFrame(rows)

    # Add benchmark label if requested
    if benchmark_col:
        df[benchmark_col] = "plantwild"

    # Add missing label columns (T1, T2, T4) filled with None
    for col in ["T1", "T2", "T4"]:
        if col not in df.columns:
            df[col] = None

    # Limit examples if requested
    if max_examples is not None and len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)
        logger.info(f"Sampled {max_examples} examples")

    logger.info(f"Loaded {len(df)} PlantWild images")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Sample row:\n{df.iloc[0] if len(df) > 0 else 'N/A'}")

    return df


def get_plantwild_splits(
    data_dir: str | Path = "data/PlantWild",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load PlantWild and split into train/val/test.

    Args:
        data_dir: Path to dataset
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for testing
        seed: Random seed

    Returns:
        (train_df, val_df, test_df)
    """
    df = build_plantwild_local_dataframe(data_dir, seed=seed)

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split
    n = len(df)
    train_idx = int(n * train_frac)
    val_idx = int(n * (train_frac + val_frac))

    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]

    return train_df, val_df, test_df
