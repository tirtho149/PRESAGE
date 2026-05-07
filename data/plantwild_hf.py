"""
data/plantwild_hf.py
====================
Load PlantWild (wild plant disease images) from HuggingFace Datasets.
PlantWild provides controlled-to-wild domain shift evaluation (§6, paper).

Class names follow crop___disease format: parses to T5 (crop) and T3 (disease).
Infers T2 (pathogen class) from disease→pathogen mapping (same as PlantVillage).
"""

from __future__ import annotations

import io
import random
from typing import Optional

import numpy as np


def _require_datasets():
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "PlantWild loader requires `datasets`. "
            "Install with: pip install datasets"
        )


def _disease_to_pathogen(disease_name: str) -> str:
    """
    Map common disease names to pathogen class.
    Fallback: if not found, return "Unknown".
    """
    disease_to_pathogen_map = {
        # Fungal
        "early_blight": "Fungal",
        "late_blight": "Fungal",
        "leaf_spot": "Fungal",
        "powdery_mildew": "Fungal",
        "rust": "Fungal",
        "anthracnose": "Fungal",
        "scab": "Fungal",
        "canker": "Fungal",
        "mildew": "Fungal",
        "blight": "Fungal",
        "rot": "Fungal",
        "damping_off": "Fungal",
        "sooty_blotch": "Fungal",
        "flyspeck": "Fungal",
        "black_rot": "Fungal",
        # Bacterial
        "bacterial_leaf_scorch": "Bacterial",
        "bacterial_spot": "Bacterial",
        "bacterial_blight": "Bacterial",
        "bacterial_canker": "Bacterial",
        "bacterial_speck": "Bacterial",
        "bacterial_wilt": "Bacterial",
        "fire_blight": "Bacterial",
        "xanthomonas": "Bacterial",
        # Viral
        "mosaic": "Viral",
        "yellows": "Viral",
        "virus": "Viral",
        "leafroll": "Viral",
        "leaf_curl": "Viral",
        # Pest/Other
        "spider_mites": "Pest damage",
        "powdery": "Fungal",
        "downy_mildew": "Fungal",
        "nutrient_deficiency": "Nutrient deficiency",
    }
    disease_lower = disease_name.lower().replace(" ", "_").replace("-", "_")
    for key, pathogen in disease_to_pathogen_map.items():
        if key in disease_lower:
            return pathogen
    return "Unknown"


def build_plantwild_dataframe(
    hf_dataset_id: str,
    *,
    split: str = "test",
    max_examples: Optional[int] = None,
    seed: int = 42,
    image_col: str = "image_bytes",
    jpeg_quality: int = 95,
) -> dict:
    """
    Load PlantWild dataset from HuggingFace and return a dict with table structure.

    Args:
        hf_dataset_id: HF dataset ID, e.g. "rashikahura/plantWild" or similar.
        split: Dataset split (default "test" for OOD evaluation).
        max_examples: Max images to load (None = all).
        seed: Random seed for sampling.
        image_col: Output column name for image bytes.
        jpeg_quality: JPEG encoding quality.

    Returns:
        dict with keys: id, {image_col}, disease_name, crop_species, pathogen_class,
                       symptom_type, severity_class, benchmark.
    """
    datasets = _require_datasets()

    # Load dataset
    try:
        ds = datasets.load_dataset(hf_dataset_id, split=split)
    except Exception as e:
        raise ValueError(
            f"Failed to load PlantWild from {hf_dataset_id}: {e}. "
            "Check the dataset ID and ensure HF_TOKEN is set if gated."
        ) from e

    if max_examples and len(ds) > max_examples:
        indices = random.Random(seed).sample(range(len(ds)), max_examples)
        ds = ds.select(indices)

    # Infer column names (assume "label", "image", or standard naming)
    cols = ds.column_names
    label_col = "label" if "label" in cols else next((c for c in cols if "class" in c.lower()), None)
    image_col_src = "image" if "image" in cols else next((c for c in cols if "image" in c.lower() or "img" in c.lower()), None)

    if not image_col_src:
        raise ValueError(f"No image column found in {hf_dataset_id}. Columns: {cols}")
    if not label_col:
        raise ValueError(f"No label column found in {hf_dataset_id}. Columns: {cols}")

    # Build dataframe
    rows = []
    for i, example in enumerate(ds):
        # Parse label (assume crop___disease format like PlantVillage)
        label_str = str(example[label_col])
        if "___" in label_str:
            crop_name, disease_name = label_str.rsplit("___", 1)
        else:
            crop_name, disease_name = "Unknown", label_str

        # Normalize names
        crop_species = crop_name.replace("_", " ").title()
        disease_name = disease_name.replace("_", " ").title()

        # Infer pathogen class
        pathogen_class = _disease_to_pathogen(disease_name)

        # Load and encode image
        image_obj = example[image_col_src]
        if hasattr(image_obj, "convert"):  # PIL Image
            buf = io.BytesIO()
            image_obj.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
            image_bytes = buf.getvalue()
        elif isinstance(image_obj, bytes):
            image_bytes = image_obj
        elif isinstance(image_obj, np.ndarray):
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(image_obj.astype("uint8")).convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
            image_bytes = buf.getvalue()
        else:
            raise ValueError(f"Unsupported image type: {type(image_obj)}")

        rows.append({
            "id": f"plantwild_{i:06d}",
            image_col: image_bytes,
            "disease_name": disease_name,
            "crop_species": crop_species,
            "pathogen_class": pathogen_class,
            "symptom_type": None,  # Not provided by PlantWild
            "severity_class": None,  # Not provided by PlantWild
            "benchmark": "plantwild",
        })

    return {
        "rows": rows,
        "num_examples": len(rows),
    }
