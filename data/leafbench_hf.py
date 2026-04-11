"""
data/leafbench_hf.py
====================
`enalis/LeafBench` from Hugging Face Datasets (VQA + multiple-choice).

Requires ``datasets``, a Hugging Face token for gated access, and ``HF_TOKEN`` in
``.env`` (see ``.env.example``).

Typical use: filter to ``DC`` (disease identification) so each row has a gold
``disease_name`` (T3) from the selected option text. Other splits in the paper
can use ``PC`` (pathogen / T2), ``CSI`` (crop / T5), etc. via config.
"""

from __future__ import annotations

import io
import os
import random
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image


def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The `datasets` package is required for LeafBench. Install with:\n"
            "  pip install datasets"
        ) from e


def _resolve_hf_token(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit.strip() or None
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None


def _answer_to_choice_text(row: Dict[str, Any]) -> Optional[str]:
    letter = str(row.get("answer") or "").strip().upper()[:1]
    if letter not in ("A", "B", "C", "D"):
        return None
    raw = row.get(letter)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    return str(raw).strip()


def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# Contiguous index ranges on ``test`` (13950 rows) — avoids slow full-scan ``filter``.
_LEAFBENCH_TEST_RANGES: Dict[str, tuple[int, int]] = {
    "HDC": (0, 2580),
    "DC": (2580, 5160),
    "CSI": (5160, 7740),
    "SNC": (7740, 9810),
    "PC": (9810, 11880),
    "SI": (11880, 13950),
}


def build_leafbench_dataframe(
    *,
    hf_dataset_id: str = "enalis/LeafBench",
    hf_split: str = "test",
    hf_token: Optional[str] = None,
    question_types: Optional[List[str]] = None,
    max_examples: Optional[int] = None,
    seed: int = 42,
    image_col: str = "image_bytes",
    benchmark: str = "leafbench",
    use_index_ranges: bool = True,
) -> pd.DataFrame:
    """
    Load LeafBench into a DataFrame compatible with :class:`data.loader.PlantDiagBenchLoader`.

    Rows are one (image, question) pair. Gold labels depend on ``question_type``:

    - ``DC``: disease name → ``disease_name`` (T3)
    - ``PC``: pathogen class → ``pathogen_class`` (T2)
    - ``CSI``: crop / species → ``crop_species`` (T5)
    - ``HDC``: healthy vs diseased (binary); stored in ``meta`` only by default
    - ``SNC``: scientific name; stored in ``meta`` / optional extension

    Parameters
    ----------
    question_types
        If set, keep only these ``question_type`` values (e.g. ``[\"DC\"]``).
    hf_token
        Overrides ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` from the environment.
    """
    _require_datasets()
    from datasets import load_dataset

    try:
        from utils.env import load_project_dotenv

        load_project_dotenv()
    except ImportError:
        pass

    token = _resolve_hf_token(hf_token)
    ds = load_dataset(hf_dataset_id, token=token, split=hf_split)

    if (
        use_index_ranges
        and hf_dataset_id == "enalis/LeafBench"
        and hf_split == "test"
        and question_types is not None
        and len(question_types) == 1
    ):
        qt = str(question_types[0]).strip()
        span = _LEAFBENCH_TEST_RANGES.get(qt)
        if span is not None:
            lo, hi = span
            ds = ds.select(range(lo, hi))

    elif question_types is not None:
        allowed = {str(x).strip() for x in question_types}
        ds = ds.filter(lambda ex: ex.get("question_type") in allowed)

    n = len(ds)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_examples is not None:
        indices = indices[: int(max_examples)]

    rows: List[Dict[str, Any]] = []
    for j in indices:
        ex = ds[j]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else Image.fromarray(img)

        jpeg_bytes = _pil_to_jpeg_bytes(img)
        qid = str(ex.get("question_id", j))
        qtype = str(ex.get("question_type", ""))
        choice_text = _answer_to_choice_text(ex)

        disease_name: Optional[str] = None
        pathogen_class: Optional[str] = None
        crop_species: Optional[str] = None

        if qtype == "DC" and choice_text:
            disease_name = choice_text
        elif qtype == "PC" and choice_text:
            pathogen_class = choice_text
        elif qtype == "CSI" and choice_text:
            crop_species = choice_text

        rows.append(
            {
                "id": f"{qid}_{qtype}",
                image_col: jpeg_bytes,
                "symptom_type": None,
                "pathogen_class": pathogen_class,
                "disease_name": disease_name,
                "severity_class": None,
                "crop_species": crop_species,
                "benchmark": benchmark,
                "leafbench_question_type": qtype,
                "leafbench_question": ex.get("question"),
                "leafbench_answer_letter": ex.get("answer"),
            }
        )

    return pd.DataFrame(rows)
