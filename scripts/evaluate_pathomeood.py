"""
scripts/evaluate_pathomeood.py
==========================
Zero-shot classification eval for PathomeOOD (paper Tables 1, 3,
4, 17, 19, 20 — anything that boils down to top-1/top-5 on a
folder-per-class image set).

For each (model, eval_dataset) pair, this script:
  1. Walks <eval_root> (folder per class) and emits a CSV in BioCAP's
     ``DatasetFromFile`` format: index, filepath, class.
  2. Invokes ``evaluation.zero_shot_iid.zero_shot_eval`` programmatically
     using the produced CSV and writes per-class metrics to
     ``results/pathomeood_eval/<run_id>/<dataset>.json``.

Supported eval datasets:
  - plantvillage : folder-per-class like ``Tomato___Early_blight``
  - plantwild    : either PV-style or ``<crop>_<disease>``
  - plantdoc     : flat ``<Crop> <Disease>`` (added in Task #13)
  - bugwood_holdout : the held-out state slice from caption building

Models can be:
  - HF hub paths:  ``hf-hub:imageomics/biocap``, ``hf-hub:imageomics/bioclip``
  - Local CKPTS:   ``checkpoints/T04/T04/checkpoints/epoch_50.pt``

Usage (single model, multiple datasets):
    python scripts/evaluate_pathomeood.py \\
        --model    hf-hub:imageomics/biocap \\
        --pv-root  /path/to/PlantVillage \\
        --pw-root  /path/to/PlantWild \\
        --crop     Tomato \\
        --out-dir  results/pathomeood_eval/pathomeood_hf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Folder → (crop, disease) normalizers — same logic OBSERVE uses.
# ---------------------------------------------------------------------------

def normalize_pv_folder(folder_name: str) -> Optional[Tuple[str, str]]:
    parts = folder_name.split("___")
    if len(parts) != 2:
        return None
    crop_raw, disease_raw = parts
    crop = crop_raw.replace("_", " ").strip()
    crop = re.sub(r"\s*\(.*?\)\s*", "", crop).strip()
    if crop.lower() == "pepper, bell":
        crop = "Bell Pepper"
    elif crop.lower() in ("corn, maize", "corn"):
        crop = "Corn"
    elif crop.lower() in ("cherry, including sour", "cherry"):
        crop = "Cherry"
    disease = disease_raw.replace("_", " ").strip()
    disease = re.sub(r"\s+", " ", disease)
    disease = "healthy" if disease.lower() == "healthy" else disease.title()
    return crop, disease


def normalize_pw_folder(folder_name: str) -> Optional[Tuple[str, str]]:
    parsed = normalize_pv_folder(folder_name)
    if parsed is not None:
        return parsed
    m = re.match(r"^([^_]+)_(.+)$", folder_name)
    if not m:
        return None
    crop_part = m.group(1).replace("-", " ").title()
    disease_part = m.group(2).replace("_", " ").title()
    return crop_part, disease_part


def normalize_plantdoc_folder(folder_name: str) -> Optional[Tuple[str, str]]:
    # PlantDoc uses "Tomato Early blight leaf", "Apple Scab Leaf", etc.
    s = folder_name.replace("_", " ").strip()
    s = re.sub(r"\s+leaf\s*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ", 1)
    if len(parts) < 2:
        return None
    crop, disease = parts
    return crop.title(), disease.title() if disease.lower() != "healthy" else "healthy"


_NORMALIZERS = {
    "plantvillage": normalize_pv_folder,
    "plantwild":    normalize_pw_folder,
    "plantdoc":     normalize_plantdoc_folder,
}


# ---------------------------------------------------------------------------
# Build the CSV that BioCAP's DatasetFromFile expects
# ---------------------------------------------------------------------------

def build_eval_csv(
    *,
    eval_root: Path,
    dataset_kind: str,
    crop: str,
    out_csv: Path,
    limit_per_class: Optional[int] = None,
) -> Dict[str, int]:
    norm = _NORMALIZERS[dataset_kind]
    rows: List[Dict[str, str]] = []
    per_class: Counter = Counter()
    skipped: Counter = Counter()
    for sub in sorted(eval_root.iterdir()):
        if not sub.is_dir():
            continue
        parsed = norm(sub.name)
        if parsed is None:
            skipped["unparsable"] += 1
            continue
        folder_crop, folder_disease = parsed
        if folder_crop.lower() != crop.lower():
            skipped["other_crop"] += 1
            continue
        class_label = f"{folder_crop} {folder_disease}"  # matches taxon_text format
        files = []
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
            files.extend(sub.glob(f"*{ext}"))
        files = sorted(files)
        if limit_per_class is not None:
            files = files[:limit_per_class]
        for f in files:
            rows.append({
                "filepath": str(f.relative_to(eval_root)),
                "class":    class_label,
            })
            per_class[class_label] += 1
    if not rows:
        raise SystemExit(
            f"no eligible samples under {eval_root} for crop={crop!r} "
            f"(dataset_kind={dataset_kind})"
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "filepath", "class"])
        for i, r in enumerate(rows):
            w.writerow([i, r["filepath"], r["class"]])
    return {
        "n_samples":   len(rows),
        "n_classes":   len(per_class),
        "per_class":   dict(per_class),
        "skipped":     dict(skipped),
        "csv_path":    str(out_csv),
    }


# ---------------------------------------------------------------------------
# Programmatic invocation of BioCAP's zero_shot_eval
# ---------------------------------------------------------------------------

def run_eval(
    *,
    model_name: str,
    pretrained: str,
    data_root: Path,
    label_csv: Path,
    text_type: str,
    projector_type: str,
    batch_size: int,
    workers: int,
    device: Optional[str] = None,
) -> Dict:
    """Call evaluation.zero_shot_eval and return its metrics dict.

    Lazily imports the BioCAP modules — they have torch/openclip deps
    that aren't on the laptop, so this only runs on the GPU host.
    """
    repo_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(repo_root / "train_and_eval"))

    import torch  # noqa: F401
    from open_clip import create_model_and_transforms
    from evaluation.data    import DatasetFromFile
    from evaluation.utils   import init_device
    from evaluation.params  import parse_args as eval_parse_args
    from evaluation.zero_shot_iid import zero_shot_eval, get_dataloader

    args = eval_parse_args([
        "--model",          model_name,
        "--pretrained",     pretrained,
        "--data_root",      str(data_root),
        "--label_filename", str(label_csv),
        "--text_type",      text_type,
        "--projector_type", projector_type,
        "--batch-size",     str(batch_size),
        "--workers",        str(workers),
        "--logs",           "none",
    ])
    args.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.save_logs = False
    args.log_path = None
    init_device(args)

    model, _, preprocess_val = create_model_and_transforms(
        args.model, args.pretrained,
        precision=args.precision, device=args.device,
        output_dict=True,
    )
    model.eval()

    data = {
        "val-unseen": get_dataloader(
            DatasetFromFile(
                args.data_root, args.label_filename,
                transform=preprocess_val, classes=args.text_type,
            ),
            batch_size=args.batch_size, num_workers=args.workers,
        ),
    }
    metrics = zero_shot_eval(model, data, args)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", required=True,
                   help="HF hub path or local ckpt name passed to BioCAP")
    p.add_argument("--pretrained", default="",
                   help="Empty for HF hub; openai/laion/etc for fresh inits")
    p.add_argument("--crop", default="Tomato")
    p.add_argument("--pv-root", default=None, help="PlantVillage root")
    p.add_argument("--pw-root", default=None, help="PlantWild root")
    p.add_argument("--plantdoc-root", default=None, help="PlantDoc root")
    p.add_argument("--out-dir", required=True,
                   help="Output dir for per-dataset JSON results")
    p.add_argument("--text-type", default="asis",
                   help="DatasetFromFile classes mode")
    p.add_argument("--projector-type", default="tax",
                   choices=("tax", "caption"),
                   help="Which projector head to use at inference time")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--limit-per-class", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_csv"
    tmp_dir.mkdir(exist_ok=True)

    summary: Dict[str, Dict] = {"model": args.model, "crop": args.crop, "evals": {}}

    targets = []
    if args.pv_root:        targets.append(("plantvillage", Path(args.pv_root)))
    if args.pw_root:        targets.append(("plantwild",    Path(args.pw_root)))
    if args.plantdoc_root:  targets.append(("plantdoc",     Path(args.plantdoc_root)))
    if not targets:
        raise SystemExit("provide at least one of --pv-root / --pw-root / --plantdoc-root")

    for kind, root in targets:
        if not root.is_dir():
            print(f"  [{kind}] root not found: {root} — skipping")
            continue
        csv_path = tmp_dir / f"{kind}.csv"
        stats = build_eval_csv(
            eval_root=root, dataset_kind=kind, crop=args.crop,
            out_csv=csv_path, limit_per_class=args.limit_per_class,
        )
        print(f"=== {kind} ===")
        print(f"  root      : {root}")
        print(f"  csv       : {csv_path}")
        print(f"  samples   : {stats['n_samples']}  classes: {stats['n_classes']}")
        for cls, n in sorted(stats["per_class"].items()):
            print(f"    {cls:40s}  n={n}")

        try:
            metrics = run_eval(
                model_name=args.model,
                pretrained=args.pretrained,
                data_root=root,
                label_csv=csv_path,
                text_type=args.text_type,
                projector_type=args.projector_type,
                batch_size=args.batch_size,
                workers=args.workers,
                device=args.device,
            )
        except Exception as e:
            print(f"  [{kind}] EVAL FAILED: {e}")
            summary["evals"][kind] = {"error": str(e), "stats": stats}
            continue

        out_json = out_dir / f"{kind}.json"
        payload = {
            "model":     args.model,
            "crop":      args.crop,
            "stats":     stats,
            "metrics":   metrics,
            "config": {
                "text_type":      args.text_type,
                "projector_type": args.projector_type,
                "batch_size":     args.batch_size,
            },
        }
        out_json.write_text(json.dumps(payload, indent=2))
        summary["evals"][kind] = metrics
        print(f"  metrics   : {metrics}")
        print(f"  wrote     : {out_json}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  summary -> {summary_path}")


if __name__ == "__main__":
    main()
