"""
scripts/evaluate_biocap_fewshot.py
==================================
Few-shot classification eval for BioCAP-on-Bugwood (paper Tables 18, 20).

Standard "prototype-mean" K-shot protocol:
    1. Encode every image in the eval set with the frozen visual encoder.
    2. For each random seed in [0..n_seeds):
       - For each class, sample K shots (the "support" set).
       - Class prototype = mean of K support features.
       - The remaining images in that class form the query set.
       - Predict argmax cosine similarity to class prototypes.
    3. Report top-1 mean ± std across seeds, per dataset.

Supports any model accepted by ``open_clip.create_model_and_transforms``
(HF hub paths, local checkpoints). The same folder normalizers used by
``scripts/evaluate_biocap.py`` apply.

Usage:
    python scripts/evaluate_biocap_fewshot.py \\
        --model    hf-hub:imageomics/biocap \\
        --pv-root  /path/to/PlantVillage \\
        --pw-root  /path/to/PlantWild \\
        --crop     Tomato --shots 1 5 --n-seeds 5 \\
        --out-dir  results/biocap_eval/biocap_hf
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_biocap import (
    normalize_pv_folder, normalize_pw_folder, normalize_plantdoc_folder,
)


_NORMALIZERS = {
    "plantvillage": normalize_pv_folder,
    "plantwild":    normalize_pw_folder,
    "plantdoc":     normalize_plantdoc_folder,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", required=True)
    p.add_argument("--pretrained", default="")
    p.add_argument("--crop", default="Tomato")
    p.add_argument("--pv-root", default=None)
    p.add_argument("--pw-root", default=None)
    p.add_argument("--plantdoc-root", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args()


def collect_paths(root: Path, kind: str, crop: str) -> Dict[str, List[Path]]:
    """Walk folder-per-class root → {class_label: [image_paths...]}."""
    norm = _NORMALIZERS[kind]
    out: Dict[str, List[Path]] = defaultdict(list)
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        parsed = norm(sub.name)
        if parsed is None:
            continue
        folder_crop, folder_disease = parsed
        if folder_crop.lower() != crop.lower():
            continue
        label = f"{folder_crop} {folder_disease}"
        files = []
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
            files.extend(sub.glob(f"*{ext}"))
        out[label].extend(sorted(files))
    return out


def encode_images(model, preprocess, device, paths: List[Path], batch_size: int):
    import torch
    from PIL import Image
    feats: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = (
                __import__("torch").stack(imgs).to(device)
                if imgs else None
            )
            if batch is None:
                continue
            feat = model.encode_image(batch)
            if isinstance(feat, tuple):
                feat = feat[0]
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat.cpu())
    if not feats:
        return None
    return __import__("torch").cat(feats, dim=0)


def kshot_accuracy(features_by_class: Dict[str, "torch.Tensor"], k: int, n_seeds: int) -> List[float]:
    import torch
    classes = sorted(features_by_class.keys())
    accs: List[float] = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        protos: List["torch.Tensor"] = []
        queries: List[Tuple[int, "torch.Tensor"]] = []
        for idx, cls in enumerate(classes):
            feats = features_by_class[cls]
            n = feats.shape[0]
            if n <= k:
                continue
            perm = list(range(n))
            rng.shuffle(perm)
            support = feats[perm[:k]]
            query = feats[perm[k:]]
            proto = support.mean(dim=0)
            proto = proto / proto.norm()
            protos.append(proto)
            for q in query:
                queries.append((idx, q))
        if not protos or not queries:
            continue
        P = torch.stack(protos, dim=0)  # [C, D]
        correct = 0
        for true_idx, q in queries:
            sims = P @ q
            pred = int(sims.argmax().item())
            if pred == true_idx:
                correct += 1
        accs.append(correct / len(queries))
    return accs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.pv_root:       targets.append(("plantvillage", Path(args.pv_root)))
    if args.pw_root:       targets.append(("plantwild",    Path(args.pw_root)))
    if args.plantdoc_root: targets.append(("plantdoc",     Path(args.plantdoc_root)))
    if not targets:
        raise SystemExit("provide at least one of --pv-root / --pw-root / --plantdoc-root")

    # Lazy imports — torch only available on the GPU host.
    repo_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(repo_root / "train_and_eval"))
    import torch
    import open_clip

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained or None,
    )
    model = model.to(device).eval()

    summary: Dict[str, Dict] = {"model": args.model, "crop": args.crop, "evals": {}}

    for kind, root in targets:
        if not root.is_dir():
            print(f"  [{kind}] root not found: {root} — skipping")
            continue
        paths_by_cls = collect_paths(root, kind, args.crop)
        if not paths_by_cls:
            print(f"  [{kind}] no samples for crop={args.crop}")
            continue
        print(f"=== {kind} few-shot ===")
        feats_by_cls: Dict[str, "torch.Tensor"] = {}
        for cls, paths in paths_by_cls.items():
            f = encode_images(model, preprocess, device, paths, args.batch_size)
            if f is None:
                continue
            feats_by_cls[cls] = f
            print(f"  {cls:40s}  n_samples={len(paths)}")
        per_shot: Dict[str, Dict[str, float]] = {}
        for k in args.shots:
            accs = kshot_accuracy(feats_by_cls, k=k, n_seeds=args.n_seeds)
            if not accs:
                per_shot[f"{k}_shot"] = {"mean": 0.0, "std": 0.0, "n_seeds": 0}
                continue
            per_shot[f"{k}_shot"] = {
                "mean":    mean(accs),
                "std":     stdev(accs) if len(accs) > 1 else 0.0,
                "n_seeds": len(accs),
                "all":     accs,
            }
            print(f"  k={k}: {mean(accs):.3f} ± {stdev(accs) if len(accs)>1 else 0.0:.3f} over {len(accs)} seeds")
        summary["evals"][kind] = per_shot
        (out_dir / f"fewshot_{kind}.json").write_text(json.dumps(per_shot, indent=2))

    (out_dir / "fewshot_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  summary -> {out_dir / 'fewshot_summary.json'}")


if __name__ == "__main__":
    main()
