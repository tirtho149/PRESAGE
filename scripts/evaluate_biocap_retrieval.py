"""
scripts/evaluate_biocap_retrieval.py
====================================
Bugwood text↔image retrieval mini-bench (paper Table 2 analog).

Bench construction
------------------
At caption-build time, ``scripts/build_biocap_captions.py --holdout-state X``
marks all rows from state ``X`` as ``split=holdout``. Those rows form
the retrieval bench: each held-out image's caption (KB-canonical +
its state's deltas) is the query, and the model must retrieve the
right image from the held-out pool, and vice versa.

What this script does
---------------------
  1. Reads the captions parquet/TSV
  2. Filters to ``split == "holdout"``
  3. Writes a CSV with columns ``id, captions`` consumable by
     ``evaluation.retrieval_openclip``
  4. Runs the same R@k computation as the paper, but with a dataset
     class that accepts an absolute image_path (Bugwood cache has
     mixed extensions, unlike Cornell-Bird which is all .jpg)
  5. Writes I2T + T2I R@{1,5,10} to a JSON file

Usage:
    python scripts/evaluate_biocap_retrieval.py \\
        --model    hf-hub:imageomics/biocap \\
        --captions data/bugwood_captions/Tomato_canonical_deltas_3.parquet \\
        --out-dir  results/biocap_eval/biocap_hf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model",      required=True,
                   help="HF hub path or open_clip model name (passed to create_model_and_transforms)")
    p.add_argument("--pretrained", default="")
    p.add_argument("--captions",   required=True,
                   help="parquet/TSV produced by build_biocap_captions.py")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def _read_captions(path: Path) -> List[Dict[str, str]]:
    suf = path.suffix.lower()
    if suf == ".parquet":
        import pyarrow.parquet as pq  # type: ignore
        return pq.read_table(path).to_pylist()
    if suf in (".tsv", ".csv"):
        delim = "\t" if suf == ".tsv" else ","
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter=delim))
    raise SystemExit(f"unsupported captions format: {path}")


def _r_at_k(similarity, k: int) -> float:
    """For each query i (row), is i among top-k columns of similarity[i]?"""
    n = similarity.shape[0]
    hits = 0
    for i in range(n):
        idx = similarity[i].argsort()[-k:]
        if i in idx.tolist():
            hits += 1
    return hits / n


def main() -> None:
    args = parse_args()
    captions_path = Path(args.captions)
    rows = [r for r in _read_captions(captions_path) if r.get("split") == "holdout"]
    if not rows:
        raise SystemExit(
            f"no holdout rows in {captions_path}. Re-run build_biocap_captions.py "
            f"with --holdout-state <S>."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Emit the retrieval CSV (matches retrieval_openclip's format).
    csv_path = out_dir / "_retrieval_bench.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "captions", "image_path"])
        for r in rows:
            w.writerow([r["image_id"], r["caption_text"], r["image_path"]])
    print(f"=== evaluate_biocap_retrieval ===")
    print(f"  captions      : {captions_path}")
    print(f"  holdout pairs : {len(rows)}")
    print(f"  bench CSV     : {csv_path}")

    # 2. Lazy import (torch / open_clip / PIL needed only at run-time).
    repo_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(repo_root / "train_and_eval"))

    import torch
    import open_clip
    from PIL import Image

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained or None,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    img_feats: List[torch.Tensor] = []
    txt_feats: List[torch.Tensor] = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            img_path = Path(r["image_path"])
            if not img_path.is_file():
                # Try alt extensions in same dir.
                stem = img_path.with_suffix("")
                for ext in ("jpg", "jpeg", "png", "webp"):
                    cand = stem.with_suffix("." + ext)
                    if cand.is_file():
                        img_path = cand
                        break
                else:
                    raise SystemExit(f"missing retrieval image: {img_path}")
            img = Image.open(img_path).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(device)
            txt_input = tokenizer([f"a photo of {r['caption_text']}"]).to(device)
            img_feat = model.encode_image(img_input)
            if isinstance(img_feat, tuple):
                img_feat = img_feat[0]
            txt_feat = model.encode_text(txt_input)
            if isinstance(txt_feat, tuple):
                txt_feat = txt_feat[0]
            img_feats.append(img_feat)
            txt_feats.append(txt_feat)

    img_feats = torch.stack(img_feats).squeeze(1)
    txt_feats = torch.stack(txt_feats).squeeze(1)
    img_feats /= img_feats.norm(dim=-1, keepdim=True)
    txt_feats /= txt_feats.norm(dim=-1, keepdim=True)

    sim_i2t = (img_feats @ txt_feats.T).cpu().numpy()
    sim_t2i = sim_i2t.T

    metrics = {
        "i2t_r1":  _r_at_k(sim_i2t, 1),
        "i2t_r5":  _r_at_k(sim_i2t, 5),
        "i2t_r10": _r_at_k(sim_i2t, 10),
        "t2i_r1":  _r_at_k(sim_t2i, 1),
        "t2i_r5":  _r_at_k(sim_t2i, 5),
        "t2i_r10": _r_at_k(sim_t2i, 10),
        "n_pairs": len(rows),
    }
    out_json = out_dir / "retrieval.json"
    out_json.write_text(json.dumps({
        "model":    args.model,
        "captions": str(captions_path),
        "metrics":  metrics,
    }, indent=2))
    print(f"  metrics       : {metrics}")
    print(f"  wrote         : {out_json}")


if __name__ == "__main__":
    main()
