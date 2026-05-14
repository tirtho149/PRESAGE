"""
scripts/build_features.py
=========================
Extract fixed-dimension multimodal features for the TabPFN classifier
step of the PathomeOOD pipeline.

For each (encoder, caption_strategy) pair the script produces:

  data/bugwood_features/<encoder>_<strategy>.npz
      X_train         (N_train, D_total)    feature matrix
      y_train         (N_train,)            integer class labels
      class_names     (C,)                  list of "Crop Disease" strings
      meta            dict with feature-block widths

And separately for the eval sets:

  data/eval_features/<encoder>_<strategy>_<dataset>.npz
      X_eval          (N_eval, D_total)
      y_eval          (N_eval,)             integer class id (mapped onto
                                            class_names from the matching
                                            Bugwood file)
      class_names     same as train file
      eval_paths      list of file paths (for debugging / error analysis)

Per-image feature vector (D_total ≈ 1100 for BioCLIP + KB caption + crop):

    [ image_emb         | caption_emb       | crop_onehot ]
    (D_image ~512-1024)   (D_text ~512)       (D_crop ~200)

State and geo (lat/lon) are intentionally OMITTED because PV / PD / PW
don't carry per-image state — keeping the test-time feature pipeline
parameter-free is more important than the small Bugwood-train signal
loss. The training class signal comes through `crop_onehot` (which is
always available since folder names give it on every eval set).

Supported encoders (see ENCODERS below): bioclip, clip_vitb16,
siglip_vitb16, biocliplab2 (BioCLIP-2). Add more by extending ENCODERS.

Usage
-----
  python scripts/build_features.py \\
      --captions data/bugwood_captions/Tomato_canonical_deltas_3.parquet \\
      --encoder  bioclip \\
      --eval-pv  data/eval/PlantVillage \\
      --eval-pd  data/eval/PlantDoc/test \\
      --eval-pw  data/eval/PlantWild
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from plantswarm.captioning import (
    build_disease_caption, build_fallback_caption, build_healthy_caption,
    load_kb_profiles, taxon_text,
)
from scripts.evaluate_pathomeood import (
    normalize_pv_folder, normalize_pw_folder, normalize_plantdoc_folder,
)


# ---------------------------------------------------------------------------
# Encoder registry — maps user-facing name to (open_clip_model, pretrained_tag)
# ---------------------------------------------------------------------------

ENCODERS: Dict[str, Tuple[str, Optional[str]]] = {
    "bioclip":         ("hf-hub:imageomics/bioclip",        None),
    "bioclip2":        ("hf-hub:imageomics/bioclip-2",      None),
    "clip_vitb16":     ("ViT-B-16",                         "openai"),
    "siglip_vitb16":   ("hf-hub:timm/ViT-B-16-SigLIP-256",  None),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--captions", required=True,
                   help="path to a captions parquet/TSV (output of "
                        "build_pathomeood_captions.py)")
    p.add_argument("--encoder", required=True, choices=sorted(ENCODERS),
                   help="frozen visual+text encoder used to produce the "
                        "image_emb and caption_emb columns")
    p.add_argument("--kb-root", default="artifacts/pathome_kb")
    p.add_argument("--strategy", default=None,
                   help="caption strategy for eval-set text encoding "
                        "(default: inferred from captions file name)")
    p.add_argument("--out-train", default=None,
                   help="output train .npz (default: "
                        "data/bugwood_features/<encoder>_<strategy>.npz)")
    p.add_argument("--out-eval-root", default="data/eval_features",
                   help="output dir for eval .npz files")
    p.add_argument("--eval-pv", default=None, help="PlantVillage root")
    p.add_argument("--eval-pd", default=None, help="PlantDoc/test root")
    p.add_argument("--eval-pw", default=None, help="PlantWild root")
    p.add_argument("--crop-filter", default=None,
                   help="restrict eval to one crop (e.g. Tomato)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Encoder loader
# ---------------------------------------------------------------------------

def load_encoder(name: str, device: Optional[str] = None):
    """Returns (visual_encoder, text_encoder, preprocess, tokenizer, device).

    The same `open_clip` model object is used for both image and text
    encoding so the embeddings live in a shared space — critical for
    the caption_emb feature to be meaningful when concatenated with the
    image_emb.
    """
    try:
        import torch
        import open_clip
    except ImportError as e:
        raise SystemExit(
            f"build_features needs torch + open_clip_torch installed "
            f"(import failed: {e})"
        )
    model_name, pretrained = ENCODERS[name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def encode_images(
    model, preprocess, device, paths: Sequence[Path], batch_size: int,
) -> "np.ndarray":
    import torch
    from PIL import Image
    feats: List["np.ndarray"] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(imgs).to(device)
            f = model.encode_image(x)
            # BioCAP-fork returns a tuple when --dual-projector is active;
            # off-shelf encoders return a single tensor. Handle both.
            if isinstance(f, tuple):
                f = f[0]
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)


def encode_texts(
    model, tokenizer, device, texts: Sequence[str], batch_size: int,
) -> "np.ndarray":
    import torch
    feats: List["np.ndarray"] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i + batch_size])
            tokens = tokenizer(batch_texts).to(device)
            f = model.encode_text(tokens)
            if isinstance(f, tuple):
                f = f[0]
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)


# ---------------------------------------------------------------------------
# Captions / KB
# ---------------------------------------------------------------------------

def _read_caption_rows(path: Path) -> List[Dict[str, str]]:
    """Load the captions parquet/TSV produced by build_pathomeood_captions.py."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        import pyarrow.parquet as pq  # type: ignore
        return pq.read_table(path).to_pylist()
    delim = "\t" if suf == ".tsv" else ","
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delim))


def _resolve_strategy(captions_path: Path, override: Optional[str]) -> str:
    if override:
        return override
    # Filename pattern: <crop>_<strategy>.{parquet,tsv}
    name = captions_path.stem
    if "_" in name:
        # everything after the first underscore is the strategy
        return name.split("_", 1)[1]
    return "canonical_full"


def _build_class_universe(captions_rows: List[Dict[str, str]]) -> List[str]:
    """Class universe = unique 'Crop Disease' strings from training rows."""
    seen = []
    for r in captions_rows:
        label = f"{r['crop']} {r['disease']}"
        if label not in seen:
            seen.append(label)
    return seen


def _build_crop_vocabulary(captions_rows: List[Dict[str, str]]) -> List[str]:
    seen = []
    for r in captions_rows:
        if r["crop"] not in seen:
            seen.append(r["crop"])
    return seen


def _onehot(value: str, vocab: List[str]) -> "np.ndarray":
    out = np.zeros((len(vocab),), dtype=np.float32)
    if value in vocab:
        out[vocab.index(value)] = 1.0
    return out


# ---------------------------------------------------------------------------
# Build train features
# ---------------------------------------------------------------------------

def build_train_features(
    captions_path: Path,
    encoder_name: str,
    out_path: Path,
    batch_size: int,
    device: Optional[str],
) -> Dict:
    rows = _read_caption_rows(captions_path)
    rows = [r for r in rows if (r.get("split") or "train") != "holdout"]
    if not rows:
        raise SystemExit(f"no non-holdout rows in {captions_path}")

    classes = _build_class_universe(rows)
    crops = _build_crop_vocabulary(rows)
    class_id = {c: i for i, c in enumerate(classes)}

    # Load encoder + run forward passes on images + captions.
    print(f"  loading encoder: {encoder_name}")
    model, preprocess, tokenizer, dev = load_encoder(encoder_name, device)

    img_paths = [Path(r["image_path"]) for r in rows]
    keep_mask = np.array([p.is_file() for p in img_paths])
    if not keep_mask.all():
        print(f"  WARNING: {(~keep_mask).sum()}/{len(rows)} image paths "
              f"missing on disk; dropping those rows")
    rows = [r for r, k in zip(rows, keep_mask) if k]
    img_paths = [Path(r["image_path"]) for r in rows]

    print(f"  encoding {len(img_paths)} images")
    image_emb = encode_images(model, preprocess, dev, img_paths, batch_size)

    print(f"  encoding {len(rows)} captions")
    captions = [r["caption_text"] for r in rows]
    caption_emb = encode_texts(model, tokenizer, dev, captions, batch_size)

    # Metadata: crop one-hot only (state isn't reliably available on eval).
    crop_oh = np.stack([_onehot(r["crop"], crops) for r in rows], axis=0)

    X = np.concatenate([image_emb, caption_emb, crop_oh], axis=1).astype(np.float32)
    y = np.array([class_id[f"{r['crop']} {r['disease']}"] for r in rows], dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(
        encoder=encoder_name,
        d_image=int(image_emb.shape[1]),
        d_caption=int(caption_emb.shape[1]),
        d_crop=int(crop_oh.shape[1]),
        n_classes=len(classes),
        crops=crops,
    )
    np.savez_compressed(
        out_path,
        X=X, y=y,
        class_names=np.array(classes, dtype=object),
        meta=json.dumps(meta),
    )
    print(f"  wrote {out_path}  X={X.shape}  y={y.shape}  C={len(classes)}")
    return meta


# ---------------------------------------------------------------------------
# Build eval features
# ---------------------------------------------------------------------------

_NORMALIZERS = {
    "plantvillage": normalize_pv_folder,
    "plantdoc":     normalize_plantdoc_folder,
    "plantwild":    normalize_pw_folder,
}


def _collect_eval_paths(root: Path, kind: str, crop_filter: Optional[str]):
    """Walk folder-per-class root → (paths, crop_disease_pairs)."""
    norm = _NORMALIZERS[kind]
    paths: List[Path] = []
    labels: List[Tuple[str, str]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        parsed = norm(sub.name)
        if parsed is None:
            continue
        folder_crop, folder_disease = parsed
        if crop_filter and folder_crop.lower() != crop_filter.lower():
            continue
        files = []
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
            files.extend(sub.glob(f"*{ext}"))
        files = sorted(files)
        for f in files:
            paths.append(f)
            labels.append((folder_crop, folder_disease))
    return paths, labels


def build_eval_features(
    eval_root: Path,
    eval_kind: str,
    encoder_name: str,
    strategy: str,
    train_meta: Dict,
    train_classes: List[str],
    kb_root: Path,
    out_path: Path,
    batch_size: int,
    device: Optional[str],
    crop_filter: Optional[str] = None,
) -> None:
    paths, labels = _collect_eval_paths(eval_root, eval_kind, crop_filter)
    if not paths:
        print(f"  [{eval_kind}] no images in {eval_root} for crop={crop_filter}")
        return
    print(f"  [{eval_kind}] {len(paths)} images across "
          f"{len(set(labels))} (crop, disease) pairs")

    # Load encoder again for eval-set encoding.
    model, preprocess, tokenizer, dev = load_encoder(encoder_name, device)

    # Image embeddings.
    image_emb = encode_images(model, preprocess, dev, paths, batch_size)

    # Caption embeddings: look up KB for each (crop, disease); use fallback
    # template when missing. SAME strategy as the train captions.
    profiles = load_kb_profiles(str(kb_root))
    captions: List[str] = []
    for crop, disease in labels:
        if disease.lower() == "healthy":
            cap = build_healthy_caption(crop)
        else:
            rec = profiles.get((crop, disease))
            if rec is None:
                cap = build_fallback_caption(crop, disease, strategy)
            else:
                try:
                    cap = build_disease_caption(
                        crop=crop, disease=disease,
                        disease_record=rec, strategy=strategy, state=None,
                    )
                except ValueError:
                    # missing deltas for a delta strategy → fall back
                    cap = build_fallback_caption(crop, disease, strategy)
        captions.append(cap)
    caption_emb = encode_texts(model, tokenizer, dev, captions, batch_size)

    # crop one-hot in the same vocabulary as train.
    crops_vocab = train_meta["crops"]
    crop_oh = np.stack([_onehot(crop, crops_vocab) for crop, _ in labels], axis=0)

    X = np.concatenate([image_emb, caption_emb, crop_oh], axis=1).astype(np.float32)

    # Map test labels to train-class ids; -1 means class not seen in training.
    class_to_id = {c: i for i, c in enumerate(train_classes)}
    y = np.array([
        class_to_id.get(f"{crop} {disease}", -1)
        for crop, disease in labels
    ], dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X, y=y,
        class_names=np.array(train_classes, dtype=object),
        eval_paths=np.array([str(p) for p in paths], dtype=object),
        eval_labels=np.array(labels, dtype=object),
    )
    print(f"  wrote {out_path}  X={X.shape}  y={y.shape}  "
          f"in-train={(y >= 0).sum()} OOC-classes={(y < 0).sum()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    captions_path = Path(args.captions)
    strategy = _resolve_strategy(captions_path, args.strategy)
    encoder_tag = args.encoder

    # Train features.
    out_train = Path(args.out_train) if args.out_train else (
        Path("data/bugwood_features") / f"{encoder_tag}_{strategy}.npz"
    )
    print(f"=== build_features (train) ===")
    print(f"  captions  : {captions_path}")
    print(f"  encoder   : {encoder_tag}")
    print(f"  strategy  : {strategy}")
    print(f"  out_train : {out_train}")
    meta = build_train_features(
        captions_path, encoder_tag, out_train, args.batch_size, args.device,
    )

    # Load train classes for ID mapping in eval files.
    npz = np.load(out_train, allow_pickle=True)
    train_classes = npz["class_names"].tolist()

    # Eval features (each test set).
    eval_root = Path(args.out_eval_root)
    for kind, src in (
        ("plantvillage", args.eval_pv),
        ("plantdoc",     args.eval_pd),
        ("plantwild",    args.eval_pw),
    ):
        if not src:
            continue
        root = Path(src)
        if not root.is_dir():
            print(f"  [{kind}] root not found: {root} — skipping")
            continue
        out_eval = eval_root / f"{encoder_tag}_{strategy}_{kind}.npz"
        print(f"\n=== build_features (eval: {kind}) ===")
        print(f"  root      : {root}")
        print(f"  out       : {out_eval}")
        build_eval_features(
            root, kind, encoder_tag, strategy, meta, train_classes,
            Path(args.kb_root), out_eval, args.batch_size, args.device,
            crop_filter=args.crop_filter,
        )


if __name__ == "__main__":
    main()
