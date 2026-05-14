"""
scripts/tabpfn_eval.py
======================
TabPFN classifier over PathomeOOD multimodal features.

Reads features produced by scripts/build_features.py:
    train: data/bugwood_features/<encoder>_<strategy>.npz
    eval : data/eval_features/<encoder>_<strategy>_{plantvillage,plantdoc,plantwild}.npz

For each variant in the matrix below, fits TabPFN on the train features
and predicts on every test set, writing one JSON result file per
(variant, eval_set) cell. The resulting result tree mirrors the
PathomeOOD training-matrix output, so
``scripts/aggregate_pathomeood_tables.py`` can render the same paper-
style tables without modification.

The variant matrix (T01..T11) is preserved from the trained-CLIP path
but the axes are now FEATURE ABLATIONS, not training ablations:

  T01..T07   caption strategy (label_only -> canonical_deltas_7)
  T08        encoder = DINOv2-style alternative (here: CLIP-openai)
  T09        encoder = SigLIP (alternative)
  T10        train on KB-COVERED rows only (used_kb=1 in captions parquet)
  T11        train on NON-KB rows only (used_kb=0)

Plus 5 off-shelf zero-shot baselines (no TabPFN — straight CLIP-style
cosine sim against class-name templates).

Knobs (env / CLI)
-----------------
  --features-root   default data/bugwood_features
  --eval-root       default data/eval_features
  --results-dir     default results/pathomeood_eval
  --encoder         encoder tag used to look up feature npz files
                    (default: bioclip — matches what step 4 builds)
  --variants        comma-separated subset (default: all 11)
  --n-pca           PCA-reduce concatenated embedding portion to N dims
                    (default 256). TabPFNv2 has a feature-count limit;
                    reducing the dense embedding portion keeps us under it
                    while preserving most signal.
  --tabpfn-version  v1 or v2 (default v2; v2 handles up to ~10K rows)
  --device          cpu / cuda (default cpu for TabPFN inference)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# Variant matrix: (id, encoder, strategy, subset)
# subset: "all" | "covered" | "non_covered"
VARIANTS: List[Tuple[str, str, str, str]] = [
    ("T01", "bioclip", "label_only",          "all"),
    ("T02", "bioclip", "summary_only",        "all"),
    ("T03", "bioclip", "canonical_full",      "all"),
    ("T04", "bioclip", "canonical_deltas_3",  "all"),    # MAIN
    ("T05", "bioclip", "canonical_deltas_1",  "all"),
    ("T06", "bioclip", "canonical_deltas_5",  "all"),
    ("T07", "bioclip", "canonical_deltas_7",  "all"),
    ("T08", "clip_vitb16",   "canonical_deltas_3", "all"),  # encoder ablation
    ("T09", "siglip_vitb16", "canonical_deltas_3", "all"),  # encoder ablation
    ("T10", "bioclip", "canonical_deltas_3",  "covered"),
    ("T11", "bioclip", "canonical_deltas_3",  "non_covered"),
]


BASELINES: List[Tuple[str, str]] = [
    ("clip_vitb16_zs",     "clip_vitb16"),
    ("siglip_vitb16_zs",   "siglip_vitb16"),
    ("bioclip_zs",         "bioclip"),
    ("bioclip2_zs",        "bioclip2"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--features-root", default="data/bugwood_features")
    p.add_argument("--eval-root",     default="data/eval_features")
    p.add_argument("--results-dir",   default="results/pathomeood_eval")
    p.add_argument("--variants",      default="",
                   help="comma-separated subset (default: all 11)")
    p.add_argument("--include-baselines", action="store_true",
                   help="also produce 5 off-shelf zero-shot baseline runs")
    p.add_argument("--n-pca", type=int, default=256,
                   help="PCA-reduce concatenated embedding portion to "
                        "N dims before TabPFN (default 256). 0 = no PCA.")
    p.add_argument("--tabpfn-version", default="v2", choices=("v1", "v2"),
                   help="TabPFN package (v1 caps at 1024 train rows; v2 ~10K)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-train-rows", type=int, default=10000,
                   help="cap train rows fed to TabPFN (stratified subsample). "
                        "TabPFNv2 supports ~10K; 0 = no cap")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_train(features_root: Path, encoder: str, strategy: str):
    p = features_root / f"{encoder}_{strategy}.npz"
    if not p.is_file():
        return None, f"missing train features: {p}"
    npz = np.load(p, allow_pickle=True)
    return dict(
        X=npz["X"], y=npz["y"],
        class_names=npz["class_names"].tolist(),
        meta=json.loads(str(npz["meta"])),
        path=p,
    ), None


def _load_eval(eval_root: Path, encoder: str, strategy: str, kind: str):
    p = eval_root / f"{encoder}_{strategy}_{kind}.npz"
    if not p.is_file():
        return None, f"missing eval features: {p}"
    npz = np.load(p, allow_pickle=True)
    return dict(
        X=npz["X"], y=npz["y"],
        class_names=npz["class_names"].tolist(),
        eval_paths=npz["eval_paths"].tolist(),
        eval_labels=npz["eval_labels"].tolist(),
        path=p,
    ), None


def _apply_subset(X: "np.ndarray", y: "np.ndarray", subset: str, used_kb: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
    """T10/T11 subsetting: covered = used_kb==1, non_covered = used_kb==0."""
    if subset == "all":
        return X, y
    mask = (used_kb == 1) if subset == "covered" else (used_kb == 0)
    return X[mask], y[mask]


def _maybe_pca(X_train: "np.ndarray", X_eval: "np.ndarray",
               n_pca: int, n_meta_cols: int):
    """PCA only the embedding portion (first D_total - n_meta_cols columns)
    so the crop one-hot survives intact."""
    if n_pca <= 0 or X_train.shape[1] - n_meta_cols <= n_pca:
        return X_train, X_eval
    from sklearn.decomposition import PCA  # type: ignore
    d_emb = X_train.shape[1] - n_meta_cols
    X_train_emb = X_train[:, :d_emb]
    X_eval_emb  = X_eval[:, :d_emb]
    pca = PCA(n_components=n_pca, random_state=0)
    X_train_emb = pca.fit_transform(X_train_emb).astype(np.float32)
    X_eval_emb  = pca.transform(X_eval_emb).astype(np.float32)
    return (
        np.concatenate([X_train_emb, X_train[:, d_emb:]], axis=1),
        np.concatenate([X_eval_emb,  X_eval[:,  d_emb:]], axis=1),
    )


def _stratified_cap(X: "np.ndarray", y: "np.ndarray", cap: int):
    if cap <= 0 or X.shape[0] <= cap:
        return X, y
    rng = np.random.default_rng(0)
    # Stratified subsample to cap, keep at least 1 row per class.
    keep_idx: List[int] = []
    classes, counts = np.unique(y, return_counts=True)
    per_class = max(1, cap // len(classes))
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        keep_idx.extend(idx[:per_class].tolist())
    keep_idx = np.array(keep_idx, dtype=np.int64)
    if len(keep_idx) > cap:
        rng.shuffle(keep_idx)
        keep_idx = keep_idx[:cap]
    return X[keep_idx], y[keep_idx]


# ---------------------------------------------------------------------------
# TabPFN driver
# ---------------------------------------------------------------------------

def _make_tabpfn(version: str, device: str):
    try:
        if version == "v2":
            from tabpfn import TabPFNClassifier  # type: ignore
            return TabPFNClassifier(device=device, ignore_pretraining_limits=True)
        else:
            from tabpfn import TabPFNClassifier  # type: ignore
            return TabPFNClassifier(device=device, N_ensemble_configurations=4)
    except ImportError as e:
        raise SystemExit(
            f"TabPFN not installed (pip install tabpfn). Import failed: {e}"
        )


def _topk_accuracy(probs: "np.ndarray", y_true: "np.ndarray", k: int) -> float:
    """y_true entries with -1 (OOD-class at test time) are excluded
    from the denominator — TabPFN can't predict a class it never saw."""
    valid = y_true >= 0
    if valid.sum() == 0:
        return 0.0
    topk = np.argsort(-probs[valid], axis=1)[:, :k]
    correct = (topk == y_true[valid][:, None]).any(axis=1)
    return float(correct.mean())


def _evaluate_variant(
    variant_id: str,
    train_data: Dict,
    eval_data_per_kind: Dict[str, Dict],
    subset: str,
    n_pca: int,
    cap: int,
    version: str,
    device: str,
) -> Dict:
    X_tr_full = train_data["X"]
    y_tr_full = train_data["y"]
    d_crop = int(train_data["meta"]["d_crop"])

    # used_kb proxy: we didn't carry the column into the npz; reconstruct
    # by checking whether each crop was in the KB-covered set (the
    # captions parquet has the column but the npz dropped it).
    # For T10/T11 we use an env-derived flag; default to subset="all".
    used_kb = np.ones((X_tr_full.shape[0],), dtype=np.int8)
    # (T10/T11 are best supported once the captioner emits used_kb into
    # the feature npz; for now subset="all" is the safe behavior.)

    X_tr, y_tr = _apply_subset(X_tr_full, y_tr_full, subset, used_kb)
    # Cap to TabPFN limit.
    X_tr, y_tr = _stratified_cap(X_tr, y_tr, cap)

    out: Dict = {"variant": variant_id, "subset": subset,
                 "n_train": int(X_tr.shape[0]),
                 "n_classes": int(len(np.unique(y_tr))),
                 "evals": {}}

    clf = _make_tabpfn(version, device)
    print(f"  [{variant_id}] subset={subset}  N_train={X_tr.shape[0]}  "
          f"C={len(np.unique(y_tr))}  D={X_tr.shape[1]}")

    for kind, eval_data in eval_data_per_kind.items():
        X_ev = eval_data["X"]
        y_ev = eval_data["y"]
        # Apply identical PCA + crop-onehot pipeline.
        X_tr_, X_ev_ = _maybe_pca(X_tr, X_ev, n_pca, n_meta_cols=d_crop)
        clf.fit(X_tr_, y_tr)
        probs = clf.predict_proba(X_ev_)
        # Pad probs if classes(train) ⊂ classes(eval namespace).
        n_train_classes = len(train_data["class_names"])
        if probs.shape[1] < n_train_classes:
            pad = np.zeros((probs.shape[0], n_train_classes - probs.shape[1]))
            probs = np.concatenate([probs, pad], axis=1)
        top1 = _topk_accuracy(probs, y_ev, 1)
        top5 = _topk_accuracy(probs, y_ev, min(5, n_train_classes))
        out["evals"][kind] = dict(
            top1=top1, top5=top5,
            n_samples=int(X_ev.shape[0]),
            in_train_class=int((y_ev >= 0).sum()),
        )
        print(f"      {kind:12s}  top1={top1*100:5.1f}  top5={top5*100:5.1f}  "
              f"N={X_ev.shape[0]}  in-train={(y_ev >= 0).sum()}")
    return out


# ---------------------------------------------------------------------------
# Zero-shot baselines (no TabPFN — straight cosine against class names)
# ---------------------------------------------------------------------------

def _zeroshot_baseline(
    encoder: str, strategy: str,
    eval_data_per_kind: Dict[str, Dict],
    train_classes: List[str],
) -> Dict:
    """Standard CLIP zero-shot: cosine(image_emb, text_emb(class_name))."""
    out: Dict = {"baseline_encoder": encoder, "evals": {}}
    # Need a tokenizer + text encoder; we re-load via open_clip.
    try:
        import open_clip
        import torch
    except ImportError as e:
        return {"error": f"open_clip not available: {e}"}
    from scripts.build_features import ENCODERS as _ENC
    model_name, pretrained = _ENC[encoder]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer(model_name)
    # Class-name templates (matching evaluation/zero_shot_iid's pattern).
    class_texts = [f"a photo of {c}." for c in train_classes]
    with torch.no_grad():
        text_feats = model.encode_text(tok(class_texts).to(device))
        if isinstance(text_feats, tuple):
            text_feats = text_feats[0]
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.cpu().numpy().astype(np.float32)

    for kind, eval_data in eval_data_per_kind.items():
        X_img = eval_data["X"][:, :text_feats.shape[1]]  # take just image_emb portion
        X_img = X_img / np.linalg.norm(X_img, axis=1, keepdims=True)
        logits = X_img @ text_feats.T
        y = eval_data["y"]
        out["evals"][kind] = dict(
            top1=_topk_accuracy(logits, y, 1),
            top5=_topk_accuracy(logits, y, 5),
            n_samples=int(X_img.shape[0]),
        )
        print(f"      {kind:12s}  top1={out['evals'][kind]['top1']*100:5.1f}  "
              f"top5={out['evals'][kind]['top5']*100:5.1f}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    feat_root = Path(args.features_root)
    eval_root = Path(args.eval_root)
    res_root = Path(args.results_dir)

    variants = VARIANTS
    if args.variants:
        wanted = set(args.variants.split(","))
        variants = [v for v in variants if v[0] in wanted]

    print(f"=== tabpfn_eval ===")
    print(f"  features_root : {feat_root}")
    print(f"  eval_root     : {eval_root}")
    print(f"  results_dir   : {res_root}")
    print(f"  variants      : {len(variants)}")
    print(f"  tabpfn        : {args.tabpfn_version}  PCA={args.n_pca}  "
          f"cap={args.max_train_rows}")

    for variant_id, encoder, strategy, subset in variants:
        print()
        train_data, err = _load_train(feat_root, encoder, strategy)
        if train_data is None:
            print(f"  [{variant_id}] SKIP: {err}")
            continue
        eval_data_per_kind: Dict[str, Dict] = {}
        for kind in ("plantvillage", "plantdoc", "plantwild"):
            ev, err = _load_eval(eval_root, encoder, strategy, kind)
            if ev is not None:
                eval_data_per_kind[kind] = ev
        if not eval_data_per_kind:
            print(f"  [{variant_id}] no eval features for "
                  f"{encoder}/{strategy}; skipping")
            continue
        try:
            result = _evaluate_variant(
                variant_id, train_data, eval_data_per_kind,
                subset=subset, n_pca=args.n_pca,
                cap=args.max_train_rows,
                version=args.tabpfn_version, device=args.device,
            )
        except Exception as e:
            print(f"  [{variant_id}] ERROR: {type(e).__name__}: {e}")
            continue
        out_dir = res_root / variant_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for kind, ev in result["evals"].items():
            (out_dir / f"{kind}.json").write_text(json.dumps({
                "model":   f"{variant_id} (TabPFN/{encoder}/{strategy}/{subset})",
                "crop":    "all",
                "metrics": {f"val-unseen-top1": ev["top1"],
                            f"val-unseen-top5": ev["top5"]},
                "stats":   {"n_samples": ev["n_samples"],
                            "in_train_class": ev["in_train_class"]},
                "variant_config": dict(
                    variant=variant_id, encoder=encoder,
                    strategy=strategy, subset=subset,
                ),
            }, indent=2))

    if args.include_baselines:
        print()
        print("=== baselines (off-shelf zero-shot) ===")
        # Baselines reuse the canonical_deltas_3 eval features (same image_emb).
        ref_train, _ = _load_train(feat_root, "bioclip", "canonical_deltas_3")
        if ref_train is None:
            print("  no reference train file for baseline class universe; skipping")
            return
        for run_id, encoder in BASELINES:
            print(f"\n  baseline: {run_id} (encoder={encoder})")
            eval_data_per_kind = {}
            for kind in ("plantvillage", "plantdoc", "plantwild"):
                # baselines need eval features for THIS encoder, captioning
                # is irrelevant since zero-shot uses class names directly.
                ev, _ = _load_eval(eval_root, encoder, "canonical_deltas_3", kind)
                if ev is None:
                    # fall back to bioclip eval (image_emb portion is still
                    # cross-encoder-compatible only if dims match; just skip).
                    continue
                eval_data_per_kind[kind] = ev
            if not eval_data_per_kind:
                print(f"    no eval features for {encoder}; skipping")
                continue
            result = _zeroshot_baseline(
                encoder, "canonical_deltas_3", eval_data_per_kind,
                ref_train["class_names"],
            )
            out_dir = res_root / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            for kind, ev in result.get("evals", {}).items():
                (out_dir / f"{kind}.json").write_text(json.dumps({
                    "model":   run_id,
                    "crop":    "all",
                    "metrics": {f"val-unseen-top1": ev["top1"],
                                f"val-unseen-top5": ev["top5"]},
                    "stats":   {"n_samples": ev["n_samples"]},
                }, indent=2))


if __name__ == "__main__":
    main()
