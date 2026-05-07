"""
scripts/train_observe_pathome.py
================================
Two-phase OBSERVE training (paper §7.3):
  Phase A — Decision Transformer on PlantSwarm Bugwood traces.
  Phase B — GRPO refinement, KL-anchored to the Phase-A policy.

Reads a single config (``configs/bugwood_pathome.yaml``); persists checkpoints
to ``observe/checkpoints/`` per Phase.

Usage:
    python scripts/train_observe_pathome.py --config configs/bugwood_pathome.yaml
    python scripts/train_observe_pathome.py --config configs/bugwood_pathome.yaml --phase a
    python scripts/train_observe_pathome.py --config configs/bugwood_pathome.yaml --phase b --init-ckpt observe/checkpoints/observe_dt_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bugwood_pathome.yaml")
    p.add_argument("--phase", choices=["a", "b", "both"], default="both")
    p.add_argument("--traces", default=None,
                   help="override traces JSONL path (default from config)")
    p.add_argument("--init-ckpt", default=None,
                   help="checkpoint to load before Phase B (defaults to Phase A best)")
    p.add_argument("--save-dir", default="observe/checkpoints/")
    return p.parse_args()


def _load_traces(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Lazy torch imports so script syntax can be validated without it.
    import torch  # noqa: F401
    from observe.model import OBSERVE
    from observe.decision_transformer import DecisionTransformerTrainer, DTConfig
    from observe.grpo import GRPOTrainer, GRPOConfig
    from observe.loss import ObserveLossWeights
    from observe.trainer import RoutingTraceDataset, TraceAnnotation
    from torch.utils.data import DataLoader

    traces_path = args.traces or os.path.join(
        cfg["output"]["traces_dir"], "plantswarm_traces.jsonl"
    )
    if not os.path.exists(traces_path):
        raise SystemExit(f"Traces JSONL not found: {traces_path}")

    print(f"Loading traces: {traces_path}")
    traces = _load_traces(traces_path)
    print(f"  {len(traces)} routing traces")

    # ------------------------------------------------------------------
    # Trace → TraceAnnotation (label generation per paper §7.3)
    # ------------------------------------------------------------------
    annotations = []
    for t in traces:
        # Per-trace summary labels — DT consumes the full episode below;
        # this is the first cut for batched training compatibility.
        path = t.get("path", [])
        bt = int(t.get("backtrack_count", 0))
        early = bool(t.get("early_terminated", False))
        path_len = len(path)

        # Default label generation (paper §7.3)
        epsilon = min(1.0, path_len / max(cfg["routing"]["Tmax"], 1)) if bt > 0 else 0.0
        aleatoric = 0.0  # cannot be inferred without ground-truth correctness in trace
        confidence = 1.0 if early else 0.5
        oc = 0.0  # populated when ground-truth label is available alongside trace

        next_agent = path[-1] if path else "DiagnosisAgent"

        annotations.append(TraceAnnotation(
            image_id=t["image_id"],
            image_b64="",            # filled by collator if needed
            context_text=" -> ".join(path),
            next_agent=next_agent,
            backtrack=bt > 0,
            epistemic=epsilon,
            aleatoric=aleatoric,
            confidence=confidence,
            belief_state="",
        ))

    # 80/10/10 split (paper Appendix C)
    n = len(annotations)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_anns = annotations[:n_train]
    val_anns = annotations[n_train:n_train + n_val]
    held_anns = annotations[n_train + n_val:]
    print(f"  split: train={len(train_anns)}, val={len(val_anns)}, held={len(held_anns)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    backbone = cfg["observe"]["backbone"]
    print(f"\nInstantiating OBSERVE on {backbone} ...")
    model = OBSERVE(
        backbone=backbone,
        lora_r=cfg["observe"]["lora"]["r"],
        lora_alpha=cfg["observe"]["lora"]["alpha"],
        lora_dropout=cfg["observe"]["lora"]["dropout"],
        oc_threshold=cfg["observe"]["oc_threshold"],
    )

    if args.init_ckpt and os.path.exists(args.init_ckpt):
        import torch
        print(f"  loading initial weights from {args.init_ckpt}")
        sd = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)

    save_dir = args.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase A — Decision Transformer
    # ------------------------------------------------------------------
    if args.phase in ("a", "both"):
        print("\n=== Phase A: Decision Transformer ===")
        dt_cfg = cfg["observe"]["decision_transformer"]
        dt = DecisionTransformerTrainer(
            model=model,
            cfg=DTConfig(
                lr=float(dt_cfg["lr"]),
                warmup_steps=int(dt_cfg["warmup_steps"]),
                epochs=int(dt_cfg["epochs"]),
                patience=int(dt_cfg["patience"]),
                batch_size=int(dt_cfg["batch_size"]),
                grad_accum_steps=int(dt_cfg["grad_accum_steps"]),
                target_return_mean=float(dt_cfg["target_return_mean"]),
            ),
            loss_weights=ObserveLossWeights(**cfg["observe"]["loss_weights"]),
        )

        train_ds = RoutingTraceDataset(train_anns, processor=model.processor)
        val_ds = RoutingTraceDataset(val_anns, processor=model.processor)
        train_loader = DataLoader(train_ds, batch_size=dt_cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=dt_cfg["batch_size"])

        dt.fit(train_loader, val_loader, save_dir=save_dir)
        print(f"  Phase A best val ECE: {dt.best_val_ece:.4f}")

    # ------------------------------------------------------------------
    # Phase B — GRPO
    # ------------------------------------------------------------------
    if args.phase in ("b", "both"):
        print("\n=== Phase B: GRPO ===")
        grpo_cfg = cfg["observe"]["grpo"]
        grpo = GRPOTrainer(
            model=model,
            cfg=GRPOConfig(
                lr=float(grpo_cfg["lr"]),
                epochs=int(grpo_cfg["epochs"]),
                rollouts_per_instance=int(grpo_cfg["rollouts_per_instance"]),
                clip_eps=float(grpo_cfg["clip_eps"]),
                beta_kl=float(grpo_cfg["beta_kl"]),
                f1_weight=float(grpo_cfg["f1_weight"]),
                ece_weight=float(grpo_cfg["ece_weight"]),
                bt_delta_weight=float(grpo_cfg["bt_delta_weight"]),
                length_penalty_weight=float(grpo_cfg["length_penalty_weight"]),
                epsilon_match_weight=float(grpo_cfg["epsilon_match_weight"]),
                max_path_length=int(cfg["routing"]["Tmax"]),
            ),
        )

        # rollout_fn is the integration point with the agent runtime — wired
        # via plantswarm.observe_rollout (see Phase B integration in
        # plantswarm/pipeline.py for details).
        from plantswarm.observe_rollout import collect_rollout
        grpo.fit(
            train_instances=train_anns,
            rollout_fn=collect_rollout,
            save_dir=save_dir,
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
