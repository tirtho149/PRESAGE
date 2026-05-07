"""
scripts/run_pathome_traces.py
=============================
Generate PlantSwarm routing traces on Bugwood (paper §5.3): 30 stochastic
runs per training image at temperature 0.9.

For 7 training images per class × 26 classes × 30 runs = 5,460 traces.

Each trace is appended to ``traces/plantswarm_traces.jsonl`` with fsync
(resume-friendly per the existing pipeline). The Bugwood loader injects
GPS / AEZ / month into ``record.meta`` and the trace records carry these
through unchanged.

Usage:
    python scripts/run_pathome_traces.py --config configs/bugwood_pathome.yaml
    python scripts/run_pathome_traces.py --config configs/bugwood_pathome.yaml --runs-per-image 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from tqdm import tqdm

from data.bugwood_loader import BugwoodLoader
from pathome import PathomeDB
from plantswarm.autogen_pipeline import AutoGenPlantSwarmPipeline
from plantswarm.hf_pipeline import HFDirectPipeline
from utils.hf_client import HFClient
from utils.routing_trace import append_trace, existing_trace_ids
from utils.vllm_client import VLLMClient, configure_vllm_client_from_yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bugwood_pathome.yaml")
    p.add_argument("--runs-per-image", type=int, default=None,
                   help="override routing.runs_per_image from config")
    p.add_argument("--orchestrator",
                   choices=["autogen_swarm", "hf_direct"], default=None)
    p.add_argument("--subset-classes", type=int, default=None,
                   help="for smoke-testing: only N classes")
    p.add_argument("--pathome-dir", default=None,
                   help="optional pre-built PathomeDB to inject into agents")
    return p.parse_args()


def _trace_meta_from_record(rec) -> dict:
    """Subset of BugwoodRecord that downstream consumers need."""
    return {
        "crop_species": rec.crop_species,
        "disease_name": rec.disease_name,
        "lat": rec.lat,
        "lon": rec.lon,
        "month": rec.month,
        "aez_code": rec.aez_code,
        "aez_climate": rec.aez_climate,
        "src_path": rec.src_path,
    }


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    orchestrator = args.orchestrator or cfg["routing"]["orchestrator"]
    runs_per_image = args.runs_per_image or cfg["routing"].get("runs_per_image", 30)

    results_dir = cfg["output"]["results_dir"]
    traces_dir = cfg["output"]["traces_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    print("Loading Bugwood trace split...")
    loader = BugwoodLoader(cfg["data"], split="trace")
    records = list(loader)
    if args.subset_classes:
        seen_classes: set = set()
        kept = []
        for r in records:
            key = (r.crop_species, r.disease_name)
            if len(seen_classes) >= args.subset_classes and key not in seen_classes:
                continue
            seen_classes.add(key)
            kept.append(r)
        records = kept

    print(f"  {len(records)} training images × {runs_per_image} runs/image = "
          f"{len(records) * runs_per_image} traces")

    # Optional PathomeDB pre-load (used later when agents consume Layer 4 prompts)
    pathome_db = None
    if args.pathome_dir or cfg["pathome"].get("load_dir"):
        pdir = args.pathome_dir or cfg["pathome"]["load_dir"]
        if os.path.isdir(pdir):
            print(f"Loading PathomeDB from {pdir}")
            pathome_db = PathomeDB.load(pdir)

    label_space = {
        "T1": cfg["labels"]["T1"],
        "T2": cfg["labels"]["T2"],
        "T3": cfg["labels"].get("T3") or [],
        "T4": cfg["labels"]["T4"],
        "T5": cfg["labels"]["T5"],
    }

    # ------------------------------------------------------------------
    # Build pipeline (mirrors run_plantswarm.py)
    # ------------------------------------------------------------------
    if orchestrator == "hf_direct":
        client = HFClient(
            model=cfg["model"]["backbone"],
            temperature=cfg["model"]["temperature"],
            seed=cfg["model"]["seed"],
            max_new_tokens=cfg["model"]["max_new_tokens"],
        )
        pipeline = HFDirectPipeline(
            client=client,
            label_space=label_space,
            Tmax=cfg["routing"]["Tmax"],
            confidence_weights=cfg["routing"]["confidence_weights"],
        )
    else:
        client = VLLMClient(
            base_url=cfg["model"]["vllm_base_url"],
            model=cfg["model"]["backbone"],
            temperature=cfg["model"]["temperature"],
            seed=cfg["model"]["seed"],
            max_new_tokens=cfg["model"]["max_new_tokens"],
        )
        configure_vllm_client_from_yaml(client, cfg.get("model"), orchestrator=orchestrator)
        pipeline = AutoGenPlantSwarmPipeline(
            client=client,
            label_space=label_space,
            Tmax=cfg["routing"]["Tmax"],
            confidence_weights=cfg["routing"]["confidence_weights"],
        )
    if pathome_db is not None:
        # Optional handoff: pipelines that opt-in look for this attribute
        # (see plantswarm/pipeline.py for the contract).
        pipeline.pathome_db = pathome_db

    # ------------------------------------------------------------------
    # Resume support — same JSONL layout as run_plantswarm.py
    # ------------------------------------------------------------------
    traces_filename = "plantswarm_traces.jsonl"
    already_done = existing_trace_ids(traces_dir, traces_filename)
    if already_done:
        print(f"  Resuming: {len(already_done)} traces already on disk")

    # Each trace gets a unique ID: <bugwood_image_id>::run<N>
    todo = []
    for rec in records:
        for run_idx in range(runs_per_image):
            tid = f"{rec.image_id}::run{run_idx:02d}"
            if tid in already_done:
                continue
            todo.append((tid, rec))
    print(f"  pending: {len(todo)} (skip={runs_per_image*len(records)-len(todo)})")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    n_failed = 0
    for trace_id, rec in tqdm(todo):
        try:
            trace = pipeline.run(image_id=trace_id, image_b64=rec.image_b64)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            tqdm.write(f"  [skip] {trace_id}: {type(e).__name__}: {e}")
            continue

        # Attach Bugwood metadata (paper §5.4 — GPS, AEZ, month live in trace)
        meta = _trace_meta_from_record(rec)
        trace.entropy_field = trace.entropy_field or []  # ensure list, not None
        try:
            # routing_signal already set by the pipeline; we add geospatial meta
            # by stashing on the dataclass — append_trace will not pick this up
            # automatically (it serialises the known fields), so write directly.
            from utils.routing_trace import _trace_to_record  # type: ignore
            record = _trace_to_record(trace)
            record["bugwood_meta"] = meta
            with open(os.path.join(traces_dir, traces_filename), "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  [warn] persist fail {trace_id}: {e}")

    if n_failed:
        print(f"  {n_failed} runs skipped due to errors.")
    print(f"\nTraces written to {os.path.join(traces_dir, traces_filename)}")


if __name__ == "__main__":
    main()
