# Pathome smoke test (2 crops, full pipeline)

A miniature end-to-end run of the Pathome pipeline on **Tomato + Soybean only** (~25 (crop, disease) classes after threshold≥15: 15 Tomato + 10 Soybean). Designed to validate every code path with as little compute as possible.

Override the threshold with `SMOKE_THRESHOLD=10 bash smoke/run_smoke.sh` for a wider smoke (~65 classes), or `SMOKE_THRESHOLD=20` for narrower (~15 classes).

```
smoke/
├── BugWood_Diseases_smoke.csv         1,200 raw rows (Tomato + Soybean only)
├── BugWood_Diseases_smoke_usable.csv  produced by Setup phase
├── bugwood_pathome_smoke.yaml         training config (small budgets)
├── plantvillage_smoke_eval.yaml       PV eval (200-image subset)
├── plantwild_smoke_eval.yaml          PW eval (200-image subset)
├── run_smoke.sh                       Bash chain — runs all 6 phases as plain `python`
├── submit_smoke.sh                    SLURM wrapper for Nova
└── README.md                          (this file)

Outputs (under smoke/, gitignored):
  smoke/artifacts/pathome_seed/symptoms_seed.json
  smoke/artifacts/pathome_v1_seed/{symptoms.json, refs/, ...}
  smoke/artifacts/pathome_v1_enhanced/{symptoms.json, refs/, ...}
  smoke/results/traces/plantswarm_traces.jsonl
  smoke/observe/checkpoints/{seed,enhanced}/observe_grpo_epoch_*.pt
  smoke/results/compare/comparison.{json,md,tex}
```

## What's downscaled vs production

| Knob | Production | Smoke |
|---|---|---|
| Crops | 197 | 2 (Tomato + Soybean) |
| Classes | 484 | ~25 (15 Tomato + 10 Soybean) |
| `per_class` / `trace_split` | 10 / 7 | 4 / 3 |
| `runs_per_image` | 30 | 3 |
| `max_new_tokens` | 512 | 256 |
| `Tmax` (max routing path) | 15 | 8 |
| Phase 0 mode | full | `--quick` (3 sources/crop) |
| Phase 4 DT epochs | 50 | 3 |
| Phase 4 GRPO epochs | 10 | 1 |
| LoRA rank | 16 | 8 |
| Phase 5 PV/PW eval images | 54,306 / 18,000 | 200 / 200 |
| `bootstrap_n` | 1,000 | 100 |

Total trace volume: ~25 classes × 3 trace seeds × 3 runs ≈ **225 traces**.

## How to run

The smoke pipeline splits the same way as production: Phase 0 runs locally, the rest runs on Nova.

### Step 1 — Phase 0 on your local machine (~5 min)

```bash
bash smoke/run_phase0_local.sh
# Produces: smoke/artifacts/pathome_seed/symptoms_seed.json (and per-crop
# artefacts under smoke/artifacts/pathome_kb/<Crop>/)
```

Then push the seed file to GitHub:
```bash
git add -f smoke/artifacts/pathome_seed/symptoms_seed.json \
           smoke/BugWood_Diseases_smoke_usable.csv
git commit -m "smoke phase 0 seed" && git push origin main
```

### Step 2 — Phases 1–5 on Nova (~60–90 min, single A100)

```bash
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main
sbatch smoke/submit_smoke.sh
tail -f logs/pathome_smoke-*.out
```

The Nova job bails out with a clear error if the seed file isn't yet on disk — meaning your `git push` hasn't reached the remote, or you haven't `git pull`-ed on Nova.

### Local-only debug (no Nova, no GPU)

If you don't have a GPU and just want to validate KB plumbing:
```bash
bash smoke/run_smoke.sh
# Setup → Phase 0 (if claude is auth'd) → Phase 1 → Phase 3.
# Phases 2/4/5 auto-skip on CPU-only machines.
# ~10–15 min total.
```

### Skip / resume / orchestrator

```bash
SMOKE_SKIP_0=1 bash smoke/run_smoke.sh        # skip Phase 0 (seed already present)
SMOKE_SKIP_2=1 bash smoke/run_smoke.sh        # skip just Phase 2
SMOKE_FROM=4 bash smoke/run_smoke.sh          # restart at training (assumes traces exist)
SMOKE_ORCH=autogen_swarm bash smoke/run_smoke.sh   # use vLLM instead of hf_direct
```

## Auth requirements (local only)

Phase 0 of the smoke (the SAGE-ported pathome_kb pipeline) runs **locally** and needs:

1. The `claude` CLI on PATH and authenticated (`claude auth login`).
2. `ANTHROPIC_API_KEY` in env or repo-root `.env` (used by the Anthropic SDK in the extraction + reconciliation stages).

If either is missing, `smoke/run_phase0_local.sh` errors out with the install commands. The Nova-side `submit_smoke.sh` does **not** check for the CLI — it relies on the seed file already being on disk from a `git pull`.

## What "success" looks like

After a clean run:

```
smoke/results/compare/comparison.md
```

contains the seed-vs-enhanced delta table for the smoke-sized PV + PW evals. The numbers are tiny and not statistically meaningful (n=200 each, 1 GRPO epoch) — but the pipeline producing the table end-to-end is the point.

If you see a non-empty `comparison.md` plus matching JSON + LaTeX siblings, every phase wired correctly. From there, the production chain (`scripts/submit_pathome_all.sh`) runs the same code at full scale.
