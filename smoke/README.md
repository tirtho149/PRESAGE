# Pathome smoke test (2 crops, full pipeline)

A miniature end-to-end run of the Pathome pipeline on **Tomato + Soybean only** (~25 (crop, disease) classes after threshold ≥ 15: 15 Tomato + 10 Soybean). Designed to validate every code path with as little compute as possible.

**Why smoke first.** The full pipeline costs ~$50–150 in API spend (Phase 0) plus ~36–50 hours of A100 time (Phase 2) plus ~24 hours of A100 time (Phase 4). A smoke run completes in ~60–90 minutes for ~$5 and exercises every phase, so plumbing issues surface before you commit to the full spend.

```
smoke/
├── BugWood_Diseases_smoke.csv         1,200 raw rows (Tomato + Soybean only)
├── BugWood_Diseases_smoke_usable.csv  produced by Setup phase
├── bugwood_pathome_smoke.yaml         training config (small budgets)
├── plantvillage_smoke_eval.yaml       PV eval (200-image subset)
├── plantwild_smoke_eval.yaml          PW eval (200-image subset)
├── run_phase0_local.sh                LOCAL — Phase 0 wrapper
├── run_smoke.sh                       Bash chain — every phase as plain `python`
├── submit_smoke.sh                    NOVA SLURM wrapper (Phases 1–5)
└── README.md                          (this file)

Outputs (under smoke/, gitignored except seed):
  smoke/artifacts/pathome_kb/<Crop>/...                  (LOCAL audit trail)
  smoke/artifacts/pathome_seed/symptoms_seed.json        (LOCAL → push via git -f)
  smoke/artifacts/pathome_v1_seed/{symptoms.json, refs/} (NOVA, Phase 1)
  smoke/artifacts/pathome_v1_enhanced/{symptoms.json, …} (NOVA, Phase 3)
  smoke/results/traces/plantswarm_traces.jsonl          (NOVA, Phase 2)
  smoke/observe/checkpoints/{seed,enhanced}/...         (NOVA, Phase 4)
  smoke/results/compare/comparison.{json,md,tex}        (NOVA, Phase 5)
```

---

## What's downscaled vs production

| Knob | Production | Smoke |
|---|---|---|
| Crops | 197 | 2 (Tomato + Soybean) |
| Classes | 484 | ~25 (15 Tomato + 10 Soybean at threshold≥15) |
| `per_class` / `trace_split` | 10 / 7 | 4 / 3 |
| `runs_per_image` | 30 | 3 |
| `max_new_tokens` | 512 | 256 |
| `Tmax` (max routing path) | 15 | 8 |
| Phase 0 mode | full | `--quick` (3 sources/crop) |
| Phase 4 DT epochs | 50 | 3 |
| Phase 4 GRPO epochs | 10 | 1 |
| LoRA rank | 16 | 8 |
| Phase 5 PV / PW eval images | 54,306 / 18,000 | 200 / 200 |
| `bootstrap_n` | 1,000 | 100 |

Total trace volume: ~25 classes × 3 trace seeds × 3 runs ≈ **225 traces**.

Override the threshold with `SMOKE_THRESHOLD=10` for a wider smoke (~65 classes), or `SMOKE_THRESHOLD=20` for narrower (~15 classes).

---

## Where each phase runs

```
   LOCAL machine                         GitHub               Nova compute
   ─────────────                         ──────               ────────────

   bash run_phase0_local.sh   ──push──→  git pull   ──→   sbatch submit_smoke.sh
   (~5 min, claude -p)                   symptoms_seed.json    Phases 1–5 (~60-90 min)
                                                              (single A100 job)
```

Same split as production: Phase 0 needs the `claude` CLI's OAuth login (Nova compute can't run it), everything else runs as ordinary GPU jobs.

---

## Step 1 — LOCAL: Phase 0 (~5 min)

### Prerequisites

```bash
# Claude Code CLI auth'd
claude --version
claude auth login    # if not already done

# Anthropic SDK key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env       # at repo root
```

### Run

```bash
bash smoke/run_phase0_local.sh
```

This does:
1. `filter_bugwood_csv.py` if `smoke/BugWood_Diseases_smoke_usable.csv` doesn't exist yet.
2. `python -m pathome_kb --quick --only-crops "Tomato,Soybean"`.

Output: `smoke/artifacts/pathome_seed/symptoms_seed.json` (the seed file Nova needs) plus per-crop audit artefacts under `smoke/artifacts/pathome_kb/{Tomato,Soybean}/`.

### Push to GitHub

```bash
git add -f smoke/artifacts/pathome_seed/symptoms_seed.json \
           smoke/BugWood_Diseases_smoke_usable.csv
git commit -m "smoke phase 0 seed"
git push origin main
```

The wrapper script prints these exact commands when it finishes — copy-paste them.

---

## Step 2 — NOVA: Phases 1–5 (~60–90 min, single A100)

```bash
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main
sbatch smoke/submit_smoke.sh
tail -f logs/pathome_smoke-*.out
```

`submit_smoke.sh` is a single A100 job (4 h walltime budget) that internally chains Setup → Phase 1 → 2 → 3 → 4 → 5 by invoking `bash smoke/run_smoke.sh` with `SMOKE_SKIP_0=1` (Phase 0 already done locally).

### Pre-flight

The job bails out with a clear error message if `smoke/artifacts/pathome_seed/symptoms_seed.json` isn't on disk — meaning your `git push` hasn't reached the remote, or Nova hasn't `git pull`-ed yet.

### What success looks like

After a clean run:
```
smoke/results/compare/comparison.md
```
contains the seed-vs-enhanced delta table for the smoke-sized PV + PW evals. Numbers are tiny and not statistically meaningful (n=200 each, 1 GRPO epoch) — the pipeline producing the table end-to-end is the point.

If `comparison.md` is non-empty plus matching `.json` + `.tex` siblings, every phase wired correctly. From there, the production chain (`scripts/submit_pathome_all.sh`) runs the same code at full scale.

---

## Local-only debug (no Nova, no GPU)

If you don't have a GPU and just want to validate KB plumbing:

```bash
bash smoke/run_smoke.sh
# Setup → Phase 0 (if claude is auth'd) → Phase 1 → Phase 3.
# Phases 2/4/5 auto-skip on CPU-only machines with a clear [skip] message.
# ~10–15 min total.
```

Useful for:
- Confirming the SAGE port works against the Anthropic API
- Smoke-testing changes to `pathome/symptoms.py`, `data/bugwood_loader.py`, the build/enhance scripts
- Running on CI

---

## Skip / resume / orchestrator knobs

```bash
# Skip individual phases by ID:
SMOKE_SKIP_0=1   bash smoke/run_smoke.sh    # skip Phase 0 (seed already on disk)
SMOKE_SKIP_2=1   bash smoke/run_smoke.sh    # skip the trace generation
SMOKE_SKIP_4=1   bash smoke/run_smoke.sh    # skip OBSERVE training
SMOKE_SKIP_5=1   bash smoke/run_smoke.sh    # skip eval+compare

# Restart from a specific phase:
SMOKE_FROM=4     bash smoke/run_smoke.sh    # restart at training (assumes traces exist)

# Switch trace orchestrator:
SMOKE_ORCH=autogen_swarm bash smoke/run_smoke.sh   # use vLLM (default: hf_direct)

# Adjust the Setup-stage class count threshold:
SMOKE_THRESHOLD=10 bash smoke/run_smoke.sh   # wider smoke (~65 classes)
SMOKE_THRESHOLD=20 bash smoke/run_smoke.sh   # narrower (~15 classes)
```

---

## Auth requirements (LOCAL only)

Phase 0 of the smoke (the SAGE-ported `pathome_kb` pipeline) needs:

1. The `claude` CLI on PATH and authenticated (`claude auth login`).
2. `ANTHROPIC_API_KEY` in env or `.env` at repo root (used by the Anthropic SDK in extraction + reconciliation).

If either is missing, `smoke/run_phase0_local.sh` errors out with the install commands. The Nova-side `submit_smoke.sh` does **not** check for the CLI — it relies on the seed file already being on disk from a `git pull`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `smoke/artifacts/pathome_seed/symptoms_seed.json not found` (Nova) | You forgot to `git push` the seed locally, or Nova hasn't `git pull`-ed. Run `git status` on Nova to confirm the file is present. |
| `claude CLI not on PATH` (local) | `curl -fsSL https://claude.ai/install.sh \| bash` then `claude auth login` |
| `ANTHROPIC_API_KEY not set` (local) | `echo "ANTHROPIC_API_KEY=sk-ant-..." > .env` at repo root |
| Phase 2 hangs or OOMs on Nova | Try `SMOKE_ORCH=hf_direct` to bypass vLLM. The HFClient is patched against the cross-image OOM. |
| Phase 4 fails to load checkpoint | The smoke is configured for 1 GRPO epoch; checkpoint name is `observe_grpo_epoch_01.pt`. If you bumped epochs, adjust `SEED_CKPT` / `ENH_CKPT` in `run_smoke.sh`. |
| `comparison.md` only has the trace columns, no eval rows | Phase 5 was skipped (no GPU? no checkpoints?). Check the Phase 4 output and `logs/pathome_smoke-*.out`. |
