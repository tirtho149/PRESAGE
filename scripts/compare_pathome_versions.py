"""
scripts/compare_pathome_versions.py
====================================
Phase 3 of the Pathome pipeline: side-by-side comparison of OBSERVE
metrics and PlantSwarm trace quality, before vs. after the trace-based
enhancement of the Claude-seeded PathomeDB.

Inputs (paths are CLI-supplied; pass either pair or both):

  --seed-eval        results/<seed_run>/pathome_eval.json     (Phase 0+1)
  --enhanced-eval    results/<enhanced_run>/pathome_eval.json (Phase 2+3)
  --seed-traces      results/<seed_run>/traces/plantswarm_traces.jsonl
  --enhanced-traces  results/<enhanced_run>/traces/plantswarm_traces.jsonl

Outputs (under --out-dir):

  comparison.json    machine-readable deltas
  comparison.md      Markdown table for PRs / notebooks
  comparison.tex     LaTeX fragment macros (\PathomeDeltaT3F1, …) suitable
                     for \input into the paper

Either eval or trace inputs may be omitted; the script reports only the
sections it has data for. Re-run after every fresh eval / trace pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Eval-metric handling
# ---------------------------------------------------------------------------

EVAL_FIELDS: List[Tuple[str, str, str]] = [
    # (path-to-value, label, format)
    ("T3.macro_f1",                     "T3 macro F1 (overall)",          "{:.4f}"),
    ("T3.ece",                          "T3 ECE",                         "{:.4f}"),
    ("T3.tpcp",                         "T3 TPCP",                        "{:.4f}"),
    ("T3_slices.seen.macro_f1",         "T3 F1 (seen classes)",           "{:.4f}"),
    ("T3_slices.unseen.macro_f1",       "T3 F1 (unseen / zero-shot)",     "{:.4f}"),
    ("T2.macro_f1",                     "T2 (pathogen) macro F1",         "{:.4f}"),
    ("T1.macro_f1",                     "T1 (symptom) macro F1",          "{:.4f}"),
    ("T4.macro_f1",                     "T4 (severity) macro F1",         "{:.4f}"),
]


def _dig(obj: Any, dotted: str) -> Optional[float]:
    cur: Any = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def load_eval(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    if not path.is_file():
        print(f"  [warn] eval not found: {path}")
        return None
    return json.loads(path.read_text())


def diff_eval(seed: dict, enhanced: dict) -> List[dict]:
    out: List[dict] = []
    for dotted, label, fmt in EVAL_FIELDS:
        s = _dig(seed, dotted)
        e = _dig(enhanced, dotted)
        if s is None and e is None:
            continue
        delta = (e - s) if (s is not None and e is not None) else None
        out.append({
            "field": dotted,
            "label": label,
            "fmt": fmt,
            "seed": s,
            "enhanced": e,
            "delta": delta,
        })
    return out


# ---------------------------------------------------------------------------
# Trace-quality handling
# ---------------------------------------------------------------------------

def iter_traces(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def trace_summary(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    if not path.is_file():
        print(f"  [warn] traces not found: {path}")
        return None

    n = 0
    path_lengths: List[int] = []
    bt_count = 0
    early_term = 0
    correct_t3 = 0
    have_gt = 0
    for t in iter_traces(path):
        n += 1
        path_lengths.append(int(t.get("path_length") or 0))
        if int(t.get("backtrack_count") or 0) > 0:
            bt_count += 1
        if t.get("early_terminated"):
            early_term += 1
        meta = t.get("bugwood_meta") or {}
        gt = (meta.get("disease_name") or "").strip()
        if gt:
            have_gt += 1
            preds = t.get("final_predictions") or {}
            pred = (preds.get("T3") or "").strip()
            if pred and pred == gt:
                correct_t3 += 1

    if n == 0:
        return None
    return {
        "n_traces":             n,
        "avg_path_length":      sum(path_lengths) / n,
        "backtrack_rate":       bt_count / n,
        "high_confidence_rate": early_term / n,
        "t3_top1_acc":          (correct_t3 / have_gt) if have_gt else None,
        "n_with_gt":            have_gt,
    }


TRACE_FIELDS: List[Tuple[str, str, str]] = [
    ("n_traces",             "trace count",             "{:.0f}"),
    ("avg_path_length",      "avg path length",         "{:.2f}"),
    ("backtrack_rate",       "backtrack rate",          "{:.3f}"),
    ("high_confidence_rate", "early-termination rate",  "{:.3f}"),
    ("t3_top1_acc",          "trace T3 top-1 accuracy", "{:.4f}"),
]


def diff_traces(seed: dict, enhanced: dict) -> List[dict]:
    out: List[dict] = []
    for key, label, fmt in TRACE_FIELDS:
        s = seed.get(key)
        e = enhanced.get(key)
        if s is None and e is None:
            continue
        delta = (e - s) if (isinstance(s, (int, float)) and isinstance(e, (int, float))) else None
        out.append({
            "field": key,
            "label": label,
            "fmt": fmt,
            "seed": s,
            "enhanced": e,
            "delta": delta,
        })
    return out


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def fmt(value: Optional[float], spec: str) -> str:
    if value is None:
        return "—"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def to_markdown(rows_eval: List[dict], rows_trace: List[dict]) -> str:
    lines: List[str] = []
    lines.append("# PathomeDB before/after — comparison\n")

    if rows_eval:
        lines.append("## OBSERVE held-out evaluation\n")
        lines.append("| metric | seed | enhanced | Δ |")
        lines.append("|---|---:|---:|---:|")
        for r in rows_eval:
            d = r["delta"]
            d_str = fmt(d, r["fmt"]) if d is not None else "—"
            if d is not None and d != 0:
                d_str = ("+" if d > 0 else "") + d_str
            lines.append(
                f"| {r['label']} "
                f"| {fmt(r['seed'], r['fmt'])} "
                f"| {fmt(r['enhanced'], r['fmt'])} "
                f"| {d_str} |"
            )
        lines.append("")

    if rows_trace:
        lines.append("## PlantSwarm trace quality\n")
        lines.append("| metric | seed | enhanced | Δ |")
        lines.append("|---|---:|---:|---:|")
        for r in rows_trace:
            d = r["delta"]
            d_str = fmt(d, r["fmt"]) if d is not None else "—"
            if d is not None and d != 0:
                d_str = ("+" if d > 0 else "") + d_str
            lines.append(
                f"| {r['label']} "
                f"| {fmt(r['seed'], r['fmt'])} "
                f"| {fmt(r['enhanced'], r['fmt'])} "
                f"| {d_str} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def to_latex(rows_eval: List[dict], rows_trace: List[dict]) -> str:
    """Emit \\newcommand macros for \\input into the paper."""
    macro_safe = {
        "T3.macro_f1": "PathomeDeltaTthreeF",
        "T3.ece": "PathomeDeltaTthreeECE",
        "T3.tpcp": "PathomeDeltaTthreeTPCP",
        "T3_slices.seen.macro_f1": "PathomeDeltaTthreeFseen",
        "T3_slices.unseen.macro_f1": "PathomeDeltaTthreeFunseen",
        "T2.macro_f1": "PathomeDeltaTtwoF",
        "T1.macro_f1": "PathomeDeltaToneF",
        "T4.macro_f1": "PathomeDeltaTfourF",
        "n_traces": "PathomeDeltaTraceN",
        "avg_path_length": "PathomeDeltaPathLen",
        "backtrack_rate": "PathomeDeltaBacktrack",
        "high_confidence_rate": "PathomeDeltaHighConf",
        "t3_top1_acc": "PathomeDeltaTraceTop",
    }
    lines: List[str] = ["% PathomeDB before/after macros (auto-generated)"]
    for r in (rows_eval + rows_trace):
        name = macro_safe.get(r["field"])
        if name is None or r["delta"] is None:
            continue
        lines.append(f"\\newcommand{{\\{name}}}{{{r['fmt'].format(r['delta'])}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seed-eval", default=None,
                   help="pathome_eval.json from the seed-DB run")
    p.add_argument("--enhanced-eval", default=None,
                   help="pathome_eval.json from the enhanced-DB run")
    p.add_argument("--seed-traces", default=None,
                   help="plantswarm_traces.jsonl from the seed-DB run")
    p.add_argument("--enhanced-traces", default=None,
                   help="plantswarm_traces.jsonl from the enhanced-DB run")
    p.add_argument("--out-dir", default="results/pathome_compare",
                   help="output directory for comparison.{json,md,tex}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_eval = load_eval(Path(args.seed_eval)) if args.seed_eval else None
    enh_eval  = load_eval(Path(args.enhanced_eval)) if args.enhanced_eval else None
    seed_tr   = trace_summary(Path(args.seed_traces)) if args.seed_traces else None
    enh_tr    = trace_summary(Path(args.enhanced_traces)) if args.enhanced_traces else None

    rows_eval = diff_eval(seed_eval or {}, enh_eval or {}) if (seed_eval or enh_eval) else []
    rows_trace = diff_traces(seed_tr or {}, enh_tr or {}) if (seed_tr or enh_tr) else []

    if not rows_eval and not rows_trace:
        raise SystemExit("nothing to compare — provide at least one --*-eval or --*-traces pair")

    payload = {
        "eval":   {"seed": seed_eval, "enhanced": enh_eval, "rows": rows_eval},
        "traces": {"seed": seed_tr,   "enhanced": enh_tr,   "rows": rows_trace},
    }

    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "comparison.md").write_text(to_markdown(rows_eval, rows_trace))
    (out_dir / "comparison.tex").write_text(to_latex(rows_eval, rows_trace))

    print(f"wrote: {out_dir / 'comparison.json'}")
    print(f"wrote: {out_dir / 'comparison.md'}")
    print(f"wrote: {out_dir / 'comparison.tex'}")
    print()
    print(to_markdown(rows_eval, rows_trace))


if __name__ == "__main__":
    main()
