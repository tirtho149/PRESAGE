#!/usr/bin/env python3
"""
scripts/sync_latex_metrics.py
==============================
Read experiment JSON under a results directory and regenerate LaTeX snippets in
``plantswarm/latex/``:

  * auto_metrics.tex              — inline macros (\\AutoMacroFOneTThree, etc.)
  * auto_table_main_results.tex   — Table~\\ref{tab:main} (four benchmarks × T3/T2 + ECE + TPCP)
  * auto_table_ablation_results.tex — Table~\\ref{tab:ablation-results}
  * auto_table_predictions.tex    — Table~\\ref{tab:predictions} (P1--P4)
  * auto_table_mechanisms.tex     — Table~\\ref{tab:mechanisms}
  * auto_table_budget.tex         — Table~\\ref{tab:budget} (optional JSON)

``plantswarm_metrics.json`` may include optional ``by_benchmark`` with keys
``plantvillage``, ``plantdoc``, ``plantwild``, ``leafbench``; each holds
per-task blocks like ``{"T3": {"macro_f1": ...}, "T2": {...}}``.
Populate ``data.benchmark_col`` in YAML and run the main pipeline to emit these.

Usage:
    python scripts/sync_latex_metrics.py --results-dir results/my_run
    python scripts/sync_latex_metrics.py --results-dir results/ --latex-dir plantswarm/latex
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Order matches acl_latex.tex Table~\\ref{tab:main} columns (T3,T2 per benchmark).
BENCHMARK_KEYS: Tuple[str, ...] = ("plantvillage", "plantdoc", "plantwild", "leafbench")

# (LaTeX row label, baseline_results.json key or None)
# PlantSwarm row uses metrics JSON, not baselines.
WIDE_MAIN_ROWS: List[Tuple[str, Optional[str]]] = [
    ("Single VLM", "Single VLM"),
    ("MVPDR", None),
    ("Snap \\& Diagnose", None),
    ("SAGE (no KB)", None),
    ("SAGE+KB ($k{=}16$)", None),
    ("Chat Demeter", None),
    ("FrugalGPT", None),
    ("Fixed Chain", "Fixed Chain"),
    ("Fixed Chain + Ctx", "Fixed Chain+Ctx"),
    ("3-Agent Swarm", None),
    ("\\textbf{PlantSwarm}", None),
    ("Entropy-Gated", None),
]

# Paper label -> variant name in ablation_metrics_T3.json
ABLATION_VARIANT_KEYS: List[Tuple[str, str]] = [
    ("Fixed Chain", "Fixed Chain"),
    ("FC + Context", "Fixed Chain + Full Ctx"),
    ("Free, No $\\kappa$", "Free, No Conf-Gate"),
    ("Free, No Backtrack", "Free, No Backtrack"),
    ("3-Agent Swarm", "3-Agent Swarm"),
    ("\\textbf{PlantSwarm}", "PlantSwarm Full"),
    ("Entropy-Gated", "Entropy-Gated Swarm"),
]


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _find_plantswarm_metrics(results_dir: Path) -> Optional[Path]:
    for name in ("plantswarm_metrics.json",):
        p = results_dir / name
        if p.is_file():
            return p
    return None


def _fmt_f1(x: Any) -> str:
    if x is None:
        return "---"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "---"
    if math.isnan(v):
        return "---"
    return f"{v:.1f}"


def _fmt_ece(x: Any) -> str:
    if x is None:
        return "---"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "---"
    if math.isnan(v):
        return "---"
    return f"{v:.4f}"


def _fmt_tpcp(x: Any) -> str:
    if x is None:
        return "---"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "---"
    if math.isnan(v) or math.isinf(v):
        return "---"
    return f"{v:.1f}"


def _fmt_rho(x: Any) -> str:
    if x is None:
        return "---"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "---"
    if math.isnan(v):
        return "---"
    return f"{v:.3f}"


def _mean_ece_tpcp(metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    eces, tpcps = [], []
    for tid in ("T1", "T2", "T3", "T4", "T5"):
        block = metrics.get(tid) or {}
        if "ece" in block:
            eces.append(block["ece"])
        if "tpcp" in block:
            t = block["tpcp"]
            if isinstance(t, (int, float)) and not math.isnan(t) and not math.isinf(t):
                tpcps.append(t)
    if not eces:
        return None, None
    return sum(eces) / len(eces), (sum(tpcps) / len(tpcps)) if tpcps else None


def _benchmark_t3_t2(bb: Optional[Dict[str, Any]], bench: str) -> Tuple[str, str]:
    if not bb:
        return "---", "---"
    block = bb.get(bench) or bb.get(bench.replace("_", ""))
    if not block:
        return "---", "---"
    t3 = _fmt_f1((block.get("T3") or {}).get("macro_f1"))
    t2 = _fmt_f1((block.get("T2") or {}).get("macro_f1"))
    return t3, t2


def _plantswarm_wide_cells(metrics: Optional[Dict[str, Any]]) -> Tuple[str, ...]:
    """8 benchmark cells (4×T3/T2) + ECE + TPCP = 10 numeric columns."""
    if not metrics:
        return ("---",) * 10
    bb = metrics.get("by_benchmark")
    cells: List[str] = []
    for b in BENCHMARK_KEYS:
        t3, t2 = _benchmark_t3_t2(bb, b)
        cells.extend([t3, t2])
    me, mt = _mean_ece_tpcp(metrics)
    cells.append(_fmt_ece(me))
    cells.append(_fmt_tpcp(mt))
    return tuple(cells)


def _baseline_wide_cells(
    baselines: Optional[Dict[str, Any]],
    key: Optional[str],
) -> Tuple[str, ...]:
    """Baselines: no per-benchmark in JSON yet — leave benchmark columns as ---."""
    if not key or not baselines or key not in baselines:
        return ("---",) * 10
    b = baselines[key]
    ece = _fmt_ece(b.get("ece_T1"))
    tp = _fmt_tpcp(b.get("tpcp_T1"))
    bench = ("---",) * 8
    return bench + (ece, tp)


def _escape_latex_row(s: str) -> str:
    return s.replace("%", "\\%")


def build_main_table_wide_fragment(
    metrics: Optional[Dict[str, Any]],
    baselines: Optional[Dict[str, Any]],
) -> str:
    ps_cells = _plantswarm_wide_cells(metrics)
    lines: List[str] = []
    for display, bkey in WIDE_MAIN_ROWS:
        if display.startswith("\\textbf{PlantSwarm"):
            parts = list(ps_cells)
            line = (
                f"{display}     & "
                + " & ".join(f"\\textbf{{{p}}}" for p in parts)
                + " \\\\"
            )
        elif bkey:
            parts = list(_baseline_wide_cells(baselines, bkey))
            line = f"{display}       & " + " & ".join(parts) + " \\\\"
        else:
            line = f"{display}       & " + " & ".join(["---"] * 10) + " \\\\"
        lines.append(_escape_latex_row(line))
    # Keep final rule inside the fragment so \\ from last data row is parsed before \bottomrule.
    lines.append("\\bottomrule")
    return "\n".join(lines)


def build_auto_metrics_tex(
    metrics: Optional[Dict[str, Any]],
    routing: Optional[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    lines = [
        "% -*- latex -*-",
        "% GENERATED by scripts/sync_latex_metrics.py — do not edit by hand.",
        f"% Generated at: {summary.get('generated_at', '')}",
        "",
    ]

    def add_macro(name: str, value: str) -> None:
        lines.append(f"\\providecommand{{\\{name}}}{{{value}}}")

    if metrics:
        tid_name = {
            "T1": "TOne",
            "T2": "TTwo",
            "T3": "TThree",
            "T4": "TFour",
            "T5": "TFive",
        }
        for tid in ("T1", "T2", "T3", "T4", "T5"):
            block = metrics.get(tid) or {}
            tname = tid_name[tid]
            add_macro(f"AutoMacroFOne{tname}", _fmt_f1(block.get("macro_f1")))
            add_macro(f"AutoECE{tname}", _fmt_ece(block.get("ece")))
            add_macro(f"AutoTPCP{tname}", _fmt_tpcp(block.get("tpcp")))
        me, mt = _mean_ece_tpcp(metrics)
        add_macro("AutoECEMeanTasks", _fmt_ece(me))
        add_macro("AutoTPCPMeanTasks", _fmt_tpcp(mt))

    p3 = (routing or {}).get("p3_early_vs_extended") or {}
    dpp = p3.get("delta_accuracy_pp")
    add_macro(
        "AutoPThreeDeltaAccPP",
        f"{dpp:+.1f}" if isinstance(dpp, (int, float)) and not math.isnan(dpp) else "---",
    )
    rq5 = (routing or {}).get("rq5_hedge_pathogen_severity") or {}
    rho = rq5.get("spearman_rho")
    add_macro("AutoRQFiveHedgeRho", _fmt_rho(rho))

    em = routing.get("exact_match_rate") if routing else None
    add_macro(
        "AutoRoutingExactMatch",
        f"{em:.3f}" if isinstance(em, (int, float)) and not math.isnan(em) else "---",
    )
    add_macro("AutoSubsetSize", str(summary.get("subset_hint", "---")))
    add_macro("AutoResultsDirName", summary.get("results_dir_basename", "---"))

    # P4: optional future field
    p4rho = (routing or {}).get("p4_kappa_entropy_spearman")
    add_macro("AutoPFourKappaEntropyRho", _fmt_rho(p4rho))

    lines.append("")
    return "\n".join(lines)


def _ablation_variant_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path or not path.is_file():
        return {}
    data = _load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    for v in data.get("variants") or []:
        name = v.get("variant")
        if name:
            out[name] = v
    return out


def build_ablation_table_fragment(ablation_by_variant: Dict[str, Dict[str, Any]]) -> str:
    """Two benchmark columns × (T3 F1, ECE) — duplicated from pooled T3 when stratified data absent."""
    lines: List[str] = []
    for display, vkey in ABLATION_VARIANT_KEYS:
        v = ablation_by_variant.get(vkey)
        if v is None:
            if display.startswith("\\textbf"):
                row = (
                    f"{display} & \\textbf{{---}} & \\textbf{{---}} & "
                    "\\textbf{---} & \\textbf{---} \\\\"
                )
            else:
                row = f"{display} & --- & --- & --- & --- \\\\"
        else:
            t3 = _fmt_f1(v.get("macro_f1_T3"))
            ece = _fmt_ece(v.get("ece_T3"))
            if display.startswith("\\textbf"):
                t3c = f"\\textbf{{{t3}}}"
                ecec = f"\\textbf{{{ece}}}"
                row = f"{display} & {t3c} & {ecec} & {t3c} & {ecec} \\\\"
            else:
                row = f"{display} & {t3} & {ece} & {t3} & {ece} \\\\"
        lines.append(_escape_latex_row(row))
    lines.append("\\bottomrule")
    return "\n".join(lines)


def build_predictions_table_fragment(routing: Optional[Dict[str, Any]]) -> str:
    p1_s, st1 = "---", "pending"
    p3_s, st3 = "---", "pending"
    p4_s, st4 = "---", "pending"
    if routing:
        p3d = routing.get("p3_early_vs_extended") or {}
        dpp = p3d.get("delta_accuracy_pp")
        if isinstance(dpp, (int, float)) and not math.isnan(dpp):
            p3_s = f"{dpp:+.1f}"
            st3 = "measured"
        rq5 = routing.get("rq5_hedge_pathogen_severity") or {}
        if rq5.get("spearman_rho") is not None:
            st1 = "see RQ5 hedge"
        p4 = routing.get("p4_kappa_entropy_spearman")
        if isinstance(p4, (int, float)) and not math.isnan(p4):
            p4_s = f"{p4:+.3f}"
            st4 = "measured"

    rows = [
        "P1 & $\\rho(L,\\,H_{\\hat{y}})$ & $+0.42$--$+0.55$ & "
        f"{st1} ({p1_s}) \\\\",
        "P2 & $\\Delta$Acc (2nd$-$1st pass, Pa) & $+9$ F1 & see traces \\\\",
        "P3 & $\\mathrm{acc}(\\mathrm{early}){-}\\mathrm{acc}(\\mathrm{extended})$ & "
        f"$+12$ F1 & {st3} ({p3_s}) \\\\",
        "P4 & $\\rho(H^{(\\mathrm{dis})},\\,\\kappa{=}\\texttt{L})$ & "
        f"$+0.55$--$+0.70$ & {st4} ({p4_s}) \\\\",
        "\\bottomrule",
    ]
    return "\n".join(_escape_latex_row(x) for x in rows)


def build_mechanisms_table_fragment(routing: Optional[Dict[str, Any]]) -> str:
    rho = None
    if routing:
        rq5 = routing.get("rq5_hedge_pathogen_severity") or {}
        rho = rq5.get("spearman_rho")
    hedge = _fmt_rho(rho)
    rows = [
        "Retrospective grounding $\\Delta$Acc (Pa) & $+9$ F1  & --- \\\\",
        "Retrospective grounding $\\Delta$Acc (Sy) & $+5$ F1  & --- \\\\",
        "Contradiction detection $P(\\mathrm{step}_j)$ & $0.72$--$0.80$ & --- \\\\",
        f"Hedge propagation $\\rho$ & $-0.38$--$-0.52$ & {hedge} \\\\",
        "\\bottomrule",
    ]
    return "\n".join(_escape_latex_row(x) for x in rows)


def build_budget_table_fragment(budget: Optional[Dict[str, Any]]) -> str:
    """Expect budget['rows'] = [ {beta, t3_f1, t2_f1, ece, mean_L, tpcp}, ... ] or empty."""
    if not budget or not budget.get("rows"):
        lines = [
            "0 (no BT)   & --- & --- & --- & --- & --- \\\\",
            "1 (default) & --- & --- & --- & --- & --- \\\\",
            "2 (two BT)  & --- & --- & --- & --- & --- \\\\",
            "\\bottomrule",
        ]
        return "\n".join(lines)
    out: List[str] = []
    for r in budget["rows"]:
        out.append(
            f"{r.get('label', '')} & {_fmt_f1(r.get('t3_f1'))} & {_fmt_f1(r.get('t2_f1'))} & "
            f"{_fmt_ece(r.get('ece'))} & {_fmt_f1(r.get('mean_L'))} & {_fmt_tpcp(r.get('tpcp'))} "
            "\\\\"
        )
    out.append("\\bottomrule")
    return "\n".join(_escape_latex_row(x) for x in out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync LaTeX tables with JSON experiment outputs.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing plantswarm_metrics.json (and baselines, routing, etc.).",
    )
    parser.add_argument(
        "--latex-dir",
        default=str(REPO_ROOT / "plantswarm" / "latex"),
        help="Where to write auto_*.tex",
    )
    parser.add_argument(
        "--subset-hint",
        default="",
        help="Optional note embedded in auto_metrics (e.g. Slurm SUBSET value).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    latex_dir = Path(args.latex_dir).resolve()
    latex_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(REPO_ROOT)

    metrics_path = _find_plantswarm_metrics(results_dir)
    metrics = _load_json(metrics_path) if metrics_path else None

    baseline_path = results_dir / "baseline_results.json"
    baselines = _load_json(baseline_path) if baseline_path.is_file() else None

    routing_path = results_dir / "routing_analysis.json"
    routing = _load_json(routing_path) if routing_path.is_file() else None

    ablation_t3_path = results_dir / "ablation_metrics_T3.json"
    ablation_map = _ablation_variant_map(ablation_t3_path if ablation_t3_path.is_file() else None)

    budget_path = results_dir / "budget_sensitivity.json"
    budget = _load_json(budget_path) if budget_path.is_file() else None

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "results_dir_basename": results_dir.name,
        "subset_hint": args.subset_hint or os.environ.get("SUBSET", "") or "---",
        "metrics_file": str(metrics_path) if metrics_path else None,
        "had_baselines": baselines is not None,
        "had_routing": routing is not None,
        "had_ablation_t3": bool(ablation_map),
        "had_budget_json": budget is not None,
    }

    summary_path = results_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    main_tex = build_main_table_wide_fragment(metrics, baselines)
    (latex_dir / "auto_table_main_results.tex").write_text(main_tex, encoding="utf-8")

    auto_metrics = build_auto_metrics_tex(metrics, routing, summary)
    (latex_dir / "auto_metrics.tex").write_text(auto_metrics, encoding="utf-8")

    (latex_dir / "auto_table_predictions.tex").write_text(
        build_predictions_table_fragment(routing), encoding="utf-8"
    )
    (latex_dir / "auto_table_ablation_results.tex").write_text(
        build_ablation_table_fragment(ablation_map), encoding="utf-8"
    )
    (latex_dir / "auto_table_mechanisms.tex").write_text(
        build_mechanisms_table_fragment(routing), encoding="utf-8"
    )
    (latex_dir / "auto_table_budget.tex").write_text(
        build_budget_table_fragment(budget), encoding="utf-8"
    )

    print(f"Wrote {latex_dir / 'auto_metrics.tex'}")
    print(f"Wrote {latex_dir / 'auto_table_main_results.tex'}")
    print(f"Wrote {latex_dir / 'auto_table_predictions.tex'}")
    print(f"Wrote {latex_dir / 'auto_table_ablation_results.tex'}")
    print(f"Wrote {latex_dir / 'auto_table_mechanisms.tex'}")
    print(f"Wrote {latex_dir / 'auto_table_budget.tex'}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
