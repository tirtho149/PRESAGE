"""
scripts/viz/kb_stats.py
=======================
Summary visualizations for the KB seed JSON:

  - per-crop disease count
  - verification status pie (verified / weakly / provisional / novel /
    unverified / contradictory)
  - swarm_support histogram
  - per-field delta count
  - per-state delta coverage

Outputs:
  results/figures/kb_*.png
  plantswarm/latex/auto_kb_stats.tex     (table + \\includegraphics blocks)

Usage:
  python scripts/viz/kb_stats.py --seed artifacts/pathome_seed/symptoms_seed.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.viz._common import (
    ensure_dirs, fig_path, figure_includegraphics, get_mpl,
    have_matplotlib, latex_escape, write_tex,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seed", required=True,
                   help="Path to symptoms_seed.json")
    p.add_argument("--name", default="kb_stats",
                   help="Output basename")
    return p.parse_args()


def _load_seed(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _aggregate(seed: dict) -> dict:
    """Roll up summary stats from a SymptomLibrary seed JSON."""
    profiles = seed.get("profiles") or []
    crop_counts:        Counter = Counter()
    status_counts:      Counter = Counter()
    field_counts:       Counter = Counter()
    state_counts:       Counter = Counter()
    support_values:     list    = []
    n_canonical_filled: int = 0
    n_with_regional:    int = 0
    n_total_deltas:     int = 0

    for p in profiles:
        crop_counts[p.get("crop", "?")] += 1
        canonical = p.get("canonical") or {}
        if canonical.get("summary"):
            n_canonical_filled += 1
        regional = p.get("regional_observations") or {}
        if regional:
            n_with_regional += 1
        for state, rec in regional.items():
            for d in (rec or {}).get("deltas") or []:
                n_total_deltas += 1
                field_counts[d.get("field", "other")] += 1
                state_counts[state] += 1
                status_counts[d.get("verification_status", "unverified")] += 1
                s = d.get("swarm_support") or d.get("support") or 0
                if isinstance(s, (int, float)):
                    support_values.append(int(s))
    return {
        "n_profiles":         len(profiles),
        "n_canonical_filled": n_canonical_filled,
        "n_with_regional":    n_with_regional,
        "n_total_deltas":     n_total_deltas,
        "crop_counts":        crop_counts,
        "status_counts":      status_counts,
        "field_counts":       field_counts,
        "state_counts":       state_counts,
        "support_values":     support_values,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_status_pie(plt, status_counts: Counter, out_png: Path) -> bool:
    if not status_counts:
        return False
    labels = list(status_counts.keys())
    sizes  = [status_counts[k] for k in labels]
    color_map = {
        "verified":          "#4CAF50",
        "weakly_supported":  "#8BC34A",
        "provisional":       "#FFC107",
        "novel_plausible":   "#FF9800",
        "unverified":        "#9E9E9E",
        "contradictory":     "#F44336",
        "duplicate_existing": "#607D8B",
    }
    colors = [color_map.get(k, "#90A4AE") for k in labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
           startangle=90, wedgeprops=dict(width=0.4, edgecolor="white"))
    ax.set_title("Regional delta verification status")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_field_count(plt, field_counts: Counter, out_png: Path) -> bool:
    if not field_counts:
        return False
    items = field_counts.most_common()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([f for f, _ in items][::-1], [c for _, c in items][::-1],
            color="#3949AB")
    ax.set_xlabel("Number of deltas")
    ax.set_title("Regional deltas per canonical field")
    for i, (_, c) in enumerate(items[::-1]):
        ax.text(c, i, f" {c}", va="center", fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_support_hist(plt, support_values: list, out_png: Path) -> bool:
    if not support_values:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    max_s = max(support_values)
    bins  = list(range(1, max(max_s + 2, 3)))
    ax.hist(support_values, bins=bins, color="#00897B", edgecolor="white")
    ax.set_xlabel("swarm_support (K-of-N agreement count)")
    ax.set_ylabel("Number of deltas")
    ax.set_title("Swarm support distribution")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_state_coverage(plt, state_counts: Counter, out_png: Path) -> bool:
    if not state_counts:
        return False
    items = state_counts.most_common(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([s for s, _ in items][::-1], [c for _, c in items][::-1],
            color="#D81B60")
    ax.set_xlabel("Number of deltas")
    ax.set_title("Top 20 states by regional delta count")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# LaTeX snippet
# ---------------------------------------------------------------------------

def _build_tex(name: str, stats: dict, figures_written: list[str]) -> str:
    """Build a LaTeX snippet with a summary table + figure includes."""
    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/viz/kb_stats.py — do not edit by hand.")
    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{lr}")
    lines.append(r"    \toprule")
    lines.append(r"    Metric & Value \\")
    lines.append(r"    \midrule")
    lines.append(f"    Profiles            & {stats['n_profiles']} \\\\")
    lines.append(f"    With canonical text & {stats['n_canonical_filled']} \\\\")
    lines.append(f"    With regional KB    & {stats['n_with_regional']} \\\\")
    lines.append(f"    Regional deltas     & {stats['n_total_deltas']} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{PathomeDB seed summary.}")
    lines.append(r"  \label{tab:kb_stats}")
    lines.append(r"\end{table}")
    lines.append("")
    # Per-status counts table
    if stats["status_counts"]:
        lines.append(r"\begin{table}[t]")
        lines.append(r"  \centering")
        lines.append(r"  \small")
        lines.append(r"  \begin{tabular}{lr}")
        lines.append(r"    \toprule")
        lines.append(r"    Verification status & Deltas \\")
        lines.append(r"    \midrule")
        for status, n in stats["status_counts"].most_common():
            lines.append(f"    {latex_escape(status)} & {n} \\\\")
        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"  \caption{Regional deltas by verification status.}")
        lines.append(r"  \label{tab:kb_status}")
        lines.append(r"\end{table}")
        lines.append("")
    # Figure includes
    for fig_name, caption, label in figures_written:
        lines.append(figure_includegraphics(fig_name, caption, label))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not Path(args.seed).is_file():
        raise SystemExit(f"seed JSON not found: {args.seed}")

    seed  = _load_seed(args.seed)
    stats = _aggregate(seed)

    print("=== KB stats ===")
    print(f"  profiles total       : {stats['n_profiles']}")
    print(f"  w/ canonical summary : {stats['n_canonical_filled']}")
    print(f"  w/ regional KB       : {stats['n_with_regional']}")
    print(f"  total regional deltas: {stats['n_total_deltas']}")
    print(f"  by status            : {dict(stats['status_counts'])}")
    print(f"  by field             : {dict(stats['field_counts'])}")
    print(f"  by state (top 5)     : {stats['state_counts'].most_common(5)}")

    figures_written: list[tuple[str, str, str]] = []

    if have_matplotlib():
        _, plt = get_mpl()
        plots = [
            (f"{args.name}_status",       _plot_status_pie,    stats["status_counts"],
             "Distribution of verification statuses for regional deltas.",
             "kb_status_pie"),
            (f"{args.name}_field",        _plot_field_count,   stats["field_counts"],
             "Number of regional deltas emitted per canonical field.",
             "kb_field_count"),
            (f"{args.name}_support",      _plot_support_hist,  stats["support_values"],
             "Swarm support distribution across regional deltas.",
             "kb_support_hist"),
            (f"{args.name}_state",        _plot_state_coverage, stats["state_counts"],
             "Top 20 states ranked by regional delta count.",
             "kb_state_coverage"),
        ]
        for figname, fn, data, caption, label in plots:
            out_png = fig_path(figname)
            ok = fn(plt, data, out_png) if data else False
            if ok:
                print(f"  wrote {out_png}")
                figures_written.append((figname, caption, label))
    else:
        print("  matplotlib not installed — skipping figures, writing text tables only")

    tex_out = write_tex(args.name, _build_tex(args.name, stats, figures_written))
    print(f"  wrote {tex_out}")


if __name__ == "__main__":
    main()
