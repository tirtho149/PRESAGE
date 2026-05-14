"""
scripts/aggregate_pathomeood_tables.py
==================================
Walk ``results/pathomeood_eval/<run_id>/*.json`` from all variants and
baselines, and produce a per-paper-table markdown summary under
``results/tables/`` plus a master ``results/pathomeood_report.md`` that
maps directly to the BioCAP paper's tables.

Tables reproduced (Bugwood/KB analog of each paper table):
    Table 1   Zero-shot classification top-1 (PV / PW / PlantDoc)
    Table 2   Retrieval (Bugwood held-out R@1/5/10)
    Table 3   Caption-strategy ablation
    Table 4   KB-covered vs non-covered split
    Table 6   #-of-deltas ablation
    Table 8   KB coverage stats (descriptive)
    Table 13  Eval dataset stats (descriptive)
    Table 17  Few-image × covered split
    Table 18  Few-shot top-1 (1-shot / 5-shot)
    Table 19  PlantDoc (zero-shot)
    Table 20  Caption strategy × few-shot
    Figure 3  Recipe ablation bars (single/dual proj, 50/100 ep)

Skipped (and explained in the report):
    Tables 5, 11, 21  — human-eval, no annotators
    Table 7           — MLLM-captioner ablation, user chose KB-only path
    Table 12          — retrieval bench stats (covered by Table 2)
    Table 14          — CUB localization, no bounding boxes
    Tables 15, 16     — format-example ablations, N/A for KB path

Usage:
    python scripts/aggregate_pathomeood_tables.py [--results-dir results/pathomeood_eval]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# Variant tag → feature-ablation config (TabPFN matrix; mirrors
# scripts/tabpfn_eval.py::VARIANTS). Axes are FEATURE-LEVEL ablations:
# encoder choice + caption strategy + KB-covered/non-covered subset.
# No training-loop variants anymore (the trained-CLIP path is
# deprecated; everything goes through frozen-encoder + TabPFN now).
VARIANTS: Dict[str, Dict[str, object]] = {
    "T01": dict(encoder="bioclip",       strategy="label_only",         subset="all"),
    "T02": dict(encoder="bioclip",       strategy="summary_only",       subset="all"),
    "T03": dict(encoder="bioclip",       strategy="canonical_full",     subset="all"),
    "T04": dict(encoder="bioclip",       strategy="canonical_deltas_3", subset="all"),   # MAIN
    "T05": dict(encoder="bioclip",       strategy="canonical_deltas_1", subset="all"),
    "T06": dict(encoder="bioclip",       strategy="canonical_deltas_5", subset="all"),
    "T07": dict(encoder="bioclip",       strategy="canonical_deltas_7", subset="all"),
    "T08": dict(encoder="clip_vitb16",   strategy="canonical_deltas_3", subset="all"),   # encoder ablation
    "T09": dict(encoder="siglip_vitb16", strategy="canonical_deltas_3", subset="all"),   # encoder ablation
    "T10": dict(encoder="bioclip",       strategy="canonical_deltas_3", subset="covered"),
    "T11": dict(encoder="bioclip",       strategy="canonical_deltas_3", subset="non_covered"),
}

# Off-shelf zero-shot baselines (cosine-sim against class-name templates;
# no TabPFN). Produced by scripts/tabpfn_eval.py --include-baselines.
BASELINES = ["clip_vitb16_zs", "siglip_vitb16_zs", "bioclip_zs", "bioclip2_zs"]

# Paper-canonical column order for zero-shot results
EVAL_DATASETS_T1 = ["plantvillage", "plantwild"]
EVAL_DATASETS_T19 = ["plantdoc"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--results-dir", default="results/pathomeood_eval")
    p.add_argument("--out-dir", default="results/tables")
    p.add_argument("--report", default="results/pathomeood_report.md")
    return p.parse_args()


def _safe_load(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _zero_shot_top1(results_dir: Path, run_id: str, dataset: str) -> Optional[float]:
    """Returns top1 metric (in [0,1]) or None if missing."""
    payload = _safe_load(results_dir / run_id / f"{dataset}.json")
    if payload is None:
        return None
    metrics = payload.get("metrics") or {}
    # zero_shot_eval returns keys like "val-unseen-top1", "val-unseen-top5".
    for k, v in metrics.items():
        if k.endswith("-top1"):
            return float(v)
    return None


def _retrieval(results_dir: Path, run_id: str) -> Optional[dict]:
    payload = _safe_load(results_dir / run_id / "retrieval.json")
    if payload is None:
        return None
    return payload.get("metrics") or {}


def _fewshot(results_dir: Path, run_id: str, dataset: str, k: int) -> Optional[dict]:
    payload = _safe_load(results_dir / run_id / f"fewshot_{dataset}.json")
    if payload is None:
        return None
    return (payload or {}).get(f"{k}_shot")


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v*100:.1f}" if isinstance(v, (int, float)) else "—"


def _fmt_mean_std(d: Optional[dict]) -> str:
    if not d:
        return "—"
    return f"{d.get('mean', 0)*100:.1f} ±{d.get('std', 0)*100:.1f}"


# ---------------------------------------------------------------------------
# Per-table builders
# ---------------------------------------------------------------------------

def build_table_01(results_dir: Path) -> str:
    rows: List[List[str]] = [["Model", "PlantVillage", "PlantWild", "Mean"]]
    runs = BASELINES + ["T04"]   # main PathomeOOD = T04
    for r in runs:
        pv = _zero_shot_top1(results_dir, r, "plantvillage")
        pw = _zero_shot_top1(results_dir, r, "plantwild")
        vals = [v for v in (pv, pw) if v is not None]
        mean = sum(vals)/len(vals) if vals else None
        rows.append([r, _fmt_pct(pv), _fmt_pct(pw), _fmt_pct(mean)])
    return _md_table("Table 1 — Zero-shot classification top-1 (%)", rows)


def build_table_02(results_dir: Path) -> str:
    rows = [["Model", "I2T R@1", "I2T R@5", "I2T R@10", "T2I R@1", "T2I R@5", "T2I R@10"]]
    for r in BASELINES + ["T04"]:
        m = _retrieval(results_dir, r)
        if m is None:
            rows.append([r] + ["—"] * 6)
            continue
        rows.append([
            r,
            _fmt_pct(m.get("i2t_r1")),  _fmt_pct(m.get("i2t_r5")),  _fmt_pct(m.get("i2t_r10")),
            _fmt_pct(m.get("t2i_r1")),  _fmt_pct(m.get("t2i_r5")),  _fmt_pct(m.get("t2i_r10")),
        ])
    return _md_table("Table 2 — Bugwood held-out retrieval R@k (%)", rows)


def build_table_03(results_dir: Path) -> str:
    """Caption-strategy ablation: T01-T04 plus T05-T07 (number of deltas
    is Table 6, but the row spans Table 3 too)."""
    rows = [["Variant", "Strategy", "PlantVillage", "PlantWild", "I2T R@10", "T2I R@10"]]
    for v in ["T01", "T02", "T03", "T04", "T05", "T06", "T07"]:
        info = VARIANTS[v]
        pv = _zero_shot_top1(results_dir, v, "plantvillage")
        pw = _zero_shot_top1(results_dir, v, "plantwild")
        rt = _retrieval(results_dir, v) or {}
        rows.append([
            v, str(info["strategy"]),
            _fmt_pct(pv), _fmt_pct(pw),
            _fmt_pct(rt.get("i2t_r10")), _fmt_pct(rt.get("t2i_r10")),
        ])
    return _md_table("Table 3 — Caption-strategy ablation (%)", rows)


def build_table_04(results_dir: Path) -> str:
    rows = [["Variant", "Subset",       "PlantVillage", "PlantWild"]]
    for v in ["T04", "T10", "T11"]:
        info = VARIANTS[v]
        pv = _zero_shot_top1(results_dir, v, "plantvillage")
        pw = _zero_shot_top1(results_dir, v, "plantwild")
        rows.append([v, str(info["subset"]), _fmt_pct(pv), _fmt_pct(pw)])
    return _md_table("Table 4 — KB-covered vs non-covered subset (%)", rows)


def build_table_06(results_dir: Path) -> str:
    """#-of-deltas ablation: T05 (1), T04 (3), T06 (5), T07 (7)."""
    rows = [["k deltas", "Variant", "PlantVillage", "PlantWild"]]
    for k, v in [(1, "T05"), (3, "T04"), (5, "T06"), (7, "T07")]:
        pv = _zero_shot_top1(results_dir, v, "plantvillage")
        pw = _zero_shot_top1(results_dir, v, "plantwild")
        rows.append([str(k), v, _fmt_pct(pv), _fmt_pct(pw)])
    return _md_table("Table 6 — Number-of-deltas ablation (%)", rows)


def build_table_18(results_dir: Path) -> str:
    rows = [["Model", "PV 1-shot", "PV 5-shot", "PW 1-shot", "PW 5-shot"]]
    for r in BASELINES + ["T04"]:
        rows.append([
            r,
            _fmt_mean_std(_fewshot(results_dir, r, "plantvillage", 1)),
            _fmt_mean_std(_fewshot(results_dir, r, "plantvillage", 5)),
            _fmt_mean_std(_fewshot(results_dir, r, "plantwild",    1)),
            _fmt_mean_std(_fewshot(results_dir, r, "plantwild",    5)),
        ])
    return _md_table("Table 18 — Few-shot top-1 (mean ± std %)", rows)


def build_table_19(results_dir: Path) -> str:
    rows = [["Model", "PlantDoc"]]
    for r in BASELINES + ["T04"]:
        pd = _zero_shot_top1(results_dir, r, "plantdoc")
        rows.append([r, _fmt_pct(pd)])
    return _md_table("Table 19 — Beyond classification: PlantDoc (%)", rows)


def build_table_20(results_dir: Path) -> str:
    rows = [["Variant", "Strategy", "PV 1-shot", "PV 5-shot", "PW 1-shot", "PW 5-shot"]]
    for v in ["T01", "T02", "T03", "T04", "T05", "T06", "T07"]:
        info = VARIANTS[v]
        rows.append([
            v, str(info["strategy"]),
            _fmt_mean_std(_fewshot(results_dir, v, "plantvillage", 1)),
            _fmt_mean_std(_fewshot(results_dir, v, "plantvillage", 5)),
            _fmt_mean_std(_fewshot(results_dir, v, "plantwild",    1)),
            _fmt_mean_std(_fewshot(results_dir, v, "plantwild",    5)),
        ])
    return _md_table("Table 20 — Caption strategy × few-shot (mean ± std %)", rows)


def build_figure_03(results_dir: Path) -> str:
    """Encoder ablation: T04 (BioCLIP), T08 (CLIP-openai), T09 (SigLIP).

    Replaces the original BioCAP "projector mode × epochs" ablation
    (which is meaningless for the TabPFN pipeline since there's no
    projector or training loop) with the more useful encoder-choice
    ablation.
    """
    rows = [["Variant", "Encoder", "Strategy", "PlantVillage", "PlantDoc", "PlantWild"]]
    for v in ["T04", "T08", "T09"]:
        info = VARIANTS[v]
        pv = _zero_shot_top1(results_dir, v, "plantvillage")
        pd = _zero_shot_top1(results_dir, v, "plantdoc")
        pw = _zero_shot_top1(results_dir, v, "plantwild")
        rows.append([
            v, str(info["encoder"]), str(info["strategy"]),
            _fmt_pct(pv), _fmt_pct(pd), _fmt_pct(pw),
        ])
    return _md_table("Figure 3 — Encoder ablation (frozen + TabPFN, %)", rows)


def build_table_08() -> str:
    """KB coverage descriptive table. Reads artifacts/pathome_kb/*/final_registry.json."""
    repo_root = Path(__file__).parent.parent.resolve()
    rows = [["Crop", "# diseases", "# with deltas", "# total deltas"]]
    kb_root = repo_root / "artifacts" / "pathome_kb"
    if not kb_root.is_dir():
        return _md_table("Table 8 — KB coverage (KB root missing)", rows)
    for crop_dir in sorted(kb_root.iterdir()):
        reg = crop_dir / "final_registry.json"
        if not reg.is_file():
            continue
        d = json.loads(reg.read_text())
        diseases = d.get("diseases") or []
        with_deltas = 0
        total_deltas = 0
        for dis in diseases:
            ro = dis.get("regional_observations") or {}
            states_with = [s for s, rec in ro.items() if isinstance(rec, dict) and rec.get("deltas")]
            if states_with:
                with_deltas += 1
                for s in states_with:
                    total_deltas += len(ro[s]["deltas"])
        rows.append([crop_dir.name, str(len(diseases)), str(with_deltas), str(total_deltas)])
    return _md_table("Table 8 — KB coverage (descriptive)", rows)


def build_table_13() -> str:
    """Eval dataset stats: walked from data/eval/ if present."""
    rows = [["Dataset", "Path", "Class folders"]]
    paths = [
        ("PlantVillage", "data/eval/PlantVillage"),
        ("PlantWild",    "data/eval/PlantWild"),
        ("PlantDoc",     "data/eval/PlantDoc/test"),
    ]
    for name, rel in paths:
        p = Path(rel)
        if p.is_dir():
            n = sum(1 for x in p.iterdir() if x.is_dir())
            rows.append([name, rel, str(n)])
        else:
            rows.append([name, rel, "(missing)"])
    return _md_table("Table 13 — Eval datasets (descriptive)", rows)


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _md_table(title: str, rows: List[List[str]]) -> str:
    if not rows:
        return f"### {title}\n\n*(no data)*\n"
    header, *body = rows
    out = [f"### {title}", ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "table_01": lambda: build_table_01(results_dir),
        "table_02": lambda: build_table_02(results_dir),
        "table_03": lambda: build_table_03(results_dir),
        "table_04": lambda: build_table_04(results_dir),
        "table_06": lambda: build_table_06(results_dir),
        "table_08": build_table_08,
        "table_13": build_table_13,
        "table_18": lambda: build_table_18(results_dir),
        "table_19": lambda: build_table_19(results_dir),
        "table_20": lambda: build_table_20(results_dir),
        "figure_03": lambda: build_figure_03(results_dir),
    }

    pieces: List[str] = []
    for name, builder in builders.items():
        md = builder()
        (out_dir / f"{name}.md").write_text(md)
        pieces.append(md)
        print(f"  wrote {out_dir / (name + '.md')}")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# PathomeOOD — paper-table reproduction\n"
        "\n"
        "This report reproduces every reproducible table from the BioCAP paper\n"
        "(Zhang et al., arXiv:2510.20095) on PlantSwarm's Bugwood + PathomeDB-KB\n"
        "data, using KB-derived hybrid captions (canonical + per-state regional\n"
        "deltas, no MLLM caption generator).\n"
        "\n"
        "## Skipped paper tables\n"
        "\n"
        "  - **Tables 5, 11, 21**: human-evaluation tables — need human raters\n"
        "  - **Table 7**: MLLM-captioner family/size ablation — user chose KB-only path\n"
        "  - **Table 12**: retrieval-bench stats — equivalent info in Table 2\n"
        "  - **Table 14**: CUB localization — Bugwood has no bounding boxes\n"
        "  - **Tables 15, 16**: format-example ablations — N/A for KB path\n"
        "\n"
        "## Reproduced tables\n"
        "\n"
    )
    report.write_text(header + "\n".join(pieces))
    print(f"\n  wrote master report: {report}")


if __name__ == "__main__":
    main()
