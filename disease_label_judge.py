"""
Two-layer LLM-as-Judge for crop disease dataset quality.

LAYER 1 — Crop Validation (decision tree)
  Node A: Is this entry a real plant/organism at all?
    → NO  → INVALID (e.g. spreadsheet artefacts like "TOTAL")
    → YES → Node B
  Node B: Is it a cultivated crop used in agriculture or horticulture?
    → NO  → NON_CROP (wild plant, purely decorative, non-crop organism)
    → YES → Node C
  Node C: Is the crop name correctly spelled and standardized?
    → NO  → MISSPELLED / UNSTANDARDIZED
    → YES → VALID

LAYER 2 — Disease Label Quality (only for VALID / MISSPELLED crops)
  For each disease label:
    CORRECT / INCORRECT / QUESTIONABLE
  Plus: similar / duplicate group detection.

Uses `claude -p` headless subprocess — no API key needed.
Results stream live to a progress txt and a JSON report.
Automatically resumes from saved results on re-run.

Two run modes:
  1. Standalone:
        python disease_label_judge.py \
            --csv crop_disease_registry.csv \
            --output disease_label_full_report.json
     Reads a flat CSV with "Crop" and "Disease" columns and writes
     a JSON report.

  2. Bugwood pipeline integration:
        python disease_label_judge.py \
            --csv BugWood_Diseases_usable.csv \
            --crop-col NormCrop --disease-col NormDisease \
            --output artifacts/bugwood_judgement.json \
            --apply --apply-out BugWood_Diseases_judged.csv
     Same judge over the already-filtered Bugwood rows, then rewrite
     the CSV dropping INVALID / NON_CROP crops and INCORRECT
     diseases. Canonicalises MISSPELLED crop names in place.

The module also exposes ``judge_crop_disease_map`` and
``apply_judgement_to_rows`` so other scripts (e.g.
``scripts/filter_bugwood_csv.py --judge``) can call the pipeline
in-process.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Default file paths — used when invoked standalone with no overrides.
_DEFAULT_CSV      = Path(__file__).parent / "crop_disease_registry.csv"
_DEFAULT_OUTPUT   = Path(__file__).parent / "disease_label_full_report.json"
_DEFAULT_PROGRESS = Path(__file__).parent / "disease_label_full_progress.txt"

# ── prompts ──────────────────────────────────────────────────────────────────

CROP_VALIDATION_PROMPT = """\
You are an expert agronomist. Apply the decision tree below to the crop name given.

DECISION TREE:
  Node A: Is this entry a real plant or plant-based crop (not a spreadsheet label, number, or non-plant)?
    → NO  → verdict: INVALID   (e.g. "TOTAL", "N/A", a row header)
    → YES → Node B
  Node B: Is it grown as an agricultural or horticultural crop (food, fibre, spice, timber, ornamental)?
    → NO  → verdict: NON_CROP  (purely wild plant, a pathogen name, not cultivated)
    → YES → Node C
  Node C: Is the common crop name spelled correctly and reasonably standardised?
    → NO  → verdict: MISSPELLED_OR_UNSTANDARDISED
    → YES → verdict: VALID

Return ONLY valid JSON (no markdown, no fences):
{
  "crop": "<exact input name>",
  "node_a": "YES" | "NO",
  "node_b": "YES" | "NO" | "N/A",
  "node_c": "YES" | "NO" | "N/A",
  "verdict": "VALID" | "INVALID" | "NON_CROP" | "MISSPELLED_OR_UNSTANDARDISED",
  "canonical_name": "<corrected / standard name, or same if VALID>",
  "category": "<one of: cereal, legume, fruit, vegetable, root_tuber, tree_crop, fiber_crop, oilseed, herb_spice, ornamental, beverage_crop, nut_crop, forage, other>",
  "notes": "<one sentence — only if something is unusual or needs clarification, else empty string>"
}

Crop to evaluate:
"""

DISEASE_JUDGE_PROMPT = """\
You are an expert plant pathologist. Quality-check disease labels for the crop below.

Return ONLY valid JSON (no markdown, no fences):
{
  "crop": "<crop name>",
  "disease_verdicts": [
    {
      "disease": "<label>",
      "verdict": "CORRECT" | "INCORRECT" | "QUESTIONABLE",
      "reason": "<one sentence>"
    }
  ],
  "similar_groups": [
    {
      "diseases": ["<label A>", "<label B>"],
      "reason": "<why these overlap or duplicate>"
    }
  ],
  "summary": "<2-3 sentence quality assessment>"
}

Verdict rules:
  CORRECT      — well-documented disease known to affect this crop.
  INCORRECT    — wrong crop; disease of a completely different plant; or entry is not a disease at all.
  QUESTIONABLE — real disease but label is vague, misspelled, names a pathogen genus instead of a
                 disease, refers to an insect pest or beneficial organism, or is an unusual
                 / unverified association with this crop.

"""

# ── helpers ───────────────────────────────────────────────────────────────────

def load_crop_disease_map(
    csv_path: Path,
    crop_col: str = "Crop",
    disease_col: str = "Disease",
) -> Dict[str, List[str]]:
    """
    Build {crop: [disease, ...]} from a flat CSV. Tolerates the
    original ``crop_disease_registry.csv`` format (per-section blank
    cells inherit the most recent non-blank value) and also works on
    fully-populated CSVs like ``BugWood_Diseases_usable.csv``.
    """
    crop_disease: Dict[str, List[str]] = {}
    current_crop = ""
    current_disease = ""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop    = (row.get(crop_col)    or "").strip()
            disease = (row.get(disease_col) or "").strip()
            if crop:    current_crop    = crop
            if disease: current_disease = disease
            if not current_crop or not current_disease:
                continue
            crop_disease.setdefault(current_crop, [])
            if current_disease not in crop_disease[current_crop]:
                crop_disease[current_crop].append(current_disease)
    return crop_disease


def call_claude(prompt: str, retries: int = 3) -> str:
    """
    Call `claude -p <prompt>` in plain-text mode.
    Returns the raw text response, or raises RuntimeError after all retries.
    """
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["claude", "--model", "claude-opus-4-7", "-p", prompt],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        if attempt < retries:
            time.sleep(4 * attempt)

    raise RuntimeError(
        f"claude CLI failed after {retries} retries. "
        f"stderr: {result.stderr.strip()[:200]}"
    )


def parse_json_from_text(text: str) -> dict:
    """Strip markdown fences and parse JSON; try partial extraction as fallback."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise

# ── Layer 1: crop validation ──────────────────────────────────────────────────

def validate_crop(crop: str) -> dict:
    prompt = CROP_VALIDATION_PROMPT + crop
    try:
        raw = call_claude(prompt)
        result = parse_json_from_text(raw)
        result.setdefault("crop", crop)
        return result
    except Exception as e:
        return {"crop": crop, "verdict": "ERROR", "error": str(e), "parse_error": True}


def _fmt_crop_validation(result: dict, idx: int, total: int) -> str:
    crop    = result.get("crop", "?")
    verdict = result.get("verdict", "ERROR")
    canon   = result.get("canonical_name", "")
    cat     = result.get("category", "")
    notes   = result.get("notes", "")
    node_a  = result.get("node_a", "?")
    node_b  = result.get("node_b", "?")
    node_c  = result.get("node_c", "?")

    lines = [
        f"\n{'─'*70}",
        f"[{idx}/{total}]  CROP: {crop}",
        f"{'─'*70}",
        f"  Node A (real plant?)     : {node_a}",
        f"  Node B (cultivated crop?): {node_b}",
        f"  Node C (name correct?)   : {node_c}",
        f"  Verdict   : {verdict}",
        f"  Category  : {cat}",
    ]
    if canon and canon != crop:
        lines.append(f"  Canonical : {canon}")
    if notes:
        lines.append(f"  Notes     : {notes}")
    return "\n".join(lines)

# ── Layer 2: disease label check ──────────────────────────────────────────────

def judge_diseases(crop: str, diseases: List[str]) -> dict:
    disease_list = "\n".join(f"- {d}" for d in diseases)
    prompt = DISEASE_JUDGE_PROMPT + f"Crop: {crop}\n\nDiseases:\n{disease_list}"
    try:
        raw = call_claude(prompt)
        result = parse_json_from_text(raw)
        result.setdefault("crop", crop)
        return result
    except Exception as e:
        return {"crop": crop, "error": str(e), "parse_error": True}


def _fmt_disease_report(result: dict, idx: int, total: int) -> str:
    crop = result.get("crop", "?")
    lines = [f"\n{'='*70}", f"[{idx}/{total}]  DISEASES — CROP: {crop}", "=" * 70]

    if result.get("parse_error"):
        lines.append(f"  [ERROR] {result.get('error') or '(empty)'}")
        return "\n".join(lines)

    verdicts     = result.get("disease_verdicts", [])
    incorrect    = [v for v in verdicts if v["verdict"] == "INCORRECT"]
    questionable = [v for v in verdicts if v["verdict"] == "QUESTIONABLE"]
    correct      = [v for v in verdicts if v["verdict"] == "CORRECT"]

    lines += [
        f"  Total: {len(verdicts)}  |  CORRECT: {len(correct)}  "
        f"QUESTIONABLE: {len(questionable)}  INCORRECT: {len(incorrect)}",
    ]
    if incorrect:
        lines.append("\n  INCORRECT:")
        for v in incorrect:
            lines.append(f"    x  {v['disease']}: {v['reason']}")
    if questionable:
        lines.append("\n  QUESTIONABLE:")
        for v in questionable:
            lines.append(f"    ?  {v['disease']}: {v['reason']}")
    for g in result.get("similar_groups", []):
        lines.append(f"\n  ~ SIMILAR: {' | '.join(g['diseases'])}")
        lines.append(f"    {g['reason']}")
    if result.get("summary"):
        lines.append(f"\n  SUMMARY: {result['summary']}")

    return "\n".join(lines)

# ── core driver (reusable from other scripts) ─────────────────────────────────

def judge_crop_disease_map(
    crop_disease_map: Dict[str, List[str]],
    output_json: Path,
    progress_path: Optional[Path] = None,
    resume: bool = True,
    sleep_sec: float = 1.0,
) -> List[dict]:
    """
    Run the two-layer judge over every crop in ``crop_disease_map`` and
    write a JSON list to ``output_json`` (incrementally, so the run
    is resumable). Returns the in-memory result list.

    Each element looks like:
        {
          "crop": "<input name>",
          "crop_validation": { ... Layer 1 ... },
          "disease_check":   { ... Layer 2 ... }  # or {"skipped": True, ...}
        }
    """
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    def _log(text: str) -> None:
        print(text)
        if progress_path is not None:
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
                f.flush()

    if progress_path is not None:
        Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
        Path(progress_path).write_text("", encoding="utf-8")

    _log("TWO-LAYER CROP DISEASE LABEL QUALITY REPORT")
    _log("=" * 70)
    _log(f"Crops to judge: {len(crop_disease_map)}, "
         f"{sum(len(v) for v in crop_disease_map.values())} unique disease labels.\n")

    # Resume from prior partial run
    saved: Dict[str, dict] = {}
    if resume and output_json.exists():
        try:
            with open(output_json, encoding="utf-8") as f:
                for r in json.load(f):
                    crop = r.get("crop", "")
                    if crop and not r.get("parse_error"):
                        saved[crop] = r
            if saved:
                _log(f"Resuming: {len(saved)} crops already completed — skipping.\n")
        except Exception:
            pass

    crops = sorted(crop_disease_map.keys())
    total = len(crops)
    all_results: List[dict] = []

    for idx, crop in enumerate(crops, start=1):
        if crop in saved:
            all_results.append(saved[crop])
            print(f"[{idx}/{total}] {crop} — skipped (saved)")
            continue

        record: dict = {"crop": crop}

        _log(f"\n{'#'*70}")
        _log(f"# [{idx}/{total}]  {crop}")
        _log(f"# LAYER 1 — Crop Validation")
        _log(f"{'#'*70}")

        crop_val = validate_crop(crop)
        record["crop_validation"] = crop_val
        _log(_fmt_crop_validation(crop_val, idx, total))

        verdict = crop_val.get("verdict", "ERROR")
        if verdict in ("INVALID", "NON_CROP", "ERROR"):
            _log(f"\n  → Skipping disease check (crop verdict: {verdict})")
            record["disease_check"] = {"skipped": True,
                                       "reason": f"Crop verdict is {verdict}"}
        else:
            _log(f"\n# LAYER 2 — Disease Label Check")
            diseases = crop_disease_map[crop]
            print(f"  Judging {len(diseases)} diseases...", end=" ", flush=True)
            disease_result = judge_diseases(crop, diseases)
            record["disease_check"] = disease_result
            print("done")
            _log(_fmt_disease_report(disease_result, idx, total))

        all_results.append(record)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        if sleep_sec:
            time.sleep(sleep_sec)

    _log(f"\n\nJSON report saved to: {output_json}")
    return all_results


# ── apply judgement back to a CSV ─────────────────────────────────────────────

DROP_CROP_VERDICTS = {"INVALID", "NON_CROP", "ERROR"}


def _build_disease_lookup(judgements: Iterable[dict]) -> Dict[str, Dict[str, str]]:
    """{crop -> {disease -> verdict}}"""
    out: Dict[str, Dict[str, str]] = {}
    for rec in judgements:
        crop = rec.get("crop", "")
        dc = rec.get("disease_check", {}) or {}
        if dc.get("skipped"):
            continue
        out[crop] = {
            (v.get("disease") or "").strip(): v.get("verdict", "CORRECT")
            for v in dc.get("disease_verdicts", []) or []
        }
    return out


def _build_crop_lookup(judgements: Iterable[dict]) -> Dict[str, dict]:
    return {rec["crop"]: rec.get("crop_validation", {}) for rec in judgements
            if "crop" in rec}


def apply_judgement_to_rows(
    rows: List[dict],
    judgements: List[dict],
    crop_col: str,
    disease_col: str,
    drop_questionable: bool = False,
    canonicalize_misspelled: bool = True,
) -> Dict[str, object]:
    """
    Filter and (optionally) rename rows in-place using the judgement
    report. Returns a stats dict.
    """
    crop_lookup    = _build_crop_lookup(judgements)
    disease_lookup = _build_disease_lookup(judgements)

    kept: List[dict] = []
    dropped_invalid_crop = 0
    dropped_incorrect = 0
    dropped_questionable = 0
    renamed_crops = 0

    for row in rows:
        crop    = (row.get(crop_col)    or "").strip()
        disease = (row.get(disease_col) or "").strip()
        cv = crop_lookup.get(crop, {})
        verdict = cv.get("verdict", "VALID")

        if verdict in DROP_CROP_VERDICTS:
            dropped_invalid_crop += 1
            continue

        if (verdict == "MISSPELLED_OR_UNSTANDARDISED"
                and canonicalize_misspelled):
            canon = (cv.get("canonical_name") or "").strip()
            if canon and canon != crop:
                row[crop_col] = canon
                renamed_crops += 1
                crop = canon

        dverdict = (disease_lookup.get(crop, {}).get(disease)
                    or disease_lookup.get(row.get(crop_col, ""), {}).get(disease)
                    or "CORRECT")
        if dverdict == "INCORRECT":
            dropped_incorrect += 1
            continue
        if dverdict == "QUESTIONABLE" and drop_questionable:
            dropped_questionable += 1
            continue

        kept.append(row)

    return {
        "rows_in":              len(rows),
        "rows_out":             len(kept),
        "dropped_invalid_crop": dropped_invalid_crop,
        "dropped_incorrect":    dropped_incorrect,
        "dropped_questionable": dropped_questionable,
        "renamed_crops":        renamed_crops,
        "kept":                 kept,
    }


def write_filtered_csv(rows: List[dict], fieldnames: List[str],
                       output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-layer Claude LLM judge for crop / disease labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",         default=str(_DEFAULT_CSV))
    p.add_argument("--crop-col",    default="Crop")
    p.add_argument("--disease-col", default="Disease")
    p.add_argument("--output",      default=str(_DEFAULT_OUTPUT),
                   help="path to write the JSON judgement report")
    p.add_argument("--progress",    default=str(_DEFAULT_PROGRESS),
                   help="path to stream live progress text")
    p.add_argument("--no-resume",   action="store_true",
                   help="ignore any existing --output JSON; start fresh")
    p.add_argument("--apply",       action="store_true",
                   help="after judging, rewrite the CSV without "
                        "INVALID / NON_CROP crops and INCORRECT diseases "
                        "(canonicalises MISSPELLED crop names too)")
    p.add_argument("--apply-out",   default=None,
                   help="output CSV path for --apply "
                        "(default: <csv>.judged.csv)")
    p.add_argument("--drop-questionable", action="store_true",
                   help="also drop QUESTIONABLE diseases under --apply")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"input CSV not found: {csv_path}")

    crop_disease_map = load_crop_disease_map(
        csv_path, crop_col=args.crop_col, disease_col=args.disease_col,
    )

    judgements = judge_crop_disease_map(
        crop_disease_map,
        output_json=Path(args.output),
        progress_path=Path(args.progress) if args.progress else None,
        resume=not args.no_resume,
    )

    # ── aggregate statistics ──
    total_crops = len(judgements)
    invalid_crops = sum(
        1 for r in judgements
        if r.get("crop_validation", {}).get("verdict") in ("INVALID", "NON_CROP", "ERROR")
    )
    misnamed_crops = sum(
        1 for r in judgements
        if r.get("crop_validation", {}).get("verdict") == "MISSPELLED_OR_UNSTANDARDISED"
    )
    total_d = total_inc = total_q = total_sim = 0
    for r in judgements:
        dc = r.get("disease_check", {})
        if dc.get("skipped"):
            continue
        vs = dc.get("disease_verdicts", [])
        total_d   += len(vs)
        total_inc += sum(1 for v in vs if v["verdict"] == "INCORRECT")
        total_q   += sum(1 for v in vs if v["verdict"] == "QUESTIONABLE")
        total_sim += len(dc.get("similar_groups", []))

    print("\n" + "=" * 70)
    print("OVERALL DATASET QUALITY SUMMARY")
    print("=" * 70)
    print(f"  LAYER 1 — Crop Labels")
    print(f"    Total crops        : {total_crops}")
    print(f"    Invalid / Non-crop : {invalid_crops}")
    print(f"    Misspelled         : {misnamed_crops}")
    print(f"    Valid              : {total_crops - invalid_crops - misnamed_crops}")
    print(f"\n  LAYER 2 — Disease Labels (valid crops only)")
    print(f"    Total diseases     : {total_d}")
    print(f"    Incorrect          : {total_inc}  "
          f"({100*total_inc/max(total_d,1):.1f}%)")
    print(f"    Questionable       : {total_q}  "
          f"({100*total_q/max(total_d,1):.1f}%)")
    print(f"    Similar groups     : {total_sim}")

    # ── optional: apply judgement back to the CSV ──
    if args.apply:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        stats = apply_judgement_to_rows(
            rows, judgements,
            crop_col=args.crop_col, disease_col=args.disease_col,
            drop_questionable=args.drop_questionable,
        )

        apply_out = Path(args.apply_out
                         or csv_path.with_suffix(".judged.csv"))
        write_filtered_csv(stats["kept"], fieldnames, apply_out)

        print(f"\n--apply: wrote {apply_out}")
        print(f"  rows in            : {stats['rows_in']}")
        print(f"  rows out           : {stats['rows_out']}")
        print(f"  dropped (bad crop) : {stats['dropped_invalid_crop']}")
        print(f"  dropped (INCORRECT): {stats['dropped_incorrect']}")
        if args.drop_questionable:
            print(f"  dropped (QUEST.)   : {stats['dropped_questionable']}")
        print(f"  crop renames       : {stats['renamed_crops']}")


if __name__ == "__main__":
    main()
