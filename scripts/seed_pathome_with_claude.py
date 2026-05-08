"""
scripts/seed_pathome_with_claude.py
====================================
Phase 0 of the Pathome pipeline: seed the VisualSymptom block on every
SymptomProfile by shelling out to ``claude -p`` (Claude Code headless).

Output is the JSON file expected by ``PathomeDB.build_from_bugwood``'s
``symptoms_path`` argument:

    artifacts/pathome_seed/symptoms_seed.json
        {
          "min_observations": 3,
          "profiles": [
            { "profile_id": "Cucumber::Cucurbit Downy Mildew",
              "crop": "Cucumber",
              "disease": "Cucurbit Downy Mildew",
              "visual": { "plant_parts": ["leaf"], "color": [...], ... },
              "state_counts": {}, "aez_counts": {}, "total_observations": 0,
              "reference_ids": [], "reobservation_prompt": "" },
            ...
          ]
        }

State-count / reference-id fields are intentionally left empty — they
are auto-derived later by ``SymptomLibrary.update_from_records`` when
the seed file is layered against the Bugwood trace+reference records in
``build_pathome.py``.

Usage:
    # smoke test on first 3 profiles
    python scripts/seed_pathome_with_claude.py --limit 3

    # full run with 4 parallel workers, sonnet 4.6, resumable
    python scripts/seed_pathome_with_claude.py --workers 4 --model sonnet

Authentication:
    Uses your local ``claude`` CLI login. Run ``claude login`` first.

Each call is independent. Failures are logged to a sidecar ``failed.json``
and skipped on retry — re-run the script to pick them up.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from data.bugwood_loader import BugwoodLoader  # noqa: E402
from pathome.symptoms import SymptomLibrary, SymptomProfile, VisualSymptom  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SCHEMA_KEYS = [
    "plant_parts", "color", "shape", "margin", "texture",
    "sporulation", "distinctive_signs", "progression",
    "confusion_diseases", "notes",
]

PROMPT_TEMPLATE = """You are a plant pathology reference. For one (crop, disease) pair, return a single JSON object describing the visual symptoms a field diagnostician would look for. No prose, no markdown fences, no commentary — JSON only.

Crop: {crop}
Disease: {disease}

Schema (use these exact keys; values are arrays of short strings unless noted):

{{
  "plant_parts":       ["leaf" | "stem" | "fruit" | "root" | "flower" | ...],
  "color":             ["brown", "yellow halo", "black", ...],
  "shape":             "circular" | "angular" | "irregular" | "elongated" | "" ,
  "margin":            "diffuse" | "sharp" | "halo" | "" ,
  "texture":           ["sunken", "raised", "powdery", "downy", ...],
  "sporulation":       ["orange masses", "white powder", "salmon spores", ...],
  "distinctive_signs": ["concentric rings", "vein clearing", "angular margins", ...],
  "progression":       "expanding" | "systemic" | "static" | "" ,
  "confusion_diseases":["disease_name_a", "disease_name_b", ...],
  "notes":             "one or two sentences of practical context"
}}

Rules:
- If the disease is rare, ambiguous, or you are uncertain, return empty arrays / empty strings rather than guessing.
- Confusion diseases should be the diseases a field diagnostician most often misroutes this disease to (or from).
- Output the JSON object and nothing else.
"""


# ---------------------------------------------------------------------------
# Claude headless invocation
# ---------------------------------------------------------------------------

DISALLOWED_TOOLS = "Bash,Edit,Write,Read,Grep,Glob,WebFetch,WebSearch"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def call_claude(prompt: str, model: Optional[str], timeout: float) -> Tuple[Optional[dict], str]:
    """Run one ``claude -p`` invocation. Returns (parsed_json, raw_text)."""
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--disallowedTools", DISALLOWED_TOOLS,
    ]
    if model:
        cmd.extend(["--model", model])
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "<timeout>"
    if proc.returncode != 0:
        return None, f"<rc={proc.returncode}> {proc.stderr.strip()[:400]}"

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, f"<bad envelope> {proc.stdout[:200]}"
    text = envelope.get("result", "")
    if not isinstance(text, str):
        return None, "<missing result>"
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned), text
    except json.JSONDecodeError:
        # Try to salvage the first {...} block.
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                return json.loads(m.group(0)), text
            except json.JSONDecodeError:
                pass
        return None, text


def coerce_visual(raw: dict) -> VisualSymptom:
    """Validate Claude's JSON against the VisualSymptom schema, dropping junk."""
    def _strs(v) -> List[str]:
        if not isinstance(v, list):
            return []
        out = []
        for x in v:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out

    def _str(v) -> str:
        return v.strip() if isinstance(v, str) else ""

    return VisualSymptom(
        plant_parts=_strs(raw.get("plant_parts")),
        color=_strs(raw.get("color")),
        shape=_str(raw.get("shape")),
        margin=_str(raw.get("margin")),
        texture=_strs(raw.get("texture")),
        sporulation=_strs(raw.get("sporulation")),
        distinctive_signs=_strs(raw.get("distinctive_signs")),
        progression=_str(raw.get("progression")),
        confusion_diseases=_strs(raw.get("confusion_diseases")),
        notes=_str(raw.get("notes")),
    )


# ---------------------------------------------------------------------------
# Discovery + persistence
# ---------------------------------------------------------------------------

def discover_classes(config_path: str) -> List[Tuple[str, str]]:
    """Return distinct (crop, disease) pairs from the loader's "all" split."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg_data = dict(cfg["data"])
    cfg_data["download"] = {"enabled": False}    # discovery only — no images
    loader = BugwoodLoader(cfg_data, split="all")
    seen: List[Tuple[str, str]] = []
    seen_set = set()
    for rec in loader:
        key = (rec.crop_species, rec.disease_name)
        if key in seen_set:
            continue
        seen_set.add(key)
        seen.append(key)
    return seen


def load_seed(path: Path) -> SymptomLibrary:
    if path.exists():
        return SymptomLibrary.load(str(path))
    return SymptomLibrary()


def save_seed(lib: SymptomLibrary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    lib.save(str(tmp))
    tmp.replace(path)


def append_failure(failed_path: Path, profile_id: str, reason: str) -> None:
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"profile_id": profile_id, "reason": reason, "ts": time.time()}
    with open(failed_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def seed_one(
    crop: str, disease: str, model: Optional[str], timeout: float,
) -> Tuple[str, Optional[VisualSymptom], str]:
    pid = SymptomProfile.make_id(crop, disease)
    prompt = PROMPT_TEMPLATE.format(crop=crop, disease=disease)
    parsed, raw = call_claude(prompt, model=model, timeout=timeout)
    if parsed is None:
        return pid, None, raw
    if not isinstance(parsed, dict):
        return pid, None, f"<not a dict> {raw[:200]}"
    return pid, coerce_visual(parsed), ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--config", default="configs/bugwood_pathome.yaml")
    p.add_argument("--out", default="artifacts/pathome_seed/symptoms_seed.json")
    p.add_argument("--failed", default="artifacts/pathome_seed/failed.jsonl")
    p.add_argument("--model", default="sonnet",
                   help="Claude model alias (sonnet | opus | haiku) or full ID")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel `claude -p` processes")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="per-call timeout in seconds")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after seeding this many new profiles (0 = all)")
    p.add_argument("--retry-failed", action="store_true",
                   help="reprocess profile_ids in failed.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    failed_path = Path(args.failed)

    classes = discover_classes(args.config)
    print(f"discovered {len(classes)} (crop, disease) pairs")

    lib = load_seed(out_path)
    already = {p.profile_id for p in lib if not p.visual.is_empty()}
    print(f"already seeded: {len(already)}")

    if args.retry_failed and failed_path.exists():
        failed_ids = {json.loads(line)["profile_id"]
                      for line in failed_path.read_text().splitlines() if line.strip()}
        already -= failed_ids
        print(f"will retry {len(failed_ids)} failed profile_ids")

    todo = [
        (crop, disease) for (crop, disease) in classes
        if SymptomProfile.make_id(crop, disease) not in already
    ]
    if args.limit > 0:
        todo = todo[: args.limit]
    print(f"to seed this run: {len(todo)}")

    if not todo:
        print("nothing to do.")
        return

    n_ok = n_fail = 0
    save_every = max(5, len(todo) // 50)   # ~50 checkpoints
    t0 = time.time()

    def _record(pid: str, visual: Optional[VisualSymptom], reason: str) -> None:
        nonlocal n_ok, n_fail
        if visual is None:
            n_fail += 1
            append_failure(failed_path, pid, reason)
            return
        crop, disease = pid.split("::", 1)
        prof = lib.get_or_create(crop, disease)
        prof.visual = visual
        prof.reobservation_prompt = visual.auto_reobservation_prompt()
        n_ok += 1

    if args.workers <= 1:
        for i, (crop, disease) in enumerate(todo, 1):
            pid, visual, reason = seed_one(crop, disease, args.model, args.timeout)
            _record(pid, visual, reason)
            if i % save_every == 0 or i == len(todo):
                save_seed(lib, out_path)
                rate = i / max(time.time() - t0, 1e-3)
                print(f"  [{i}/{len(todo)}] ok={n_ok} fail={n_fail} "
                      f"rate={rate:.2f}/s")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(seed_one, crop, disease, args.model, args.timeout): (crop, disease)
                for crop, disease in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                pid, visual, reason = fut.result()
                _record(pid, visual, reason)
                if i % save_every == 0 or i == len(todo):
                    save_seed(lib, out_path)
                    rate = i / max(time.time() - t0, 1e-3)
                    print(f"  [{i}/{len(todo)}] ok={n_ok} fail={n_fail} "
                          f"rate={rate:.2f}/s")

    save_seed(lib, out_path)
    elapsed = time.time() - t0
    print(f"\nseeded {n_ok} new visual blocks ({n_fail} failures) in {elapsed:.0f}s")
    print(f"output: {out_path}")
    if n_fail:
        print(f"failures logged to: {failed_path}")
        print(f"retry with: python {sys.argv[0]} --retry-failed")


if __name__ == "__main__":
    main()
