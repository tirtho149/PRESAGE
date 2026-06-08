"""
scripts/backfill_references.py
==============================
Backfill empty ``web_support`` on regional deltas using URLs that
``scripts/validate_kb.py`` (step 3) already attached to sibling deltas.

Step 3 runs the Claude+WebSearch verifier per (crop, disease, state)
tuple. Whatever URLs it finds get written onto the deltas it could
actually support (verified / weakly_supported). The remaining deltas
in the same tuple — provisional / novel_plausible / unverified — get
no citations even though the verifier's web hits are equally relevant
to them. This script copies those URLs over.

Pooling scope, in priority order:
    1. (disease, state)  — sibling deltas verified in the same tuple
    2. (disease)         — same disease, other states
A delta is left as-is if neither pool yields any URL.

Each inherited entry is tagged ``"inherited": true`` and carries a
``"inherited_from": "<disease>::<state>"`` field so the provenance of
borrowed citations stays distinct from URLs the verifier attached
directly. ``verification_status`` is NOT changed — borrowing a
general-disease reference does not constitute new evidence for that
specific observation.

Dry-run by default; pass --write to persist.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _iter_deltas(registry: Dict[str, Any]):
    for disease in registry.get("diseases") or []:
        d_name = (disease.get("disease_name") or "").strip()
        ro = disease.get("regional_observations") or {}
        for state, block in ro.items():
            if not isinstance(block, dict):
                continue
            for delta in block.get("deltas") or []:
                if isinstance(delta, dict):
                    yield d_name, state, delta


def _build_pools(
    registry: Dict[str, Any],
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, str]]],
           Dict[str, List[Dict[str, str]]]]:
    """Return ((disease,state) -> urls, disease -> urls). Dedup by URL."""
    by_ds: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = defaultdict(dict)
    by_d:  Dict[str, Dict[str, Dict[str, str]]]             = defaultdict(dict)
    for disease, state, delta in _iter_deltas(registry):
        for item in delta.get("web_support") or []:
            if not isinstance(item, dict):
                continue
            if item.get("inherited"):
                continue
            url = (item.get("url") or "").strip()
            if not url:
                continue
            quote = (item.get("quote") or "").strip()
            entry = {"url": url, "quote": quote}
            by_ds[(disease, state)].setdefault(url, entry)
            by_d[disease].setdefault(url, entry)
    pooled_ds = {k: list(v.values()) for k, v in by_ds.items()}
    pooled_d  = {k: list(v.values()) for k, v in by_d.items()}
    return pooled_ds, pooled_d


def _backfill(registry: Dict[str, Any]) -> Dict[str, int]:
    pooled_ds, pooled_d = _build_pools(registry)
    stats = {
        "deltas_total":          0,
        "deltas_with_refs":      0,
        "filled_from_state":     0,
        "filled_from_disease":   0,
        "left_empty":            0,
        "refs_added":            0,
    }
    for disease, state, delta in _iter_deltas(registry):
        stats["deltas_total"] += 1
        web = delta.get("web_support") or []
        if web:
            stats["deltas_with_refs"] += 1
            continue
        pool = pooled_ds.get((disease, state)) or []
        source_key = f"{disease}::{state}"
        if not pool:
            pool = pooled_d.get(disease) or []
            source_key = f"{disease}::*"
            if pool:
                stats["filled_from_disease"] += 1
        else:
            stats["filled_from_state"] += 1
        if not pool:
            stats["left_empty"] += 1
            continue
        inherited = [
            {**item, "inherited": True, "inherited_from": source_key}
            for item in pool
        ]
        delta["web_support"] = inherited
        stats["refs_added"] += len(inherited)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--kb-root", default="artifacts/pathome_kb",
                        help="root containing <Crop>/final_registry.json")
    parser.add_argument("--crops", default="",
                        help="comma-separated crop allowlist (default: all)")
    parser.add_argument("--write", action="store_true",
                        help="persist changes (default: dry-run)")
    args = parser.parse_args()

    kb_root = Path(args.kb_root)
    if not kb_root.is_dir():
        raise SystemExit(f"KB root not found: {kb_root}")

    crop_filter = {c.strip() for c in args.crops.split(",") if c.strip()}

    registries: List[Path] = []
    for crop_dir in sorted(kb_root.iterdir()):
        if not crop_dir.is_dir():
            continue
        if crop_filter and crop_dir.name not in crop_filter:
            continue
        reg = crop_dir / "final_registry.json"
        if reg.is_file():
            registries.append(reg)
    if not registries:
        raise SystemExit(f"no final_registry.json under {kb_root}")

    print(f"=== backfill_references ({'WRITE' if args.write else 'DRY-RUN'}) ===")
    print(f"  kb_root    : {kb_root}")
    print(f"  registries : {len(registries)}")
    print()

    grand = {k: 0 for k in (
        "deltas_total", "deltas_with_refs",
        "filled_from_state", "filled_from_disease",
        "left_empty", "refs_added",
    )}
    for reg in registries:
        data = json.loads(reg.read_text(encoding="utf-8"))
        stats = _backfill(data)
        for k, v in stats.items():
            grand[k] += v
        print(
            f"  {reg.parent.name:25} "
            f"total={stats['deltas_total']:4} "
            f"had_refs={stats['deltas_with_refs']:4} "
            f"filled_state={stats['filled_from_state']:4} "
            f"filled_disease={stats['filled_from_disease']:4} "
            f"still_empty={stats['left_empty']:4} "
            f"refs_added={stats['refs_added']:4}"
        )
        if args.write and stats["refs_added"]:
            reg.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print()
    print(f"  TOTAL "
          f"deltas={grand['deltas_total']}  had_refs={grand['deltas_with_refs']}  "
          f"filled_state={grand['filled_from_state']}  "
          f"filled_disease={grand['filled_from_disease']}  "
          f"still_empty={grand['left_empty']}  "
          f"refs_added={grand['refs_added']}")
    if not args.write:
        print()
        print("  (dry-run — pass --write to persist)")


if __name__ == "__main__":
    main()
