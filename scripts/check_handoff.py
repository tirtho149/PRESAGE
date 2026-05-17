"""
scripts/check_handoff.py
========================
Step-handoff guard for the 6-step LOCAL/NOVA pipeline.

Each step pulls the previous step's artifact, consumes it, and pushes its
own. Nothing checked that the *input* of step N actually exists / is
well-formed (e.g. Step 2 sbatching the swarm against a canonical KB that
Step 1 never pushed), nor that the *output* is sane before `git push`.
This module is that contract — a tiny stdlib-only checker, importable +
CLI, that fails fast with a one-line actionable message.

Modes
-----
    canonical-kb       --kb-root --crops   each crop has final_registry.json
                                           with diseases[] and ≥1 populated
                                           visual_symptoms.summary
                                           (Step 1 post / Step 2 pre)
    unverified-deltas  --kb-root --crops   ≥1 delta still unverified
                                           (Step 3 pre)
    verified-kb        --kb-root --crops   ZERO deltas still unverified
                                           (Step 3 post / Step 4-5 pre)
    checkpoint         --path              file exists, size above a floor
                                           (Step 5 pre, PATHOMEOOD_CKPT)

Exit code 0 = ok, 4 = handoff precondition/postcondition failed,
2 = bad invocation.

Bypass: set SKIP_HANDOFF_CHECK=1 (callers gate on this) — this script
itself does not read that env; the shell wrappers decide whether to call
it at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

FAIL = 4
BADCALL = 2

# A truncated / empty checkpoint is the realistic failure; real ones are
# ~600 MB. 1 MB floor catches the empties without being brittle.
_CKPT_MIN_BYTES = 1_000_000


def _resolve_crops(raw: str) -> List[str]:
    """Same semantics as scripts/validate_kb.py: smoke -> Soybean,Tomato;
    all / empty -> [] (no filter); otherwise comma list."""
    raw = (raw or "").strip()
    if not raw or raw == "all":
        return []
    if raw == "smoke":
        return ["Soybean", "Tomato"]
    return [c.strip() for c in raw.split(",") if c.strip()]


def _registries(kb_root: Path, crops: List[str]) -> List[Path]:
    out: List[Path] = []
    if not kb_root.is_dir():
        return out
    for crop_dir in sorted(kb_root.iterdir()):
        if not crop_dir.is_dir():
            continue
        if crops and crop_dir.name not in crops:
            continue
        reg = crop_dir / "final_registry.json"
        if reg.is_file():
            out.append(reg)
    return out


def _load(reg: Path) -> Dict[str, Any]:
    try:
        return json.loads(reg.read_text())
    except Exception as e:  # noqa: BLE001 - any unreadable registry is a fail
        raise ValueError(f"{reg}: {type(e).__name__}: {e}") from e


def _summary_value(disease_record: Dict[str, Any]) -> str:
    """visual_symptoms.summary may be a bare string or a {"value": ...}
    cited field (same shape validate_kb._flatten_canonical unwraps)."""
    vs = disease_record.get("visual_symptoms") or {}
    s = vs.get("summary")
    if isinstance(s, dict):
        s = s.get("value")
    return str(s or "").strip()


def _iter_deltas(data: Dict[str, Any]):
    for dr in data.get("diseases") or []:
        ro = dr.get("regional_observations") or {}
        if not isinstance(ro, dict):
            continue
        for state_block in ro.values():
            if not isinstance(state_block, dict):
                continue
            for d in state_block.get("deltas") or []:
                if isinstance(d, dict):
                    yield d


def _is_unverified(delta: Dict[str, Any]) -> bool:
    return (delta.get("verification_status") or "").lower() in ("", "unverified")


def _missing_crops(kb_root: Path, crops: List[str]) -> List[str]:
    if not crops:
        return []
    return [c for c in crops
            if not (kb_root / c / "final_registry.json").is_file()]


def check_canonical_kb(kb_root: Path, crops: List[str]) -> Tuple[bool, str]:
    miss = _missing_crops(kb_root, crops)
    if miss:
        return False, (f"missing final_registry.json for {miss} under "
                       f"{kb_root} — run Step 1 (sh_01_phase0_local.sh) "
                       f"and git push before this step")
    regs = _registries(kb_root, crops)
    if not regs:
        return False, (f"no final_registry.json under {kb_root} — Step 1 "
                       f"(sh_01_phase0_local.sh) has not run / not pulled")
    for reg in regs:
        data = _load(reg)
        diseases = data.get("diseases") or []
        if not diseases:
            return False, (f"{reg} has empty diseases[] — Step 1 produced no "
                            f"canonical KB for {reg.parent.name}")
        if not any(_summary_value(dr) for dr in diseases):
            return False, (f"{reg}: no disease has a populated "
                            f"visual_symptoms.summary — canonical KB is a "
                            f"stub; re-run Step 1 for {reg.parent.name}")
    return True, f"canonical KB ok ({len(regs)} registr{'y' if len(regs)==1 else 'ies'})"


def check_unverified_deltas(kb_root: Path, crops: List[str]) -> Tuple[bool, str]:
    regs = _registries(kb_root, crops)
    if not regs:
        return False, (f"no final_registry.json under {kb_root} — Step 2 "
                       f"(swarm) output not present / not pulled")
    total = 0
    for reg in regs:
        total += sum(1 for d in _iter_deltas(_load(reg)) if _is_unverified(d))
    if total == 0:
        return False, ("no unverified deltas found — Step 2 (Phase 0R swarm) "
                       "produced nothing to verify, or this KB is already "
                       "verified (nothing for Step 3 to do)")
    return True, f"{total} unverified delta(s) ready for Step 3"


def check_verified_kb(kb_root: Path, crops: List[str]) -> Tuple[bool, str]:
    regs = _registries(kb_root, crops)
    if not regs:
        return False, (f"no final_registry.json under {kb_root} — verified "
                       f"KB not present / not pulled")
    leftover = 0
    for reg in regs:
        leftover += sum(1 for d in _iter_deltas(_load(reg)) if _is_unverified(d))
    if leftover > 0:
        return False, (f"{leftover} delta(s) still 'unverified' — Step 3 "
                       f"(sh_03_validate_local.sh) did not finish "
                       f"successfully; do not train/eval on a half-verified KB")
    return True, f"verified KB ok ({len(regs)} registr{'y' if len(regs)==1 else 'ies'}, 0 unverified)"


def check_checkpoint(path: Path) -> Tuple[bool, str]:
    if not path.is_file():
        return False, (f"checkpoint not found: {path} — run Step 4 "
                       f"(sh_04_train_encoder_nova.sh) or set PATHOMEOOD_CKPT")
    size = path.stat().st_size
    if size < _CKPT_MIN_BYTES:
        return False, (f"checkpoint {path} is only {size} bytes "
                       f"(< {_CKPT_MIN_BYTES}) — likely truncated; re-run "
                       f"Step 4 or re-scp the checkpoint")
    return True, f"checkpoint ok ({path}, {size} bytes)"


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="mode", required=True)

    for name in ("canonical-kb", "unverified-deltas", "verified-kb"):
        sp = sub.add_parser(name)
        sp.add_argument("--kb-root", default="artifacts/pathome_kb")
        sp.add_argument("--crops", default=os.environ.get("CROPS", ""))

    cp = sub.add_parser("checkpoint")
    cp.add_argument("--path", required=True)

    args = p.parse_args(argv)

    try:
        if args.mode == "checkpoint":
            ok, msg = check_checkpoint(Path(args.path))
        else:
            kb_root = Path(args.kb_root)
            crops = _resolve_crops(args.crops)
            fn = {
                "canonical-kb":      check_canonical_kb,
                "unverified-deltas": check_unverified_deltas,
                "verified-kb":       check_verified_kb,
            }[args.mode]
            ok, msg = fn(kb_root, crops)
    except ValueError as e:
        print(f"[handoff:{args.mode}] FAIL — unreadable registry: {e}")
        return FAIL

    tag = "OK" if ok else "FAIL"
    print(f"[handoff:{args.mode}] {tag} — {msg}")
    return 0 if ok else FAIL


if __name__ == "__main__":
    sys.exit(main())
