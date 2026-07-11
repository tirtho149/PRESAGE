#!/usr/bin/env python
"""
reset_uncited.py — mark every regional delta that lacks web_support
citations as verification_status="unverified" so validate_kb.py will
re-verify it. Deltas that already carry web_support (your prior real
verification) are left untouched.

Usage:
    python scripts/reset_uncited.py [--kb-root artifacts/pathome_kb] [--crops A,B]
"""
import argparse, glob, json, os


def deltas_of(reg):
    for dd in reg.get("diseases", []):
        ro = dd.get("regional_observations") or {}
        if isinstance(ro, dict):
            for _st, obs in ro.items():
                dl = obs.get("deltas") if isinstance(obs, dict) else (obs if isinstance(obs, list) else [])
                for x in (dl or []):
                    if isinstance(x, dict):
                        yield x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-root", default="artifacts/pathome_kb")
    ap.add_argument("--crops", default="")
    args = ap.parse_args()
    allow = {c.strip() for c in args.crops.split(",") if c.strip()}

    total_reset = 0
    files = sorted(glob.glob(f"{args.kb_root}/*/final_registry.json"))
    for f in files:
        crop = os.path.basename(os.path.dirname(f))
        if allow and crop not in allow:
            continue
        reg = json.load(open(f))
        n = 0
        for x in deltas_of(reg):
            ws = x.get("web_support")
            cited = isinstance(ws, list) and len(ws) > 0
            if not cited:
                x["verification_status"] = "unverified"
                x["reasoning"] = ""
                n += 1
        if n:
            json.dump(reg, open(f, "w"), indent=2)
            total_reset += n
            print(f"  {crop}: reset {n} uncited deltas -> unverified")
    print(f"TOTAL reset: {total_reset} deltas across {len(files)} registries")


if __name__ == "__main__":
    main()
