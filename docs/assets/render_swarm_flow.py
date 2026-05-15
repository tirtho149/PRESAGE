"""
docs/assets/render_swarm_flow.py
================================
Render an animated GIF that walks a viewer through the current Phase 0R
**organ-routed decision tree** (DR.Arti-style):

    Act 1 — Setup
            Static context: canonical visual_symptoms slice + ONE
            Bugwood field photograph (one photo = one organ).

    Act 2 — Organ detection (the routing root)
            OrganDetectionAgent makes ONE visual call and classifies
            the dominant organ: leaf / stem / root / crown / flower /
            fruit / whole_plant / other. Pure visual triage.

    Act 3 — Routing
            route_for_organ() activates ONLY the detected organ's DEEP
            single-feature specialists + always-on cross-cutters
            (ColorPalette / Severity / LookAlikeCoT / Sporulation).
            Every other branch stays dormant.

    Act 4 — Deep dive (round 1 + round 2 blackboard)
            The activated branch fans out in parallel, writes deltas
            to the running log, then a blackboard round-2 refines
            (support / challenge / withdraw) among the active agents.

    Act 5 — Consolidate -> K-of-N -> verifier -> merge -> KB
            VisualDiagnosisAgent consolidates; cross-pass agreement,
            Claude web verifier, conservative merge into
            final_registry.json.

The animation is intentionally SLOW (2 fps / 500 ms per frame) so each
step is readable.

Output: docs/assets/swarm_flow.gif
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

OUT = Path(__file__).resolve().parent / "swarm_flow.gif"


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

NODE_STATIC = dict(
    pos=(0.31, 0.915, 0.56, 0.062),
    color="#6366f1", fill="#e0e7ff",
    title="STATIC CONTEXT",
    sub="canonical visual_symptoms slice + ONE field photo (one photo = one organ)",
)
NODE_DETECT = dict(
    pos=(0.31, 0.81, 0.40, 0.07),
    color="#b45309", fill="#fef3c7",
    title="OrganDetectionAgent  (1 visual call)",
    sub="which single organ dominates the frame?",
)
NODE_CONSOL = dict(
    pos=(0.31, 0.155, 0.56, 0.075),
    color="#0d9488", fill="#ccfbf1",
    title="VisualDiagnosisAgent — CoT consolidator",
    sub="dedup + decisive forks → K-of-N → web verifier → merge",
)
NODE_OUTPUT = dict(
    pos=(0.31, 0.05, 0.56, 0.05),
    color="#16a34a", fill="#dcfce7",
    title="final_registry.json → regional_observations[<state>].deltas[]",
    sub="",
)

# Eight organ branches. Each maps to the deep specialists it activates.
ORGAN_ORDER = ["leaf", "stem", "root", "crown", "flower", "fruit",
               "whole_plant", "other"]

_CROSS = ["ColorPalette", "Severity", "LookAlikeCoT", "Sporulation"]

BRANCHES: Dict[str, Dict] = {
    "leaf": dict(
        color="#16a34a", fill="#dcfce7", n=13,
        deep=["LeafLesionShape", "LeafLesionColor", "LeafLesionTexture",
              "LeafChlorosis", "LeafNecrosis", "LeafCurl",
              "LeafVeinPattern", "LeafGeometry", "ConcentricPattern"]
             + _CROSS,
    ),
    "stem": dict(color="#dc2626", fill="#fee2e2", n=8,
                 deep=["StemLesion", "StemPith", "StemSurface",
                       "StemDiscoloration"] + _CROSS),
    "root": dict(color="#a16207", fill="#fef3c7", n=5,
                 deep=["Root"] + _CROSS),
    "crown": dict(color="#a16207", fill="#fef3c7", n=5,
                  deep=["CrownCollar"] + _CROSS),
    "flower": dict(color="#7c3aed", fill="#ede9fe", n=5,
                   deep=["Flower"] + _CROSS),
    "fruit": dict(color="#7c3aed", fill="#ede9fe", n=6,
                  deep=["Fruit", "ConcentricPattern"] + _CROSS),
    "whole_plant": dict(color="#2563eb", fill="#dbeafe", n=7,
                        deep=["Wilting", "Defoliation", "SpatialPattern"]
                             + _CROSS),
    "other": dict(color="#64748b", fill="#f1f5f9", n=24,
                  deep=["(all 24 — safe fallback)"]),
}

# The organ this walkthrough demonstrates.
DETECTED = "leaf"

# Branch chip geometry: a row of 8 chips.
_CHIP_Y = 0.645
_CHIP_W = 0.105
_CHIP_H = 0.058
_CHIP_X0 = 0.045
_CHIP_DX = 0.118


def _chip_pos(idx: int) -> Tuple[float, float, float, float]:
    return (_CHIP_X0 + idx * _CHIP_DX, _CHIP_Y, _CHIP_W, _CHIP_H)


# Deltas the leaf branch produces (right-side running log).
DELTAS: List[Tuple[str, str]] = [
    ("leaf_chlorosis",     "interveinal yellowing; veins stay green"),
    ("leaf_necrosis",      "interveinal flecks coalescing outward"),
    ("leaf_vein_pattern",  "chlorotic banding aligned along major veins"),
    ("color_palette",      "lemon-yellow grading to tan-brown necrosis"),
    ("look_alikes_visual", "foliar pattern visually identical to SDS; "
                           "decisive split-stem not in this frame"),
    ("leaf_curl",          "leaflet tips cupping upward before necrosis"),
    ("severity_visible",   "medium — ~25% of visible blade affected"),
]
DEDUP_DROP = {5}        # leaf_curl folded into chlorosis at consolidation
KOFN_DROP  = {6}        # severity below K-of-N support across seeds


# ----------------------------------------------------------------------
# Frame budget — SLOW (2 fps, 500 ms/frame)
# ----------------------------------------------------------------------

FPS = 2
FRAME_MS = int(1000 / FPS)          # 500 ms per frame

ACT1 = 4    # setup
ACT2 = 6    # organ detection
ACT3 = 5    # routing
ACT4 = 10   # deep dive (round 1 + round 2)
ACT5 = 8    # consolidate + K-of-N + merge
HOLD = 6    # end hold
TOTAL = ACT1 + ACT2 + ACT3 + ACT4 + ACT5 + HOLD


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def _setup_axes():
    fig, ax = plt.subplots(figsize=(13.5, 7.4), dpi=110)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _box(ax, x, y, w, h, *, color, fill, lw=1.4, alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=lw, edgecolor=color, facecolor=fill, alpha=alpha,
        zorder=z))


def _glow(ax, x, y, w, h, *, color, alpha=0.30):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2 - 0.013, y - h / 2 - 0.013), w + 0.026, h + 0.026,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=0, facecolor=color, alpha=alpha, zorder=1))


def _node(ax, node, *, dim=False, active=False):
    x, y, w, h = node["pos"]
    color = "#cbd5e1" if dim else node["color"]
    fill = "#f8fafc" if dim else node["fill"]
    if active:
        _glow(ax, x, y, w, h, color=node["color"])
    _box(ax, x, y, w, h, color=color, fill=fill, lw=2.4 if active else 1.4)
    ax.text(x, y + h / 2 - 0.012, node["title"], ha="center", va="top",
            fontsize=9.4, fontweight="bold",
            color="#94a3b8" if dim else "#0f172a")
    if node["sub"]:
        ax.text(x, y - h / 2 + 0.013, node["sub"], ha="center",
                va="bottom", fontsize=7.4, style="italic",
                color="#cbd5e1" if dim else "#334155")


def _arrow(ax, p, q, *, color="#94a3b8", lw=1.0, dashed=False, alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="->,head_length=4,head_width=3",
        color=color, lw=lw, alpha=alpha,
        linestyle=(0, (4, 3)) if dashed else "-",
        mutation_scale=12, zorder=2))


def _chip(ax, organ, idx, *, state):
    """state: 'idle' | 'scan' | 'picked' | 'dim'"""
    x, y, w, h = _chip_pos(idx)
    b = BRANCHES[organ]
    if state == "picked":
        _glow(ax, x, y, w, h, color=b["color"])
        color, fill, lw, tcol = b["color"], b["fill"], 2.6, "#0f172a"
    elif state == "scan":
        color, fill, lw, tcol = b["color"], b["fill"], 1.8, "#0f172a"
    elif state == "dim":
        color, fill, lw, tcol = "#e2e8f0", "#f8fafc", 1.0, "#cbd5e1"
    else:
        color, fill, lw, tcol = b["color"], b["fill"], 1.3, "#0f172a"
    _box(ax, x, y, w, h, color=color, fill=fill, lw=lw)
    ax.text(x, y + 0.011, organ, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=tcol)
    ax.text(x, y - 0.014, f"{b['n']} agents", ha="center", va="center",
            fontsize=6.8, color=tcol if state != "dim" else "#cbd5e1")


def _deep_panel(ax, organ, *, reveal=1.0, round2=False):
    """Left panel listing the activated branch's deep specialists."""
    b = BRANCHES[organ]
    x0, y0, w, h = 0.045, 0.560, 0.515, 0.30
    _box(ax, x0 + w / 2, y0 - h / 2, w, h,
         color=b["color"], fill="#ffffff", lw=1.6)
    ax.text(x0 + w / 2, y0 - 0.018,
            f"ACTIVE branch: {organ}  →  {b['n']} deep agents"
            + ("   (round 2: blackboard refine)" if round2 else ""),
            ha="center", va="top", fontsize=8.6, fontweight="bold",
            color="#0f172a")
    deep = b["deep"]
    n_show = max(1, int(round(len(deep) * reveal)))
    col_n = (len(deep) + 1) // 2
    for i, name in enumerate(deep[:n_show]):
        col = 0 if i < col_n else 1
        row = i if i < col_n else i - col_n
        ax.text(x0 + 0.018 + col * 0.255, y0 - 0.05 - row * 0.0265,
                "• " + name, ha="left", va="top", fontsize=7.4,
                color="#334155")


def _dormant_note(ax, organ):
    others = [o for o in ORGAN_ORDER if o != organ]
    ax.text(0.045, 0.245,
            "DORMANT (not called this image): "
            + ", ".join(others),
            ha="left", va="top", fontsize=7.0, style="italic",
            color="#94a3b8")


def _log(ax, entries: List[Tuple[str, str]]):
    x0, w = 0.585, 0.40
    ax.add_patch(FancyBboxPatch(
        (x0, 0.05), w, 0.90,
        boxstyle="round,pad=0.005,rounding_size=0.006",
        linewidth=1.0, edgecolor="#cbd5e1", facecolor="#ffffff", zorder=1))
    ax.text(x0 + w / 2, 0.955, "Running delta log (leaf branch)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#0f172a")
    y = 0.915
    for text, color in entries:
        ax.text(x0 + 0.012, y, text, ha="left", va="top",
                fontsize=7.2, color=color, wrap=True)
        y -= 0.052


def _title(ax, t):
    ax.text(0.5, 0.992, t, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#0f172a")


def _caption(ax, c, step):
    ax.text(0.01, 0.006, f"step {step:02d}", ha="left", va="bottom",
            fontsize=7.5, color="#64748b")
    ax.text(0.99, 0.006, c, ha="right", va="bottom",
            fontsize=8.4, color="#0f172a", style="italic")


def _skeleton(ax, *, detect=False, chips_state=None, consol=False,
              output=False):
    _node(ax, NODE_STATIC)
    _arrow(ax, (NODE_STATIC["pos"][0], NODE_STATIC["pos"][1] - 0.035),
           (NODE_DETECT["pos"][0], NODE_DETECT["pos"][1] + 0.035),
           color="#cbd5e1", lw=1.2)
    _node(ax, NODE_DETECT, active=detect)
    # detector -> chips
    dx, dy = NODE_DETECT["pos"][0], NODE_DETECT["pos"][1] - 0.035
    cs = chips_state or {o: "idle" for o in ORGAN_ORDER}
    for i, o in enumerate(ORGAN_ORDER):
        cx, cy, _, ch = _chip_pos(i)
        _arrow(ax, (dx, dy), (cx, cy + ch / 2),
               color="#e2e8f0", lw=0.7, dashed=True)
        _chip(ax, o, i, state=cs.get(o, "idle"))
    _node(ax, NODE_CONSOL, active=consol)
    _node(ax, NODE_OUTPUT, dim=not output)


def _finalize(fig) -> Image.Image:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf,
                           "raw", "RGBA", 0, 1).convert("RGB")
    plt.close(fig)
    return img


# ----------------------------------------------------------------------
# Acts
# ----------------------------------------------------------------------

def _act1(step):
    out = []
    caps = [
        "One Bugwood photo shows essentially ONE organ — a leaf shot is just a leaf.",
        "Static context: the canonical visual_symptoms slice + that single photo.",
        "Running every agent on every image is wasteful and vague — instead, route.",
        "DR.Arti-style decision tree: detect the organ, then deep-dive ONE branch.",
    ]
    for i in range(ACT1):
        fig, ax = _setup_axes()
        _title(ax, "Phase 0R — organ-routed visual-symptom swarm")
        _skeleton(ax, chips_state={o: "idle" for o in ORGAN_ORDER})
        _log(ax, [])
        _caption(ax, caps[min(i, len(caps) - 1)], step + i)
        out.append(_finalize(fig))
    return out


def _act2(step):
    out = []
    for i in range(ACT2):
        fig, ax = _setup_axes()
        _title(ax, "Act 2 — OrganDetectionAgent: which organ is this? (1 visual call)")
        # scan chips left→right, then lock onto 'leaf'.
        cs = {}
        scan_idx = i  # advance the scan highlight
        for j, o in enumerate(ORGAN_ORDER):
            if i >= ACT2 - 2:
                cs[o] = "picked" if o == DETECTED else "dim"
            elif j == scan_idx % len(ORGAN_ORDER):
                cs[o] = "scan"
            else:
                cs[o] = "idle"
        _skeleton(ax, detect=True, chips_state=cs)
        _log(ax, [])
        if i >= ACT2 - 2:
            cap = ("Detected organ = 'leaf' (high confidence). "
                   "Pure visual triage — no KB, no diagnosis.")
        else:
            cap = "Scanning candidate organs: leaf / stem / root / crown / flower / fruit / whole_plant."
        _caption(ax, cap, step + i)
        out.append(_finalize(fig))
    return out


def _act3(step):
    out = []
    caps = [
        "route_for_organ('leaf') selects ONLY the leaf branch.",
        "8 deep leaf specialists activate (lesion shape/color/texture, chlorosis, ...).",
        "+ always-on cross-cutters: ColorPalette, Severity, LookAlikeCoT, Sporulation.",
        "= 13 active agents. The other 7 branches stay DORMANT this image.",
        "No fruit/root/stem agent wastes a call saying 'not visible'.",
    ]
    for i in range(ACT3):
        fig, ax = _setup_axes()
        _title(ax, "Act 3 — Routing: activate only the detected organ's deep specialists")
        cs = {o: ("picked" if o == DETECTED else "dim") for o in ORGAN_ORDER}
        _skeleton(ax, chips_state=cs)
        _deep_panel(ax, DETECTED, reveal=(i + 1) / ACT3)
        _dormant_note(ax, DETECTED)
        _log(ax, [])
        _caption(ax, caps[min(i, len(caps) - 1)], step + i)
        out.append(_finalize(fig))
    return out


def _act4(step):
    out = []
    n = len(DELTAS)
    for i in range(ACT4):
        fig, ax = _setup_axes()
        round2 = i >= ACT4 // 2
        _title(ax, "Act 4 — Deep dive: leaf branch fans out "
                   + ("(round 2 — blackboard refine)" if round2 else "(round 1 — independent)"))
        cs = {o: ("picked" if o == DETECTED else "dim") for o in ORGAN_ORDER}
        _skeleton(ax, chips_state=cs)
        _deep_panel(ax, DETECTED, reveal=1.0, round2=round2)
        _dormant_note(ax, DETECTED)
        if not round2:
            done = min(n, int((i + 1) / max(1, ACT4 // 2) * n))
            entries = [(f"[{f}] {t}", BRANCHES[DETECTED]["color"])
                       for f, t in DELTAS[:done]]
            cap = f"Round 1: {done}/{n} deltas emitted (each vs canonical visual_symptoms)."
        else:
            entries = [(f"[{f}] {t}", BRANCHES[DETECTED]["color"])
                       for f, t in DELTAS]
            entries.append(("  ✓ SUPPORT  LeafNecrosis → LeafChlorosis "
                            "(same interveinal progression)", "#16a34a"))
            entries.append(("  ⚠ CHALLENGE ColorPalette → LeafLesionColor "
                            "('tan, not brown')", "#dc2626"))
            cap = ("Round 2: blackboard visible — active agents refine / "
                   "support / challenge each other.")
        _log(ax, entries)
        _caption(ax, cap, step + i)
        out.append(_finalize(fig))
    return out


def _act5(step):
    out = []
    caps = [
        "VisualDiagnosisAgent consolidates the leaf-branch outputs (dedup, decisive forks).",
        "Dedup: 'leaf_curl' folded into 'leaf_chlorosis' (overlapping observation).",
        "Cross-pass K-of-N (N=10, K=3): drop deltas not robust across seeds.",
        "'severity_visible' fell below K — dropped as a per-pass artifact.",
        "Claude web verifier checks each survivor (extension / APS / CABI).",
        "Conservative merge into final_registry.json — existing preserved, support bumped.",
        "Result: regional_observations[<state>].deltas[] for this (crop, disease).",
        "Done. 1 photo → 1 organ → 1 deep branch → verified visual deltas.",
    ]
    for i in range(ACT5):
        fig, ax = _setup_axes()
        _title(ax, "Act 5 — Consolidate → K-of-N → web verifier → merge → KB")
        cs = {o: ("picked" if o == DETECTED else "dim") for o in ORGAN_ORDER}
        _skeleton(ax, chips_state=cs,
                  consol=i < ACT5 - 2, output=i >= ACT5 - 3)
        entries: List[Tuple[str, str]] = []
        for k, (f, t) in enumerate(DELTAS):
            if i >= 1 and k in DEDUP_DROP:
                if i >= 2:
                    continue
                entries.append((f"[{f}] {t}  ← dedup", "#9ca3af"))
                continue
            if i >= 3 and k in KOFN_DROP:
                if i >= 4:
                    continue
                entries.append((f"[{f}] {t}  ← K-of-N drop", "#9ca3af"))
                continue
            entries.append((f"[{f}] {t}", BRANCHES[DETECTED]["color"]))
        _log(ax, entries)
        if i >= ACT5 - 3:
            js = [
                '"<state>": {',
                '  "deltas": [',
                '    {"field": "leaf_chlorosis",',
                '     "image_shows": "interveinal yellowing; veins stay green",',
                '     "__support__": 9, "verification_status": "verified"},',
                '    {"field": "look_alikes_visual", ...},',
                '    {"field": "color_palette",      ...} ] }',
            ]
            for k, line in enumerate(js):
                ax.text(0.05, 0.40 - k * 0.026, line, ha="left", va="top",
                        fontsize=7.0, family="monospace", color="#0f172a")
        _caption(ax, caps[min(i, len(caps) - 1)], step + i)
        out.append(_finalize(fig))
    return out


def main():
    print(f"Rendering {OUT.name} — {TOTAL} frames @ {FPS} fps "
          f"({FRAME_MS} ms/frame, ~{TOTAL / FPS:.0f} s)")
    frames: List[Image.Image] = []
    s = 1
    for fn, n in ((_act1, ACT1), (_act2, ACT2), (_act3, ACT3),
                  (_act4, ACT4), (_act5, ACT5)):
        frames.extend(fn(s))
        s += n
    frames.extend([frames[-1]] * HOLD)
    print(f"  composing {len(frames)} frames")
    frames[0].save(
        OUT, save_all=True, append_images=frames[1:],
        duration=FRAME_MS, loop=0, optimize=True)
    mb = OUT.stat().st_size / (1024 * 1024)
    print(f"  wrote {OUT} ({mb:.1f} MB, {len(frames)} frames, "
          f"~{len(frames) * FRAME_MS / 1000:.0f} s playback)")


if __name__ == "__main__":
    main()
