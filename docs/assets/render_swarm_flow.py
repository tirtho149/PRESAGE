"""
docs/assets/render_swarm_flow.py
================================
Render an animated GIF that walks a viewer through the current Phase 0R
**real swarm** — 24 specialists running TWO rounds with a shared
blackboard, cross-talk (support / challenge / withdraw), and a CoT
consolidator.

Five acts:

    Act 1 — Setup
            Static context (canonical KB + field photograph + state)
            7 organ-family group cards introduce the 24 specialists

    Act 2 — Round 1: independent observation
            All 24 specialists fan out in parallel
            Each writes its delta to the running log
            No inter-agent visibility yet

    Act 3 — Blackboard cross-talk (round 2 — the "real swarm" round)
            Every round-1 output is published to a shared blackboard
            All 24 specialists run AGAIN with the blackboard visible
            Animated cross-arrows show SUPPORT (green), CHALLENGE
            (red), and WITHDRAW (gray) actions between specialists

    Act 4 — CoT consolidator (VisualDiagnosisAgent)
            5-step CoT: triage -> decisive forks -> adjudicate
            cross-refs -> dedup -> emit final deltas + trace

    Act 5 — K-of-N + Merge + Output
            Cross-pass agreement filter, Claude web verifier,
            conservative merge into final_registry.json

Output: docs/assets/swarm_flow.gif
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

OUT = Path(__file__).resolve().parent / "swarm_flow.gif"


# ----------------------------------------------------------------------
# Layout — 24 specialists grouped into 7 organ families
# ----------------------------------------------------------------------

LEFT_W = 0.62  # x-fraction of the swarm panel

# 7 organ-family GROUP cards. Each card lists its specialists in small
# text. Coordinates are (x_center, y_center, width, height) in [0,1].
GROUPS: Dict[str, Dict] = {
    "LEAF": dict(
        pos=(0.18, 0.77, 0.20, 0.12),
        color="#16a34a",
        fill="#dcfce7",
        title="LEAF (8)",
        agents=[
            "LeafLesionShape", "LeafLesionColor", "LeafLesionTexture",
            "LeafChlorosis",   "LeafNecrosis",    "LeafCurl",
            "LeafVeinPattern", "LeafGeometry",
        ],
    ),
    "STEM": dict(
        pos=(0.42, 0.77, 0.20, 0.12),
        color="#dc2626",
        fill="#fee2e2",
        title="STEM (4)",
        agents=[
            "StemLesion", "StemPith", "StemSurface", "StemDiscoloration",
        ],
    ),
    "BELOW": dict(
        pos=(0.18, 0.59, 0.20, 0.10),
        color="#a16207",
        fill="#fef3c7",
        title="BELOW-GROUND (2)",
        agents=["Root", "CrownCollar"],
    ),
    "REPRO": dict(
        pos=(0.42, 0.59, 0.20, 0.10),
        color="#7c3aed",
        fill="#ede9fe",
        title="REPRODUCTIVE (2)",
        agents=["Flower", "Fruit"],
    ),
    "SIGNS": dict(
        pos=(0.18, 0.45, 0.20, 0.08),
        color="#0d9488",
        fill="#ccfbf1",
        title="PATHOGEN SIGNS (1)",
        agents=["Sporulation"],
    ),
    "PAT": dict(
        pos=(0.42, 0.45, 0.20, 0.10),
        color="#2563eb",
        fill="#dbeafe",
        title="WHOLE-PLANT PATTERNS (3)",
        agents=["Wilting", "Defoliation", "SpatialPattern"],
    ),
    "DIAG": dict(
        pos=(0.30, 0.30, 0.32, 0.10),
        color="#c026d3",
        fill="#fae8ff",
        title="DIAGNOSTIC CROSS-CUTTERS (4)",
        agents=[
            "ConcentricPattern", "ColorPalette (encoder)",
            "LookAlikeCoT", "SeverityVisual",
        ],
    ),
}

# Static / consolidator / output nodes.
NODE_STATIC = dict(
    pos=(0.30, 0.93, 0.55, 0.07),
    color="#6366f1",
    fill="#e0e7ff",
    title="STATIC CONTEXT",
    sub="canonical visual_symptoms + field photo + state",
)
NODE_CONSOL = dict(
    pos=(0.30, 0.15, 0.55, 0.08),
    color="#0d9488",
    fill="#ccfbf1",
    title="VisualDiagnosisAgent — CoT consolidator",
    sub="(1) triage  (2) decisive forks  (3) dedup  (4) emit",
)
NODE_OUTPUT = dict(
    pos=(0.30, 0.045, 0.55, 0.05),
    color="#16a34a",
    fill="#dcfce7",
    title="state regional_observations.deltas[]",
    sub="",
)


# ----------------------------------------------------------------------
# Sample deltas — one per group, used to populate the running log
# ----------------------------------------------------------------------

DELTAS: List[Tuple[str, str]] = [
    ("LEAF",  "[leaf_chlorosis] interveinal yellowing on younger leaves; veins stay green"),
    ("LEAF",  "[leaf_necrosis] tip-and-margin necrosis with crisp bleached edge"),
    ("LEAF",  "[leaf_vein_pattern] vein-aligned chlorotic banding crosses major veins"),
    ("STEM",  "[stem_pith] split lower stem: WHITE pith inside chocolate-brown vascular ring"),
    ("STEM",  "[stem_lesion] no surface canker visible on the photographed stem"),
    ("BELOW", "[root_visible] BLUE fungal masses near taproot at the crown"),
    ("REPRO", "[flower] no flowers in this frame"),
    ("SIGNS", "[sporulation] white aerial mycelium at petiole bases"),
    ("PAT",   "[defoliation] bare-petiole skeletons: blades dropped while petioles remained attached"),
    ("PAT",   "[wilting] hemispheric wilting on the lower canopy only"),
    ("DIAG",  "[concentric_pattern] no target-spot ringing in lesions"),
    ("DIAG",  "[color_palette] tan center + chocolate margin + chlorotic yellow halo"),
    ("DIAG",  "[look_alikes_visual] white pith + bare-petiole + blue roots match SDS, not BSR"),
]
# Items dropped by the consolidator's dedup step (Step 3).
DROPPED_INDEX = {4, 6, 10}   # stem_lesion-no, flower-no, concentric-no
# Items that survive K-of-N agreement filter in Act 4. Drop a couple
# more to demonstrate the filter (simulating cross-pass disagreement).
KOFN_DROPPED_INDEX = {1}     # leaf_necrosis tip-and-margin dropped


# ----------------------------------------------------------------------
# Frame budgets per act
# ----------------------------------------------------------------------

FPS = 6
ACT1_FRAMES   = 18   # ~3 s  Setup
ACT2_FRAMES   = 24   # ~4 s  Round 1: independent fan-out
ACT3_FRAMES   = 30   # ~5 s  Round 2: blackboard cross-talk (the "real swarm" round)
ACT4_FRAMES   = 24   # ~4 s  CoT consolidator
ACT5_FRAMES   = 18   # ~3 s  K-of-N + merge + output
HOLD_FRAMES   = 12   # ~2 s  end hold
TOTAL_FRAMES  = (ACT1_FRAMES + ACT2_FRAMES + ACT3_FRAMES + ACT4_FRAMES
                 + ACT5_FRAMES + HOLD_FRAMES)


# Cross-ref actions to animate in Act 3 (round 2).
# Each tuple: (from_group, to_group, action, rationale_short)
CROSS_REFS = [
    ("STEM",  "PAT",   "support",   "StemPith white → SDS; supports Defoliation's bare-petiole SDS call"),
    ("BELOW", "STEM",  "support",   "Root sees blue masses → SDS; confirms StemPith"),
    ("DIAG",  "LEAF",  "challenge", "ColorPalette: tan dominant, not brown — challenge LeafLesionColor"),
    ("DIAG",  "STEM",  "support",   "LookAlikeCoT walks: pith + petiole + roots all → SDS"),
    ("LEAF",  None,    "withdraw",  "LeafLesionColor withdraws round-1 'brown' after ColorPalette challenge"),
    ("PAT",   "BELOW", "support",   "Wilting pattern hemispheric → consistent with SDS root rot"),
]


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def ease(p: float) -> float:
    p = max(0.0, min(1.0, p))
    return 0.5 - 0.5 * math.cos(math.pi * p)


def _node_box(ax, x, y, w, h, *, color, fill, lw=1.4, alpha=1.0, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=lw, edgecolor=color, facecolor=fill,
        alpha=alpha, zorder=zorder,
    ))


def _glow(ax, x, y, w, h, *, color, alpha=0.25):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2 - 0.014, y - h / 2 - 0.014), w + 0.028, h + 0.028,
        boxstyle="round,pad=0.01,rounding_size=0.014",
        linewidth=0, facecolor=color, alpha=alpha, zorder=1,
    ))


def _draw_group(ax, gid: str, *, active: bool = False, dim: bool = False):
    g = GROUPS[gid]
    x, y, w, h = g["pos"]
    color = g["color"]
    fill  = g["fill"]
    if active:
        _glow(ax, x, y, w, h, color=color, alpha=0.30)
        lw = 2.6
    elif dim:
        color = "#cbd5e1"
        fill = "#f8fafc"
        lw = 1.0
    else:
        lw = 1.4
    _node_box(ax, x, y, w, h, color=color, fill=fill, lw=lw)
    ax.text(x, y + h / 2 - 0.014, g["title"],
            ha="center", va="top", fontsize=8.8, fontweight="bold",
            color="#0f172a" if not dim else "#94a3b8")
    # Specialist names listed in two columns underneath the title.
    names = g["agents"]
    half = (len(names) + 1) // 2
    col_left  = names[:half]
    col_right = names[half:]
    name_y = y - 0.005
    for i, n in enumerate(col_left):
        ax.text(x - w / 2 + 0.012, name_y - i * 0.013, "• " + n,
                ha="left", va="top", fontsize=6.5,
                color="#334155" if not dim else "#cbd5e1")
    for i, n in enumerate(col_right):
        ax.text(x + 0.002, name_y - i * 0.013, "• " + n,
                ha="left", va="top", fontsize=6.5,
                color="#334155" if not dim else "#cbd5e1")


def _draw_static_node(ax, node: Dict, *, dim: bool = False):
    x, y, w, h = node["pos"]
    color = node["color"] if not dim else "#cbd5e1"
    fill  = node["fill"]  if not dim else "#f8fafc"
    _node_box(ax, x, y, w, h, color=color, fill=fill, lw=1.4)
    ax.text(x, y + h / 2 - 0.012, node["title"],
            ha="center", va="top", fontsize=9.0, fontweight="bold",
            color="#0f172a" if not dim else "#94a3b8")
    if node["sub"]:
        ax.text(x, y - h / 2 + 0.014, node["sub"],
                ha="center", va="bottom", fontsize=7.5,
                color="#334155" if not dim else "#cbd5e1",
                style="italic")


def _draw_arrow(ax, src_xy, dst_xy, *, color="#94a3b8", alpha=1.0, lw=1.0, dashed=False):
    style = "->,head_length=4,head_width=3"
    ls = (0, (4, 3)) if dashed else "-"
    ax.add_patch(FancyArrowPatch(
        src_xy, dst_xy, arrowstyle=style,
        color=color, alpha=alpha, linewidth=lw,
        linestyle=ls, mutation_scale=12,
        zorder=2,
    ))


def _idle_edges(ax):
    # Static -> all groups
    sx, sy, _, sh = NODE_STATIC["pos"]
    for gid in GROUPS:
        gx, gy, _, gh = GROUPS[gid]["pos"]
        _draw_arrow(ax, (sx, sy - sh / 2), (gx, gy + gh / 2),
                    color="#e2e8f0", lw=0.8, dashed=True)
    # Groups -> consolidator
    cx, cy, _, ch = NODE_CONSOL["pos"]
    for gid in GROUPS:
        gx, gy, _, gh = GROUPS[gid]["pos"]
        _draw_arrow(ax, (gx, gy - gh / 2), (cx, cy + ch / 2),
                    color="#e2e8f0", lw=0.8, dashed=True)
    # Consolidator -> output
    ox, oy, _, oh = NODE_OUTPUT["pos"]
    _draw_arrow(ax, (cx, cy - ch / 2), (ox, oy + oh / 2),
                color="#e2e8f0", lw=0.8, dashed=True)


def _draw_log(ax, entries: List[Tuple[str, str]]):
    """Right-side running delta log."""
    x0 = 0.65
    y0 = 0.93
    w  = 0.33
    ax.add_patch(FancyBboxPatch(
        (x0, 0.04), w, 0.92,
        boxstyle="round,pad=0.005,rounding_size=0.006",
        linewidth=1.0, edgecolor="#cbd5e1", facecolor="#ffffff",
        zorder=1,
    ))
    ax.text(x0 + w / 2, y0 + 0.01, "Running delta log (per pass)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#0f172a")
    y = y0 - 0.02
    line_h = 0.025
    for text, color in entries:
        ax.text(x0 + 0.008, y, text,
                ha="left", va="top", fontsize=6.6, color=color,
                wrap=True)
        y -= line_h


def _draw_title(ax, t: str):
    ax.text(0.5, 0.985, t, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color="#0f172a")


def _draw_caption(ax, c: str, step: int):
    ax.text(0.01, 0.005, f"step {step:02d}", ha="left", va="bottom",
            fontsize=7.5, color="#64748b")
    ax.text(0.99, 0.005, c, ha="right", va="bottom",
            fontsize=8.0, color="#0f172a", style="italic")


# ----------------------------------------------------------------------
# Per-act frame generation
# ----------------------------------------------------------------------

def _setup_axes():
    fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=110)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_aspect("auto")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _common_skeleton(ax, *, active_groups=None, dim_groups=None,
                     consol_active=False, output_active=False,
                     show_static=True):
    active_groups = set(active_groups or [])
    dim_groups    = set(dim_groups or [])
    _idle_edges(ax)
    _draw_static_node(ax, NODE_STATIC, dim=not show_static)
    for gid in GROUPS:
        _draw_group(ax, gid,
                    active=gid in active_groups,
                    dim=gid in dim_groups)
    _draw_static_node(ax, NODE_CONSOL,
                      dim=not consol_active and not output_active)
    if consol_active:
        cx, cy, cw, ch = NODE_CONSOL["pos"]
        _glow(ax, cx, cy, cw, ch, color="#0d9488", alpha=0.30)
    _draw_static_node(ax, NODE_OUTPUT, dim=not output_active)


def _act1_frames(step: int) -> List:
    out = []
    titles = [
        "Phase 0R — Qwen visual-symptom swarm (24 specialists + CoT consolidator)",
    ]
    captions = [
        "Static context: a canonical KB block (Phase 0 / Claude) + a Bugwood field photograph.",
        "Static context: a canonical KB block + a field photograph (visual_symptoms slice only).",
        "Specialists are grouped by organ family. Each owns ONE delta field.",
        "LEAF group: 8 specialists (lesion shape / color / texture, chlorosis, necrosis, curl, vein, geometry).",
        "STEM group: 4 specialists (lesion, pith, surface, discoloration) — pith is the decisive SDS↔BSR fork.",
        "BELOW-GROUND group: roots (cysts → SCN; blue masses → SDS) + crown / collar.",
        "REPRODUCTIVE group: flower + fruit. Skipped when not visible.",
        "PATHOGEN SIGNS group: visible mycelium / spores / pycnidia / ooze.",
        "WHOLE-PLANT PATTERNS group: wilting, defoliation (petiole-attached SDS fork), spatial pattern.",
        "DIAGNOSTIC CROSS-CUTTERS group: concentric pattern, color encoder, look-alike CoT, severity.",
    ]
    intro_order = ["LEAF", "STEM", "BELOW", "REPRO", "SIGNS", "PAT", "DIAG"]

    for i in range(ACT1_FRAMES):
        fig, ax = _setup_axes()
        _draw_title(ax, titles[0])

        # First 2 frames just show the static context.
        if i < 2:
            _common_skeleton(ax, active_groups=[], dim_groups=intro_order)
            _draw_log(ax, [])
            _draw_caption(ax, captions[min(i, 1)], step + i)
        else:
            # Reveal groups one at a time.
            n_revealed = min(i - 1, len(intro_order))
            revealed = intro_order[:n_revealed]
            hidden   = intro_order[n_revealed:]
            _common_skeleton(ax, active_groups=[], dim_groups=hidden)
            _draw_log(ax, [])
            cap_idx = min(1 + n_revealed, len(captions) - 1)
            _draw_caption(ax, captions[cap_idx], step + i)
        out.append(_finalize(fig))
    return out


def _act2_frames(step: int) -> List:
    """Parallel fan-out: groups light up simultaneously, deltas
    accumulate in the log."""
    out = []
    n_deltas = len(DELTAS)
    deltas_per_frame = max(1, n_deltas // (ACT2_FRAMES - 6))

    for i in range(ACT2_FRAMES):
        fig, ax = _setup_axes()
        _draw_title(ax, "Act 2 — Round 1: 24 specialists fan out in parallel (independent observation)")
        # First 4 frames: all 7 groups light up simultaneously.
        if i < 4:
            actives = list(GROUPS) if i >= 1 else []
            _common_skeleton(ax, active_groups=actives)
            _draw_log(ax, [])
            cap = ("All 24 specialists run in parallel on the same input "
                   "(image + canonical KB + existing KB)."
                   if i >= 1 else
                   "Pass begins. Running log is empty.")
            _draw_caption(ax, cap, step + i)
        else:
            # Stream deltas into the log.
            done = min(n_deltas, (i - 3) * deltas_per_frame)
            entries: List[Tuple[str, str]] = []
            for k in range(done):
                gid, text = DELTAS[k]
                color = GROUPS[gid]["color"]
                entries.append((text, color))
            actives = sorted({DELTAS[k][0] for k in range(done)})
            _common_skeleton(ax, active_groups=actives)
            _draw_log(ax, entries)
            _draw_caption(
                ax,
                f"{done}/{n_deltas} deltas emitted — each tagged with its specialist's field.",
                step + i,
            )
        out.append(_finalize(fig))
    return out


def _draw_cross_ref_arrow(ax, from_gid: str, to_gid: str, action: str, *, alpha=0.9):
    """Curved arrow between two group cards illustrating a round-2
    support / challenge / withdraw action."""
    if from_gid not in GROUPS:
        return
    sx, sy, sw, sh = GROUPS[from_gid]["pos"]
    if to_gid and to_gid in GROUPS:
        dx, dy, dw, dh = GROUPS[to_gid]["pos"]
    else:
        # Self-withdraw → draw a loop staying on the from_gid card.
        dx, dy = sx + 0.045, sy - 0.05
    color = {"support": "#16a34a",
             "challenge": "#dc2626",
             "withdraw": "#6b7280"}.get(action, "#94a3b8")
    style = "->,head_length=5,head_width=4"
    # Use a curved patch so multiple arrows don't overlap badly.
    ax.add_patch(FancyArrowPatch(
        (sx, sy), (dx, dy),
        connectionstyle=f"arc3,rad={0.18 if to_gid else 0.6}",
        arrowstyle=style, color=color, alpha=alpha,
        linewidth=2.0, mutation_scale=14, zorder=4,
    ))


def _act3_frames(step: int) -> List:
    """Round 2 — Blackboard cross-talk. Specialists see each other's
    round-1 outputs and emit support / challenge / withdraw actions."""
    out = []
    n_deltas = len(DELTAS)
    captions = [
        "Round 2 begins. Every round-1 output has been published to a shared blackboard.",
        "All 24 specialists run AGAIN — each now reads peers' findings on the blackboard.",
        "Cross-talk #1: StemPithAgent (white pith → SDS) SUPPORTS DefoliationAgent (bare petioles).",
        "Cross-talk #2: RootAgent (blue masses) SUPPORTS StemPithAgent — converging on SDS.",
        "Cross-talk #3: ColorPaletteAgent CHALLENGES LeafLesionColorAgent — 'tan dominant, not brown'.",
        "Cross-talk #4: LookAlikeCoTAgent SUPPORTS StemPith — decisive SDS-vs-BSR fork.",
        "Cross-talk #5: LeafLesionColorAgent WITHDRAWS round-1 'brown' after the color-encoder challenge.",
        "Cross-talk #6: WiltingAgent SUPPORTS RootAgent — hemispheric wilt matches SDS root rot.",
        "Round 2 complete. The blackboard now carries refined deltas + 4 SUPPORTS, 1 CHALLENGE, 1 WITHDRAW.",
    ]
    per_cap = max(1, ACT3_FRAMES // len(captions))

    # Each cross-ref animation lasts 3 frames.
    cross_ref_schedule: List[Tuple[int, int]] = []
    n_refs = len(CROSS_REFS)
    for i in range(n_refs):
        start = 6 + i * 3
        cross_ref_schedule.append((start, start + 3))

    for i in range(ACT3_FRAMES):
        cap_idx = min(i // per_cap, len(captions) - 1)
        fig, ax = _setup_axes()
        _draw_title(ax, "Act 3 — Round 2: blackboard cross-talk (the 'real swarm' round)")

        # Round-2 log: round-1 entries + cross_ref annotations.
        entries: List[Tuple[str, str]] = []
        for k in range(n_deltas):
            gid, text = DELTAS[k]
            color = GROUPS[gid]["color"]
            entries.append((text, color))
        # Append cross-ref entries as they fire.
        for ridx, (start, stop) in enumerate(cross_ref_schedule):
            if i >= start:
                src, tgt, action, rationale = CROSS_REFS[ridx]
                ack = {"support": "✓ SUPPORT",
                       "challenge": "⚠ CHALLENGE",
                       "withdraw": "↩ WITHDRAW"}[action]
                arrow = f"{src} → {tgt or 'self'}" if tgt is not None else f"{src} → self"
                col = {"support": "#16a34a",
                       "challenge": "#dc2626",
                       "withdraw": "#6b7280"}[action]
                entries.append((f"  {ack}  {arrow}: {rationale}", col))

        # Active groups: anything currently involved in a cross-ref or
        # all groups during the second half.
        actives = []
        for ridx, (start, stop) in enumerate(cross_ref_schedule):
            if start <= i < stop:
                src, tgt, _, _ = CROSS_REFS[ridx]
                actives.extend([g for g in (src, tgt) if g])
        if i >= 24:
            actives = list(GROUPS)
        _common_skeleton(ax, active_groups=set(actives))

        # Live cross-ref arrows for the current animation window.
        for ridx, (start, stop) in enumerate(cross_ref_schedule):
            if start <= i < stop:
                src, tgt, action, _rationale = CROSS_REFS[ridx]
                _draw_cross_ref_arrow(ax, src, tgt, action, alpha=0.95)

        _draw_log(ax, entries)
        _draw_caption(ax, captions[cap_idx], step + i)
        out.append(_finalize(fig))
    return out


def _act4_frames_cot(step: int) -> List:
    """CoT consolidator: VisualDiagnosisAgent walks 5 steps."""
    out = []
    n_deltas = len(DELTAS)
    cot_captions = [
        "Step 1 — Triage: which organs are visible? leaf ✓, cut stem ✓, root ✓, flower ✗.",
        "Step 2 — Decisive forks: white pith + bare-petioles + blue roots → SDS, not BSR.",
        "Step 3 — Adjudicate cross_refs: SUPPORTS raise confidence; CHALLENGE/WITHDRAW resolved.",
        "Step 4 — Dedup: drop overlapping items and anything that just restates canonical.",
        "Step 5 — Emit final deltas plus a CoT trace explaining the decision.",
    ]
    per_step = ACT4_FRAMES // 5

    for i in range(ACT4_FRAMES):
        step_idx = min(i // per_step, 4)
        fig, ax = _setup_axes()
        _draw_title(ax, "Act 4 — VisualDiagnosisAgent walks the decision-graph CoT")

        entries: List[Tuple[str, str]] = []
        for k in range(n_deltas):
            gid, text = DELTAS[k]
            color = GROUPS[gid]["color"]
            # Step 4 dims dropped entries; Step 5 keeps survivors only.
            if step_idx >= 3 and k in DROPPED_INDEX:
                color = "#9ca3af"
                text = text + "  ← drop (overlaps / not new)"
            if step_idx >= 4 and k in DROPPED_INDEX:
                continue
            entries.append((text, color))

        _common_skeleton(ax, active_groups=[], consol_active=True)
        _draw_log(ax, entries)
        _draw_caption(ax, cot_captions[step_idx], step + i)
        out.append(_finalize(fig))
    return out


def _act5_frames(step: int) -> List:
    """K-of-N filter + merge + final JSON output."""
    out = []
    survived = [d for k, d in enumerate(DELTAS)
                if k not in DROPPED_INDEX and k not in KOFN_DROPPED_INDEX]
    captions = [
        "Act 5 — Cross-pass K-of-N agreement filter (passes run with N seeds, e.g. N=10, K=3).",
        "Spurious per-pass hallucinations dropped; survivors are robust across seeds.",
        "Claude web verifier checks each survivor against extension / APS / CABI sources.",
        "Conservative merge into final_registry.json — existing deltas preserved, support bumped on overlap.",
        "Result: final regional_observations[<state>].deltas[] entry for this (crop, disease) profile.",
    ]
    per_cap = ACT5_FRAMES // len(captions)

    for i in range(ACT5_FRAMES):
        cap_idx = min(i // per_cap, len(captions) - 1)
        fig, ax = _setup_axes()
        _draw_title(ax, "Act 5 — K-of-N agreement + web verifier + merge → KB")

        entries: List[Tuple[str, str]] = []
        for gid, text in survived:
            color = GROUPS[gid]["color"]
            entries.append((text, color))

        _common_skeleton(ax, active_groups=[],
                         consol_active=cap_idx < 4,
                         output_active=cap_idx >= 3)
        _draw_log(ax, entries)

        # JSON snippet bottom-overlay when reaching the final cap.
        if cap_idx >= 3:
            json_lines = [
                '"<state>": {',
                '  "image_ids": ["bugwood::1234567"],',
                '  "deltas": [',
                '    {"field": "stem_pith",',
                '     "canonical_says": "(not specified)",',
                '     "image_shows": "white pith inside chocolate-brown vascular ring",',
                '     "image_quote": "split lower stem shows white center",',
                '     "support": 4, "verification_status": "verified"},',
                '    {"field": "look_alikes_visual", ...},',
                '    {"field": "defoliation",        ...},',
                '    ...',
                '  ]',
                '}',
            ]
            for k, line in enumerate(json_lines):
                ax.text(0.02, 0.225 - k * 0.014, line,
                        ha="left", va="top", fontsize=6.4,
                        family="monospace", color="#0f172a")
        _draw_caption(ax, captions[cap_idx], step + i)
        out.append(_finalize(fig))
    return out


def _finalize(fig) -> Image.Image:
    fig.canvas.draw()
    # matplotlib >= 3.8 removed tostring_rgb; use buffer_rgba and drop alpha.
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer(
        "RGBA", fig.canvas.get_width_height(), buf, "raw", "RGBA", 0, 1,
    ).convert("RGB")
    plt.close(fig)
    return img


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    print(f"Rendering {OUT.name} — {TOTAL_FRAMES} frames @ {FPS} fps")
    frames: List[Image.Image] = []
    step = 1
    frames.extend(_act1_frames(step));         step += ACT1_FRAMES
    frames.extend(_act2_frames(step));         step += ACT2_FRAMES
    frames.extend(_act3_frames(step));         step += ACT3_FRAMES
    frames.extend(_act4_frames_cot(step));     step += ACT4_FRAMES
    frames.extend(_act5_frames(step));         step += ACT5_FRAMES
    # End-hold: repeat last frame.
    frames.extend([frames[-1]] * HOLD_FRAMES)

    print(f"  composing {len(frames)} frames into {OUT.name}")
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    sz_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"  wrote {OUT} ({sz_mb:.1f} MB, {len(frames)} frames, "
          f"~{len(frames)/FPS:.1f} s)")


if __name__ == "__main__":
    main()
