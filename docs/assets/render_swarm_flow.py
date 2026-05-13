"""
docs/assets/render_swarm_flow.py
================================
Render an animated GIF that walks a viewer through the full Phase 2
handoff-swarm pipeline, in four acts:

    Act 1 — Setup           : the static context + agent line-up
    Act 2 — One full run    : Triage → specialists → Verifier (with
                              web-search rejection + refinement loop)
                              → Consolidator → delta set
    Act 3 — Cross-run filter: the same run repeated N times with
                              different seeds; deltas surviving in
                              K-of-N runs are kept
    Act 4 — Merge + output  : conservative merge with existing KB,
                              final regional_observations JSON

The output is a single GIF intended to stand alone — the viewer should
understand the whole Phase 2 design from the animation without reading
any prose.

Output: docs/assets/swarm_flow.gif
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

OUT = Path(__file__).resolve().parent / "swarm_flow.gif"

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
LEFT_W = 0.60   # width fraction of the left (network) panel

# Single-run agent network coordinates.
NODES = {
    "STATIC":   (0.50, 0.93, 0.42, 0.10),
    "TRIAGE":   (0.50, 0.78, 0.20, 0.07),
    "MOR":      (0.13, 0.61, 0.18, 0.07),
    "SYM":      (0.37, 0.61, 0.18, 0.07),
    "PAT":      (0.61, 0.61, 0.18, 0.07),
    "SEV":      (0.85, 0.61, 0.18, 0.07),
    "VERIFIER": (0.50, 0.40, 0.32, 0.09),
    "CONS":     (0.50, 0.21, 0.30, 0.07),
    "OUTPUT":   (0.50, 0.06, 0.42, 0.07),
}
LABELS = {
    "STATIC":   "Static context\ncanonical KB block + field photograph",
    "TRIAGE":   "Triage\npicks first specialist",
    "MOR":      "Morphology\nlesion shape, organ",
    "SYM":      "Symptom\nspread, diagnostic",
    "PAT":      "Pathogen\nlook-alikes, type",
    "SEV":      "Severity\ncoverage, treatments",
    "VERIFIER": "Verifier\nClaude headless + web search",
    "CONS":     "Consolidator\ndedupe, drop restatements",
    "OUTPUT":   "Regional deltas for this state",
}
FILL = {
    "STATIC":   "#e0e7ff",
    "TRIAGE":   "#fef3c7",
    "MOR":      "#dbeafe",
    "SYM":      "#d1fae5",
    "PAT":      "#ede9fe",
    "SEV":      "#fee2e2",
    "VERIFIER": "#ffedd5",
    "CONS":     "#ccfbf1",
    "OUTPUT":   "#dcfce7",
}
EDGE = {
    "STATIC":   "#6366f1",
    "TRIAGE":   "#d97706",
    "MOR":      "#2563eb",
    "SYM":      "#059669",
    "PAT":      "#7c3aed",
    "SEV":      "#dc2626",
    "VERIFIER": "#ea580c",
    "CONS":     "#0d9488",
    "OUTPUT":   "#16a34a",
}

# Sample deltas used across the story.
D_MOR  = "[Morphology] yellow halos around dark sunken lesions"
D_MOR2 = "[Morphology] chlorotic halos surround dark sunken lesions (refined)"
D_SYM  = "[Symptom] lesions concentrate near leaf margins, late-season onset"
D_PAT  = "[Pathogen] distinguishable from Septoria by halo colour"
D_SEV  = "[Severity] ~30% canopy affected at first observation"
D_DUP  = "[Pathogen] (dup) similar to Septoria leaf spot"   # dropped by consolidator

# Web-search source snippets shown next to the verifier.
WEB_SOURCES = [
    ("extension.umn.edu",  "...chlorotic halos with sunken lesions appear in late season..."),
    ("apsnet.org",         "...halos are diagnostic; sunken centres distinguish from Septoria..."),
    ("ipm.ucanr.edu",      "...affected canopy commonly ~20-40% in field observations..."),
]

# Cross-run mini-deltas for Act 3. Each run is a list of which deltas
# survived (mor/sym/pat/sev) — a tick = present, blank = absent.
RUNS = [
    {"mor": True,  "sym": True,  "pat": True,  "sev": True },   # run 1 (the one we just watched)
    {"mor": True,  "sym": True,  "pat": False, "sev": True },   # run 2 — pat dropped
    {"mor": True,  "sym": True,  "pat": True,  "sev": False},   # run 3 — sev dropped
    {"mor": True,  "sym": True,  "pat": True,  "sev": True },   # run 4
    {"mor": False, "sym": True,  "pat": True,  "sev": True },   # run 5 — mor dropped (spurious)
]
# Survival counts: mor 4/5, sym 5/5, pat 4/5, sev 4/5. With K=3, all survive.

# ----------------------------------------------------------------------
# Story
# ----------------------------------------------------------------------
# Each stage tuple: (scene, kind, payload, caption, log, extra?)
#   scene in {"intro", "single", "crossrun", "merge"}
#   kind: scene-specific, see render code
#
# Captions sit at the bottom. The step counter is auto-numbered.

STAGES = [
    # ----- ACT 1 — setup -----
    ("intro", "title", "How Phase 2 builds the regional half of PathomeDB",
        "We extract image-grounded deltas one (crop, disease, state) tuple at a time.", []),
    ("intro", "context", None,
        "Static context: a text-grounded canonical block (from Phase 1) and a field photograph.", []),
    ("intro", "agents", None,
        "The swarm has four specialists, a triage agent, a verifier, and a consolidator.", []),

    # ----- ACT 2 — one full run -----
    ("single", "idle", None, "One run begins. The running delta log on the right starts empty.", []),
    ("single", "active", "TRIAGE",
        "Triage inspects the photograph and chooses the first specialist.", []),
    ("single", "flow", ("TRIAGE", "MOR"), "Handoff to Morphology.", []),
    ("single", "active_write", ("MOR", D_MOR),
        "Morphology writes the first delta into the shared log.",
        [(D_MOR, "#1f2937")]),
    ("single", "flow", ("MOR", "SYM"),
        "Handoff to Symptom. Symptom will read Morphology's delta as context.",
        [(D_MOR, "#1f2937")]),
    ("single", "active_write", ("SYM", D_SYM),
        "Symptom adds a complementary observation that builds on Morphology.",
        [(D_MOR, "#1f2937"), (D_SYM, "#1f2937")]),
    ("single", "flow", ("SYM", "VERIFIER"),
        "Mid-run handoff to the Verifier — Symptom wants its delta checked early.",
        [(D_MOR, "#1f2937"), (D_SYM, "#1f2937")]),
    ("single", "verify_web", None,
        "Verifier consults the open web for each delta.",
        [(D_MOR, "#1f2937"), (D_SYM, "#1f2937")]),
    ("single", "verify_reject", "MOR",
        "Verifier rejects the Morphology delta — phrasing is under-specified.",
        [(D_MOR, "#b91c1c"), (D_SYM, "#1f2937")]),
    ("single", "flow_back", ("VERIFIER", "MOR"),
        "Hand the rejected delta back to Morphology with an explanation.",
        [(D_MOR, "#b91c1c"), (D_SYM, "#1f2937")]),
    ("single", "active_write", ("MOR", D_MOR2),
        "Morphology refines and re-submits.",
        [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937")]),
    ("single", "flow", ("MOR", "PAT"), "Handoff to Pathogen.",
        [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937")]),
    ("single", "active_write", ("PAT", D_PAT),
        "Pathogen reads the log and flags a likely look-alike.",
        [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937")]),
    ("single", "flow", ("PAT", "SEV"), "Handoff to Severity.",
        [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937")]),
    ("single", "active_write", ("SEV", D_SEV),
        "Severity scores lesion coverage and timing.",
        [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937"), (D_SEV, "#1f2937")]),
    ("single", "flow", ("SEV", "VERIFIER"), "Final handoff to the Verifier.",
        [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937"), (D_SEV, "#1f2937")]),
    ("single", "verify_web", None,
        "Verifier re-checks every delta against the web.",
        [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937"), (D_SEV, "#1f2937")]),
    ("single", "verify_pass", None,
        "All four deltas verified — each is tagged with the URLs that support it.",
        [(D_MOR2, "#15803d"), (D_SYM, "#15803d"), (D_PAT, "#15803d"), (D_SEV, "#15803d")]),
    ("single", "flow", ("VERIFIER", "CONS"), "Handoff to Consolidator.",
        [(D_MOR2, "#15803d"), (D_SYM, "#15803d"), (D_PAT, "#15803d"), (D_SEV, "#15803d")]),
    ("single", "consolidate", D_DUP,
        "Consolidator deduplicates and drops anything that restates canonical.",
        [(D_MOR2, "#15803d"), (D_SYM, "#15803d"), (D_PAT, "#15803d"), (D_SEV, "#15803d"),
         (D_DUP,  "#9ca3af")]),
    ("single", "flow", ("CONS", "OUTPUT"), "Emit the run's delta set.",
        [(D_MOR2, "#15803d"), (D_SYM, "#15803d"), (D_PAT, "#15803d"), (D_SEV, "#15803d")]),

    # ----- ACT 3 — agreement filter across N runs -----
    ("crossrun", "open", None,
        "That was run 1 of N. The whole swarm runs N times with different seeds.", None),
    ("crossrun", "fill", None,
        "Each run is independent: agent order, specialist routing, and generation are stochastic.", None),
    ("crossrun", "filter", None,
        "Cross-run agreement filter: keep only deltas that appear in at least K of N runs.", None),
    ("crossrun", "result", None,
        "Spurious per-run hallucinations are removed; the survivors are robust across seeds.", None),

    # ----- ACT 4 — merge + final output -----
    ("merge",  "show", None,
        "Conservative merge: add new deltas to the existing regional record without overwriting.", None),
    ("merge",  "json", None,
        "Final regional_observations entry for this state — each delta carries its support and provenance.", None),
]

FRAMES_PER_STAGE = 8        # → 8/6 ≈ 1.33 s per stage (slow enough to read)
FPS = 6
HOLD_FRAMES = FPS * 3       # ~3 s end hold

# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------
def ease(p):
    p = max(0.0, min(1.0, p))
    return 0.5 - 0.5 * math.cos(math.pi * p)

def _draw_node(ax, nid, active=False, dim=False, label_override=None):
    x, y, w, h = NODES[nid]
    if active:
        edge = EDGE[nid]; lw = 3.0; fill = FILL[nid]
        ax.add_patch(FancyBboxPatch(
            (x - w/2 - 0.012, y - h/2 - 0.012), w + 0.024, h + 0.024,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=0, facecolor=EDGE[nid], alpha=0.20, zorder=1,
        ))
    elif dim:
        edge = "#cbd5e1"; lw = 1.0; fill = "#f8fafc"
    else:
        edge = EDGE[nid]; lw = 1.2; fill = FILL[nid]
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2,
    ))
    lbl = label_override or LABELS[nid]
    ax.text(x, y, lbl, ha="center", va="center",
            fontsize=8.4, color="#0f172a" if not dim else "#94a3b8",
            zorder=3, fontweight="bold" if active else "normal")

def _draw_idle_edges(ax):
    pairs = [
        ("STATIC","TRIAGE"),
        ("TRIAGE","MOR"),("TRIAGE","SYM"),("TRIAGE","PAT"),("TRIAGE","SEV"),
        ("MOR","SYM"),("SYM","PAT"),("PAT","SEV"),
        ("MOR","VERIFIER"),("SYM","VERIFIER"),
        ("PAT","VERIFIER"),("SEV","VERIFIER"),
        ("VERIFIER","CONS"),("CONS","OUTPUT"),
    ]
    for s, d in pairs:
        sx, sy, _, _ = NODES[s]
        dx, dy, _, _ = NODES[d]
        ax.plot([sx, dx], [sy, dy], color="#e2e8f0", linewidth=1.0, zorder=1)

def _draw_arrow(ax, src, dst, progress=1.0, color="#0ea5e9", style="-"):
    sx, sy, _, _ = NODES[src]
    dx, dy, _, _ = NODES[dst]
    vec = np.array([dx - sx, dy - sy])
    L = np.linalg.norm(vec) or 1.0
    u = vec / L
    p1 = np.array([sx + u[0]*0.06, sy + u[1]*0.06])
    p2 = np.array([dx - u[0]*0.06, dy - u[1]*0.06])
    cur = p1 + (p2 - p1) * ease(progress)
    ax.add_patch(FancyArrowPatch(
        tuple(p1), tuple(cur),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=2.4, color=color, zorder=4, linestyle=style,
    ))

def _draw_log(ax, lines, title="Running delta log", fade_in_last=1.0):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor="#94a3b8", facecolor="#f8fafc", zorder=1,
    ))
    ax.text(0.5, 0.96, title, ha="center", va="top",
            fontsize=10.5, color="#0f172a", fontweight="bold")
    if not lines:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                fontsize=10, color="#94a3b8", style="italic")
        return
    y = 0.88
    for i, (text, color) in enumerate(lines):
        is_last = i == len(lines) - 1
        alpha = fade_in_last if is_last else 1.0
        ax.text(0.06, y, text, ha="left", va="top",
                fontsize=8.2, color=color, family="monospace",
                alpha=alpha, wrap=True)
        y -= 0.085

def _draw_web_panel(ax, alpha=1.0):
    """Small popup beside the verifier showing 'web search results'."""
    bx, by, bw, bh = 0.68, 0.50, 0.30, 0.18
    ax.add_patch(FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        linewidth=1.0, edgecolor="#fb923c", facecolor="#fff7ed",
        alpha=alpha, zorder=5,
    ))
    ax.text(bx + bw/2, by + bh - 0.018, "Web sources",
            ha="center", va="top", fontsize=8.5, color="#9a3412",
            fontweight="bold", alpha=alpha, zorder=6)
    y = by + bh - 0.05
    for url, snippet in WEB_SOURCES:
        ax.text(bx + 0.012, y, f"• {url}", ha="left", va="top",
                fontsize=7.4, color="#7c2d12", alpha=alpha, zorder=6)
        ax.text(bx + 0.024, y - 0.02, f"  \"{snippet[:48]}…\"",
                ha="left", va="top", fontsize=6.8, color="#9a3412",
                style="italic", alpha=alpha, zorder=6,
                family="monospace")
        y -= 0.05

# ----------------------------------------------------------------------
# Scene renderers
# ----------------------------------------------------------------------
def _scene_intro(ax_net, ax_log, kind, payload, frame, total_frames):
    progress = ease((frame + 1) / total_frames)
    ax_net.set_xlim(0, 1); ax_net.set_ylim(0, 1); ax_net.set_axis_off()
    if kind == "title":
        ax_net.text(0.5, 0.62, payload, ha="center", va="center",
                    fontsize=15, fontweight="bold", color="#0f172a",
                    alpha=progress)
        ax_net.text(0.5, 0.46,
                    "PathomeDB = Canonical (text) + Regional deltas (image-grounded)",
                    ha="center", va="center", fontsize=10.5, color="#475569",
                    alpha=progress)
        # Decorative phase strip.
        for i, (name, color) in enumerate([
            ("Phase 1\nCanonical KB", "#6366f1"),
            ("Phase 2\nRegional deltas (this animation)", "#ea580c"),
            ("Phase 3\nOBSERVE classifier", "#16a34a"),
        ]):
            x = 0.18 + i * 0.32
            ax_net.add_patch(FancyBboxPatch(
                (x - 0.13, 0.20), 0.26, 0.10,
                boxstyle="round,pad=0.01,rounding_size=0.012",
                linewidth=1.5, edgecolor=color,
                facecolor=color + "22", alpha=progress, zorder=2,
            ))
            ax_net.text(x, 0.25, name, ha="center", va="center",
                        fontsize=8.4, color=color, fontweight="bold",
                        alpha=progress)
        _draw_log(ax_log, [], title="Running delta log")
        return

    if kind == "context":
        # Show the static-context box, then a canonical-block summary
        # on the left and a photo placeholder on the right (non-overlapping).
        _draw_node(ax_net, "STATIC", active=True)
        ax_net.text(0.05, 0.78,
                    "Canonical block (text-grounded)",
                    ha="left", va="center", fontsize=9.6,
                    color="#3730a3", fontweight="bold", alpha=progress)
        cb_lines = [
            "pathogen:   Macrophomina phaseolina",
            "type:       Fungal",
            "symptoms:   microsclerotia in stem",
            "look-alikes: Sudden Death Syndrome",
            "parts:      Stem, Root, Foliar",
        ]
        for i, ln in enumerate(cb_lines):
            ax_net.text(0.05, 0.72 - i*0.045, ln,
                        ha="left", va="center", fontsize=8.0,
                        family="monospace", color="#1f2937", alpha=progress)
        # Photo placeholder on the right (well clear of canonical text).
        ax_net.add_patch(FancyBboxPatch(
            (0.62, 0.46), 0.34, 0.26,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=1.5, edgecolor="#475569", facecolor="#e5e7eb",
            alpha=progress, zorder=2,
        ))
        ax_net.text(0.79, 0.62, "field photograph", ha="center", va="center",
                    fontsize=9.5, color="#374151", alpha=progress,
                    fontweight="bold")
        ax_net.text(0.79, 0.56, "Alabama, Soybean", ha="center", va="center",
                    fontsize=8.2, color="#6b7280", alpha=progress)
        _draw_log(ax_log, [], title="Running delta log")
        return

    if kind == "agents":
        # Show the line-up of all agents with short owned-field tags.
        _draw_idle_edges(ax_net)
        for nid in NODES:
            _draw_node(ax_net, nid, active=False)
        # Two-line owned-field tags, kept short so they don't overlap.
        owned = {
            "MOR": "lesion morph,\naffected organs",
            "SYM": "spread pattern,\ndiagnostic feats",
            "PAT": "look-alikes,\ndisease type",
            "SEV": "severity,\ntreatments",
        }
        for nid, txt in owned.items():
            x, y, _, _ = NODES[nid]
            ax_net.text(x, y - 0.055, txt,
                        ha="center", va="top", fontsize=6.6,
                        color="#475569", style="italic", alpha=progress,
                        linespacing=1.1)
        _draw_log(ax_log, [], title="Running delta log")
        return


def _scene_single(ax_net, ax_log, kind, payload, frame, total_frames,
                  log_lines):
    progress = (frame + 1) / total_frames
    ax_net.set_xlim(0, 1); ax_net.set_ylim(0, 1); ax_net.set_axis_off()
    _draw_idle_edges(ax_net)

    active = set()
    flow = None
    flow_color = "#0ea5e9"
    flow_style = "-"
    web_alpha = 0.0

    if kind == "idle":
        pass
    elif kind == "active":
        active.add(payload)
    elif kind == "flow":
        flow = payload
    elif kind == "flow_back":
        flow = payload
        flow_color = "#dc2626"
        flow_style = "--"
    elif kind == "active_write":
        active.add(payload[0])
    elif kind == "verify_web":
        active.add("VERIFIER")
        web_alpha = ease(progress)
    elif kind == "verify_reject":
        active.add("VERIFIER")
        web_alpha = 1.0
    elif kind == "verify_pass":
        active.add("VERIFIER")
        web_alpha = 1.0
    elif kind == "consolidate":
        active.add("CONS")

    for nid in NODES:
        _draw_node(ax_net, nid, active=(nid in active))

    if flow is not None:
        _draw_arrow(ax_net, flow[0], flow[1],
                    progress=ease(progress),
                    color=flow_color, style=flow_style)

    if web_alpha > 0:
        _draw_web_panel(ax_net, alpha=web_alpha)

    # log: fade-in last line during 'active_write' / 'consolidate'
    fade_last = ease(progress) if kind in ("active_write", "consolidate") else 1.0
    _draw_log(ax_log, log_lines or [], fade_in_last=fade_last)


def _scene_crossrun(ax_net, ax_log, kind, payload, frame, total_frames):
    progress = ease((frame + 1) / total_frames)
    ax_net.set_xlim(0, 1); ax_net.set_ylim(0, 1); ax_net.set_axis_off()
    ax_net.text(0.5, 0.95, "Repeat the whole swarm N=5 times with different seeds",
                ha="center", va="top", fontsize=11, color="#0f172a",
                fontweight="bold")

    # 5 mini-run boxes arranged horizontally.
    box_y, box_h = 0.30, 0.55
    box_w = 0.16
    gap   = (1.0 - 5*box_w) / 6.0
    fields = ["mor", "sym", "pat", "sev"]
    field_color = {"mor": "#2563eb", "sym": "#059669",
                   "pat": "#7c3aed", "sev": "#dc2626"}
    survival = {f: 0 for f in fields}

    if kind == "open":
        n_show = 1
    elif kind == "fill":
        n_show = min(5, 1 + int(progress * 5))
    else:
        n_show = 5

    for i in range(5):
        x = gap + i * (box_w + gap)
        alpha = 1.0 if (i + 1) <= n_show else 0.15
        ax_net.add_patch(FancyBboxPatch(
            (x, box_y), box_w, box_h,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=1.4, edgecolor="#475569",
            facecolor="#f8fafc", alpha=alpha, zorder=2,
        ))
        ax_net.text(x + box_w/2, box_y + box_h - 0.03, f"run {i+1}",
                    ha="center", va="top", fontsize=9.2,
                    color="#0f172a", alpha=alpha, fontweight="bold")
        # Tick / dash for each field.
        for j, f in enumerate(fields):
            ty = box_y + box_h - 0.10 - j*0.10
            present = RUNS[i][f]
            if present:
                survival[f] += 1 if (i + 1) <= n_show else 0
            mark   = "✓" if present else "—"
            color  = field_color[f] if present else "#9ca3af"
            ax_net.text(x + 0.025, ty, mark,
                        ha="left", va="top", fontsize=12,
                        color=color, alpha=alpha, fontweight="bold")
            ax_net.text(x + 0.060, ty, f,
                        ha="left", va="top", fontsize=8.2,
                        color="#1f2937" if present else "#9ca3af",
                        alpha=alpha, family="monospace")

    # Filter / result band at the bottom of the network panel.
    band_y = 0.12
    show_filter = kind in ("filter", "result")
    show_result = kind == "result"

    if show_filter:
        ax_net.add_patch(FancyBboxPatch(
            (0.04, band_y), 0.92, 0.12,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=1.4, edgecolor="#0ea5e9", facecolor="#ecfeff",
            alpha=progress if kind == "filter" else 1.0, zorder=3,
        ))
        ax_net.text(0.06, band_y + 0.08,
                    "K-of-N filter  (K = 3)",
                    ha="left", va="center", fontsize=9.5,
                    color="#0369a1", fontweight="bold",
                    alpha=progress if kind == "filter" else 1.0)
        xs = [0.32, 0.50, 0.68, 0.86]
        for j, f in enumerate(fields):
            ax_net.text(xs[j], band_y + 0.08,
                        f"{f}: {survival[f]}/5",
                        ha="center", va="center", fontsize=9.0,
                        color=field_color[f],
                        alpha=progress if kind == "filter" else 1.0)
            verdict = "kept" if survival[f] >= 3 else "dropped"
            vcolor  = "#15803d" if verdict == "kept" else "#b91c1c"
            if show_result:
                ax_net.text(xs[j], band_y + 0.03, verdict,
                            ha="center", va="center", fontsize=8.6,
                            color=vcolor, fontweight="bold")

    if show_result:
        survivors = [
            (D_MOR2, "#15803d"),
            (D_SYM,  "#15803d"),
            (D_PAT,  "#15803d"),
            (D_SEV,  "#15803d"),
        ]
        _draw_log(ax_log, survivors, title="Cross-run survivors")
    else:
        _draw_log(ax_log, [], title="Cross-run survivors")


def _scene_merge(ax_net, ax_log, kind, payload, frame, total_frames):
    progress = ease((frame + 1) / total_frames)
    ax_net.set_xlim(0, 1); ax_net.set_ylim(0, 1); ax_net.set_axis_off()
    if kind == "show":
        # Diagram: new survivors (left) + existing KB (right) -> merged
        ax_net.text(0.5, 0.95, "Conservative merge into existing KB",
                    ha="center", va="top", fontsize=11.5,
                    color="#0f172a", fontweight="bold")
        # Left: new survivors
        ax_net.add_patch(FancyBboxPatch(
            (0.05, 0.40), 0.28, 0.45,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=1.4, edgecolor="#15803d", facecolor="#dcfce7",
            zorder=2,
        ))
        ax_net.text(0.19, 0.81, "New survivors\n(this Phase 2 run)",
                    ha="center", va="top", fontsize=9.0,
                    color="#14532d", fontweight="bold")
        for j, t in enumerate(["mor", "sym", "pat", "sev"]):
            ax_net.text(0.07, 0.71 - j*0.06, f"• {t}",
                        ha="left", va="center", fontsize=8.4,
                        color="#1f2937", family="monospace")
        # Right: existing KB
        ax_net.add_patch(FancyBboxPatch(
            (0.67, 0.40), 0.28, 0.45,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=1.4, edgecolor="#6366f1", facecolor="#e0e7ff",
            zorder=2,
        ))
        ax_net.text(0.81, 0.81, "Existing KB entry\n(prior Phase 2 runs)",
                    ha="center", va="top", fontsize=9.0,
                    color="#312e81", fontweight="bold")
        for j, t in enumerate(["mor (support=3)", "pat (support=2)"]):
            ax_net.text(0.69, 0.71 - j*0.06, f"• {t}",
                        ha="left", va="center", fontsize=8.4,
                        color="#1f2937", family="monospace")
        # Arrows to merged
        for sx in (0.33, 0.67):
            ax_net.add_patch(FancyArrowPatch(
                (sx, 0.50), (0.50, 0.30),
                arrowstyle="-|>", mutation_scale=12,
                linewidth=2.0, color="#0f172a", zorder=3,
            ))
        ax_net.add_patch(FancyBboxPatch(
            (0.30, 0.06), 0.40, 0.26,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=1.6, edgecolor="#0f172a", facecolor="#ffffff",
            zorder=4,
        ))
        ax_net.text(0.50, 0.28, "Merged regional_observations",
                    ha="center", va="top", fontsize=9.6,
                    color="#0f172a", fontweight="bold", zorder=6)
        rules = [
            "• overlap         → support++",
            "• new field       → appended",
            "• existing fields → never overwritten",
        ]
        for i, r in enumerate(rules):
            ax_net.text(0.32, 0.22 - i*0.045, r,
                        ha="left", va="top", fontsize=7.8,
                        color="#334155", family="monospace", zorder=6)
        _draw_log(ax_log, [], title="Merged delta record")
        return

    if kind == "json":
        ax_net.text(0.5, 0.95, "Final output for one (crop, disease, state)",
                    ha="center", va="top", fontsize=11.5,
                    color="#0f172a", fontweight="bold")
        json_lines = [
            "\"regional_observations\": {",
            "  \"Alabama\": {",
            "    \"deltas\": [",
            "      {",
            "        \"field\":              \"lesion_morphology\",",
            "        \"canonical_says\":     \"(not specified)\",",
            "        \"image_shows\":        \"chlorotic halos surround dark sunken lesions\",",
            "        \"image_quote\":        \"...\",",
            "        \"image_evidence_id\":  \"bugwood::1568038\",",
            "        \"swarm_support\":      4,         // appeared in 4 of 5 runs",
            "        \"verification_status\": \"verified\",",
            "        \"web_support\": [",
            "          { \"url\": \"https://extension.umn.edu/...\", \"quote\": \"...\" },",
            "          { \"url\": \"https://apsnet.org/...\",       \"quote\": \"...\" }",
            "        ],",
            "        \"handoff_provenance\": [\"morphology\", \"verifier\",",
            "                                \"morphology\", \"verifier\"]",
            "      },",
            "      ... (3 more deltas)",
            "    ]",
            "  }",
            "}",
        ]
        y = 0.86
        for ln in json_lines:
            ax_net.text(0.04, y, ln, ha="left", va="top",
                        fontsize=7.6, family="monospace", color="#1f2937")
            y -= 0.034
        _draw_log(ax_log, [], title="Phase 2 complete")


# ----------------------------------------------------------------------
# Frame composer
# ----------------------------------------------------------------------
def render_frame(stage_idx, frame_in_stage, total_stages):
    fig = plt.figure(figsize=(13.5, 7.0), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax_top = fig.add_axes([0.02, 0.93, 0.96, 0.06]); ax_top.set_axis_off()
    ax_top.text(0.5, 0.5, "Phase 2 — handoff swarm: from canonical KB + field photo to verified regional deltas",
                ha="center", va="center", fontsize=12.5,
                fontweight="bold", color="#0f172a")
    ax_net = fig.add_axes([0.02, 0.12, LEFT_W - 0.04, 0.80])
    ax_log = fig.add_axes([LEFT_W + 0.01, 0.12, 1 - LEFT_W - 0.03, 0.80])
    ax_cap = fig.add_axes([0.02, 0.01, 0.96, 0.10]); ax_cap.set_axis_off()

    stage = STAGES[stage_idx]
    scene, kind, payload, caption = stage[0], stage[1], stage[2], stage[3]
    log_lines = stage[4] if len(stage) > 4 else None

    if scene == "intro":
        _scene_intro(ax_net, ax_log, kind, payload,
                     frame_in_stage, FRAMES_PER_STAGE)
    elif scene == "single":
        _scene_single(ax_net, ax_log, kind, payload,
                      frame_in_stage, FRAMES_PER_STAGE,
                      log_lines=log_lines or [])
    elif scene == "crossrun":
        _scene_crossrun(ax_net, ax_log, kind, payload,
                        frame_in_stage, FRAMES_PER_STAGE)
    elif scene == "merge":
        _scene_merge(ax_net, ax_log, kind, payload,
                     frame_in_stage, FRAMES_PER_STAGE)

    progress_label = f"Step {stage_idx+1}/{total_stages}  ·  Act {' '.join(['I','II','III','IV'][:1] if scene == 'intro' else (['II'] if scene == 'single' else (['III'] if scene == 'crossrun' else ['IV'])))}"
    ax_cap.text(0.01, 0.5, progress_label, ha="left", va="center",
                fontsize=9.5, color="#475569", fontweight="bold")
    ax_cap.text(0.5, 0.5, caption or "", ha="center", va="center",
                fontsize=11, color="#0f172a")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba).convert("RGB")


def main():
    frames = []
    total = len(STAGES)
    for s in range(total):
        for f in range(FRAMES_PER_STAGE):
            frames.append(render_frame(s, f, total))
    for _ in range(HOLD_FRAMES):
        frames.append(frames[-1])
    duration_ms = int(1000 / FPS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True, disposal=2,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024 / 1024:.2f} MB, "
          f"{len(frames)} frames, {len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
