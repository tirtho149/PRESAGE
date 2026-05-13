"""
docs/assets/render_swarm_flow.py
================================
Render an animated GIF that visualises the handoff-swarm flow:

  Triage -> first specialist -> next specialist -> ... -> Verifier
  (Verifier can hand a rejected delta back to a specialist for refinement)
  -> Consolidator -> Regional deltas.

The animation is a sequence of stages. Each stage either:
  - highlights one agent ("active"), or
  - shows a handoff arrow flowing from one agent to another, or
  - shows the verifier doing a web-search check, or
  - shows the consolidator emitting the final output.

A running delta-log panel on the right grows line by line as each
specialist writes. The rejected line briefly turns red when the
verifier rejects it, then re-appears in green when refined.

Output: docs/assets/swarm_flow.gif
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import io

OUT = Path(__file__).resolve().parent / "swarm_flow.gif"

# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
# Coordinates are in [0, 1] x [0, 1] within the network panel.
LEFT_W = 0.62   # left panel (network) width fraction
NODES = {
    # node_id: (x, y, width, height)
    "STATIC":   (0.50, 0.92, 0.42, 0.10),
    "TRIAGE":   (0.50, 0.77, 0.20, 0.07),
    "MOR":      (0.13, 0.60, 0.18, 0.07),
    "SYM":      (0.36, 0.60, 0.18, 0.07),
    "PAT":      (0.59, 0.60, 0.18, 0.07),
    "SEV":      (0.82, 0.60, 0.18, 0.07),
    "VERIFIER": (0.50, 0.40, 0.30, 0.08),
    "CONS":     (0.50, 0.22, 0.30, 0.07),
    "OUTPUT":   (0.50, 0.07, 0.42, 0.07),
}
LABELS = {
    "STATIC":   "Static context\ncanonical KB + field photo",
    "TRIAGE":   "Triage",
    "MOR":      "Morphology",
    "SYM":      "Symptom",
    "PAT":      "Pathogen",
    "SEV":      "Severity",
    "VERIFIER": "Verifier\n(Claude headless + web search)",
    "CONS":     "Consolidator",
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


# -------------------------------------------------------------------
# Story (sequence of stages)
# -------------------------------------------------------------------
# Each stage is one of:
#   ("active", node_id, caption)
#   ("flow",   (src, dst), caption, color)
#   ("verify_web", caption)         (verifier "running web search" pulse)
#   ("output_pulse", caption)
#
# After each stage we also pass the current state of the log
# (list of (text, color)) — we duplicate that in the stages list
# to make it explicit, but actually we'll build it programmatically.

# Each scripted delta
D_MOR  = "[Morphology] yellow halos around dark sunken lesions"
D_MOR2 = "[Morphology] (refined) chlorotic halos surround dark sunken lesions"
D_SYM  = "[Symptom] lesions concentrate near the leaf margin"
D_PAT  = "[Pathogen] similar to Septoria; distinguish by halo colour"
D_SEV  = "[Severity] ~30% canopy affected; mid-late season"

# Stage list. Each entry: (kind, payload, caption, log_after)
STAGES = [
    ("idle",  None, "Static context: canonical KB block + field photograph for this state.", []),
    ("active","TRIAGE","Triage agent inspects the photograph and picks the first specialist.", []),
    ("flow",  ("TRIAGE","MOR"), "Handoff -> Morphology", []),
    ("active","MOR","Morphology writes the first delta into the running log.",
                  [(D_MOR, "#1f2937")]),
    ("flow",  ("MOR","SYM"), "Handoff -> Symptom (it now reads Morphology's delta)",
                  [(D_MOR, "#1f2937")]),
    ("active","SYM","Symptom reads the log, adds a complementary observation.",
                  [(D_MOR, "#1f2937"), (D_SYM, "#1f2937")]),
    ("flow",  ("SYM","VERIFIER"), "Mid-run handoff to the Verifier",
                  [(D_MOR, "#1f2937"), (D_SYM, "#1f2937")]),
    ("verify_web", None, "Verifier consults the open web for each delta.",
                  [(D_MOR, "#1f2937"), (D_SYM, "#1f2937")]),
    ("verify_reject", "MOR", "Verifier rejects the Morphology delta (under-specified).",
                  [(D_MOR, "#b91c1c"), (D_SYM, "#1f2937")]),
    ("flow",  ("VERIFIER","MOR"), "Hand back to Morphology with explanation",
                  [(D_MOR, "#b91c1c"), (D_SYM, "#1f2937")], "#dc2626"),
    ("active","MOR","Morphology refines its delta using the feedback.",
                  [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937")]),
    ("flow",  ("MOR","PAT"), "Handoff -> Pathogen",
                  [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937")]),
    ("active","PAT","Pathogen reads the log, flags a likely look-alike.",
                  [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937")]),
    ("flow",  ("PAT","SEV"), "Handoff -> Severity",
                  [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937")]),
    ("active","SEV","Severity scores the lesion coverage and timing.",
                  [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937"), (D_SEV, "#1f2937")]),
    ("flow",  ("SEV","VERIFIER"), "Final handoff to the Verifier",
                  [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937"), (D_SEV, "#1f2937")]),
    ("verify_web", None, "Verifier re-checks every delta against the web.",
                  [(D_MOR2, "#1f2937"), (D_SYM, "#1f2937"), (D_PAT, "#1f2937"), (D_SEV, "#1f2937")]),
    ("verify_pass", None, "All four deltas verified.",
                  [(D_MOR2, "#15803d"), (D_SYM, "#15803d"), (D_PAT, "#15803d"), (D_SEV, "#15803d")]),
    ("flow",  ("VERIFIER","CONS"), "Handoff -> Consolidator",
                  [(D_MOR2, "#15803d"), (D_SYM, "#15803d"), (D_PAT, "#15803d"), (D_SEV, "#15803d")]),
    ("active","CONS","Consolidator deduplicates and drops canonical restatements.",
                  [(D_MOR2, "#15803d"), (D_SYM, "#15803d"), (D_PAT, "#15803d"), (D_SEV, "#15803d")]),
    ("flow",  ("CONS","OUTPUT"), "Emit the final delta set",
                  [(D_MOR2, "#15803d"), (D_SYM, "#15803d"), (D_PAT, "#15803d"), (D_SEV, "#15803d")]),
    ("output_pulse", None, "Run complete: 4 verified regional deltas for this state.",
                  [(D_MOR2, "#15803d"), (D_SYM, "#15803d"), (D_PAT, "#15803d"), (D_SEV, "#15803d")]),
]

FRAMES_PER_STAGE = 6        # = 6/12 fps = 0.5s per stage
FPS = 12

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _draw_node(ax, nid, active=False, dim=False):
    x, y, w, h = NODES[nid]
    if active:
        edge = EDGE[nid]; lw = 3.0; fill = FILL[nid]; alpha = 1.0
        # halo
        halo = FancyBboxPatch(
            (x - w/2 - 0.012, y - h/2 - 0.012), w + 0.024, h + 0.024,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            linewidth=0, facecolor=EDGE[nid], alpha=0.20, zorder=1,
        )
        ax.add_patch(halo)
    elif dim:
        edge = "#cbd5e1"; lw = 1.0; fill = "#f8fafc"; alpha = 0.6
    else:
        edge = EDGE[nid]; lw = 1.2; fill = FILL[nid]; alpha = 0.95
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=lw, edgecolor=edge, facecolor=fill, alpha=alpha, zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x, y, LABELS[nid], ha="center", va="center",
        fontsize=8.6 if "\n" in LABELS[nid] else 9.2,
        color="#0f172a" if not dim else "#94a3b8",
        zorder=3, fontweight="bold" if active else "normal",
    )

def _draw_arrow(ax, src, dst, progress=1.0, color="#475569", style="-"):
    sx, sy, sw, sh = NODES[src]
    dx, dy, dw, dh = NODES[dst]
    # Approximate edge points: shorten line so it ends at the box edge.
    vec = np.array([dx - sx, dy - sy])
    L = np.linalg.norm(vec) or 1.0
    u = vec / L
    p1 = np.array([sx + u[0]*0.06, sy + u[1]*0.06])
    p2 = np.array([dx - u[0]*0.06, dy - u[1]*0.06])
    cur = p1 + (p2 - p1) * np.clip(progress, 0, 1)
    arrow = FancyArrowPatch(
        tuple(p1), tuple(cur),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=2.2, color=color, zorder=4,
        linestyle=style,
    )
    ax.add_patch(arrow)

def _draw_idle_edges(ax):
    # Show the static handoff topology faintly (always visible).
    pairs = [
        ("STATIC","TRIAGE"),
        ("TRIAGE","MOR"),("TRIAGE","SYM"),("TRIAGE","PAT"),("TRIAGE","SEV"),
        ("MOR","SYM"),("SYM","PAT"),("PAT","SEV"),
        ("MOR","VERIFIER"),("SYM","VERIFIER"),
        ("PAT","VERIFIER"),("SEV","VERIFIER"),
        ("VERIFIER","CONS"),
        ("CONS","OUTPUT"),
    ]
    for s, d in pairs:
        sx, sy, _, _ = NODES[s]
        dx, dy, _, _ = NODES[d]
        ax.plot([sx, dx], [sy, dy], color="#e2e8f0", linewidth=1.0, zorder=1)

def _draw_log(ax, lines, title="Running delta log"):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    bg = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor="#94a3b8", facecolor="#f8fafc", zorder=1,
    )
    ax.add_patch(bg)
    ax.text(0.5, 0.96, title, ha="center", va="top",
            fontsize=11, color="#0f172a", fontweight="bold")
    if not lines:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                fontsize=10, color="#94a3b8", style="italic")
        return
    y = 0.88
    for text, color in lines:
        ax.text(0.06, y, text, ha="left", va="top",
                fontsize=8.6, color=color, family="monospace",
                wrap=True)
        y -= 0.10

def _stage_payload(stage):
    """Return (kind, payload, caption, log, extra)."""
    if len(stage) == 4:
        return (*stage, None)
    if len(stage) == 5:
        return stage
    return (*stage, None)

# -------------------------------------------------------------------
# Frame renderer
# -------------------------------------------------------------------
def render_frame(stage_idx, frame_in_stage, total_stages):
    fig = plt.figure(figsize=(11, 6.4), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    # Title strip at top so set_title doesn't clip.
    ax_top = fig.add_axes([0.02, 0.93, 0.96, 0.06]); ax_top.set_axis_off()
    ax_top.text(0.5, 0.5, "Phase 2 — handoff swarm",
                ha="center", va="center",
                fontsize=14, fontweight="bold", color="#0f172a")
    ax_net = fig.add_axes([0.02, 0.12, LEFT_W - 0.04, 0.80])
    ax_log = fig.add_axes([LEFT_W + 0.01, 0.12, 1 - LEFT_W - 0.03, 0.80])
    ax_cap = fig.add_axes([0.02, 0.01, 0.96, 0.10]); ax_cap.set_axis_off()

    ax_net.set_xlim(0, 1); ax_net.set_ylim(0, 1); ax_net.set_axis_off()

    _draw_idle_edges(ax_net)

    kind, payload, caption, log, extra = _stage_payload(STAGES[stage_idx])
    progress = (frame_in_stage + 1) / FRAMES_PER_STAGE
    # ease-in-out
    progress = 0.5 - 0.5 * math.cos(math.pi * progress)

    active = set()
    flow   = None
    flow_color = "#0ea5e9"
    flow_style = "-"
    verify_pulse = 0.0

    if kind == "idle":
        pass
    elif kind == "active":
        active.add(payload)
    elif kind == "flow":
        flow = payload
        if extra is not None:
            flow_color = extra
            flow_style = "--"
    elif kind == "verify_web":
        active.add("VERIFIER")
        verify_pulse = progress
    elif kind == "verify_reject":
        active.add("VERIFIER")
    elif kind == "verify_pass":
        active.add("VERIFIER")
    elif kind == "output_pulse":
        active.add("OUTPUT")

    # Draw all nodes — active highlighted, others normal.
    for nid in NODES:
        _draw_node(ax_net, nid, active=(nid in active), dim=False)

    # Draw the current flow arrow on top.
    if flow is not None:
        _draw_arrow(ax_net, flow[0], flow[1], progress=progress,
                    color=flow_color, style=flow_style)

    # If verifier is web-searching, draw small spinner-dots near it.
    if kind == "verify_web":
        x, y, _, h = NODES["VERIFIER"]
        for k in range(3):
            a = 0.3 + 0.7 * ((math.sin(progress*6 + k*1.3) + 1) / 2)
            ax_net.plot(
                x + 0.10 + 0.025*k, y + h/2 + 0.03,
                "o", color="#ea580c", alpha=a, markersize=6,
            )

    # Draw the log panel.
    _draw_log(ax_log, log or [])

    # Caption strip
    progress_label = f"Step {stage_idx+1}/{total_stages}"
    ax_cap.text(0.01, 0.5, progress_label, ha="left", va="center",
                fontsize=10, color="#475569", fontweight="bold")
    ax_cap.text(0.5, 0.5, caption or "", ha="center", va="center",
                fontsize=11, color="#0f172a")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba).convert("RGB")


def main():
    frames = []
    for s in range(len(STAGES)):
        for f in range(FRAMES_PER_STAGE):
            frames.append(render_frame(s, f, len(STAGES)))
    # Hold the final frame a bit longer.
    for _ in range(FPS * 2):
        frames.append(frames[-1])
    duration_ms = int(1000 / FPS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True,
        disposal=2,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024 / 1024:.2f} MB, "
          f"{len(frames)} frames, {len(frames)/FPS:.1f}s)")


if __name__ == "__main__":
    main()
