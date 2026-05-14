"""
scripts/viz/_common.py
======================
Shared utilities for visualization scripts.

- Output paths: PNG figures to results/figures/, LaTeX snippets to
  paper/auto_<name>.tex.
- Soft matplotlib import — viz scripts degrade gracefully to text-only
  LaTeX tables if matplotlib isn't installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR  = ROOT / "results" / "figures"
TEX_DIR  = ROOT / "plantswarm" / "latex"


def have_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def get_mpl():
    """Lazy matplotlib import — returns (matplotlib, pyplot) or (None, None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")     # headless safe
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError:
        return None, None


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)


def fig_path(name: str) -> Path:
    return FIG_DIR / f"{name}.png"


def tex_path(name: str) -> Path:
    return TEX_DIR / f"auto_{name}.tex"


def write_tex(name: str, content: str) -> Path:
    """Write a LaTeX snippet that the paper can ``\\input{auto_<name>}``."""
    ensure_dirs()
    p = tex_path(name)
    p.write_text(content)
    return p


def figure_includegraphics(
    name: str, caption: str, label: str,
    width: str = r"\linewidth",
) -> str:
    """Return a \\begin{figure} ... \\end{figure} block that includes
    the PNG generated for ``name``."""
    return (
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width={width}]{{figures/{name}.png}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{label}}}\n"
        "\\end{figure}\n"
    )


def latex_escape(s: str) -> str:
    """Minimal LaTeX-safe escape for table cells."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&",  r"\&")
        .replace("%",  r"\%")
        .replace("$",  r"\$")
        .replace("#",  r"\#")
        .replace("_",  r"\_")
        .replace("{",  r"\{")
        .replace("}",  r"\}")
        .replace("~",  r"\textasciitilde{}")
        .replace("^",  r"\textasciicircum{}")
    )
