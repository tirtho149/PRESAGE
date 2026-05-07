"""
pathome/layer4_decision_graph.py
================================
Layer 4: Diagnostic decision graph (paper §6.4).

A NetworkX DiGraph with six mandatory branch axes traversed in order:
  host → plant_part → symptom_category → progression → pathogen_signs → environment

Angular vs. circular lesion margin is the single most informative visual
feature separating bacterial from fungal etiology and acts as a hard branch
condition. Terminal conditions enable early routing (e.g. orange sporulation
is definitive for Colletotrichum regardless of host).

Each graph node carries:
  - responsible agent
  - visual features required for branch resolution
  - posterior updates
  - targeted re-observation instruction injected into C_{t+1} on backtrack
    (paper §6.4: "examine lesion margins for orange or salmon masses
    specifically" rather than a generic retry)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DGNode:
    node_id: str
    axis: str                         # one of: host, plant_part, symptom_category,
                                      # progression, pathogen_signs, environment
    responsible_agent: str            # MorphologyAgent | SymptomAgent | ...
    required_features: List[str] = field(default_factory=list)
    posterior_update: Dict[str, float] = field(default_factory=dict)  # disease → log-odds delta
    reobservation_prompt: str = ""    # injected into context buffer on backtrack
    terminal_disease: Optional[str] = None   # if set, immediate diagnosis
    children: List[str] = field(default_factory=list)  # outgoing edges (node_ids)


class DiagnosticDecisionGraph:
    """Layer 4 graph. Wraps NetworkX when present, falls back to dict otherwise."""

    BRANCH_ORDER = [
        "host",
        "plant_part",
        "symptom_category",
        "progression",
        "pathogen_signs",
        "environment",
    ]

    def __init__(self):
        self._nodes: Dict[str, DGNode] = {}
        self._root: Optional[str] = None
        try:
            import networkx as nx  # type: ignore
            self._G = nx.DiGraph()
        except ImportError:
            self._G = None

    # ------------------------------------------------------------------

    def add_node(self, node: DGNode, parent: Optional[str] = None) -> None:
        if node.axis not in self.BRANCH_ORDER:
            raise ValueError(f"unknown axis {node.axis!r}")
        self._nodes[node.node_id] = node
        if self._G is not None:
            self._G.add_node(node.node_id, **node.__dict__)
        if parent is None:
            if self._root is None:
                self._root = node.node_id
        else:
            if parent not in self._nodes:
                raise KeyError(f"parent {parent!r} not in graph")
            self._nodes[parent].children.append(node.node_id)
            if self._G is not None:
                self._G.add_edge(parent, node.node_id)

    def get(self, node_id: str) -> Optional[DGNode]:
        return self._nodes.get(node_id)

    def root(self) -> Optional[DGNode]:
        return self._nodes.get(self._root) if self._root else None

    def children_of(self, node_id: str) -> List[DGNode]:
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[c] for c in node.children if c in self._nodes]

    # ------------------------------------------------------------------
    # Routing helpers consumed by OBSERVE
    # ------------------------------------------------------------------

    def reobservation_for(self, node_id: str) -> str:
        """Targeted re-observation prompt for OBSERVE-driven backtracks."""
        n = self._nodes.get(node_id)
        return n.reobservation_prompt if n else ""

    def terminal_diagnosis(self, node_id: str) -> Optional[str]:
        n = self._nodes.get(node_id)
        return n.terminal_disease if n else None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "root": self._root,
            "nodes": [n.__dict__ for n in self._nodes.values()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DiagnosticDecisionGraph":
        with open(path) as f:
            data = json.load(f)
        g = cls()
        # First pass: instantiate nodes without parent linkage
        for n in data.get("nodes", []):
            children = list(n.pop("children", []))
            node = DGNode(**n)
            g._nodes[node.node_id] = node
            if g._G is not None:
                g._G.add_node(node.node_id, **node.__dict__)
            node.children = children
        # Second pass: build edges in graph
        if g._G is not None:
            for n in g._nodes.values():
                for c in n.children:
                    g._G.add_edge(n.node_id, c)
        g._root = data.get("root")
        return g


# ---------------------------------------------------------------------------
# Worked-example builder for a small subgraph (paper §6.4 example)
# ---------------------------------------------------------------------------

def build_demo_graph() -> DiagnosticDecisionGraph:
    """Tiny graph demonstrating the angular-vs-circular branch."""
    g = DiagnosticDecisionGraph()
    g.add_node(DGNode(
        node_id="root_host",
        axis="host",
        responsible_agent="SeverityAgent",
        required_features=["crop species"],
    ))
    g.add_node(DGNode(
        node_id="leaf_part",
        axis="plant_part",
        responsible_agent="MorphologyAgent",
        required_features=["primary tissue affected"],
        reobservation_prompt="Identify whether symptoms are on leaf, stem, fruit, or root.",
    ), parent="root_host")
    g.add_node(DGNode(
        node_id="lesion_margin",
        axis="symptom_category",
        responsible_agent="MorphologyAgent",
        required_features=["lesion margin shape"],
        reobservation_prompt=(
            "Examine lesion margin closely: angular (vein-bounded, suggesting "
            "bacterial) or circular (radial, suggesting fungal)."
        ),
    ), parent="leaf_part")
    g.add_node(DGNode(
        node_id="orange_sporulation",
        axis="pathogen_signs",
        responsible_agent="PathogenAgent",
        required_features=["orange/salmon spore mass on lesion centre"],
        reobservation_prompt=(
            "Look for orange or salmon masses at lesion centre — diagnostic of "
            "Colletotrichum acervuli regardless of host."
        ),
        terminal_disease="Anthracnose",
    ), parent="lesion_margin")
    return g
