"""
pathome/
========
PathomeDB: a five-layer mechanistic cross-crop pathogen knowledge base.

Paper §6 (pathome_final). Each layer is a distinct, independently-versioned
component:

  Layer 1 — Mechanistic infection pathway (per pathogen, crop-agnostic)
  Layer 2 — Cross-crop manifestation (host-specific lesion / sporulation maps)
  Layer 3 — Geospatially-grounded regional epidemiology (P̂(d|r,σ) from
            Bugwood GPS density; validated against EPPO at r ≥ 0.71)
  Layer 4 — Diagnostic decision graph (NetworkX DiGraph)
  Layer 5 — Geo-tagged reference library (78 Bugwood references, FAISS retrieval)

The top-level ``PathomeDB`` orchestrator wires the layers together and is
the object passed to ``PlantSwarmPipeline`` and ``OBSERVE`` at construction.
"""

from .database import PathomeDB
from .layer1_pathway import MechanisticPathway, PathwayStep
from .layer2_manifestation import CrossCropManifestation
from .layer3_geo import RegionalEpidemiology
from .layer4_decision_graph import DiagnosticDecisionGraph
from .layer5_references import ReferenceLibrary

__all__ = [
    "PathomeDB",
    "MechanisticPathway",
    "PathwayStep",
    "CrossCropManifestation",
    "RegionalEpidemiology",
    "DiagnosticDecisionGraph",
    "ReferenceLibrary",
]
