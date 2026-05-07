"""
pathome/database.py
===================
Top-level orchestrator for the five PathomeDB layers (paper §6).

Used by:
  * ``PlantSwarmPipeline`` / ``AutoGenPlantSwarmPipeline`` — agents consult
    Layer 4 for the next required visual feature; Layer 1+2 for mechanism
    grounding; Layer 3 for the geo prior.
  * ``OBSERVE`` (paper §7) — Layer 3 phi_geo, Layer 5 Ref_{1:3}, Layer 4
    G_t are all inputs alongside image + context.

Build / load:

    from pathome import PathomeDB
    db = PathomeDB.build_from_bugwood(records, layer1_path, ...)
    db.save("artifacts/pathome_v1/")

    db = PathomeDB.load("artifacts/pathome_v1/")
    prior = db.geo_prior(disease="Anthracnose", lat=6.5, lon=3.4, month=7)
    refs = db.retrieve_references(image, lat=6.5, lon=3.4, top_k=3)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image

from .layer1_pathway import (
    MechanisticPathway,
    all_builtin as _builtin_pathways,
    load as _load_pathways,
    save as _save_pathways,
)
from .layer2_manifestation import CrossCropManifestation
from .layer3_geo import RegionalEpidemiology
from .layer4_decision_graph import DiagnosticDecisionGraph, build_demo_graph
from .layer5_references import ReferenceLibrary, ReferenceImage, RetrievalHit
from utils.geo import aez_lookup


@dataclass
class GeoPriorResult:
    """What OBSERVE consumes when querying Layer 3."""
    disease: str
    aez_code: Optional[str]
    month: Optional[int]
    prior: Optional[float]      # P̂(d|r,σ); None when sparse → use ``global_prior``
    global_prior: float
    is_sparse: bool


class PathomeDB:
    """All five layers behind one object."""

    def __init__(
        self,
        layer1: Optional[List[MechanisticPathway]] = None,
        layer2: Optional[CrossCropManifestation] = None,
        layer3: Optional[RegionalEpidemiology] = None,
        layer4: Optional[DiagnosticDecisionGraph] = None,
        layer5: Optional[ReferenceLibrary] = None,
        version: str = "v1.0",
    ):
        self.version = version
        self.layer1: List[MechanisticPathway] = layer1 if layer1 is not None else _builtin_pathways()
        self.layer2: CrossCropManifestation = layer2 if layer2 is not None else CrossCropManifestation()
        self.layer3: RegionalEpidemiology = layer3 if layer3 is not None else RegionalEpidemiology()
        self.layer4: DiagnosticDecisionGraph = layer4 if layer4 is not None else build_demo_graph()
        self.layer5: ReferenceLibrary = layer5 if layer5 is not None else ReferenceLibrary()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build_from_bugwood(
        cls,
        trace_records: Iterable,
        reference_records: Iterable,
        layer1_path: Optional[str] = None,
        layer2_path: Optional[str] = None,
        version: str = "v1.0",
    ) -> "PathomeDB":
        """
        Build a fresh PathomeDB from the 7-trace and 3-reference Bugwood splits.

        Layer 1 / Layer 2 come from the curated knowledge files; Layer 3 is
        built directly from the trace records' GPS metadata; Layer 5 is built
        from the reference records.
        """
        l1 = _load_pathways(layer1_path) if layer1_path else _builtin_pathways()
        l2 = CrossCropManifestation.load(layer2_path) if layer2_path else CrossCropManifestation()
        l3 = RegionalEpidemiology()
        l3.ingest_records(trace_records)
        l5 = ReferenceLibrary()
        for r in reference_records:
            ref = ReferenceImage(
                ref_id=r.image_id,
                image_path=r.src_path,
                crop_species=r.crop_species,
                disease_name=r.disease_name,
                lat=r.lat, lon=r.lon, aez_code=r.aez_code,
            )
            l5.add(ref)
        return cls(layer1=l1, layer2=l2, layer3=l3, layer5=l5, version=version)

    # ------------------------------------------------------------------
    # Query helpers used by PlantSwarm / OBSERVE
    # ------------------------------------------------------------------

    def pathway(self, pathogen_genus: str) -> Optional[MechanisticPathway]:
        for pw in self.layer1:
            if pw.pathogen_genus.lower() == pathogen_genus.lower():
                return pw
        return None

    def manifestation(self, pathogen_genus: str, host_crop: str):
        return self.layer2.get(pathogen_genus, host_crop)

    def geo_prior(
        self,
        disease: str,
        lat: Optional[float],
        lon: Optional[float],
        month: Optional[int],
    ) -> GeoPriorResult:
        aez = aez_lookup(lat, lon) if (lat is not None and lon is not None) else None
        aez_code = aez.code if aez else None
        prior = (
            self.layer3.prior(disease, aez_code, month)
            if (aez_code and month) else None
        )
        return GeoPriorResult(
            disease=disease,
            aez_code=aez_code,
            month=month,
            prior=prior,
            global_prior=self.layer3.global_prior(disease),
            is_sparse=(aez_code is None or month is None
                       or self.layer3.is_sparse(aez_code, month)),
        )

    def retrieve_references(
        self,
        query_image: Image.Image,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        top_k: int = 3,
        constrain_disease: Optional[str] = None,
    ) -> List[RetrievalHit]:
        return self.layer5.retrieve(
            query_image=query_image,
            query_lat=lat, query_lon=lon,
            top_k=top_k, constrain_disease=constrain_disease,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dirpath: str) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        _save_pathways(self.layer1, str(d / "layer1_pathways.json"))
        self.layer2.save(str(d / "layer2_manifestations.json"))
        self.layer3.save(str(d / "layer3_geo.json"))
        self.layer4.save(str(d / "layer4_decision_graph.json"))
        self.layer5.save(str(d / "layer5_refs"))
        with open(d / "version.txt", "w") as f:
            f.write(self.version + "\n")

    @classmethod
    def load(cls, dirpath: str) -> "PathomeDB":
        d = Path(dirpath)
        l1 = _load_pathways(str(d / "layer1_pathways.json")) if (d / "layer1_pathways.json").exists() else _builtin_pathways()
        l2 = (CrossCropManifestation.load(str(d / "layer2_manifestations.json"))
              if (d / "layer2_manifestations.json").exists() else CrossCropManifestation())
        l3 = (RegionalEpidemiology.load(str(d / "layer3_geo.json"))
              if (d / "layer3_geo.json").exists() else RegionalEpidemiology())
        l4 = (DiagnosticDecisionGraph.load(str(d / "layer4_decision_graph.json"))
              if (d / "layer4_decision_graph.json").exists() else build_demo_graph())
        l5 = (ReferenceLibrary.load(str(d / "layer5_refs"))
              if (d / "layer5_refs").exists() else ReferenceLibrary())
        version = (d / "version.txt").read_text().strip() if (d / "version.txt").exists() else "unknown"
        return cls(layer1=l1, layer2=l2, layer3=l3, layer4=l4, layer5=l5, version=version)
