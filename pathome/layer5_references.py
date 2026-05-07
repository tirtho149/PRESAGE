"""
pathome/layer5_references.py
============================
Layer 5: Geo-tagged reference library (paper §6.5).

The 3 images per disease class held out from Bugwood (78 total) are staged
using Qwen2.5-VL-7B prompted with Layer 1 pathway descriptions, then indexed
by CLIP embedding for retrieval. These are the ONLY reference images in
PathomeDB — no PlantVillage or PlantWild image leakage.

Geo-weighted retrieval (Eq. retrieval, paper §6.5):

    score(q, r_i) = 0.7 * cos(e_q, e_{r_i})  +  0.3 * ClimSim(phi_q, phi_{r_i})

A query from Nigeria retrieves references from similar humid-tropical
field conditions rather than visually similar but climatically distant
Himalayan images.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from utils.geo import clim_sim, climate_vector


@dataclass
class ReferenceImage:
    ref_id: str
    image_path: str
    crop_species: str
    disease_name: str
    pathogen_genus: str = ""
    disease_stage: str = "mid"          # "early" | "mid" | "late"
    lat: Optional[float] = None
    lon: Optional[float] = None
    aez_code: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    climate_vec: Optional[np.ndarray] = field(default=None, repr=False)
    notes: str = ""


@dataclass
class RetrievalHit:
    reference: ReferenceImage
    score: float
    cos_score: float
    climsim_score: float


class ReferenceLibrary:
    """
    Layer 5 store. Build embeddings once; FAISS-index for sub-ms retrieval.

    Without FAISS / CLIP available, retrieval falls back to brute-force NumPy
    cosine, which is fine for a 78-entry library.
    """

    EMB_DIM = 512                          # CLIP ViT-B/32
    COS_WEIGHT = 0.7
    CLIM_WEIGHT = 0.3

    def __init__(self):
        self.refs: List[ReferenceImage] = []
        self._faiss_index = None
        self._clip_model = None
        self._clip_preprocess = None

    # ------------------------------------------------------------------

    def add(self, ref: ReferenceImage) -> None:
        if ref.embedding is None:
            ref.embedding = self._embed_image(ref.image_path)
        if ref.climate_vec is None and ref.lat is not None and ref.lon is not None:
            ref.climate_vec = climate_vector(ref.lat, ref.lon)
        self.refs.append(ref)
        self._faiss_index = None  # invalidate index

    def __len__(self) -> int:
        return len(self.refs)

    # ------------------------------------------------------------------
    # Retrieval (paper Eq. retrieval)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_image: Image.Image,
        query_lat: Optional[float] = None,
        query_lon: Optional[float] = None,
        top_k: int = 3,
        constrain_disease: Optional[str] = None,
    ) -> List[RetrievalHit]:
        if not self.refs:
            return []

        q_emb = self._embed_pil(query_image)
        q_clim = (
            climate_vector(query_lat, query_lon)
            if (query_lat is not None and query_lon is not None)
            else np.zeros(6, dtype=np.float32)
        )

        candidates = self.refs
        if constrain_disease:
            candidates = [r for r in self.refs if r.disease_name == constrain_disease]
            if not candidates:
                return []

        hits: List[RetrievalHit] = []
        for ref in candidates:
            if ref.embedding is None:
                continue
            cos = float(np.dot(q_emb, ref.embedding) /
                        (np.linalg.norm(q_emb) * np.linalg.norm(ref.embedding) + 1e-9))
            cs = (
                clim_sim(q_clim, ref.climate_vec)
                if (ref.climate_vec is not None and q_clim.size)
                else 0.0
            )
            score = self.COS_WEIGHT * cos + self.CLIM_WEIGHT * cs
            hits.append(RetrievalHit(reference=ref, score=score,
                                     cos_score=cos, climsim_score=cs))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    # ------------------------------------------------------------------
    # Overconfidence support (paper §7.2)
    # ------------------------------------------------------------------

    def max_match_for_claim(
        self,
        query_image: Image.Image,
        claimed_disease: str,
    ) -> float:
        """
        Highest cosine similarity between the query and any reference of the
        claimed disease. If <0.55, OBSERVE raises P(OC_t) regardless of agent
        confidence (paper §7.2).
        """
        hits = self.retrieve(
            query_image,
            top_k=3,
            constrain_disease=claimed_disease,
        )
        if not hits:
            return 0.0
        return max(h.cos_score for h in hits)

    # ------------------------------------------------------------------
    # Embedding (CLIP if available, else random for stub testing)
    # ------------------------------------------------------------------

    def _ensure_clip(self) -> bool:
        if self._clip_model is not None:
            return True
        try:
            import clip  # type: ignore
            import torch
        except ImportError:
            return False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=device)
        self._clip_device = device
        return True

    def _embed_image(self, path: str) -> np.ndarray:
        try:
            return self._embed_pil(Image.open(path).convert("RGB"))
        except Exception:
            return np.zeros(self.EMB_DIM, dtype=np.float32)

    def _embed_pil(self, img: Image.Image) -> np.ndarray:
        if not self._ensure_clip():
            # Stub fallback — deterministic hash-based pseudo-embedding so
            # the retrieval API is exercisable in tests without CLIP installed.
            arr = np.array(img.resize((32, 32))).astype(np.float32).flatten()
            arr = arr[: self.EMB_DIM] if arr.size >= self.EMB_DIM else \
                np.concatenate([arr, np.zeros(self.EMB_DIM - arr.size)])
            n = np.linalg.norm(arr) + 1e-9
            return (arr / n).astype(np.float32)

        import torch
        with torch.no_grad():
            t = self._clip_preprocess(img).unsqueeze(0).to(self._clip_device)
            e = self._clip_model.encode_image(t).cpu().numpy().flatten().astype(np.float32)
        n = np.linalg.norm(e) + 1e-9
        return e / n

    # ------------------------------------------------------------------
    # FAISS indexing (optional speedup; ignored if FAISS missing)
    # ------------------------------------------------------------------

    def build_faiss_index(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError:
            return
        if not self.refs:
            return
        mat = np.stack([r.embedding for r in self.refs if r.embedding is not None])
        index = faiss.IndexFlatIP(self.EMB_DIM)
        index.add(mat.astype(np.float32))
        self._faiss_index = index

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dirpath: str) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        manifest = []
        emb_path = d / "embeddings.npy"
        embeddings: List[np.ndarray] = []
        for i, ref in enumerate(self.refs):
            entry = {
                "ref_id": ref.ref_id,
                "image_path": ref.image_path,
                "crop_species": ref.crop_species,
                "disease_name": ref.disease_name,
                "pathogen_genus": ref.pathogen_genus,
                "disease_stage": ref.disease_stage,
                "lat": ref.lat, "lon": ref.lon,
                "aez_code": ref.aez_code,
                "notes": ref.notes,
                "emb_idx": i,
            }
            manifest.append(entry)
            embeddings.append(
                ref.embedding if ref.embedding is not None else np.zeros(self.EMB_DIM, dtype=np.float32)
            )
        if embeddings:
            np.save(emb_path, np.stack(embeddings))
        else:
            # Empty library: persist a 0xEMB_DIM array so load() roundtrips cleanly.
            np.save(emb_path, np.zeros((0, self.EMB_DIM), dtype=np.float32))
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, dirpath: str) -> "ReferenceLibrary":
        d = Path(dirpath)
        with open(d / "manifest.json") as f:
            manifest = json.load(f)
        emb_mat = np.load(d / "embeddings.npy")
        out = cls()
        for entry in manifest:
            idx = entry.pop("emb_idx")
            ref = ReferenceImage(
                **entry,
                embedding=emb_mat[idx].astype(np.float32),
            )
            if ref.lat is not None and ref.lon is not None:
                ref.climate_vec = climate_vector(ref.lat, ref.lon)
            out.refs.append(ref)
        return out
