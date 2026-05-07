"""
pathome/layer2_manifestation.py
===============================
Layer 2: Cross-crop manifestation — host-specific presentations of
crop-agnostic pathogen mechanisms (paper §6.2).

For a given (pathogen_genus, host_crop) pair, Layer 2 records:
  - typical lesion diameter range
  - sporulation timing
  - dominant tissue affected
  - host-pigmentation-driven colour shift
  - confusion pairs (other diseases on the same host that look similar)

These maps let OBSERVE diagnose the 12 unseen PlantVillage classes (paper P5)
by composing Layer 1 mechanism + Layer 2 host-specific manifestation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ManifestationEntry:
    pathogen_genus: str
    host_crop: str
    lesion_diameter_mm: Tuple[float, float] = (0.0, 0.0)   # (min, max)
    sporulation_time_days: Tuple[int, int] = (0, 0)
    dominant_tissue: str = "leaf"                          # "leaf" | "stem" | "fruit" | "root"
    color_shift: str = ""                                  # baseline → diseased colour
    confusion_pairs: List[str] = field(default_factory=list)  # other diseases on same host
    notes: str = ""


class CrossCropManifestation:
    """Layer 2 KV store keyed by (pathogen_genus, host_crop)."""

    def __init__(self, entries: Optional[List[ManifestationEntry]] = None):
        self._entries: Dict[Tuple[str, str], ManifestationEntry] = {}
        for e in entries or _BUILTIN_ENTRIES:
            self.add(e)

    # ------------------------------------------------------------------

    def add(self, entry: ManifestationEntry) -> None:
        self._entries[(entry.pathogen_genus.lower(), entry.host_crop.lower())] = entry

    def get(self, pathogen_genus: str, host_crop: str) -> Optional[ManifestationEntry]:
        return self._entries.get((pathogen_genus.lower(), host_crop.lower()))

    def for_pathogen(self, pathogen_genus: str) -> List[ManifestationEntry]:
        return [e for (g, _h), e in self._entries.items() if g == pathogen_genus.lower()]

    def for_host(self, host_crop: str) -> List[ManifestationEntry]:
        return [e for (_g, h), e in self._entries.items() if h == host_crop.lower()]

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                [
                    {
                        "pathogen_genus": e.pathogen_genus,
                        "host_crop": e.host_crop,
                        "lesion_diameter_mm": list(e.lesion_diameter_mm),
                        "sporulation_time_days": list(e.sporulation_time_days),
                        "dominant_tissue": e.dominant_tissue,
                        "color_shift": e.color_shift,
                        "confusion_pairs": e.confusion_pairs,
                        "notes": e.notes,
                    }
                    for e in self._entries.values()
                ],
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "CrossCropManifestation":
        with open(path) as f:
            data = json.load(f)
        entries = [
            ManifestationEntry(
                pathogen_genus=d["pathogen_genus"],
                host_crop=d["host_crop"],
                lesion_diameter_mm=tuple(d.get("lesion_diameter_mm", (0.0, 0.0))),
                sporulation_time_days=tuple(d.get("sporulation_time_days", (0, 0))),
                dominant_tissue=d.get("dominant_tissue", "leaf"),
                color_shift=d.get("color_shift", ""),
                confusion_pairs=d.get("confusion_pairs", []),
                notes=d.get("notes", ""),
            )
            for d in data
        ]
        return cls(entries=entries)


# ---------------------------------------------------------------------------
# Worked examples (paper §6.2): Colletotrichum on three hosts.
# ---------------------------------------------------------------------------

_BUILTIN_ENTRIES: List[ManifestationEntry] = [
    # --- Colletotrichum ---
    ManifestationEntry(
        pathogen_genus="Colletotrichum",
        host_crop="Mango",
        lesion_diameter_mm=(20.0, 40.0),
        sporulation_time_days=(5, 7),
        dominant_tissue="fruit",
        color_shift="green/yellow → sunken black with orange centre",
        confusion_pairs=["Bacterial Black Spot", "Stem-end Rot"],
    ),
    ManifestationEntry(
        pathogen_genus="Colletotrichum",
        host_crop="Soybean",
        lesion_diameter_mm=(8.0, 22.0),
        sporulation_time_days=(7, 10),
        dominant_tissue="stem",
        color_shift="green → reddish-brown blotches with acervuli",
        confusion_pairs=["Phomopsis Stem Canker"],
    ),
    ManifestationEntry(
        pathogen_genus="Colletotrichum",
        host_crop="Strawberry",
        lesion_diameter_mm=(4.0, 12.0),
        sporulation_time_days=(3, 5),
        dominant_tissue="fruit",
        color_shift="red → light tan sunken with salmon mass",
        confusion_pairs=["Botrytis Fruit Rot"],
    ),
    # --- Phytophthora ---
    ManifestationEntry(
        pathogen_genus="Phytophthora",
        host_crop="Tomato",
        lesion_diameter_mm=(10.0, 30.0),
        sporulation_time_days=(3, 6),
        dominant_tissue="leaf",
        color_shift="green → grey-green water-soaked → dark brown with white sporulation underside",
        confusion_pairs=["Early Blight", "Septoria Leaf Spot"],
    ),
    ManifestationEntry(
        pathogen_genus="Phytophthora",
        host_crop="Potato",
        lesion_diameter_mm=(10.0, 40.0),
        sporulation_time_days=(3, 5),
        dominant_tissue="leaf",
        color_shift="green → grey-green → black-brown with white halo on underside",
        confusion_pairs=["Early Blight", "Brown Spot"],
    ),
    # --- Xanthomonas ---
    ManifestationEntry(
        pathogen_genus="Xanthomonas",
        host_crop="Tomato",
        lesion_diameter_mm=(2.0, 5.0),
        sporulation_time_days=(0, 0),
        dominant_tissue="leaf",
        color_shift="green → angular dark spots with yellow halo",
        confusion_pairs=["Bacterial Speck (Pseudomonas)"],
    ),
    ManifestationEntry(
        pathogen_genus="Xanthomonas",
        host_crop="Pepper",
        lesion_diameter_mm=(3.0, 7.0),
        sporulation_time_days=(0, 0),
        dominant_tissue="leaf",
        color_shift="green → angular brown lesions with chlorotic margin",
        confusion_pairs=["Bacterial Spot"],
    ),
    ManifestationEntry(
        pathogen_genus="Xanthomonas",
        host_crop="Rice",
        lesion_diameter_mm=(0.0, 0.0),
        sporulation_time_days=(0, 0),
        dominant_tissue="leaf",
        color_shift="green → yellow leaf-tip blight extending downward",
        confusion_pairs=["Bacterial Leaf Streak"],
    ),
    # --- Botrytis ---
    ManifestationEntry(
        pathogen_genus="Botrytis",
        host_crop="Strawberry",
        lesion_diameter_mm=(5.0, 25.0),
        sporulation_time_days=(2, 4),
        dominant_tissue="fruit",
        color_shift="red → tan with dense grey fuzz",
        confusion_pairs=["Anthracnose Fruit Rot"],
    ),
    ManifestationEntry(
        pathogen_genus="Botrytis",
        host_crop="Grape",
        lesion_diameter_mm=(0.0, 0.0),
        sporulation_time_days=(2, 5),
        dominant_tissue="fruit",
        color_shift="purple → tan-brown with dense grey conidial mass",
        confusion_pairs=["Black Rot"],
    ),
    # --- Puccinia ---
    ManifestationEntry(
        pathogen_genus="Puccinia",
        host_crop="Wheat",
        lesion_diameter_mm=(1.0, 3.0),
        sporulation_time_days=(7, 14),
        dominant_tissue="leaf",
        color_shift="green → orange/yellow stripe of pustules",
        confusion_pairs=["Leaf Rust", "Stem Rust"],
    ),
    # --- Alternaria ---
    ManifestationEntry(
        pathogen_genus="Alternaria",
        host_crop="Tomato",
        lesion_diameter_mm=(5.0, 15.0),
        sporulation_time_days=(5, 8),
        dominant_tissue="leaf",
        color_shift="green → dark concentric rings with chlorotic halo (target spot)",
        confusion_pairs=["Late Blight", "Septoria Leaf Spot"],
    ),
    ManifestationEntry(
        pathogen_genus="Alternaria",
        host_crop="Potato",
        lesion_diameter_mm=(5.0, 12.0),
        sporulation_time_days=(5, 8),
        dominant_tissue="leaf",
        color_shift="green → dark concentric rings with chlorotic halo",
        confusion_pairs=["Late Blight"],
    ),
    # --- Fusarium ---
    ManifestationEntry(
        pathogen_genus="Fusarium",
        host_crop="Tomato",
        lesion_diameter_mm=(0.0, 0.0),
        sporulation_time_days=(0, 0),
        dominant_tissue="stem",
        color_shift="green → unilateral wilt with brown vascular streaks",
        confusion_pairs=["Verticillium Wilt"],
    ),
    ManifestationEntry(
        pathogen_genus="Fusarium",
        host_crop="Banana",
        lesion_diameter_mm=(0.0, 0.0),
        sporulation_time_days=(0, 0),
        dominant_tissue="stem",
        color_shift="green → progressive yellowing + brown vascular streak (Panama disease)",
        confusion_pairs=["Bacterial Wilt"],
    ),
    # --- Cercospora ---
    ManifestationEntry(
        pathogen_genus="Cercospora",
        host_crop="Corn",
        lesion_diameter_mm=(2.0, 30.0),
        sporulation_time_days=(7, 14),
        dominant_tissue="leaf",
        color_shift="green → tan rectangular lesions parallel to veins (grey leaf spot)",
        confusion_pairs=["Northern Leaf Blight"],
    ),
    # --- Erysiphe ---
    ManifestationEntry(
        pathogen_genus="Erysiphe",
        host_crop="Cucumber",
        lesion_diameter_mm=(0.0, 0.0),
        sporulation_time_days=(3, 5),
        dominant_tissue="leaf",
        color_shift="green → white powdery patches on upper leaf surface",
        confusion_pairs=["Downy Mildew (Pseudoperonospora)"],
    ),
    ManifestationEntry(
        pathogen_genus="Erysiphe",
        host_crop="Apple",
        lesion_diameter_mm=(0.0, 0.0),
        sporulation_time_days=(3, 6),
        dominant_tissue="leaf",
        color_shift="green → white felty growth on shoots and leaves",
        confusion_pairs=["Apple Scab (early)"],
    ),
    # --- Tomato Mosaic Virus ---
    ManifestationEntry(
        pathogen_genus="TomatoMosaic",
        host_crop="Tomato",
        lesion_diameter_mm=(0.0, 0.0),
        sporulation_time_days=(0, 0),
        dominant_tissue="leaf",
        color_shift="uniform green → mosaic mottle, leaf curling, fern-leaf appearance",
        confusion_pairs=["Cucumber Mosaic Virus", "Yellow Leaf Curl"],
    ),
]
