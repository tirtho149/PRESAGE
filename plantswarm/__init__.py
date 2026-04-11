"""PlantSwarm: confidence-gated multi-agent VLM pipeline for plant disease diagnosis."""

from .autogen_pipeline import AutoGenPlantSwarmPipeline
from .entropy_pipeline import EntropyPlantSwarmPipeline
from .pipeline import PlantSwarmPipeline, RoutingTrace

__all__ = [
    "AutoGenPlantSwarmPipeline",
    "EntropyPlantSwarmPipeline",
    "PlantSwarmPipeline",
    "RoutingTrace",
]
