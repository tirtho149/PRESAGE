"""PlantSwarm package.

Submodules:
    delta_pipeline   Qwen-driven regional delta extraction for PathomeDB
    captioning       KB-derived caption builder for PathomeOOD training

Both submodules are imported explicitly by their consumers (e.g.
``from plantswarm.delta_pipeline import run_batch``); nothing is
re-exported at the package level so that environments needing only
one submodule do not have to install the other's dependencies.
"""
