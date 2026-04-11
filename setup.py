"""
setup.py
========
PlantSwarm package setup (PlantSwarm paper).
"""

from setuptools import find_packages, setup

setup(
    name="plantswarm",
    version="0.2.0",
    description=(
        "PlantSwarm: Confidence-Gated Emergent Routing in Multi-Agent VLM Swarms "
        "for Calibrated Plant Disease Diagnosis"
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "vllm>=0.4.0",
        "transformers>=4.40.0",
        "torch>=2.1.0",
        "pyarrow>=14.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "autogen-agentchat>=0.4.0",
        "autogen-ext[openai]>=0.4.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
)
