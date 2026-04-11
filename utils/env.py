"""Load repository `.env` so scripts pick up HF_TOKEN and other secrets."""

from __future__ import annotations

from pathlib import Path


def load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
