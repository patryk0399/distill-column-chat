"""Application configuration.

Maps environment variables and .env values into a typed AppConfig object.
This module is used by all layers (data, RAG, agents, UI) as a single
source of truth for runtime configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class AppConfig(BaseModel):
    """Typed application configuration.

    Fields are intentionally small and generic in Iteration 0C.
    Later iterations may extend this model, but existing fields
    should remain stable.
    """

    env: Literal["dev", "prod"] = "dev"
    data_dir: Path = Path("data")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    llm_backend: str = "dummy"


def load_config() -> AppConfig:
    """Load application configuration from environment and .env file.

    Load .env first, then read environment variables, and finally
    validate them into an AppConfig instance.
    """
    # Load .env from the current working directory if present.
    load_dotenv()

    # Collect raw values from environment with explicit defaults.
    raw_config = {
        "env": os.getenv("ENV", "dev"),
        "data_dir": os.getenv("DATA_DIR", "data"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "llm_backend": os.getenv("LLM_BACKEND", "dummy"),
    }

    try:
        config = AppConfig(**raw_config)
    except ValidationError as exc:
        # Fail fast and show what went wrong with configuration.
        print("[config] Failed to validate AppConfig:")
        print(exc)
        raise

    print("[config] Loaded AppConfig:", config)
    return config


__all__ = ["AppConfig", "load_config"]