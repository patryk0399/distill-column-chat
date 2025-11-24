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
        Define existing fields + defaults/fallbacks.
        Will be used by all modules to handle paths, variables, etc.
    """

    env: Literal["dev", "prod"] = "dev"
    data_dir: Path = Path("data")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    llm_backend: str = "dummy"
    #NOTE: for llama.cpp specific (fir now)
    llm_model_path: Path = Path("models") / "llamacpp" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    llm_context_window: int = 2048
    llm_n_gpu_layers: int = 10      # 0=CPU only, >0=some layers on GPU
    llm_n_threads: int = 4         #  threading hint


def load_config() -> AppConfig:
    """Load application configuration from environment and .env file.

    Load .env first, then read environment variables, and finally
    validate them into an AppConfig instance.
    """
    # Load .env from the current working directory if present.
    load_dotenv(override=True)

    raw_config = {
        "env": os.getenv("ENV", "dev"),
        "data_dir": os.getenv("DATA_DIR", "data"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "llm_backend": os.getenv("LLM_BACKEND", "dummy"),
        "embedding_model_name": os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
    }

    # If no values are set in .env use raw_config
    llm_model_path_env = os.getenv("LLM_MODEL_PATH")
    if llm_model_path_env is not None:
        raw_config["llm_model_path"] = llm_model_path_env

    llm_context_window_env = os.getenv("LLM_CONTEXT_WINDOW")
    if llm_context_window_env is not None:
        raw_config["llm_context_window"] = int(llm_context_window_env)

    llm_n_gpu_layers_env = os.getenv("LLM_N_GPU_LAYERS")
    if llm_n_gpu_layers_env is not None:
        raw_config["llm_n_gpu_layers"] = int(llm_n_gpu_layers_env)

    llm_n_threads_env = os.getenv("LLM_N_THREADS")
    if llm_n_threads_env is not None:
        raw_config["llm_n_threads"] = int(llm_n_threads_env)

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