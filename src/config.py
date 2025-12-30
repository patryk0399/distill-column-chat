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

    data_dir: Path = os.getenv("DATA_DIR", Path("data"))
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L12-v2")
    # llm_backend: str = os.getenv("LLM_BACKEND", "llama_cpp")
    # llm_model_path: Path = os.getenv("LLM_MODEL_PATH", Path("models") / "llamacpp" / "Llama-3-Groq-8B-Tool-Use-Q8_0.gguf")
    llm_model_path: Path = os.getenv("LLM_MODEL_PATH", Path("models") / "llamacpp" / "Hermes-2-Pro-Mistral-7B.Q8_0.gguf")
    # llm_model_path: Path = os.getenv("LLM_MODEL_PATH", Path("models") / "openai" / "gpt-oss-20b-mxfp4.gguf")
    llm_context_window: int = os.getenv("LLM_CONTEXT_WINDOW", 2048)
    llm_n_gpu_layers: int = os.getenv("LLM_N_GPU_LAYERS", 10)            #  0=CPU only, >0=some layers on GPU
    llm_n_threads: int = os.getenv("LLM_N_THREADS", 4)                   #  threading hint
    max_messages: int = os.getenv("MAX_MESSAGES", 20)                    #  max number of displayed messages (not used for context)
    llm_n_batch: int = os.getenv("LLM_N_BATCH", 8)
    llm_use_mmap: bool = os.getenv("LLM_USE_MMAP", False)
    llm_use_mlock: bool = os.getenv("LLM_USE_MLOCK", False)
    #todo  prompt limit for context / "short-term-memory"


def load_config() -> AppConfig:
    """Load application configuration from environment and .env file.

    Load .env first, then read environment variables, and finally
    validate them into an AppConfig instance.
    """
    try:
        # Load .env from the current working directory if present.
        load_dotenv(override=True)
        config = AppConfig()
    except ValidationError as exc:
        print("[config] Failed to validate AppConfig or .env is missing:")
        print(exc)
        raise

    print("[config] Loaded AppConfig:", config)
    return config


__all__ = ["AppConfig", "load_config"]