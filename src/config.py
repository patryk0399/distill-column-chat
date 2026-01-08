"""Application configuration.

Maps environment variables and .env values into a typed AppConfig object.
This module is used by all layers (data, RAG, agents, UI) as a single
source of truth for runtime configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError


class AppConfig(BaseModel):
    """Typed application configuration.
        Define existing fields + defaults/fallbacks.
        Will be used by all modules to handle paths, variables, etc.
    """

    # IF YOU CHANGE THE MODEL ALSO CHANGE chat_format IN llm_backend.py if needed!!!!
    # Cretain models need or only work with specific chat_format(s) !!!
    model: str = "Hermes-3-Llama-3.2-3B.Q8_0.gguf"
    # model: str = "Hermes-2-Pro-Mistral-7B.Q8_0.gguf"
    # model: str = "Hermes-2-Pro-Mistral-7B.Q8_0.gguf"
    # model: str = "Llama-3-Groq-8B-Tool-Use-Q8_0.gguf"
    # model: str = "Llama-3-SauerkrautLM-8b-Instruct-Q8_0_L.gguf"
    # model: str = 
    llm_model_path: Path = Path("models") / "llamacpp" / model
    data_dir: Path = "data"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L12-v2"

    llm_context_window: int = 10004 #M
    llm_n_gpu_layers: int = 20 #HW #M            #  0=CPU only, >0=some layers on GPU
    llm_n_threads: int = 6 #HW                  #  threading hint
    max_messages: int = 20                 #  max number of displayed messages (not used for context)
    llm_n_batch: int = 8
    llm_use_mmap: bool = True
    llm_use_mlock: bool = False
    
    retrieve_k_min: int = 3 
    retrieve_k_max: int = 5 # not used right now
    index_chunk_size: int = 1000
    index_chunk_overlap: int = 150

    #todo  prompt limit for context / "short-term-memory"


def load_config() -> AppConfig:
    """Load application configuration from environment and .env file.

    Load .env first, then read environment variables, and finally
    validate them into an AppConfig instance.
    """
    try:
        # Load .env from the current working directory if present.
        # load_dotenv(override=True)
        config = AppConfig()
    except ValidationError as exc:
        print("[config] Failed to validate AppConfig or .env is missing:")
        print(exc)
        raise

    print("[config] Loaded AppConfig:", config)
    return config


__all__ = ["AppConfig", "load_config"]