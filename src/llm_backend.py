from __future__ import annotations

"""LLM backend registry for LLM layer.

This module provides a "registry" for local LLM backends.
Core "managing" code and agents only depends on the registry functions and
the BaseLanguageModel interface, not on specific backend implementations.

Layer mapping:
- LLM layer: This module lives in the LLM layer and exposes get_local_llm().
- Data / context layers: no Dependency.
- Agents / UI layers: Use get_local_llm() to obtain a configured model.
"""

from collections.abc import Callable
from typing import Dict

from langchain_core.language_models import BaseLanguageModel
from langchain_community.llms.fake import FakeListLLM
from langchain_community.llms.human import HumanInputLLM
from langchain_community.llms import LlamaCpp

from src.config import AppConfig, load_config

# Registry of available LLM backends.
LLM_BACKENDS: Dict[str, Callable[[AppConfig], BaseLanguageModel]] = {}


def register_llm_backend(
    name: str,
    builder: Callable[[AppConfig], BaseLanguageModel],
) -> None:
    """Register a new LLM backend builder.

    Parameters
    ----------
    name:
        Registry key used in config.llm_backend.
    builder:
        Callable that receives an AppConfig and returns a configured
        BaseLanguageModel instance.
    """
    if name in LLM_BACKENDS:
        print(f"[llm] Overwriting existing LLM backend registration: {name}")
    else:
        print(f"[llm] Registering LLM backend: {name}")
    LLM_BACKENDS[name] = builder


def get_local_llm(cfg: AppConfig) -> BaseLanguageModel:
    """Return a configured local LLM via AppConfig.llm_backend.

    Note: This function is the single entrypoint that the other layers should use to
    obtain an LLM instance. Adding new backends only requires registering them
    here in this module -> no changes are needed in the callers.
    """
    backend_name = cfg.llm_backend
    builder = LLM_BACKENDS.get(backend_name)

    if builder is None:
        known = ", ".join(sorted(LLM_BACKENDS)) or "<none>"
        message = (
            f"Unknown LLM backend '{backend_name}'. "
            f"Known backends: {known}. "
            "Check LLM_BACKEND in .env or AppConfig.llm_backend."
        )
        raise ValueError(message)

    print(f"[llm] Initialising LLM backend '{backend_name}'.")
    return builder(cfg)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _build_dummy_llm(cfg: AppConfig) -> BaseLanguageModel:
    """Dummy"""
    _ = cfg  # Builder signature

    responses = [
        (
            "Dummy LLM response: this is backend for testing the pipeline.\n"
            "No language generation here."
        )
    ]
    return FakeListLLM(responses=responses)


def _build_human_llm(cfg: AppConfig) -> BaseLanguageModel:
    """Dummy"""
    _ = cfg  # builder signature.
    return HumanInputLLM()

def _build_llama_cpp_llm(cfg: AppConfig) -> BaseLanguageModel:
    """Local llama.cpp-based LLM backend.

    Here specificly: Uses a GGUF model file via llama-cpp-python. All settings come from
    AppConfig so the model can be switched via configuration only
    """
    model_path = str(cfg.llm_model_path)

    return LlamaCpp(
        model_path=model_path,
        n_ctx=cfg.llm_context_window,
        n_gpu_layers=cfg.llm_n_gpu_layers,
        n_threads=cfg.llm_n_threads,
        #todo: generation behaviour here later (temperature, top_p etc.)
    )


def _register_default_backends() -> None:
    """Register built-in backends for the early iterations."""
    register_llm_backend("dummy", _build_dummy_llm)
    register_llm_backend("human", _build_human_llm)
    register_llm_backend("llama_cpp", _build_llama_cpp_llm)



# Registering backends at import-time so get_local_llm() can be used immediately
_register_default_backends()


def main() -> None:
    """DEbug CLI to verify that LLM backend selection works."""
    cfg = load_config()
    print("[llm] Loaded AppConfig:", cfg)

    llm = get_local_llm(cfg)
    print("[llm] Backend class:", type(llm).__name__)

    test_prompt = "Quick test of llm_backend main()."
    try:
        result = llm.invoke(test_prompt)
    except Exception as exc:  # pragma: no cover - debug print for manual runs
        print("[llm] Error while invoking LLM:", exc)
    else:
        # Truncate to keep console output compact.
        preview = repr(result)
        if len(preview) > 200:
            preview = preview[:200] + "...'"
        print("[llm] Sample respinse:", preview)


if __name__ == "__main__":
    main()
