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
from langchain_community.llms import LlamaCpp

from src.config import AppConfig, load_config

def get_local_llm(cfg: AppConfig) -> BaseLanguageModel:
    """Return a configured local LLM via AppConfig.llm_backend.

    Note: This function is the single entrypoint that the other layers should use to
    obtain an LLM instance. Adding new backends only requires switching the helper function
    for the builder variable, e.g. _build_llama_cpp_llm().
    """
    builder = _build_llama_cpp_llm(cfg)

    print(f"[llm] Initialising LLM backend '{cfg.llm_model_path}'.")
    return builder


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
        n_batch = cfg.llm_n_batch,
        use_mmap = cfg.llm_use_mmap,
        use_mlock = cfg.llm_use_mlock
        #todo: generation behaviour here later (temperature, top_p etc.)
    )
      
def main() -> None:
    """Debug CLI to verify that LLM backend selection works."""
    cfg = load_config()
    print("[llm] Loaded AppConfig:", cfg)

    llm = get_local_llm(cfg)
    print("[llm] Backend class:", type(llm).__name__)

    system_prompt = """\
        Du bist ein hilfreicher Assistent.
        Nutze den folgenden Kontext, um die Frage zu beantworten.
        Wenn die Antwort nicht klar im Kontext vorhanden ist, sag: "Ich weiß es nicht."
        Antworte ausschließlich auf Deutsch.
        """
    context_text = "Kontext zum Testen."

    query = "Das ist ein Test." # Der Teil, der "tatsächlich" in die UI eingegeben wird.

    user_prompt = f"""\
        Kontext:
        {context_text}

        Frage:
        {query}
        """

    llm_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    try:
        result = llm.invoke(llm_prompt)
    except Exception as exc: 
        print("[llm] Error while invoking LLM:", exc)
    else:
        preview = repr(result)
        print("[llm] Sample respinse:", preview[:200] + "...'")


if __name__ == "__main__":
    main()
