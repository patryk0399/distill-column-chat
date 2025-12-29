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

from langchain_core.language_models import BaseLanguageModel
from langchain_community.chat_models import ChatLlamaCpp
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI


from src.config import AppConfig


def get_local_llm(cfg: AppConfig) -> BaseLanguageModel:
    """Return a configured local LLM via AppConfig.
    """
    llm = _build_open_ai_llm(cfg)
    print(f"[llm] Initialising llama.cpp backend with model_path='{cfg.llm_model_path}'.")
    return llm

def _build_open_ai_llm(cfg: AppConfig):
    return ChatOpenAI(
        openai_api_base="http://127.0.0.1:8080/v1",
        openai_api_key="no-key",        # dummy; still required by client
        model="gpt-oss-20b-mxfp4.gguf",          # IMPORTANT: not the gguf path
        temperature=0,
        )

def _build_chat_llama_cpp_llm(cfg: AppConfig) -> BaseLanguageModel:
    """Backend
    """
   
    chat_format = getattr(cfg, "llm_chat_format", "chatml")

    return ChatLlamaCpp(
        model_path=str(cfg.llm_model_path),
        n_ctx=cfg.llm_context_window,
        n_gpu_layers=cfg.llm_n_gpu_layers,
        n_threads=cfg.llm_n_threads,
        n_batch=cfg.llm_n_batch,
        use_mmap=cfg.llm_use_mmap,
        use_mlock=cfg.llm_use_mlock,
        verbose=getattr(cfg, "llm_verbose", False),
    
        model_kwargs={
            "chat_format": chat_format,
            "temperature": getattr(cfg, "llm_temperature", 0.2),
            "top_p": getattr(cfg, "llm_top_p", 0.95),
        },
    )


def main() -> None:
    """Debug CLI to verify llama.cpp backend loads & responds."""
    from src.config import load_config
    from langchain_core.messages import SystemMessage, HumanMessage

    cfg = load_config()
    print("[llm] Loaded AppConfig:", cfg)

    llm = get_local_llm(cfg)
    print("[llm] Backend class:", type(llm).__name__)

    system = SystemMessage(
        content=(
            "Du bist ein hilfreicher Assistent. "
            "Wenn du etwas nicht weißt, sag: 'Ich weiß es nicht.' "
            "Antworte ausschließlich auf Deutsch."
        )
    )
    user = HumanMessage(content="Das ist ein Test. Antworte kurz: Was ist 2+2?")

    try:
        result = llm.invoke([system, user])
    except Exception as exc:
        print("[llm] Error while invoking LLM:", exc)
        raise
    else:
        print("[llm] Sample response:", (result.content or "").strip())


if __name__ == "__main__":
    main()