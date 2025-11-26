from __future__ import annotations

"""Single-LLM chat session.
This is a wrapper around the RAG helpers + local LLM backend configuration.

This module sits between the context / RAG layer and (#todo in the future) agent layers.

- Context layer:
  - Delegates retrieval and RAG prompt construction to rag.query.answer.
- LLM layer:
  - Saves chat history for a single user / assistant interaction loop.
  - Note: ask() for debugging via CLI (and UI).


"""

from dataclasses import dataclass
from typing import List, Literal, Optional

from src.config import AppConfig, load_config
from rag.query import answer as rag_answer


Role = Literal["user", "assistant"]


@dataclass
class ChatTurn:
    """Single turn in a chat conversation.

    Avoiding LangChain internals for easier debugging.
    """
    role: Role
    content: str


class BaseChatSession:
    """Chat session class with RAG for each user question.

    Responsibilities
    ----------------
    - Hold a short list of message turns (user question/ assistant answer).
    - For each `ask` call, call `rag_answer` with the user question query.
    - Append the user message and the assistant answer to history.
    """

    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        if cfg is None:
            cfg = load_config()
            print("[chat] Loaded AppConfig from environment.")

        self._cfg: AppConfig = cfg
        self._history: List[ChatTurn] = []

        print("[chat] Initialised BaseChatSession.")
        print("[chat] Using LLM backend:", self._cfg.llm_backend)

    @property
    def config(self) -> AppConfig:
        """Return the configuration used by this chat session."""
        return self._cfg

    @property
    def history(self) -> List[ChatTurn]:
        """Return a copy of the current chat history (e.g. for debugging/inspection).
        """

        return list(self._history)

    def reset(self) -> None:
        """Clear the chat history.

        For later UI with a "Start new conversation" action.
        """

        self._history.clear()
        print("[chat] History cleared.")

    def ask(self, user_input: str, k: int = 3) -> str:
        """Process a single user input and return the assistant's/RAG answer.

        Parameters
        ----------
        user_input:
            User question query.
        k:
            Number of chunks to retrieve for RAG context. Passed through to
            `rag_answer` (potentially doppelt-gemoppelt, we'll find out).

        Returns
        -------
        str
            Answer text by the RAG pipeline.
        """

        user_input = user_input.strip()
        if not user_input:
            print("[chat] Empty user input received. Returning empty answer.")
            return ""

        print("[chat] New user message:", repr(user_input))
        print("[chat] Previous turns in history:", len(self._history))

        # Append user prompt before calling the model so thats later logic can
        # inspect the question even if model call fails.
        self._history.append(ChatTurn(role="user", content=user_input))

        # The actual RAG work goes to rag.query.answer (for now #todo).
        # Goal: only save user questions and model answers. No retrieval or Agent/-prompt details.
        answer_text = rag_answer(user_input, k=k, cfg=self._cfg)

        self._history.append(ChatTurn(role="assistant", content=answer_text))

        print("[chat] Assistant answer length (chars):", len(answer_text))

        return answer_text
