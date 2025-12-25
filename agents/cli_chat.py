from __future__ import annotations

"""CLI chat loop.
Connects BaseChatSession to terminal interface.
"""

from typing import Optional

from src.config import AppConfig, load_config
from rag.query import answer as rag_answer



def _print_intro() -> None:
    """Print help text for CLI chat 
    (debugging + UI implementation reminders)

    """
    print("=" * 60)
    print(" CLI Chat ")
    print("=" * 60)
    print("Commands:")
    print("  :q      ->  End the chat session")
    print()

    #todo for UI: give hints for the user.
    print("Type your question and press Enter.")

def ask(user_input: str, k: int = 3, cfg: AppConfig = None) -> str:
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

        # Append user prompt before calling the model so thats later logic can
        # inspect the question even if model call fails.
       

        # The actual RAG work goes to rag.query.answer (for now #todo).
        # Goal: only save user questions and model answers. No retrieval or Agent/-prompt details.
        answer_text = rag_answer(user_input, k=k, cfg=cfg)

        print("[chat] Assistant answer length (chars):", len(answer_text))

        return answer_text


def run_cli_chat() -> None:
    """Run chat loop in the terminal."""
    cfg = load_config()
    _print_intro()

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\n[cli] Detected interrupt/EOF. Exiting chat.")
            break

        stripped = user_input.strip()

        if not stripped:
            # No Leerlauf ;)
            print("[cli] Empty input. Type :q to quit or a question to continue.")
            continue

        if (stripped == ":q"):
            # ich muss raaaaauuus
            print("[cli] User requested exit.")
            break


        print("[cli] Sending message to chat session ...")

        answer = ask(stripped, cfg = cfg)
        print()
        print("====== ANSWER ======")
        print()
        print("system:", answer)
        print()


def main() -> None:
    """CLI: python -m agents.cli_chat.

    Creates a chat session and starts the loop.
    """
    run_cli_chat()


if __name__ == "__main__":
    main()
