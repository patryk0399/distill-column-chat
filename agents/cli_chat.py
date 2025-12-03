from __future__ import annotations

"""CLI chat loop.
Connects BaseChatSession to terminal interface.
"""

from typing import Optional

from src.config import AppConfig, load_config
from agents.base_chat import BaseChatSession


def _print_intro() -> None:
    """Print help text for CLI chat 
    (debugging + UI implementation reminders)

    """
    print("=" * 60)
    print(" CLI Chat ")
    print("=" * 60)
    print("Commands:")
    print("  :q      ->  End the chat session")
    print("  :reset  ->  Clear the chat history")
    print()

    #todo for UI: give hints for the user.
    print("Type your question and press Enter.")


def _create_session(cfg: Optional[AppConfig] = None) -> BaseChatSession:
    """Create a BaseChatSession with the given or loaded configuration.
    """
    if cfg is None:
        cfg = load_config()
        print("[cli] Loaded AppConfig from environment.")

    # Creating chat session here so we can change the config
    # handling later without changing the rest of the CLI loop.
    session = BaseChatSession(cfg=cfg)
    return session


def run_cli_chat(session: Optional[BaseChatSession] = None) -> None:
    """Run chat loop in the terminal."""

    if session is None:
        session = _create_session()

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

        if (stripped == ":reset"):
            session.reset()
            print("[cli] Chat history cleared.")
            continue

        print("[cli] Sending message to chat session ...")
        answer = session.ask(stripped)
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
