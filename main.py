"""Interactive CLI REPL for the telecom RAG chatbot (FR-18, FR-19).

    python main.py

Type a question and press Enter. Answers stream as they are generated.
Type ``quit`` (or press Ctrl-C) to exit.
"""

from __future__ import annotations

import sys

from chain import stream_answer
from config import SAMPLE_QUESTIONS


def _print_banner() -> None:
    print("=" * 60)
    print(" NovaCell Telecom Support Assistant (CLI)")
    print("=" * 60)
    print("Ask a Tier-1 support question. Type 'quit' to exit.\n")
    print("Sample questions:")
    for q in SAMPLE_QUESTIONS:
        print(f"  - {q}")
    print()


def main() -> None:
    _print_banner()
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        print("Bot> ", end="", flush=True)
        try:
            for token in stream_answer(question):
                print(token, end="", flush=True)
            print("\n")
        except Exception as exc:  # surface errors without killing the REPL
            print(f"\n[error] {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
