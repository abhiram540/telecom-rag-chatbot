"""Convenience runner: ingest every knowledge source in one command.

    python ingest_all.py
"""

from __future__ import annotations

import ingest_faq
import ingest_guides
import ingest_plans
import ingest_tickets


def main() -> None:
    print("Ingesting all knowledge sources into chroma_store/ ...\n")
    ingest_faq.main()
    ingest_tickets.main()
    ingest_guides.main()
    ingest_plans.main()
    print("\nDone. The chatbot is ready to run.")


if __name__ == "__main__":
    main()
