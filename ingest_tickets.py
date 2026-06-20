"""Ingest resolved support tickets into the ``tickets`` collection (FR-15).

One resolved ticket becomes one vector document. The embedded text pairs the
reported problem with how it was resolved, so retrieval surfaces real fixes for
similar issues. Re-running is idempotent (FR-17). Support ops can seed new
tickets into data/tickets.db and re-run this (US-07).

    python ingest_tickets.py
"""

from __future__ import annotations

import sqlite3

from langchain_core.documents import Document

from config import COLLECTIONS, TICKETS_DB
from vectorstore import reset_collection


def load_documents() -> list[Document]:
    docs: list[Document] = []
    conn = sqlite3.connect(TICKETS_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT ticket_id, category, issue_type, description, resolution "
            "FROM tickets WHERE status = 'resolved'"
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        content = (
            f"Issue ({row['category']} / {row['issue_type']}): "
            f"{row['description']}\nResolution: {row['resolution']}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "TICKETS",
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "issue_type": row["issue_type"],
                },
            )
        )
    return docs


def main() -> None:
    docs = load_documents()
    store = reset_collection(COLLECTIONS["TICKETS"])
    store.add_documents(docs, ids=[d.metadata["ticket_id"] for d in docs])
    print(f"Ingested {len(docs)} resolved tickets into '{COLLECTIONS['TICKETS']}'.")


if __name__ == "__main__":
    main()
