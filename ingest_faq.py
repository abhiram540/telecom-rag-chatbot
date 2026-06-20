"""Ingest FAQ entries into the ``faq`` collection (FR-14).

One CSV row (question, answer pair) becomes one vector document. Re-running
this script is idempotent: the collection is reset first (FR-17). Support ops
can edit data/faq.csv and re-run this without an engineering release (US-06).

    python ingest_faq.py
"""

from __future__ import annotations

import csv

from langchain_core.documents import Document

from config import COLLECTIONS, FAQ_CSV
from vectorstore import reset_collection


def load_documents() -> list[Document]:
    docs: list[Document] = []
    with open(FAQ_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Embed question + answer so retrieval matches either phrasing.
            content = f"Q: {row['question']}\nA: {row['answer']}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "FAQ",
                        "id": row["id"],
                        "category": row.get("category", ""),
                        "question": row["question"],
                    },
                )
            )
    return docs


def main() -> None:
    docs = load_documents()
    store = reset_collection(COLLECTIONS["FAQ"])
    store.add_documents(docs, ids=[f"faq-{d.metadata['id']}" for d in docs])
    print(f"Ingested {len(docs)} FAQ entries into '{COLLECTIONS['FAQ']}'.")


if __name__ == "__main__":
    main()
