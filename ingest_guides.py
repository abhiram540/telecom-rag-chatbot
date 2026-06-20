"""Ingest the telecom PDF guide into the ``guides`` collection (FR-16).

The PDF is split into 600-character chunks with 100-character overlap before
embedding. Re-running is idempotent (FR-17).

    python ingest_guides.py
"""

from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTIONS, GUIDE_PDF
from vectorstore import reset_collection


def load_documents() -> list[Document]:
    pages = PyPDFLoader(str(GUIDE_PDF)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        # Keep page provenance but normalise the source label for the prompt.
        chunk.metadata["source"] = "GUIDES"
    return chunks


def main() -> None:
    docs = load_documents()
    store = reset_collection(COLLECTIONS["GUIDES"])
    store.add_documents(docs, ids=[f"guide-{i}" for i in range(len(docs))])
    print(f"Ingested {len(docs)} guide chunks into '{COLLECTIONS['GUIDES']}'.")


if __name__ == "__main__":
    main()
