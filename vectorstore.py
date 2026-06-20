"""ChromaDB vector store helpers (NFR-05).

A single persistent Chroma store on disk holds one collection per knowledge
source. To avoid index corruption, every ``Chroma`` wrapper shares ONE
underlying ``PersistentClient`` for the process — opening multiple independent
clients on the same directory can corrupt the on-disk HNSW segment.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from langchain_chroma import Chroma

from config import CHROMA_DIR
from embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_client() -> "chromadb.ClientAPI":
    """Return a process-wide singleton persistent Chroma client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_vectorstore(collection_name: str) -> Chroma:
    """Open (or create) a persistent Chroma collection on the shared client."""
    return Chroma(
        client=get_client(),
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )


def reset_collection(collection_name: str) -> Chroma:
    """Drop and recreate a collection so re-running an ingest is idempotent.

    Re-running an ingest script always yields the same end state rather than
    appending duplicate vectors (FR-17).
    """
    client = get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        # Collection may not exist yet on a first run; that's fine.
        pass
    return get_vectorstore(collection_name)
