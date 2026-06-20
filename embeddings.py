"""Local embedding model (FR-09, NFR-02).

Uses sentence-transformers/all-MiniLM-L6-v2 via HuggingFace, which runs fully
on-device. The model (~90 MB) is auto-downloaded on first use and cached.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a process-wide singleton embedding function.

    Caching avoids reloading the model (and re-paying the warm-up cost) every
    time a vector store is opened.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
