"""Merged retriever across the three knowledge collections (FR-06 – FR-08).

Each query is run against all registered collections in parallel. The top-3
documents from each are gathered (9 total) and formatted into a single,
source-labelled context block for the prompt.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

from config import COLLECTIONS, TOP_K
from vectorstore import get_vectorstore


def build_retrievers() -> RunnableParallel:
    """Build a parallel runnable that fans a query out to every collection.

    To add a new knowledge source, register its collection in
    ``config.COLLECTIONS`` (NFR-06) — it is picked up here automatically.
    """
    retrievers = {
        label: get_vectorstore(name).as_retriever(search_kwargs={"k": TOP_K})
        for label, name in COLLECTIONS.items()
    }
    return RunnableParallel(retrievers)


# Built once and reused across queries.
_RETRIEVERS = build_retrievers()


def retrieve(query: str) -> dict[str, list[Document]]:
    """Return ``{source_label: [docs]}`` for the query across all collections."""
    return _RETRIEVERS.invoke(query)


def format_context(query: str) -> str:
    """Retrieve and render a source-labelled context block for the prompt (FR-08)."""
    results = retrieve(query)
    blocks: list[str] = []
    for label, docs in results.items():
        for doc in docs:
            blocks.append(f"[{label}]\n{doc.page_content}")
    if not blocks:
        return "(No relevant knowledge found.)"
    return "\n\n".join(blocks)
