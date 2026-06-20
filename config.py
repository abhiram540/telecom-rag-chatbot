"""Central configuration for the telecom RAG chatbot.

All paths, model names, and tunables live here so the rest of the codebase
imports from a single source of truth. Secrets are read from the environment
(loaded from a local ``.env`` file) and never hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if present (NFR-03).
load_dotenv()

# --- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_store"

FAQ_CSV = DATA_DIR / "faq.csv"
TICKETS_DB = DATA_DIR / "tickets.db"
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"

# --- Embeddings (local, no external API cost — FR-09 / NFR-02) -----------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector store collections (FR-06) ------------------------------------
# Maps the human-facing source label (used in the prompt context) to the
# Chroma collection name on disk. New sources are registered here and in
# retriever.py (NFR-06).
COLLECTIONS: dict[str, str] = {
    "FAQ": "faq",
    "TICKETS": "tickets",
    "GUIDES": "guides",
    "PLANS": "plans",
}

# --- Retrieval (FR-07) ---------------------------------------------------
TOP_K = 3  # documents fetched per collection (3 collections -> 9 docs)

# --- Guide chunking (FR-16) ----------------------------------------------
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --- LLM (FR-12 / FR-13) -------------------------------------------------
LLM_MODEL = "qwen/qwen3-32b"
TEMPERATURE = 0
# Groq returns Qwen3 reasoning tokens separately so the parsed answer is clean.
REASONING_FORMAT = "parsed"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Sample questions for the sidebar / CLI hints (FR-02) ----------------
SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "How do I activate international roaming before a trip?",
    "My SIM is not being recognised — what should I do?",
    "I was charged twice this month. Why?",
    "How do I enable Wi-Fi calling?",
    "What's your cheapest unlimited plan?",
    "Do you have a family plan?",
    "How much is roaming for a week in Europe?",
]


def require_groq_key() -> str:
    """Return the Groq API key or raise a helpful error if it is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "key from https://console.groq.com."
        )
    return GROQ_API_KEY
