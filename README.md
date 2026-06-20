# NovaCell — RAG Telecom Customer-Care Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers Tier-1 telecom
support questions (connectivity, billing, SIM/eSIM, roaming, voice, account)
grounded **only** in NovaCell's verified knowledge — never inventing prices or
policies.

<img width="1907" height="857" alt="image" src="https://github.com/user-attachments/assets/2ec56012-4c5c-40aa-82a4-a98d159da1aa" />


## How it works

```
question ─▶ merged retriever ─▶ ChromaDB · faq      (top-3)
                              ├─ ChromaDB · tickets  (top-3)
                              ├─ ChromaDB · guides   (top-3)
                              └─ ChromaDB · plans    (top-3)
                       │ 12 source-labelled docs
                       ▼
              ChatPromptTemplate ─▶ Qwen3-32B (Groq, temp=0) ─▶ streamed answer
```

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally (no API cost).
- **Vector store:** ChromaDB, persisted to `chroma_store/`.
- **LLM:** `qwen/qwen3-32b` via Groq, `temperature=0`, `reasoning_format=parsed`.
- **Framework:** LangChain (LCEL). **UI:** Streamlit + a CLI.

## Setup

Requires Python 3.11+.

```bash
# 1. Install dependencies
uv sync            # or: pip install -e .

# 2. Add your Groq API key
cp .env.example .env        # then edit .env and paste your key

# 3. Ingest the knowledge sources into chroma_store/  (one-time; re-run to refresh)
python ingest_all.py
```

The embedding model (~90 MB) downloads automatically on first run.

## Run

```bash
# Browser UI
streamlit run app.py

# Or the CLI REPL  (type 'quit' to exit)
python main.py
```

## Updating the knowledge base

Support ops can refresh content without a code release. Edit the source, then
re-run the matching ingest script (re-runs are idempotent):

| Source | File | Ingest |
|---|---|---|
| FAQ | `data/faq.csv` | `python ingest_faq.py` |
| Resolved tickets | `data/tickets.db` | `python ingest_tickets.py` |
| PDF guide | `data/telecom_guide.pdf` | `python ingest_guides.py` |
| Plan & pricing catalog | `data/plans.json` | `python ingest_plans.py` |

## Project layout

| File | Purpose |
|---|---|
| `config.py` | Paths, model names, tunables, `.env` loading |
| `embeddings.py` | Local MiniLM embedding singleton |
| `vectorstore.py` | Chroma open / idempotent reset helpers |
| `ingest_faq.py` / `ingest_tickets.py` / `ingest_guides.py` / `ingest_plans.py` | Per-source ingest |
| `ingest_all.py` | Run all four ingests |
| `retriever.py` | Parallel multi-collection retrieval + context formatting |
| `chain.py` | LCEL chain (retrieve → prompt → LLM → parse) |
| `app.py` | Streamlit chat UI |
| `main.py` | CLI REPL |

To add a new knowledge source: write an `ingest_*.py` and register its
collection in `config.COLLECTIONS` — `retriever.py` picks it up automatically.
