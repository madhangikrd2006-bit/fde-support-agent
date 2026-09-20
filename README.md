# DocuBot — Grounded Q&A Agent Over Your Company's Docs

**The problem:** Every company accumulates scattered internal docs (runbooks, FAQs, onboarding guides, policy PDFs). New hires, support agents, and customers waste hours searching for answers that already exist somewhere in that pile — or worse, get a wrong answer from an LLM that's confidently hallucinating.

**What this does:** DocuBot ingests a folder of documents (`.txt`, `.md`, `.pdf`), chunks and embeds them locally, and answers natural-language questions by retrieving the most relevant passages and grounding Claude's answer in them — with citations back to the source file and chunk. It's a small, self-contained example of the kind of "connect an LLM to a customer's real data" workflow a Forward Deployed Engineer builds constantly: point it at *any* organization's docs and it works with zero retraining.

It runs two ways:
- **Retrieval-only mode** (no API key needed) — returns the best-matching passages, so you can see the retrieval quality immediately.
- **Generation mode** (Anthropic API key set) — Claude synthesizes a direct answer from those passages, with inline citations, and says "I don't know" when the docs don't cover it.

## Why this design

| Decision | Reasoning |
|---|---|
| Local embeddings (`sentence-transformers`) | Search works instantly, offline, and free — no API cost just to index documents. Mirrors how you'd deploy on-prem for a customer who won't send data to a third-party API. |
| In-memory vector store (numpy cosine similarity) | No infra dependency (no Pinecone/Weaviate account needed) — trivial to swap out for a real vector DB later; the interface (`VectorStore.add`/`.search`) is the seam. |
| Claude only used for the *final answer*, not search | Keeps the expensive/slow step optional and isolated. You can demo retrieval quality without ever spending a token. |
| Chunking with overlap | Naive fixed-size chunking breaks context across boundaries; overlap reduces the chance a fact gets split across two chunks and misses retrieval. |
| FastAPI + a single static HTML page | No frontend build step. Clone, run one command, open a browser. Minimizes friction — the thing FDEs are always optimizing for at customer onboarding. |

## Architecture

```
                 ┌─────────────┐
  documents ───► │  ingest.py  │  chunk text (with overlap)
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │   rag.py    │  embed chunks (MiniLM) + cosine-sim search
                 └──────┬──────┘
                        │ top-k relevant chunks
                 ┌──────▼──────┐
                 │   llm.py    │  Claude synthesizes grounded answer + citations
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │  main.py    │  FastAPI: /upload, /ask, serves the chat UI
                 └─────────────┘
```

## Quickstart

```bash
git clone <your-repo-url>
cd fde-support-agent
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# (optional) enable Claude-generated answers instead of raw-passage retrieval:
export ANTHROPIC_API_KEY=sk-ant-...

uvicorn app.main:app --reload
```

Open **http://localhost:8000**, click "Load sample docs" (or upload your own `.txt`/`.md`/`.pdf` files), and start asking questions.

### Try it with the included sample docs
`sample_docs/` contains a fictional company's onboarding + policy docs. Try asking:
- "How many vacation days do new hires get?"
- "What's the process for expensing a client dinner?"
- "Who approves a purchase over $5,000?"

### Docker

```bash
docker build -t docubot .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY docubot
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST (multipart) | Upload one or more `.txt`/`.md`/`.pdf` files to index |
| `/load_samples` | POST | Index the bundled sample docs |
| `/ask?q=...&k=4` | GET | Ask a question; returns answer + cited chunks |
| `/status` | GET | Number of chunks indexed, whether generation mode is active |

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests cover chunking edge cases (empty docs, docs shorter than chunk size) and retrieval ranking (that an obviously relevant chunk outranks an irrelevant one) without requiring network access or an API key.

## What I'd do with more time

- Swap the in-memory store for a persistent vector DB (pgvector/Chroma) so the index survives restarts and scales past a few hundred documents.
- Add per-source access control — real customer deployments need doc-level permissions, not just "everything is searchable by everyone."
- Stream the Claude response token-by-token instead of waiting for the full completion.
- Add eval harness: a small set of Q&A pairs with expected source docs, to catch retrieval regressions as chunking parameters change.
- Support incremental re-indexing when a source doc changes, instead of requiring a full re-upload.

## Tech stack
Python, FastAPI, sentence-transformers (MiniLM-L6-v2), numpy, Anthropic API (Claude), vanilla HTML/JS.
