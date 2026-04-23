# Self RAG

Personal Markdown RAG learning console. This project is a local-first workspace for scanning Markdown notes, previewing chunks, and comparing keyword, vector, and hybrid retrieval results before building a full answer-generation layer.

## Quick Start

Start the backend and frontend together:

```bash
bash scripts/dev.sh
```

Or start them separately.

Backend:

```bash
bash scripts/dev-backend.sh
```

Frontend:

```bash
bash scripts/dev-frontend.sh
```

Then open the frontend from your machine:

```text
http://<server-ip>:5173/
```

FastAPI docs are available at:

```text
http://<server-ip>:8800/docs
```

The frontend calls backend APIs through the Vite proxy, so the browser only needs access to port `5173`. The server itself must be able to reach `127.0.0.1:8800`.

## Architecture

Self RAG is split into a FastAPI backend and a React/Vite frontend.

```text
Browser
  |
  | http://<server-ip>:5173
  v
React + Vite frontend
  |
  | Vite proxy: /api and /health -> http://127.0.0.1:8800
  v
FastAPI backend
  |
  +-- Markdown scanner
  +-- Structure-aware chunker
  +-- Keyword retriever
  +-- Vector retriever
  +-- Hybrid retriever
  +-- SQLite schema initialization
```

The current backend performs retrieval directly over Markdown files supplied by the request. SQLite tables are initialized at startup and define the intended persistence shape for documents and chunks, but the active API path still scans and chunks files on demand.

## Project Layout

```text
app/
  main.py                  FastAPI app, request/response models, API routes
  config.py                Runtime settings loaded from .env and SELF_RAG_* env vars
  db.py                    SQLite connection helper and document/chunk schema
  documents/
    loader.py              Recursive Markdown file scanner
    models.py              MarkdownDocument and MarkdownChunk dataclasses
  indexing/
    chunker.py             Markdown section/paragraph/code-fence aware chunking
    embeddings.py          hash and sentence-transformers embedding providers
    vector_store.py        In-memory vector search store
  retrieval/
    keyword.py             Lexical retriever for exact terms and Chinese phrases
    vector.py              Embedding-based cosine retrieval
    hybrid.py              Weighted fusion of keyword and vector results
    schemas.py             Shared retrieval response models

frontend/
  src/main.tsx             React retrieval console
  src/styles.css           Application styling
  vite.config.ts           Vite server and backend proxy config

scripts/
  dev.sh                   Starts backend and frontend together
  dev-backend.sh           Starts FastAPI on port 8800
  dev-frontend.sh          Starts Vite on port 5173

tests/                     Backend unit and API tests
```

## Retrieval Flow

1. The frontend sends a Markdown root folder, query, retrieval mode, `top_k`, and chunk size settings to the backend.
2. The backend recursively loads `*.md` files from the requested root.
3. Markdown is split by headings first, then by paragraph-like blocks while preserving fenced code blocks.
4. Keyword retrieval scores exact query matches, significant terms, filenames, headings, and Chinese n-grams.
5. Vector retrieval embeds the query and searchable chunks, then ranks by vector similarity in memory.
6. Hybrid retrieval runs both retrievers, normalizes their scores, and merges them with a keyword-heavy weighting.
7. The frontend displays the trace, debug metadata, source file, line range, score, and original chunk text.

## API Surface

```text
GET  /health
POST /api/index/preview
POST /api/retrieval/keyword
POST /api/retrieval/vector
POST /api/retrieval/hybrid
```

`/api/index/preview` is useful for checking whether the Markdown scanner and chunker produce sensible evidence before running retrieval. The retrieval endpoints return ranked chunks with source metadata so results can be audited directly in the UI.

## Configuration

Runtime settings live in `app/config.py` and can be overridden with `.env` values using the `SELF_RAG_` prefix.

Common settings:

```text
SELF_RAG_DATABASE_PATH=data/sqlite/self_rag.db
SELF_RAG_DEFAULT_MARKDOWN_ROOT=/path/to/markdown
SELF_RAG_EMBEDDING_PROVIDER=bge-m3
SELF_RAG_EMBEDDING_MODEL_PATH=models/BAAI/bge-m3
```

Embedding providers:

```text
hash      Dependency-free deterministic vectors for quick local testing
bge-m3    Local sentence-transformers model loaded from SELF_RAG_EMBEDDING_MODEL_PATH
```

To use `bge-m3`, install the optional local embedding dependencies and make sure the model path exists locally. Use `hash` when you only need to verify the API and UI without model dependencies.

## Development

Run tests:

```bash
.venv/bin/python -m pytest
```

Build the frontend:

```bash
cd frontend
npm run build
```
