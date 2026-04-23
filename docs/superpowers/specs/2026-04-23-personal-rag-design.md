# Personal Markdown RAG Design

Date: 2026-04-23

## Purpose

Build a personal RAG system that starts as a learning-focused Web debugging console for a Markdown folder knowledge base, then evolves into a dual-mode personal knowledge product with both normal Q&A and RAG debug views.

The primary goal is not only to answer questions over notes, but to help the owner understand how RAG works end to end: chunking, embedding, vector retrieval, keyword retrieval, hybrid retrieval, reranking, context construction, answer generation, citation, and evaluation.

## Scope

### MVP

- Support a local folder of ordinary Markdown files.
- Scan, parse, chunk, index, and update Markdown documents.
- Support vector retrieval, keyword retrieval, and hybrid retrieval.
- Show retrieved chunks, scores, sources, and final LLM context.
- Generate answers with citations back to source Markdown files.
- Run as a single-user local Web application.

### Later

- Add reranking, query rewriting, multi-knowledge-base support, tags, metadata filters, question history, and normal chat mode.
- Add PDF, web page, and work document ingestion.
- Add periodic indexing, knowledge summaries, learning cards, and agent tool calling.

### Out Of Scope For MVP

- Multi-user authentication and permissions.
- Production deployment hardening.
- Complex document editing UI.
- Full Obsidian backlink graph support.
- Agent automation beyond retrieval-augmented question answering.

## Recommended Technical Direction

Start with a simple Python-first architecture and evolve it only when the concepts are clear.

- Backend: FastAPI.
- Frontend: lightweight Web UI, preferably React/Vite or simple server-rendered pages.
- Metadata storage: SQLite.
- Keyword search: SQLite FTS5.
- Vector index: Chroma or FAISS.
- LLM and embedding model: provider abstraction, with the concrete provider replaceable.

This keeps the first implementation small while still exposing the important RAG mechanics.

## Architecture

```text
Markdown Folder
  -> Document Loader
  -> Markdown Chunker
  -> Indexer
  -> Vector Index + Keyword Index + Metadata Store
  -> Retriever
  -> Answer Engine
  -> Debug Web UI
```

## Components

### Document Loader

Responsibilities:

- Recursively scan a configured Markdown root folder.
- Read `.md` files.
- Track file path, content hash, size, and modified time.
- Detect added, changed, and removed files for incremental indexing.

Interface:

- Input: root directory path.
- Output: document records with file metadata and raw Markdown content.

### Markdown Chunker

Responsibilities:

- Split Markdown into retrieval-friendly chunks.
- Prefer structure-aware splitting using headings, paragraphs, lists, and code fences.
- Preserve metadata for each chunk:
  - source file path
  - heading path
  - chunk index
  - line range when feasible
  - content hash

Initial strategy:

- Split by headings first.
- Keep short sections intact.
- Split long sections by paragraph with a target size.
- Avoid splitting inside fenced code blocks.
- Allow small overlap only when needed.

### Indexer

Responsibilities:

- Generate embeddings for chunks.
- Store chunk metadata in SQLite.
- Store vector embeddings in Chroma or FAISS.
- Store chunk text in SQLite FTS5 for keyword search.
- Re-index only changed chunks where possible.

### Retriever

Responsibilities:

- Accept a user query and retrieval mode.
- Return ranked candidate chunks with scores and source metadata.

Supported MVP modes:

- `vector`: semantic retrieval through embeddings.
- `keyword`: exact and lexical retrieval through SQLite FTS5.
- `hybrid`: combine vector and keyword results.

Hybrid strategy for MVP:

- Retrieve top-k candidates from vector search.
- Retrieve top-k candidates from keyword search.
- Normalize scores per retriever.
- Merge by chunk id.
- Combine scores using a configurable weight, for example `0.65 vector + 0.35 keyword`.
- Deduplicate and return top final candidates.

### Answer Engine

Responsibilities:

- Build final context from retrieved chunks.
- Call the configured LLM.
- Ask the LLM to answer only from provided context when possible.
- Require citations using chunk source identifiers.
- Return answer, citations, selected chunks, and debug metadata.

Initial answer policy:

- If evidence is insufficient, say so instead of inventing.
- Cite file path and heading path for every substantial claim.
- Keep raw retrieved evidence visible in the debug UI.

### Debug Web UI

Responsibilities:

- Provide a Web interface for indexing, asking, and inspecting RAG internals.
- Favor observability over polish in MVP.

Core areas:

- Index page: configure Markdown root, scan files, build or update index.
- Ask page: enter question and choose `vector`, `keyword`, or `hybrid`.
- Debug panel: show retrieval results, scores, retrieval mode, merged ranking, and final context.
- Source panel: show cited chunks with file path, heading path, and original text.

## Data Model

Minimal SQLite tables:

```text
documents
- id
- path
- content_hash
- modified_time
- size
- indexed_at

chunks
- id
- document_id
- chunk_index
- heading_path
- start_line
- end_line
- text
- text_hash
- embedding_id
- created_at
- updated_at

questions
- id
- query
- retrieval_mode
- answer
- created_at

retrieval_events
- id
- question_id
- chunk_id
- retriever
- raw_score
- normalized_score
- final_score
- rank
```

SQLite FTS5 should index chunk text and useful source metadata.

## Retrieval Concepts To Learn

### Why Vectors

Embeddings map text into a numeric space where semantically similar texts are near each other. This helps retrieve relevant content even when the query and document do not share exact words.

Example:

```text
Query: how to reduce model hallucination
Document terms: grounding, citation, retrieval augmentation, verification
```

Pure keyword retrieval may miss this. Vector retrieval is more likely to find it.

### Why Keywords Still Matter

Vector retrieval is weaker for exact identifiers, names, versions, paths, commands, dates, and rare domain terms. Keyword retrieval helps recover exact-match evidence.

Examples:

- `pgvector`
- `RFC 9110`
- `ERR_CONNECTION_RESET`
- `2026-04-23`
- file paths and function names

### Why Hybrid Retrieval

Hybrid retrieval combines semantic recall and exact-match precision. It is usually more robust than pure vector search for personal knowledge bases because notes often contain names, commands, filenames, concepts, and loosely phrased ideas together.

### Reranking Later

A retriever quickly finds candidates. A reranker more carefully judges which candidates actually answer the question. Reranking should be added after the basic vector, keyword, and hybrid modes are observable.

## API Sketch

```text
POST /api/index/scan
POST /api/index/rebuild
POST /api/index/update
GET  /api/index/status

POST /api/ask
GET  /api/questions/{id}
GET  /api/questions/{id}/debug

GET  /api/documents
GET  /api/chunks/{id}
```

Example ask request:

```json
{
  "query": "What are my notes about RAG evaluation?",
  "retrieval_mode": "hybrid",
  "top_k": 8,
  "debug": true
}
```

Example ask response:

```json
{
  "answer": "...",
  "citations": [
    {
      "chunk_id": "chunk_123",
      "path": "notes/rag/evaluation.md",
      "heading_path": "RAG > Evaluation"
    }
  ],
  "debug": {
    "vector_results": [],
    "keyword_results": [],
    "hybrid_results": [],
    "final_context": "..."
  }
}
```

## Error Handling

- Invalid Markdown root: show a clear path validation error.
- Empty knowledge base: ask the user to index documents first.
- Embedding provider failure: preserve document scan state and mark indexing as failed.
- Partial indexing failure: record failed files and continue indexing the rest where safe.
- Retrieval returns no results: answer that no relevant notes were found.
- LLM failure: show retrieved evidence so the user can still inspect search quality.

## Testing Strategy

### Unit Tests

- Markdown scanning.
- Markdown chunking, especially headings, code blocks, and long sections.
- Hash-based change detection.
- Score normalization and hybrid merge logic.

### Integration Tests

- Index a small fixture Markdown folder.
- Ask known questions and assert expected chunks are retrieved.
- Compare vector, keyword, and hybrid outputs.
- Verify citations map to existing chunks.

### Manual Evaluation

Create a small evaluation set:

- 10 semantic questions.
- 10 exact keyword questions.
- 10 mixed questions.
- 5 questions that should return insufficient evidence.

Track:

- Did retrieval find the right chunk?
- Did the final answer cite the right source?
- Did the answer avoid unsupported claims?

## Milestones

### Milestone 1: Minimal Observable RAG

- Scan Markdown folder.
- Chunk files.
- Store chunks.
- Run vector search.
- Show top-k chunks in Web UI.

### Milestone 2: Keyword And Hybrid Retrieval

- Add SQLite FTS5.
- Add keyword mode.
- Add hybrid merge.
- Show separate vector, keyword, and merged rankings.

### Milestone 3: Answer Generation With Citations

- Add LLM answer generation.
- Build context from retrieved chunks.
- Return citations.
- Show final prompt context in debug UI.

### Milestone 4: Dual-Mode Product

- Add normal Q&A mode.
- Keep debug mode available.
- Add question history.
- Add basic settings for retrieval weights and top-k.

### Milestone 5: Advanced RAG Learning

- Add reranker.
- Add query rewriting.
- Add metadata filters.
- Add evaluation dashboard.

## Approval Gate

This design is approved for writing an implementation plan once the user reviews the document and confirms no major changes are needed.
