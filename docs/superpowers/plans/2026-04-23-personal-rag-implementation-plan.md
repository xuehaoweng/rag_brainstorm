# Personal Markdown RAG Implementation Plan

Date: 2026-04-23

Design: `docs/superpowers/specs/2026-04-23-personal-rag-design.md`

## Goal

Implement the project in small, observable milestones so the system is useful as both a personal Markdown knowledge base and a RAG learning lab.

The first implementation should avoid unnecessary framework complexity. Each RAG stage should be inspectable through logs, API responses, or the Web debug UI.

## Milestone 0: Project Skeleton

Deliverables:

- Create Python backend project structure.
- Add FastAPI application entrypoint.
- Add minimal frontend or server-rendered UI entrypoint.
- Add configuration file support.
- Add local data directory layout.
- Add basic developer commands.

Suggested structure:

```text
self_rag/
  app/
    main.py
    config.py
    db.py
    documents/
    indexing/
    retrieval/
    answering/
    web/
  data/
    sqlite/
    vector/
  tests/
  docs/
```

Acceptance checks:

- Backend starts locally.
- Health endpoint returns OK.
- SQLite database can be initialized.

## Milestone 1: Markdown Loading And Chunking

Deliverables:

- Scan a configured Markdown root directory.
- Load `.md` files recursively.
- Store document metadata in SQLite.
- Implement structure-aware Markdown chunking.
- Store chunks with source metadata.

Core files:

- `app/documents/loader.py`
- `app/documents/models.py`
- `app/indexing/chunker.py`
- `app/db.py`

Learning focus:

- Why document parsing quality matters.
- How chunk size affects retrieval.
- Why source metadata is required for citations.

Acceptance checks:

- Given a fixture Markdown folder, scanner finds expected files.
- Chunker preserves heading path.
- Chunker does not split inside fenced code blocks.
- Changed file hash triggers re-index eligibility.

## Milestone 2: Vector Retrieval

Deliverables:

- Add embedding provider abstraction.
- Add one concrete embedding implementation.
- Add vector store abstraction.
- Index chunks into the vector store.
- Add vector search API.
- Show vector top-k results in the Web UI.

Core files:

- `app/indexing/embeddings.py`
- `app/indexing/vector_store.py`
- `app/retrieval/vector.py`
- `app/retrieval/schemas.py`

Learning focus:

- What embeddings represent.
- Why semantic search can match different wording.
- What top-k means.
- How score interpretation differs by vector backend.

Acceptance checks:

- Indexing creates one vector record per chunk.
- Query returns ranked chunks.
- API response includes score, chunk text, file path, and heading path.

## Milestone 3: Keyword Retrieval

Deliverables:

- Add SQLite FTS5 table for chunks.
- Index chunk text into FTS.
- Add keyword retrieval mode.
- Show keyword top-k results separately in the debug UI.

Core files:

- `app/indexing/keyword_index.py`
- `app/retrieval/keyword.py`

Learning focus:

- Why exact terms, names, versions, paths, and commands often need keyword search.
- Difference between lexical matching and semantic matching.

Acceptance checks:

- Exact terms from notes are retrievable.
- Queries for filenames, commands, or rare terms perform better in keyword mode than vector-only mode.

## Milestone 4: Hybrid Retrieval

Deliverables:

- Add score normalization.
- Merge vector and keyword candidates by chunk id.
- Add configurable vector and keyword weights.
- Return debug details for raw and normalized scores.
- Add `hybrid` mode to the Ask UI.

Core files:

- `app/retrieval/hybrid.py`
- `app/retrieval/scoring.py`

Learning focus:

- Why hybrid retrieval is usually more robust than pure vector retrieval.
- How score normalization changes ranking.
- How weight choices affect precision and recall.

Acceptance checks:

- Hybrid response includes vector results, keyword results, and merged results.
- Duplicate chunks are merged.
- Final ranking changes when retrieval weights change.

## Milestone 5: Answer Generation With Citations

Deliverables:

- Add LLM provider abstraction.
- Build prompt context from selected chunks.
- Generate answer from retrieved context.
- Return citations with file path and heading path.
- Show final context in debug UI.

Core files:

- `app/answering/llm.py`
- `app/answering/prompts.py`
- `app/answering/engine.py`

Learning focus:

- Difference between retrieval quality and generation quality.
- Why citations reduce unsupported answers but do not guarantee correctness.
- How context length and top-k affect answer quality.

Acceptance checks:

- Answer cites source chunks.
- If retrieval returns no evidence, answer says evidence is insufficient.
- Debug UI shows final context sent to the model.

## Milestone 6: Web Debug Console

Deliverables:

- Index page.
- Ask page.
- Retrieval debug panel.
- Source chunk panel.
- Basic settings for top-k, retrieval mode, and hybrid weights.

Learning focus:

- How to inspect a RAG failure.
- How to compare vector, keyword, and hybrid retrieval.
- How to tune retrieval before changing models.

Acceptance checks:

- User can index a Markdown folder from the UI.
- User can ask a question and inspect all intermediate retrieval results.
- User can open source chunks from citations.

## Milestone 7: Evaluation Harness

Deliverables:

- Add small evaluation dataset format.
- Add script or endpoint to run evaluation questions.
- Track expected source chunks.
- Report retrieval hit rate and citation quality.

Suggested evaluation categories:

- Semantic questions.
- Exact keyword questions.
- Mixed semantic and keyword questions.
- Insufficient-evidence questions.

Learning focus:

- How to know whether a RAG change improved the system.
- Why anecdotal testing is not enough.

Acceptance checks:

- Evaluation run produces a readable report.
- Report includes which expected chunks were found or missed.

## Implementation Order

1. Milestone 0: Project skeleton.
2. Milestone 1: Markdown loading and chunking.
3. Milestone 2: Vector retrieval.
4. Milestone 3: Keyword retrieval.
5. Milestone 4: Hybrid retrieval.
6. Milestone 5: Answer generation with citations.
7. Milestone 6: Web debug console.
8. Milestone 7: Evaluation harness.

## Recommended First Coding Slice

Start with Milestone 0 and the smallest useful part of Milestone 1:

- FastAPI app with health endpoint.
- SQLite initialization.
- Markdown scanner.
- Markdown chunker.
- One API endpoint that scans a configured folder and returns chunks.
- Unit tests for scanner and chunker.

This first slice avoids model/API dependency and lets the project validate local Markdown handling before adding embeddings.

## Risks And Mitigations

- Risk: The project becomes a generic chat app too early.
  Mitigation: Keep debug visibility as a non-negotiable MVP requirement.

- Risk: Chunking quality silently damages retrieval.
  Mitigation: Add chunk preview UI and unit tests before tuning embeddings.

- Risk: Model provider choices distract from RAG mechanics.
  Mitigation: Use provider interfaces and defer provider optimization.

- Risk: Hybrid scores are misleading.
  Mitigation: Always expose raw score, normalized score, and final score in debug output.

- Risk: Answers look correct but are unsupported.
  Mitigation: Require citations and include insufficient-evidence behavior.

## Done Criteria For MVP

The MVP is complete when a user can:

- Configure a Markdown folder.
- Build an index.
- Ask a question.
- Choose vector, keyword, or hybrid retrieval.
- Inspect retrieved chunks and scores.
- Receive an answer with citations.
- Open or identify the original Markdown source for each citation.

