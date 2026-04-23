# RAG Roadmap And Learning Plan

Date: 2026-04-23

This document records the next engineering directions for the personal Markdown RAG console and the learning path needed to build real RAG capability.

## Current State

The project currently supports:

- Markdown folder scanning.
- Structure-aware Markdown chunking.
- Local `bge-m3` embedding provider.
- Dependency-free `hash` embedding provider for tests.
- Vector retrieval.
- Keyword retrieval.
- Hybrid retrieval.
- React debug console showing the retrieval chain.
- Server-friendly startup scripts.

The system is currently a learning and debugging console, not yet a full personal knowledge assistant.

## Engineering Roadmap

### 1. Persistent Index

Goal: avoid scanning, chunking, and embedding the whole folder on every query.

Build:

- Store document metadata in SQLite.
- Store chunk metadata and text in SQLite.
- Store embeddings in a local vector index.
- Track file hash and modified time.
- Re-index only changed files.

Why it matters:

- Makes retrieval fast enough for daily use.
- Makes evaluation stable.
- Creates the foundation for multi-knowledge-base support.

Learning focus:

- Incremental indexing.
- Embedding lifecycle.
- Vector index persistence.
- Cache invalidation.

### 2. Answer Generation With Citations

Goal: turn retrieved chunks into grounded answers.

Build:

- LLM provider abstraction.
- Prompt builder.
- Context assembly from hybrid results.
- Citation format using file path, heading path, and line range.
- Insufficient-evidence behavior.
- Frontend panel: evidence -> context -> answer.

Why it matters:

- Retrieval alone is not RAG. RAG means retrieved evidence is used to ground generation.
- Citations help detect hallucination and debug weak evidence.

Learning focus:

- Context packing.
- Prompt grounding.
- Citation discipline.
- Hallucination failure modes.

### 3. Evaluation Harness

Goal: measure whether retrieval changes actually improve the system.

Build:

- A small evaluation file with query, expected files, and expected headings.
- Script to run evaluation against keyword, vector, and hybrid modes.
- Metrics:
  - hit@k
  - expected document found
  - expected chunk found
  - no-answer behavior for unsupported queries

Why it matters:

- Without evaluation, retrieval tuning becomes anecdotal.
- Small changes to chunking, weights, or thresholds can silently make other queries worse.

Learning focus:

- Retrieval evaluation.
- Precision and recall.
- Regression testing for RAG systems.
- Golden datasets.

### 4. Reranker

Goal: improve ordering after broad retrieval.

Build:

- Add reranker provider abstraction.
- Start with a local BGE reranker if available.
- Rerank top 20 hybrid candidates into final top 5.
- Display reranker scores in the retrieval chain.

Why it matters:

- Retriever finds candidates quickly.
- Reranker judges candidate-query relevance more carefully.
- This often improves answer quality more than changing the embedding model.

Learning focus:

- Retriever vs reranker.
- Cross-encoder reranking.
- Latency and quality tradeoff.

### 5. Better Chunking

Goal: reduce noisy chunks and improve evidence quality.

Build:

- Frontmatter-aware parsing.
- Remove or separately store image-only and heading-only blocks.
- Keep title and metadata as searchable context.
- Add configurable chunk size and overlap.
- Preview chunk quality in the frontend.

Why it matters:

- Bad chunking causes bad retrieval.
- Chunking errors are often mistaken for model errors.

Learning focus:

- Markdown structure.
- Chunk size tradeoffs.
- Metadata enrichment.
- Evidence granularity.

### 6. Multi-Source Knowledge

Goal: expand beyond Markdown after the Markdown pipeline is stable.

Build later:

- PDF ingestion.
- Web article ingestion.
- Obsidian vault support.
- Code repository ingestion.
- Tags and metadata filters.

Do not start here first. More data formats will amplify existing retrieval problems if the Markdown pipeline is not solid.

## Learning Roadmap

### Stage 1: Understand Retrieval

Questions to answer:

- What is a chunk?
- Why does chunk size affect retrieval?
- Why does pure vector retrieval fail on exact terms?
- Why does pure keyword retrieval miss semantic matches?
- What does `top_k` really mean?
- Why can low-scoring candidates still appear?

Experiments:

- Ask the same query in keyword, vector, and hybrid modes.
- Compare retrieved chunks side by side.
- Change top-k and observe noise.
- Search for exact filenames, commands, Chinese terms, and conceptual questions.

### Stage 2: Understand Embeddings

Questions to answer:

- What does an embedding vector represent?
- Why does `bge-m3` output 1024 dimensions?
- Why do we normalize embeddings?
- Why is cosine similarity used?
- Why is similarity score not a probability?

Experiments:

- Compare `hash` vs `bge-m3`.
- Query with synonyms.
- Query with exact technical terms.
- Query with unsupported topics and inspect false positives.

### Stage 3: Understand Hybrid Search

Questions to answer:

- What should keyword retrieval handle?
- What should vector retrieval handle?
- How should scores be normalized?
- Why does hybrid search need thresholds?
- When should keyword weight be higher than vector weight?

Experiments:

- Tune vector/keyword weights.
- Add and remove score thresholds.
- Track which queries benefit from keyword search.
- Track which queries benefit from vector search.

### Stage 4: Understand Grounded Generation

Questions to answer:

- How does retrieved evidence become LLM context?
- How much context should be passed?
- What should the model do when evidence is insufficient?
- How do citations reduce hallucination?
- Why can cited answers still be wrong?

Experiments:

- Generate answers from top 3 vs top 8 chunks.
- Force the model to cite every claim.
- Ask unsupported questions.
- Compare answers with and without source evidence.

### Stage 5: Understand Evaluation

Questions to answer:

- What is hit@k?
- What is a golden query set?
- How do we evaluate no-answer behavior?
- How do we avoid overfitting to a few manual examples?

Experiments:

- Create 30 test queries:
  - 10 exact keyword queries.
  - 10 semantic concept queries.
  - 5 mixed queries.
  - 5 unsupported queries.
- Run them after every retrieval change.
- Track whether quality improves or regresses.

## Recommended Next Step

Build persistent indexing next.

Reason:

- It removes the biggest current inefficiency.
- It makes the frontend feel much faster.
- It creates stable data needed for answer generation, reranking, and evaluation.

Suggested next milestone:

```text
Index page:
  scan Markdown folder
  write documents/chunks to SQLite
  create embeddings once
  save vector index
  show index status

Ask page:
  query existing index
  do not re-embed all chunks on every request
```

## Practical Rule

Do not add more document formats until the Markdown RAG loop is strong:

```text
good chunking -> stable index -> hybrid retrieval -> cited answer -> evaluation
```

This sequence builds real RAG capability instead of only accumulating integrations.

