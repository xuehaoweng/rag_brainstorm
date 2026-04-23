from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.db import init_db
from app.documents.loader import scan_markdown_folder
from app.documents.models import MarkdownChunk
from app.indexing.embeddings import create_embedding_provider
from app.indexing.chunker import chunk_markdown_document
from app.retrieval.schemas import RetrievedChunk, VectorSearchDebug
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.vector import VectorRetriever


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="Self RAG", lifespan=lifespan)


class IndexPreviewRequest(BaseModel):
    root: Path = Field(description="Local folder containing Markdown files")
    max_chars: int = Field(default=1_800, ge=300, le=8_000)
    overlap_chars: int = Field(default=160, ge=0, le=1_000)


class ChunkPreview(BaseModel):
    document_path: str
    chunk_index: int
    heading_path: str
    start_line: int
    end_line: int
    text: str
    text_hash: str


class IndexPreviewResponse(BaseModel):
    document_count: int
    chunk_count: int
    chunks: list[ChunkPreview]


class VectorSearchRequest(IndexPreviewRequest):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    embedding_provider: str | None = Field(
        default=None,
        description="Use 'hash' for dependency-free testing or 'bge-m3' for local model embeddings",
    )
    embedding_model_path: Path | None = None


class VectorSearchResponse(BaseModel):
    document_count: int
    chunk_count: int
    results: list[RetrievedChunk]
    debug: VectorSearchDebug


class KeywordSearchResponse(BaseModel):
    document_count: int
    chunk_count: int
    results: list[RetrievedChunk]


class HybridSearchResponse(BaseModel):
    document_count: int
    chunk_count: int
    results: list[RetrievedChunk]
    vector_results: list[RetrievedChunk]
    keyword_results: list[RetrievedChunk]
    debug: VectorSearchDebug


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/index/preview")
def preview_index(request: IndexPreviewRequest) -> IndexPreviewResponse:
    documents, chunks = _load_chunks(request)

    return IndexPreviewResponse(
        document_count=len(documents),
        chunk_count=len(chunks),
        chunks=[_chunk_preview(chunk) for chunk in chunks],
    )


@app.post("/api/retrieval/vector")
def vector_search(request: VectorSearchRequest) -> VectorSearchResponse:
    documents, chunks = _load_chunks(request)
    provider_name = request.embedding_provider or settings.embedding_provider
    model_path = request.embedding_model_path or settings.embedding_model_path

    try:
        provider = create_embedding_provider(provider_name, model_path=model_path)
        results, debug = VectorRetriever(provider).search(
            chunks=chunks,
            query=request.query,
            top_k=request.top_k,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return VectorSearchResponse(
        document_count=len(documents),
        chunk_count=len(chunks),
        results=results,
        debug=debug,
    )


@app.post("/api/retrieval/keyword")
def keyword_search(request: VectorSearchRequest) -> KeywordSearchResponse:
    documents, chunks = _load_chunks(request)
    try:
        results = KeywordRetriever().search(chunks=chunks, query=request.query, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KeywordSearchResponse(
        document_count=len(documents),
        chunk_count=len(chunks),
        results=results,
    )


@app.post("/api/retrieval/hybrid")
def hybrid_search(request: VectorSearchRequest) -> HybridSearchResponse:
    documents, chunks = _load_chunks(request)
    provider_name = request.embedding_provider or settings.embedding_provider
    model_path = request.embedding_model_path or settings.embedding_model_path

    try:
        provider = create_embedding_provider(provider_name, model_path=model_path)
        results, vector_results, keyword_results = HybridRetriever(provider).search(
            chunks=chunks,
            query=request.query,
            top_k=request.top_k,
        )
        query_vector = provider.embed_texts([request.query])[0]
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return HybridSearchResponse(
        document_count=len(documents),
        chunk_count=len(chunks),
        results=results,
        vector_results=vector_results,
        keyword_results=keyword_results,
        debug=VectorSearchDebug(
            provider=provider.name,
            indexed_chunk_count=len(chunks),
            query_vector_dimensions=len(query_vector),
        ),
    )


def _load_chunks(request: IndexPreviewRequest) -> tuple[list, list[MarkdownChunk]]:
    try:
        documents = scan_markdown_folder(request.root)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = [
        chunk
        for document in documents
        for chunk in chunk_markdown_document(
            document,
            max_chars=request.max_chars,
            overlap_chars=request.overlap_chars,
        )
    ]
    return documents, chunks


def _chunk_preview(chunk: MarkdownChunk) -> ChunkPreview:
    return ChunkPreview(
        document_path=chunk.document_path,
        chunk_index=chunk.chunk_index,
        heading_path=chunk.heading_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        text=chunk.text,
        text_hash=chunk.text_hash,
    )
