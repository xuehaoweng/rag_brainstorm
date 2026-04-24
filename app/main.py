from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.db import init_db
from app.documents.loader import scan_markdown_folder
from app.documents.models import MarkdownChunk
from app.generation.llm import create_llm_provider
from app.generation.prompt import build_answer_prompt
from app.indexing.embeddings import create_embedding_provider
from app.indexing.chunker import chunk_markdown_document
from app.indexing.persistent_index import (
    build_persistent_index,
    get_persistent_index_status,
    load_persistent_chunks,
    search_persistent_vector_index,
)
from app.retrieval.schemas import RetrievedChunk, VectorSearchDebug
from app.retrieval.hybrid import HybridRetriever, merge_results
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.vector import VectorRetriever


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="Self RAG", lifespan=lifespan)


class IndexPreviewRequest(BaseModel):
    root: Path = Field(description="Local folder containing Markdown files")
    query: str | None = Field(default=None, description="Optional query used to preview only relevant chunks")
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
    use_persistent_index: bool = Field(
        default=False,
        description="Search the previously built SQLite + FAISS index instead of scanning files on this request",
    )


class IndexBuildRequest(BaseModel):
    root: Path = Field(description="Local folder containing Markdown files")
    max_chars: int = Field(default=1_800, ge=300, le=8_000)
    overlap_chars: int = Field(default=160, ge=0, le=1_000)
    embedding_provider: str | None = Field(
        default=None,
        description="Use 'hash' for dependency-free testing or 'bge-m3' for local model embeddings",
    )
    embedding_model_path: Path | None = None


class IndexBuildResponse(BaseModel):
    document_count: int
    chunk_count: int
    indexed_chunk_count: int
    provider: str
    vector_dimensions: int
    root: str
    database_path: str
    faiss_index_path: str
    built_at: str


class IndexStatusResponse(BaseModel):
    exists: bool
    document_count: int
    chunk_count: int
    indexed_chunk_count: int
    provider: str | None
    vector_dimensions: int | None
    root: str | None
    database_path: str
    faiss_index_path: str
    built_at: str | None


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


class CitationResponse(BaseModel):
    source_id: int
    document_path: str
    heading_path: str
    start_line: int
    end_line: int


class AnswerRequest(VectorSearchRequest):
    mode: str = Field(default="hybrid", pattern="^(hybrid|vector|keyword)$")


class AnswerResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: list[RetrievedChunk]
    context: str
    mode: str
    model: str
    provider: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/index/preview")
def preview_index(request: IndexPreviewRequest) -> IndexPreviewResponse:
    documents, chunks = _load_chunks(request)
    chunk_count = len(chunks)
    if request.query and request.query.strip():
        chunks = KeywordRetriever().search(chunks=chunks, query=request.query, top_k=20)

    return IndexPreviewResponse(
        document_count=len(documents),
        chunk_count=chunk_count,
        chunks=[_chunk_preview(chunk) for chunk in chunks],
    )


@app.post("/api/index/build")
def build_index(request: IndexBuildRequest) -> IndexBuildResponse:
    provider_name = request.embedding_provider or settings.embedding_provider
    model_path = request.embedding_model_path or settings.embedding_model_path
    try:
        provider = create_embedding_provider(provider_name, model_path=model_path)
        result = build_persistent_index(
            root=request.root,
            embedding_provider=provider,
            max_chars=request.max_chars,
            overlap_chars=request.overlap_chars,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IndexBuildResponse(
        document_count=result.document_count,
        chunk_count=result.chunk_count,
        indexed_chunk_count=result.indexed_chunk_count,
        provider=result.provider,
        vector_dimensions=result.vector_dimensions,
        root=result.root,
        database_path=result.database_path,
        faiss_index_path=result.faiss_index_path,
        built_at=result.built_at,
    )


@app.get("/api/index/status")
def index_status() -> IndexStatusResponse:
    status = get_persistent_index_status()
    return IndexStatusResponse(
        exists=status.exists,
        document_count=status.document_count,
        chunk_count=status.chunk_count,
        indexed_chunk_count=status.indexed_chunk_count,
        provider=status.provider,
        vector_dimensions=status.vector_dimensions,
        root=status.root,
        database_path=status.database_path,
        faiss_index_path=status.faiss_index_path,
        built_at=status.built_at,
    )


@app.post("/api/retrieval/vector")
def vector_search(request: VectorSearchRequest) -> VectorSearchResponse:
    return _run_vector_search(request)


@app.post("/api/retrieval/keyword")
def keyword_search(request: VectorSearchRequest) -> KeywordSearchResponse:
    return _run_keyword_search(request)


@app.post("/api/retrieval/hybrid")
def hybrid_search(request: VectorSearchRequest) -> HybridSearchResponse:
    return _run_hybrid_search(request)


@app.post("/api/answer")
def answer_question(request: AnswerRequest) -> AnswerResponse:
    retrieval = _run_retrieval(mode=request.mode, request=request)
    prompt = build_answer_prompt(request.query, retrieval.results)
    try:
        provider = create_llm_provider()
        answer = provider.generate(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnswerResponse(
        answer=answer,
        citations=[
            CitationResponse(
                source_id=citation.source_id,
                document_path=citation.document_path,
                heading_path=citation.heading_path,
                start_line=citation.start_line,
                end_line=citation.end_line,
            )
            for citation in prompt.citations
        ],
        retrieved_chunks=retrieval.results,
        context=prompt.context,
        mode=request.mode,
        model=provider.model,
        provider=provider.name,
    )


def _run_retrieval(mode: str, request: VectorSearchRequest) -> VectorSearchResponse | KeywordSearchResponse | HybridSearchResponse:
    if mode == "vector":
        return _run_vector_search(request)
    if mode == "keyword":
        return _run_keyword_search(request)
    if mode == "hybrid":
        return _run_hybrid_search(request)
    raise ValueError(f"Unsupported retrieval mode: {mode}")


def _run_vector_search(request: VectorSearchRequest) -> VectorSearchResponse:
    if request.use_persistent_index:
        provider_name = request.embedding_provider or settings.embedding_provider
        model_path = request.embedding_model_path or settings.embedding_model_path
        try:
            provider = create_embedding_provider(provider_name, model_path=model_path)
            persistent_result = search_persistent_vector_index(
                query=request.query,
                top_k=request.top_k,
                embedding_provider=provider,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return VectorSearchResponse(
            document_count=persistent_result.document_count,
            chunk_count=persistent_result.chunk_count,
            results=persistent_result.results,
            debug=persistent_result.debug,
        )

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


def _run_keyword_search(request: VectorSearchRequest) -> KeywordSearchResponse:
    if request.use_persistent_index:
        status = get_persistent_index_status()
        chunks = load_persistent_chunks()
        if not chunks:
            raise HTTPException(status_code=400, detail="Persistent index does not contain chunks. Build it first.")
        try:
            results = KeywordRetriever().search(chunks=chunks, query=request.query, top_k=request.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return KeywordSearchResponse(
            document_count=status.document_count,
            chunk_count=status.chunk_count,
            results=results,
        )

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


def _run_hybrid_search(request: VectorSearchRequest) -> HybridSearchResponse:
    if request.use_persistent_index:
        provider_name = request.embedding_provider or settings.embedding_provider
        model_path = request.embedding_model_path or settings.embedding_model_path
        try:
            provider = create_embedding_provider(provider_name, model_path=model_path)
            vector_result = search_persistent_vector_index(
                query=request.query,
                top_k=max(request.top_k * 4, 20),
                embedding_provider=provider,
            )
            chunks = load_persistent_chunks()
            keyword_results = KeywordRetriever().search(
                chunks=chunks,
                query=request.query,
                top_k=max(request.top_k * 4, 20),
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        vector_results = vector_result.results
        results = merge_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=request.top_k,
        )
        return HybridSearchResponse(
            document_count=vector_result.document_count,
            chunk_count=vector_result.chunk_count,
            results=results,
            vector_results=vector_results[: request.top_k],
            keyword_results=keyword_results[: request.top_k],
            debug=vector_result.debug,
        )

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
