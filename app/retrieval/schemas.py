from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    document_path: str
    chunk_index: int
    heading_path: str
    start_line: int
    end_line: int
    text: str
    text_hash: str


class VectorSearchDebug(BaseModel):
    provider: str
    indexed_chunk_count: int
    query_vector_dimensions: int

