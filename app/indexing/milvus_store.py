from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MilvusSearchHit:
    chunk_id: int
    score: float


class MilvusVectorStore:
    """Vector store backed by a Milvus instance using MilvusClient API."""

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        dimension: int,
    ) -> None:
        self.uri = f"http://{host}:{port}"
        self.collection_name = collection_name
        self.dimension = dimension
        self._client = None

    @property
    def client(self):
        if self._client is None:
            MilvusClient = _load_milvus_client()
            self._client = MilvusClient(uri=self.uri)
        return self._client

    def ensure_collection(self) -> None:
        """Create collection if it does not exist."""
        if self.client.has_collection(self.collection_name):
            return

        from pymilvus import CollectionSchema, DataType, FieldSchema

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="document_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="heading_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="start_line", dtype=DataType.INT32),
            FieldSchema(name="end_line", dtype=DataType.INT32),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="text_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        schema = CollectionSchema(fields=fields, description="rag_learning knowledge chunks")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="IP",
            index_type="IVF_FLAT",
            params={"nlist": 128},
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )

    def drop_collection(self) -> None:
        """Drop collection for a full rebuild."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

    def insert(
        self,
        ids: list[int],
        vectors: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """Insert vectors with metadata in batches."""
        batch_size = 1000
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            data = [
                {
                    "id": ids[i],
                    "document_path": metadata[i]["document_path"],
                    "heading_path": metadata[i]["heading_path"],
                    "chunk_index": metadata[i]["chunk_index"],
                    "start_line": metadata[i]["start_line"],
                    "end_line": metadata[i]["end_line"],
                    "text": metadata[i]["text"],
                    "text_hash": metadata[i]["text_hash"],
                    "embedding": vectors[i],
                }
                for i in range(start, end)
            ]
            self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query_vector: list[float], top_k: int) -> list[MilvusSearchHit]:
        """Search for nearest vectors, return chunk IDs and scores."""
        self.client.load_collection(self.collection_name)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="embedding",
            search_params={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["id"],
        )
        return [
            MilvusSearchHit(chunk_id=int(hit["id"]), score=float(hit["distance"]))
            for hit in results[0]
        ]

    def count(self) -> int:
        """Return number of entities in collection."""
        if not self.client.has_collection(self.collection_name):
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats["row_count"])

    def has_collection(self) -> bool:
        return self.client.has_collection(self.collection_name)


def _load_milvus_client():
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError(
            "pymilvus is not installed. Install the milvus extra: pip install -e '.[milvus]'"
        ) from exc
    return MilvusClient
