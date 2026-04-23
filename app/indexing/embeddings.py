from __future__ import annotations

from hashlib import blake2b
from math import sqrt
from pathlib import Path
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Converts text into dense numeric vectors."""

    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class HashEmbeddingProvider:
    """Small deterministic provider for local tests and dependency-free demos.

    This is not a semantic model. It exists so the indexing and retrieval pipeline
    can be developed before installing model dependencies.
    """

    name = "hash"

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_normalize(_hash_embed(text, self.dimensions)) for text in texts]


class SentenceTransformersEmbeddingProvider:
    """Embedding provider backed by a local SentenceTransformers model."""

    name = "sentence-transformers"

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in embeddings]

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the local-embedding "
                "extra before using the bge-m3 provider."
            ) from exc

        self._model = SentenceTransformer(str(self.model_path))
        return self._model


def create_embedding_provider(
    provider: str,
    model_path: Path | str | None = None,
) -> EmbeddingProvider:
    if provider == "hash":
        return HashEmbeddingProvider()
    if provider in {"sentence-transformers", "bge-m3"}:
        if model_path is None:
            raise ValueError("model_path is required for sentence-transformers provider")
        return SentenceTransformersEmbeddingProvider(model_path)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _hash_embed(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return vector


def _tokens(text: str) -> list[str]:
    normalized = []
    current = []
    for char in text.lower():
        if char.isalnum() or char in {"_", "-"}:
            current.append(char)
            continue
        if current:
            normalized.append("".join(current))
            current = []
    if current:
        normalized.append("".join(current))
    return normalized


def _normalize(vector: list[float]) -> list[float]:
    magnitude = sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]

