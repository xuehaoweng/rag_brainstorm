import pytest

from app.indexing.embeddings import HashEmbeddingProvider, create_embedding_provider


def test_hash_embedding_provider_is_deterministic() -> None:
    provider = HashEmbeddingProvider(dimensions=16)

    first = provider.embed_texts(["hybrid retrieval"])[0]
    second = provider.embed_texts(["hybrid retrieval"])[0]

    assert first == second
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_create_embedding_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        create_embedding_provider("unknown")


def test_bge_provider_can_be_created() -> None:
    provider = create_embedding_provider("bge-m3", model_path="models/BAAI/bge-m3")

    assert provider.name == "sentence-transformers"
