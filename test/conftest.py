import pytest
from unittest.mock import MagicMock, patch

import query


@pytest.fixture
def chroma():
    """Patch the embedding function + Chroma client used by query.retrieve.

    Yields (collection, client) — both MagicMocks. `client.get_collection`
    already returns the collection, so tests only set `collection.query`.
    """
    collection = MagicMock(name="collection")
    client = MagicMock(name="client")
    client.get_collection.return_value = collection

    with patch.object(
        query.embedding_functions, "SentenceTransformerEmbeddingFunction"
    ), patch.object(query.chromadb, "PersistentClient", return_value=client):
        yield collection, client


@pytest.fixture
def chroma_result():
    """Return a builder for a fake Chroma query() result (list-of-lists)."""
    def _make(documents, metadatas):
        return {"documents": [documents], "metadatas": [metadatas]}

    return _make
