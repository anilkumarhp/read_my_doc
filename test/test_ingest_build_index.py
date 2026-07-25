import logging

import pytest
from unittest.mock import MagicMock, patch

import ingest
from ingest import build_index


# helpers

def make_collection():
    """A fake Chroma collection that records what was added to it."""
    return MagicMock(name="collection")


def patched_chroma(collection):
    """Patch out the embedding model and Chroma client for build_index.

    Returns a context-managing tuple of patchers already started; the caller
    is responsible for stopping them (use via the `chroma` fixture below).
    """
    client = MagicMock(name="client")
    client.create_collection.return_value = collection
    return client


@pytest.fixture
def chroma():
    """Patch embedding function + Chroma client. Yields the fake collection."""
    collection = make_collection()
    client = MagicMock(name="client")
    client.create_collection.return_value = collection

    with patch.object(
        ingest.embedding_functions, "SentenceTransformerEmbeddingFunction"
    ), patch.object(ingest.chromadb, "PersistentClient", return_value=client):
        yield collection, client


# empty / no-op cases

def test_no_documents_prints_and_returns(tmp_path, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="ingest"):
        build_index(str(tmp_path))

    out = capsys.readouterr().out
    assert "Nothing to ingest" in out
    assert "nothing to ingest" in caplog.text.lower()


def test_no_documents_never_touches_chroma(tmp_path):
    with patch.object(ingest.chromadb, "PersistentClient") as client:
        build_index(str(tmp_path))

    client.assert_not_called()


# happy path

def test_documents_are_chunked_and_added(tmp_path, chroma, capsys):
    collection, _ = chroma
    (tmp_path / "a.txt").write_text("Alpha beta gamma.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Delta epsilon zeta.", encoding="utf-8")

    build_index(str(tmp_path))

    collection.add.assert_called_once()
    kwargs = collection.add.call_args.kwargs
    assert len(kwargs["documents"]) == len(kwargs["ids"]) == len(kwargs["metadatas"])
    assert len(kwargs["documents"]) >= 2
    sources = {m["source"] for m in kwargs["metadatas"]}
    assert sources == {"a.txt", "b.txt"}

    out = capsys.readouterr().out
    assert "Ingested 2 document(s)" in out


def test_ids_are_unique(tmp_path, chroma):
    collection, _ = chroma
    (tmp_path / "a.txt").write_text("Alpha beta gamma. " * 100, encoding="utf-8")

    build_index(str(tmp_path))

    ids = collection.add.call_args.kwargs["ids"]
    assert len(ids) == len(set(ids))


def test_chunk_index_metadata_is_sequential(tmp_path, chroma):
    collection, _ = chroma
    (tmp_path / "a.txt").write_text("Alpha beta gamma. " * 100, encoding="utf-8")

    build_index(str(tmp_path))

    metadatas = collection.add.call_args.kwargs["metadatas"]
    indices = [m["chunk_index"] for m in metadatas if m["source"] == "a.txt"]
    assert indices == list(range(len(indices)))


def test_existing_collection_is_deleted_first(tmp_path, chroma):
    _, client = chroma
    (tmp_path / "a.txt").write_text("Alpha beta gamma.", encoding="utf-8")

    build_index(str(tmp_path))

    client.delete_collection.assert_called_once_with(ingest.COLLECTION_NAME)


def test_missing_collection_on_delete_is_ignored(tmp_path, chroma, caplog):
    _, client = chroma
    client.delete_collection.side_effect = Exception("does not exist")
    (tmp_path / "a.txt").write_text("Alpha beta gamma.", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        build_index(str(tmp_path))

    # First-run delete failure must not abort ingestion.
    assert "No existing collection" in caplog.text


# error propagation

def test_load_error_propagates(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_index("folder_that_does_not_exist")


def test_embedding_model_failure_is_logged_and_raised(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("Alpha beta gamma.", encoding="utf-8")

    with patch.object(
        ingest.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        side_effect=RuntimeError("model download failed"),
    ):
        with caplog.at_level(logging.ERROR, logger="ingest"):
            with pytest.raises(RuntimeError):
                build_index(str(tmp_path))

    assert "Failed to load embedding model" in caplog.text


def test_collection_add_failure_is_logged_and_raised(tmp_path, chroma, caplog):
    collection, _ = chroma
    collection.add.side_effect = RuntimeError("write failed")
    (tmp_path / "a.txt").write_text("Alpha beta gamma.", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="ingest"):
        with pytest.raises(RuntimeError):
            build_index(str(tmp_path))

    assert "Failed to add" in caplog.text


def test_documents_with_only_whitespace_add_nothing(tmp_path, chroma, capsys):
    collection, _ = chroma
    (tmp_path / "blank.txt").write_text("    \n\n   ", encoding="utf-8")

    build_index(str(tmp_path))

    collection.add.assert_not_called()
    out = capsys.readouterr().out
    assert "Nothing to ingest" in out
