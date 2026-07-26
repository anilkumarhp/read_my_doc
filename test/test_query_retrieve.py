import logging

import pytest
from unittest.mock import patch

import query
from query import IndexNotFoundError, retrieve


# argument validation

@pytest.mark.parametrize("bad_question", ["", "   ", "\n\t"])
def test_empty_question_raises_value_error(bad_question, caplog):
    with caplog.at_level(logging.ERROR, logger="query"):
        with pytest.raises(ValueError):
            retrieve(bad_question)

    assert "Empty question" in caplog.text


@pytest.mark.parametrize("bad_top_k", [0, -1, -10])
def test_non_positive_top_k_raises_value_error(bad_top_k, caplog):
    with caplog.at_level(logging.ERROR, logger="query"):
        with pytest.raises(ValueError):
            retrieve("What is this?", top_k=bad_top_k)

    assert "Invalid top_k" in caplog.text


# happy path

def test_retrieve_returns_zipped_chunks_and_metadata(chroma, chroma_result):
    collection, _ = chroma
    collection.query.return_value = chroma_result(
        ["chunk one", "chunk two"],
        [{"source": "a.txt", "chunk_index": 0}, {"source": "b.txt", "chunk_index": 1}],
    )

    result = retrieve("question", top_k=2)

    assert result == [
        ("chunk one", {"source": "a.txt", "chunk_index": 0}),
        ("chunk two", {"source": "b.txt", "chunk_index": 1}),
    ]


def test_retrieve_passes_top_k_through(chroma, chroma_result):
    collection, _ = chroma
    collection.query.return_value = chroma_result([], [])

    retrieve("question", top_k=7)

    assert collection.query.call_args.kwargs["n_results"] == 7


def test_retrieve_logs_count(chroma, chroma_result, caplog):
    collection, _ = chroma
    collection.query.return_value = chroma_result(["one"], [{"source": "a.txt"}])

    with caplog.at_level(logging.INFO, logger="query"):
        retrieve("question")

    assert "Retrieved 1 chunk(s)" in caplog.text


# empty / malformed results

def test_retrieve_empty_collection_returns_empty_list(chroma, chroma_result):
    collection, _ = chroma
    collection.query.return_value = chroma_result([], [])

    assert retrieve("question") == []


def test_retrieve_handles_none_documents(chroma):
    collection, _ = chroma
    collection.query.return_value = {"documents": None, "metadatas": None}

    assert retrieve("question") == []


def test_retrieve_handles_missing_keys(chroma):
    collection, _ = chroma
    collection.query.return_value = {}

    assert retrieve("question") == []


# index / backend errors

def test_missing_collection_raises_index_not_found(chroma, caplog):
    _, client = chroma
    client.get_collection.side_effect = Exception("collection does not exist")

    with caplog.at_level(logging.ERROR, logger="query"):
        with pytest.raises(IndexNotFoundError) as exc:
            retrieve("question")

    assert "Run ingest.py first" in str(exc.value)
    assert isinstance(exc.value.__cause__, Exception)


def test_query_failure_is_logged_and_raised(chroma, caplog):
    collection, _ = chroma
    collection.query.side_effect = RuntimeError("backend down")

    with caplog.at_level(logging.ERROR, logger="query"):
        with pytest.raises(RuntimeError):
            retrieve("question")

    assert "Query failed" in caplog.text


def test_embedding_model_failure_is_logged_and_raised(caplog):
    with patch.object(
        query.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        side_effect=RuntimeError("model load failed"),
    ):
        with caplog.at_level(logging.ERROR, logger="query"):
            with pytest.raises(RuntimeError):
                retrieve("question")

    assert "Failed to load embedding model" in caplog.text
