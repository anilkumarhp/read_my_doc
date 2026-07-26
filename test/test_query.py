import logging

import pytest
from unittest.mock import MagicMock, patch

import query
from query import (
    IndexNotFoundError,
    build_prompt,
    generate_answer,
    main,
    retrieve,
)


# helpers

def chroma_result(documents, metadatas):
    """Shape a fake Chroma query() result (list-of-lists, one per query)."""
    return {"documents": [documents], "metadatas": [metadatas]}


@pytest.fixture
def chroma():
    """Patch embedding fn + Chroma client. Yields (collection, client)."""
    collection = MagicMock(name="collection")
    client = MagicMock(name="client")
    client.get_collection.return_value = collection

    with patch.object(
        query.embedding_functions, "SentenceTransformerEmbeddingFunction"
    ), patch.object(query.chromadb, "PersistentClient", return_value=client):
        yield collection, client


# retrieve — argument validation

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


# retrieve — happy path

def test_retrieve_returns_zipped_chunks_and_metadata(chroma):
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


def test_retrieve_passes_top_k_through(chroma):
    collection, _ = chroma
    collection.query.return_value = chroma_result([], [])

    retrieve("question", top_k=7)

    assert collection.query.call_args.kwargs["n_results"] == 7


def test_retrieve_logs_count(chroma, caplog):
    collection, _ = chroma
    collection.query.return_value = chroma_result(["one"], [{"source": "a.txt"}])

    with caplog.at_level(logging.INFO, logger="query"):
        retrieve("question")

    assert "Retrieved 1 chunk(s)" in caplog.text


# retrieve — empty / malformed results

def test_retrieve_empty_collection_returns_empty_list(chroma):
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


# retrieve — index / backend errors

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


# build_prompt

def test_build_prompt_numbers_sources_from_one():
    retrieved = [
        ("first chunk", {"source": "a.txt"}),
        ("second chunk", {"source": "b.txt"}),
    ]

    prompt = build_prompt("What?", retrieved)

    assert "[1] (from a.txt) first chunk" in prompt
    assert "[2] (from b.txt) second chunk" in prompt
    assert "Question: What?" in prompt


def test_build_prompt_tolerates_missing_source():
    prompt = build_prompt("What?", [("chunk", {})])

    assert "(from unknown)" in prompt


def test_build_prompt_empty_retrieved():
    prompt = build_prompt("What?", [])

    assert "Question: What?" in prompt


# generate_answer

def test_generate_answer_returns_model_content():
    with patch.object(
        query.ollama,
        "chat",
        return_value={"message": {"content": "The answer is 42. [1]"}},
    ):
        answer = generate_answer("Q", [("chunk", {"source": "a.txt"})])

    assert answer == "The answer is 42. [1]"


def test_generate_answer_reports_failure_gracefully(caplog):
    with patch.object(query.ollama, "chat", side_effect=Exception("connection refused")):
        with caplog.at_level(logging.ERROR, logger="query"):
            answer = generate_answer("Q", [("chunk", {"source": "a.txt"})])

    assert "Generation failed" in answer
    assert "connection refused" in answer
    assert query.OLLAMA_MODEL in answer
    # The failure is surfaced to the user but also logged.
    assert "Generation failed" in caplog.text


def test_generate_answer_sends_system_and_user_messages():
    with patch.object(
        query.ollama, "chat", return_value={"message": {"content": "ok"}}
    ) as chat:
        generate_answer("Q", [("chunk", {"source": "a.txt"})])

    messages = chat.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]


# main

def test_main_prints_sources_and_answer(chroma, capsys):
    collection, _ = chroma
    collection.query.return_value = chroma_result(
        ["chunk body"], [{"source": "a.txt", "chunk_index": 3}]
    )

    with patch.object(
        query.ollama, "chat", return_value={"message": {"content": "Final answer."}}
    ):
        main("What is in the doc?")

    out = capsys.readouterr().out
    assert "Retrieved sources:" in out
    assert "a.txt (chunk 3)" in out
    assert "Final answer." in out


def test_main_handles_empty_retrieval(chroma, capsys):
    collection, _ = chroma
    collection.query.return_value = chroma_result([], [])

    main("Anything?")

    out = capsys.readouterr().out
    assert "Nothing retrieved" in out


def test_main_tolerates_missing_metadata_keys(chroma, capsys):
    collection, _ = chroma
    collection.query.return_value = chroma_result(["body"], [{}])

    with patch.object(
        query.ollama, "chat", return_value={"message": {"content": "ans"}}
    ):
        main("Q?")

    out = capsys.readouterr().out
    assert "unknown (chunk ?)" in out


def test_main_does_not_swallow_index_error(chroma):
    _, client = chroma
    client.get_collection.side_effect = Exception("no collection")

    with pytest.raises(IndexNotFoundError):
        main("Q?")
