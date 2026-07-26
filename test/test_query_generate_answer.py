import logging

from unittest.mock import patch

import query
from query import generate_answer


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


def test_generate_answer_passes_built_prompt_as_user_message():
    retrieved = [("chunk body", {"source": "a.txt"})]
    expected = query.build_prompt("My question?", retrieved)

    with patch.object(
        query.ollama, "chat", return_value={"message": {"content": "ok"}}
    ) as chat:
        generate_answer("My question?", retrieved)

    user_message = chat.call_args.kwargs["messages"][1]
    assert user_message["content"] == expected


def test_generate_answer_uses_configured_model():
    with patch.object(
        query.ollama, "chat", return_value={"message": {"content": "ok"}}
    ) as chat:
        generate_answer("Q", [("chunk", {"source": "a.txt"})])

    assert chat.call_args.kwargs["model"] == query.OLLAMA_MODEL
