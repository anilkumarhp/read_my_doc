import logging

from unittest.mock import patch

import evaluate
from evaluate import judge_faithfulness


def test_judge_returns_stripped_verdict():
    with patch.object(
        evaluate.ollama, "chat",
        return_value={"message": {"content": "  FAITHFUL: yes  \n"}},
    ):
        verdict = judge_faithfulness([("c", {})], "answer")

    assert verdict == "FAITHFUL: yes"


def test_judge_numbers_sources_in_prompt():
    with patch.object(
        evaluate.ollama, "chat", return_value={"message": {"content": "FAITHFUL: yes"}}
    ) as chat:
        judge_faithfulness([("first", {}), ("second", {})], "answer")

    prompt = chat.call_args.kwargs["messages"][0]["content"]
    assert "[1] first" in prompt
    assert "[2] second" in prompt


def test_judge_failure_is_graceful_and_logged(caplog):
    with patch.object(evaluate.ollama, "chat", side_effect=Exception("judge down")):
        with caplog.at_level(logging.ERROR, logger="evaluate"):
            verdict = judge_faithfulness([("c", {})], "answer")

    assert verdict.startswith("FAITHFUL: unknown")
    assert "judge down" in verdict
    assert "Judge call failed" in caplog.text


def test_judge_includes_answer_in_prompt():
    with patch.object(
        evaluate.ollama, "chat", return_value={"message": {"content": "FAITHFUL: yes"}}
    ) as chat:
        judge_faithfulness([("c", {})], "The tower is 300m tall.")

    prompt = chat.call_args.kwargs["messages"][0]["content"]
    assert "The tower is 300m tall." in prompt


def test_judge_uses_configured_model():
    with patch.object(
        evaluate.ollama, "chat", return_value={"message": {"content": "FAITHFUL: yes"}}
    ) as chat:
        judge_faithfulness([("c", {})], "answer")

    assert chat.call_args.kwargs["model"] == evaluate.JUDGE_MODEL
