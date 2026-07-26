import logging

import pytest
from unittest.mock import patch

import evaluate
from evaluate import run_eval
from query import IndexNotFoundError


# helpers

def write_questions(tmp_path, *questions):
    path = tmp_path / "questions.txt"
    path.write_text("\n".join(questions), encoding="utf-8")
    return str(path)


ONE_CHUNK = [("c", {"source": "a.txt", "chunk_index": 0})]


# file handling

def test_missing_questions_file_is_logged_and_raised(caplog):
    with caplog.at_level(logging.ERROR, logger="evaluate"):
        with pytest.raises(FileNotFoundError):
            run_eval("no_such_file.txt")

    assert "Questions file not found" in caplog.text


def test_unreadable_questions_file_is_logged_and_raised(tmp_path, caplog):
    path = write_questions(tmp_path, "q1")

    with caplog.at_level(logging.ERROR, logger="evaluate"):
        with patch("builtins.open", side_effect=OSError("disk error")):
            with pytest.raises(OSError):
                run_eval(path)

    assert "Failed to read questions file" in caplog.text


def test_empty_questions_file_prints_message(tmp_path, capsys, caplog):
    path = write_questions(tmp_path, "", "   ", "")

    with caplog.at_level(logging.WARNING, logger="evaluate"):
        run_eval(path)

    assert "No questions found" in capsys.readouterr().out
    assert "No questions found" in caplog.text


def test_blank_lines_are_ignored(tmp_path, capsys):
    path = write_questions(tmp_path, "First question?", "", "Second question?")

    with patch.object(evaluate, "retrieve", return_value=ONE_CHUNK), \
         patch.object(evaluate, "generate_answer", return_value="ans"), \
         patch.object(evaluate.ollama, "chat", return_value={"message": {"content": "FAITHFUL: yes"}}):
        run_eval(path)

    out = capsys.readouterr().out
    assert "Q: First question?" in out
    assert "Q: Second question?" in out
    assert "2/2 answers judged fully faithful" in out


# happy path & counting

def test_faithful_count_only_counts_yes(tmp_path, capsys):
    path = write_questions(tmp_path, "q1", "q2", "q3")
    verdicts = iter(["FAITHFUL: yes", "FAITHFUL: partial", "FAITHFUL: no"])

    with patch.object(evaluate, "retrieve", return_value=ONE_CHUNK), \
         patch.object(evaluate, "generate_answer", return_value="ans"), \
         patch.object(evaluate.ollama, "chat",
                      side_effect=lambda *a, **k: {"message": {"content": next(verdicts)}}):
        run_eval(path)

    assert "1/3 answers judged fully faithful" in capsys.readouterr().out


def test_case_insensitive_yes_is_counted(tmp_path, capsys):
    path = write_questions(tmp_path, "q1")

    with patch.object(evaluate, "retrieve", return_value=ONE_CHUNK), \
         patch.object(evaluate, "generate_answer", return_value="ans"), \
         patch.object(evaluate.ollama, "chat", return_value={"message": {"content": "faithful: YES, ok"}}):
        run_eval(path)

    assert "1/1 answers judged fully faithful" in capsys.readouterr().out


def test_answer_and_verdict_are_printed_per_question(tmp_path, capsys):
    path = write_questions(tmp_path, "q1")

    with patch.object(evaluate, "retrieve", return_value=ONE_CHUNK), \
         patch.object(evaluate, "generate_answer", return_value="The generated answer."), \
         patch.object(evaluate.ollama, "chat",
                      return_value={"message": {"content": "FAITHFUL: partial"}}):
        run_eval(path)

    out = capsys.readouterr().out
    assert "A: The generated answer." in out
    assert "Verdict: FAITHFUL: partial" in out


def test_generated_answer_is_handed_to_the_judge(tmp_path):
    path = write_questions(tmp_path, "q1")

    with patch.object(evaluate, "retrieve", return_value=ONE_CHUNK), \
         patch.object(evaluate, "generate_answer", return_value="ANSWER-SENTINEL"), \
         patch.object(evaluate.ollama, "chat",
                      return_value={"message": {"content": "FAITHFUL: yes"}}) as chat:
        run_eval(path)

    judge_prompt = chat.call_args.kwargs["messages"][0]["content"]
    assert "ANSWER-SENTINEL" in judge_prompt


# per-question resilience

def test_question_with_no_retrieval_is_skipped(tmp_path, capsys):
    path = write_questions(tmp_path, "answerable", "unanswerable")
    retrievals = iter([ONE_CHUNK, []])

    with patch.object(evaluate, "retrieve", side_effect=lambda q, *a, **k: next(retrievals)), \
         patch.object(evaluate, "generate_answer", return_value="ans"), \
         patch.object(evaluate.ollama, "chat", return_value={"message": {"content": "FAITHFUL: yes"}}):
        run_eval(path)

    out = capsys.readouterr().out
    assert "[skip] 'unanswerable', nothing retrieved" in out
    assert "1/1 answers judged fully faithful" in out


def test_retrieval_error_skips_only_that_question(tmp_path, capsys, caplog):
    path = write_questions(tmp_path, "boom", "good")
    retrievals = iter([RuntimeError("backend hiccup"), ONE_CHUNK])

    def fake_retrieve(q, *a, **k):
        val = next(retrievals)
        if isinstance(val, Exception):
            raise val
        return val

    with patch.object(evaluate, "retrieve", side_effect=fake_retrieve), \
         patch.object(evaluate, "generate_answer", return_value="ans"), \
         patch.object(evaluate.ollama, "chat", return_value={"message": {"content": "FAITHFUL: yes"}}):
        with caplog.at_level(logging.ERROR, logger="evaluate"):
            run_eval(path)

    out = capsys.readouterr().out
    assert "[skip] 'boom', retrieval error" in out
    assert "1/1 answers judged fully faithful" in out
    assert "Retrieval failed" in caplog.text


def test_all_questions_skipped_prints_no_answers(tmp_path, capsys):
    path = write_questions(tmp_path, "q1", "q2")

    with patch.object(evaluate, "retrieve", return_value=[]):
        run_eval(path)

    out = capsys.readouterr().out
    assert "No answers were evaluated" in out
    assert "faithful" not in out  # no misleading 0/0 summary line


# index-missing is fatal

def test_missing_index_aborts_the_run(tmp_path):
    path = write_questions(tmp_path, "q1", "q2")

    with patch.object(evaluate, "retrieve", side_effect=IndexNotFoundError("no index")):
        with pytest.raises(IndexNotFoundError):
            run_eval(path)


def test_missing_index_stops_before_second_question(tmp_path):
    path = write_questions(tmp_path, "q1", "q2")
    calls = []

    def fake_retrieve(q, *a, **k):
        calls.append(q)
        raise IndexNotFoundError("no index")

    with patch.object(evaluate, "retrieve", side_effect=fake_retrieve):
        with pytest.raises(IndexNotFoundError):
            run_eval(path)

    # It must not keep looping once the index is known to be missing.
    assert calls == ["q1"]
