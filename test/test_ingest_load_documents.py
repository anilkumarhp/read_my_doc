import builtins
import logging
import os

import pytest
from unittest.mock import patch

from ingest import load_documents


# helpers

def fail_open_for(filename: str, error: Exception):
    """Return an `open` replacement that raises `error` only for `filename`."""
    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        if os.path.basename(str(file)) == filename:
            raise error
        return real_open(file, *args, **kwargs)

    return fake_open


# happy path

def test_load_multiple_txt_files(tmp_path):
    (tmp_path / "a.txt").write_text("Hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("World", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert docs == [
        {"source": "a.txt", "text": "Hello"},
        {"source": "b.txt", "text": "World"},
    ]


def test_empty_txt_file(tmp_path):
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert docs == [
        {"source": "empty.txt", "text": ""}
    ]


def test_unicode_file(tmp_path):
    content = "こんにちは नमस्ते"

    (tmp_path / "unicode.txt").write_text(content, encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert docs[0]["text"] == content


def test_multiline_text(tmp_path):
    content = """
                Line 1
                Line 2
                Line 3
              """

    (tmp_path / "multi.txt").write_text(content, encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert docs[0]["text"] == content


def test_large_file(tmp_path):
    content = "Python " * 100000

    (tmp_path / "large.txt").write_text(content, encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert docs[0]["text"] == content


def test_output_structure(tmp_path):
    (tmp_path / "sample.txt").write_text("Sample", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert isinstance(docs, list)

    for doc in docs:
        assert isinstance(doc, dict)
        assert set(doc.keys()) == {"source", "text"}
        assert isinstance(doc["source"], str)
        assert isinstance(doc["text"], str)


# file filtering

def test_ignore_non_txt_files(tmp_path):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.md").write_text("Markdown", encoding="utf-8")
    (tmp_path / "c.csv").write_text("CSV", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0]["source"] == "a.txt"


def test_uppercase_extension_is_ignored(tmp_path):
    (tmp_path / "HELLO.TXT").write_text("Hello", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert docs == []


def test_hidden_txt_file(tmp_path):
    (tmp_path / ".hidden.txt").write_text("Secret", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0]["source"] == ".hidden.txt"


def test_subdirectory_named_txt_is_ignored(tmp_path):
    (tmp_path / "folder.txt").mkdir()

    docs = load_documents(str(tmp_path))

    assert docs == []


def test_empty_folder_returns_empty_list(tmp_path):
    docs = load_documents(str(tmp_path))

    assert docs == []


# ordering

def test_files_are_sorted(tmp_path):
    (tmp_path / "z.txt").write_text("Z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "m.txt").write_text("M", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert [doc["source"] for doc in docs] == [
        "a.txt",
        "m.txt",
        "z.txt",
    ]


# folder level errors

def test_non_existing_folder():
    with pytest.raises(FileNotFoundError):
        load_documents("folder_that_does_not_exist")


def test_non_existing_folder_is_logged(caplog):
    with caplog.at_level(logging.ERROR, logger="ingest"):
        with pytest.raises(FileNotFoundError):
            load_documents("folder_that_does_not_exist")

    assert "Folder not found: folder_that_does_not_exist" in caplog.text
    assert "Traceback" in caplog.text


def test_permission_error_on_folder_is_raised_and_logged(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="ingest"):
        with patch("os.listdir", side_effect=PermissionError):
            with pytest.raises(PermissionError):
                load_documents(str(tmp_path))

    assert f"Permission denied for folder: {tmp_path}" in caplog.text
    assert "Traceback" in caplog.text


# file level errors

def test_permission_error_on_file_is_skipped(tmp_path):
    (tmp_path / "a.txt").write_text("Hello", encoding="utf-8")

    with patch("builtins.open", side_effect=PermissionError):
        docs = load_documents(str(tmp_path))

    assert docs == []


def test_invalid_utf8_file_is_skipped_and_logged(tmp_path, caplog):
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe invalid")
    (tmp_path / "good.txt").write_text("Good", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="ingest"):
        docs = load_documents(str(tmp_path))

    assert docs == [{"source": "good.txt", "text": "Good"}]
    assert "Invalid UTF-8 encoding: bad.txt" in caplog.text
    assert "UnicodeDecodeError" in caplog.text


def test_permission_error_on_file_is_logged(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="ingest"):
        with patch("builtins.open", fail_open_for("a.txt", PermissionError())):
            docs = load_documents(str(tmp_path))

    assert docs == [{"source": "b.txt", "text": "B"}]
    assert "Permission denied: a.txt" in caplog.text
    assert "Traceback" in caplog.text


def test_os_error_on_file_is_skipped_and_logged(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="ingest"):
        with patch("builtins.open", fail_open_for("a.txt", OSError("disk failure"))):
            docs = load_documents(str(tmp_path))

    assert docs == [{"source": "b.txt", "text": "B"}]
    assert "Failed to read a.txt" in caplog.text
    assert "disk failure" in caplog.text


def test_read_error_after_open_is_handled(tmp_path, caplog):
    """The read itself can fail, not only the open."""
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")

    class BrokenFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            raise OSError("read failed")

    with caplog.at_level(logging.ERROR, logger="ingest"):
        with patch("builtins.open", return_value=BrokenFile()):
            docs = load_documents(str(tmp_path))

    assert docs == []
    assert "Failed to read a.txt" in caplog.text


def test_all_files_failing_returns_empty_list(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="ingest"):
        with patch("builtins.open", side_effect=OSError("boom")):
            docs = load_documents(str(tmp_path))

    assert docs == []
    assert caplog.text.count("Failed to read") == 2


# logging on the happy path

def test_subdirectory_named_txt_logs_warning(tmp_path, caplog):
    (tmp_path / "folder.txt").mkdir()

    with caplog.at_level(logging.WARNING, logger="ingest"):
        docs = load_documents(str(tmp_path))

    assert docs == []
    assert "Skipping directory" in caplog.text
    assert "folder.txt" in caplog.text


def test_start_and_summary_are_logged(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="ingest"):
        load_documents(str(tmp_path))

    assert f"Loading documents from {tmp_path}" in caplog.text
    assert "Loaded a.txt" in caplog.text
    assert "Loaded b.txt" in caplog.text
    assert "Loaded 2 document(s)" in caplog.text


def test_summary_count_excludes_failed_files(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="ingest"):
        with patch("builtins.open", fail_open_for("a.txt", OSError("boom"))):
            load_documents(str(tmp_path))

    assert "Loaded 1 document(s)" in caplog.text


def test_empty_folder_logs_zero_documents(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="ingest"):
        load_documents(str(tmp_path))

    assert "Loaded 0 document(s)" in caplog.text
