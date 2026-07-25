import logging

import pytest

from ingest import recursive_split


# helpers

def visible(chunks) -> str:
    """All non-whitespace characters joined, so text can be compared losslessly.

    Chunk boundaries and joiners only ever change whitespace, never the text.
    """
    return "".join("".join(chunk.split()) for chunk in chunks)


# argument validation

@pytest.mark.parametrize("bad_text", [None, 123, [], {}, b"bytes"])
def test_non_string_text_raises_type_error(bad_text, caplog):
    with caplog.at_level(logging.ERROR, logger="ingest"):
        with pytest.raises(TypeError):
            recursive_split(bad_text, 100, 10)

    assert "Expected text to be str" in caplog.text


@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_non_positive_chunk_size_raises_value_error(chunk_size, caplog):
    with caplog.at_level(logging.ERROR, logger="ingest"):
        with pytest.raises(ValueError):
            recursive_split("Hello", chunk_size, 0)

    assert "Invalid chunk_size" in caplog.text


@pytest.mark.parametrize("overlap", [-1, -50])
def test_negative_overlap_raises_value_error(overlap, caplog):
    with caplog.at_level(logging.ERROR, logger="ingest"):
        with pytest.raises(ValueError):
            recursive_split("Hello", 100, overlap)

    assert "Invalid overlap" in caplog.text


@pytest.mark.parametrize("overlap", [100, 101, 500])
def test_overlap_not_smaller_than_chunk_size_raises(overlap, caplog):
    with caplog.at_level(logging.ERROR, logger="ingest"):
        with pytest.raises(ValueError):
            recursive_split("Hello", 100, overlap)

    assert "Invalid overlap" in caplog.text


def test_validation_runs_before_splitting():
    """Bad arguments are rejected even when the text is fine."""
    with pytest.raises(ValueError):
        recursive_split("A perfectly valid paragraph.", 10, 10)


def test_maximum_allowed_overlap_is_accepted():
    chunks = recursive_split("Alpha beta gamma. Delta epsilon zeta.", 20, 19)

    assert chunks


# empty input

@pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t", " \n \t \n "])
def test_empty_or_whitespace_text_returns_empty_list(text, caplog):
    with caplog.at_level(logging.WARNING, logger="ingest"):
        chunks = recursive_split(text, 100, 10)

    assert chunks == []
    assert "Text is empty or whitespace only" in caplog.text


# paragraph level splitting

def test_short_text_is_a_single_chunk():
    chunks = recursive_split("Hello world.", 100, 10)

    assert chunks == ["Hello world."]


def test_paragraphs_are_merged_while_they_fit():
    text = "First para.\n\nSecond para."

    chunks = recursive_split(text, 100, 10)

    assert chunks == ["First para.\n\nSecond para."]


def test_paragraphs_split_when_they_do_not_fit():
    text = "AAAA.\n\nBBBB.\n\nCCCC."

    chunks = recursive_split(text, 12, 0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 12 for chunk in chunks)


def test_blank_paragraphs_are_dropped():
    text = "First.\n\n   \n\nSecond."

    chunks = recursive_split(text, 100, 10)

    assert chunks == ["First.\n\nSecond."]


def test_multiple_blank_lines_treated_as_one_separator():
    text = "First.\n\n\n\n\nSecond."

    chunks = recursive_split(text, 100, 10)

    assert chunks == ["First.\n\nSecond."]


def test_windows_line_endings_are_handled():
    text = "First.\r\n\r\nSecond."

    chunks = recursive_split(text, 100, 10)

    assert len(chunks) == 1
    assert "First." in chunks[0]
    assert "Second." in chunks[0]


def test_leading_and_trailing_whitespace_is_stripped():
    text = "\n\n   Hello world.   \n\n"

    chunks = recursive_split(text, 100, 10)

    assert chunks == ["Hello world."]


# sentence level fallback

def test_long_paragraph_falls_back_to_sentences(caplog):
    text = "First sentence here. Second sentence here. Third sentence here."

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        chunks = recursive_split(text, 30, 0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert "splitting by sentence" in caplog.text


@pytest.mark.parametrize("terminator", [".", "!", "?"])
def test_all_sentence_terminators_are_recognised(terminator):
    text = f"First sentence here{terminator} Second sentence here{terminator}"

    chunks = recursive_split(text, 25, 0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 25 for chunk in chunks)


# hard cut of an oversized single sentence
#
# NOTE: the current algorithm keeps only sentence[:chunk_size] and DROPS the
# remainder. These tests pin that behaviour (and its warning) as it stands,
# not as ideal. See the data-loss note flagged with this change.

def test_oversized_sentence_is_hard_cut_with_warning(caplog):
    text = "word " * 40  # one 199-char "sentence", no terminators

    with caplog.at_level(logging.WARNING, logger="ingest"):
        chunks = recursive_split(text, 20, 0)

    assert "hard-cutting drops the remainder" in caplog.text
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_hard_cut_loses_the_remainder():
    """Documents the known data loss: 'ab' with chunk_size 1 yields only 'a'."""
    chunks = recursive_split("ab", 1, 0)

    assert chunks == ["a"]


def test_no_hard_cut_warning_when_everything_fits(caplog):
    text = "First sentence. Second sentence."

    with caplog.at_level(logging.WARNING, logger="ingest"):
        recursive_split(text, 200, 10)

    assert "hard-cutting" not in caplog.text


# chunk size invariant (overlap = 0 only)
#
# With overlap the tail of the previous chunk is prepended AFTER sizing, so a
# chunk can exceed chunk_size — see the overlap section. Without overlap the
# base chunker keeps every chunk within chunk_size.

@pytest.mark.parametrize("chunk_size", [1, 5, 20, 50, 200])
def test_no_chunk_exceeds_chunk_size_without_overlap(chunk_size):
    text = (
        "Alpha beta gamma. Delta epsilon zeta.\n\n"
        "A very long paragraph that keeps going and going without stopping soon.\n\n"
        "Short.\n\n"
        + "supercalifragilistic " * 5
    )

    chunks = recursive_split(text, chunk_size, 0)

    assert all(len(chunk) <= chunk_size for chunk in chunks)


def test_no_chunk_is_blank():
    text = "One.\n\n   \n\nTwo.\n\n\n\nThree."

    chunks = recursive_split(text, 10, 0)

    assert all(chunk.strip() for chunk in chunks)


def test_chunks_are_stripped_without_overlap():
    text = "   First.   \n\n   Second.   "

    chunks = recursive_split(text, 12, 0)

    for chunk in chunks:
        assert chunk == chunk.strip()


def test_returns_list_of_strings():
    chunks = recursive_split("Hello world.", 100, 10)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


# content preservation (when no hard cut fires)

def test_lossless_when_chunk_size_exceeds_every_sentence():
    text = "One two three.\n\nFour five six seven eight nine ten."

    for chunk_size in [60, 100, 500]:
        chunks = recursive_split(text, chunk_size, 0)
        assert visible(chunks) == visible([text]), chunk_size


def test_content_is_preserved_across_paragraph_splits():
    text = "Alpha beta.\n\nGamma delta.\n\nEpsilon zeta."

    chunks = recursive_split(text, 15, 0)

    joined = " ".join(chunks)
    for word in ["Alpha", "beta", "Gamma", "delta", "Epsilon", "zeta"]:
        assert word in joined


def test_unicode_text_is_split_safely():
    text = "こんにちは世界。\n\nनमस्ते दुनिया।\n\nHello world."

    chunks = recursive_split(text, 20, 0)

    assert chunks
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_is_deterministic():
    text = "One two three.\n\nFour five six.\n\nSeven eight nine."

    assert recursive_split(text, 20, 5) == recursive_split(text, 20, 5)


# overlap
#
# Overlap is applied post-hoc: each chunk after the first is prefixed with the
# last `overlap` characters of the previous chunk. This can push a chunk over
# chunk_size, which the tests below pin as current behaviour.

def test_overlap_prepends_tail_of_previous_chunk():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa."

    chunks = recursive_split(text, 25, 6)

    for previous, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[0] in previous


def test_overlap_makes_later_chunks_longer():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa."

    without = recursive_split(text, 25, 0)
    with_overlap = recursive_split(text, 25, 6)

    assert with_overlap != without
    assert len(with_overlap[1]) > len(without[1])


def test_overlap_can_exceed_chunk_size():
    """Post-hoc overlap deliberately allows a chunk to grow past chunk_size."""
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa."

    chunks = recursive_split(text, 25, 10)

    assert any(len(chunk) > 25 for chunk in chunks)


def test_zero_overlap_leaves_chunks_untouched():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa."

    with_zero = recursive_split(text, 25, 0)
    negative_impossible = recursive_split(text, 25, 0)

    assert with_zero == negative_impossible
    assert all(len(chunk) <= 25 for chunk in with_zero)


def test_overlap_does_not_apply_to_first_chunk():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa."

    without = recursive_split(text, 25, 0)
    with_overlap = recursive_split(text, 25, 6)

    assert with_overlap[0] == without[0]


# logging

def test_entry_and_summary_are_logged(caplog):
    text = "Hello world."

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        chunks = recursive_split(text, 100, 10)

    assert "Splitting 12 char(s)" in caplog.text
    assert "chunk_size=100" in caplog.text
    assert "overlap=10" in caplog.text
    assert f"Split text into {len(chunks)} chunk(s)" in caplog.text


def test_paragraph_count_is_logged_at_debug(caplog):
    text = "One.\n\nTwo.\n\nThree."

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        recursive_split(text, 100, 10)

    assert "Found 3 paragraph(s)" in caplog.text
