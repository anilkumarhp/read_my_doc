import pytest

from evaluate import parse_verdict


# well-formed verdicts

@pytest.mark.parametrize("verdict,expected", [
    ("FAITHFUL: yes", "yes"),
    ("FAITHFUL: no", "no"),
    ("FAITHFUL: partial", "partial"),
])
def test_exact_format_is_parsed(verdict, expected):
    assert parse_verdict(verdict) == expected


def test_parsing_is_case_insensitive():
    assert parse_verdict("faithful: YES") == "yes"
    assert parse_verdict("FAITHFUL: YES, looks supported") == "yes"


def test_leading_and_trailing_whitespace_is_tolerated():
    assert parse_verdict("   FAITHFUL: partial  \n") == "partial"


# models that ramble past the label

def test_trailing_prose_after_label_is_ignored():
    verdict = "FAITHFUL: no, the answer adds a detail not in the sources."
    assert parse_verdict(verdict) == "no"


def test_missing_faithful_prefix_falls_back_to_leading_token():
    assert parse_verdict("No, because sources [1] and [2] differ.") == "no"
    assert parse_verdict("Yes. Every claim is supported.") == "yes"


def test_colon_is_optional():
    assert parse_verdict("FAITHFUL yes") == "yes"


# unparseable / failure verdicts

def test_unknown_when_no_token_present():
    assert parse_verdict("The answer seems fine but I am not certain.") == "unknown"


def test_empty_verdict_is_unknown():
    assert parse_verdict("") == "unknown"


def test_judge_failure_string_is_unknown():
    assert parse_verdict("FAITHFUL: unknown, judge call failed (boom)") == "unknown"
