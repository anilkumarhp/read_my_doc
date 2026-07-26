from query import build_prompt


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


def test_build_prompt_separates_sources_with_blank_line():
    retrieved = [("first chunk", {"source": "a.txt"}), ("second chunk", {"source": "b.txt"})]

    prompt = build_prompt("What?", retrieved)

    assert "first chunk\n\n[2]" in prompt


def test_build_prompt_has_header_and_instruction():
    prompt = build_prompt("What?", [("chunk", {"source": "a.txt"})])

    assert prompt.startswith("Sources:")
    assert prompt.rstrip().endswith("with citations.")
