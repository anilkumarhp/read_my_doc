"""
evaluate.py — A lightweight faithfulness check.

This is intentionally simple, not the full Ragas/DeepEval treatment used in
the later, production-grade projects in this series. It asks the same local
model, acting as a judge, whether the generated answer's claims are actually
supported by the retrieved sources, the core idea of faithfulness, without
the full metric suite.

Usage:
    python evaluate.py eval_questions.txt
"""

import argparse
import logging
import re
import sys
from collections import Counter

import ollama

from query import IndexNotFoundError, generate_answer, retrieve

logger = logging.getLogger(__name__)

# A stronger general model than the 3B generator makes a more reliable judge.
# Swap for a smaller one (e.g. "gemma3:4b") if you want faster, cheaper runs.
JUDGE_MODEL = "mistral-nemo"

JUDGE_PROMPT = """You are checking whether an answer is faithful to its \
sources, meaning every factual claim in the answer is actually supported \
by the numbered sources, not introduced from outside knowledge.

Sources:
{sources}

Answer to check:
{answer}

Reply with ONE line and nothing else, no explanation, in exactly this format:
FAITHFUL: yes
Use "yes" if every claim is supported by the sources, "partial" if some \
claims are supported but at least one is not, or "no" if the answer is not \
supported. The first word after the colon must be yes, no, or partial."""


def parse_verdict(verdict: str) -> str:
    """Normalize a judge verdict to 'yes' | 'no' | 'partial' | 'unknown'.

    Tolerant of models that don't follow the format exactly: it looks for a
    yes/no/partial token after "FAITHFUL:", then falls back to a leading
    token, and only gives up ("unknown") when neither is present.
    """
    text = verdict.strip().lower()

    match = re.search(r"faithful\s*:?\s*(yes|no|partial)", text)
    if match:
        return match.group(1)

    match = re.match(r"(yes|no|partial)\b", text)
    if match:
        return match.group(1)

    return "unknown"


def judge_faithfulness(retrieved, answer: str) -> str:
    sources_text = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, (chunk, _meta) in enumerate(retrieved)
    )
    prompt = JUDGE_PROMPT.format(sources=sources_text, answer=answer)
    try:
        response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        # A judge failure must not abort the run; report it as an unknown
        # verdict but log the cause so it is not lost.
        logger.exception("Judge call failed with model %s", JUDGE_MODEL)
        return f"FAITHFUL: unknown, judge call failed ({e})"


def judge_faithfulness(retrieved, answer: str) -> str:
    sources_text = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, (chunk, _meta) in enumerate(retrieved)
    )
    prompt = JUDGE_PROMPT.format(sources=sources_text, answer=answer)
    try:
        response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        # A judge failure must not abort the run; report it as an unknown
        # verdict but log the cause so it is not lost.
        logger.exception("Judge call failed with model %s", JUDGE_MODEL)
        return f"FAITHFUL: unknown, judge call failed ({e})"


def run_eval(questions_file: str) -> None:
    logger.info("Running eval from %s", questions_file)

    try:
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.exception("Questions file not found: %s", questions_file)
        raise
    except OSError:
        logger.exception("Failed to read questions file: %s", questions_file)
        raise

    if not questions:
        logger.warning("No questions found in %s", questions_file)
        print(f"No questions found in {questions_file}.")
        return

    logger.info("Loaded %d question(s)", len(questions))

    results = []
    for question in questions:
        try:
            retrieved = retrieve(question)
        except IndexNotFoundError:
            # The index is missing for every question, not just this one —
            # stop and let the boundary report it.
            raise
        except Exception:
            logger.exception("Retrieval failed, skipping question: %s", question)
            print(f"[skip] '{question}', retrieval error, see logs.")
            continue

        if not retrieved:
            logger.warning("Nothing retrieved for question: %s", question)
            print(f"[skip] '{question}', nothing retrieved, is the index built?")
            continue

        answer = generate_answer(question, retrieved)
        verdict = judge_faithfulness(retrieved, answer)
        results.append((question, verdict))
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Verdict: {verdict}")

    if not results:
        logger.warning("No answers were evaluated")
        print("No answers were evaluated.")
        return

    counts = Counter(parse_verdict(v) for _, v in results)
    faithful_count = counts["yes"]
    logger.info(
        "%d/%d answers judged fully faithful (%s)",
        faithful_count, len(results), dict(counts),
    )
    print(f"\n{'=' * 50}")
    print(f"{faithful_count}/{len(results)} answers judged fully faithful.")
    print(
        f"Breakdown: {counts['yes']} yes, {counts['partial']} partial, "
        f"{counts['no']} no, {counts['unknown']} unknown."
    )


if __name__ == "__main__":
    from logger import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Run a faithfulness check over a set of questions.")
    parser.add_argument(
        "questions_file",
        nargs="?",
        default="eval_questions.txt",
        help="A text file with one question per line.",
    )
    args = parser.parse_args()

    # Top-level boundary: turn expected failures into a clean message + exit
    # code instead of an uncaught traceback.
    try:
        run_eval(args.questions_file)
    except IndexNotFoundError as e:
        logger.error("%s", e)
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Questions file not found: %s", args.questions_file)
        print(f"Error: questions file not found: {args.questions_file}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error during evaluation")
        print("Unexpected error during evaluation. See logs for details.", file=sys.stderr)
        sys.exit(1)