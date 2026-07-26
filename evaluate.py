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
import sys

import ollama

from query import IndexNotFoundError, generate_answer, retrieve

logger = logging.getLogger(__name__)

JUDGE_MODEL = "llama3.2"

JUDGE_PROMPT = """You are checking whether an answer is faithful to its \
sources, meaning every factual claim in the answer is actually supported \
by the numbered sources, not introduced from outside knowledge.

Sources:
{sources}

Answer to check:
{answer}

Respond with exactly one line in this format:
FAITHFUL: yes|no|partial, """


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

    faithful_count = sum(1 for _, v in results if v.upper().startswith("FAITHFUL: YES"))
    logger.info(
        "%d/%d answers judged fully faithful", faithful_count, len(results)
    )
    print(f"\n{'=' * 50}")
    print(f"{faithful_count}/{len(results)} answers judged fully faithful.")


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