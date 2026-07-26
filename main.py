"""
main.py — Drive the read-my-doc RAG pipeline end to end.

The three stages are just function calls. Edit the constants below (or comment
out a stage) while iterating; no CLI needed for a demo you run yourself.

    ingest   -> build the local Chroma index from your .txt docs
    query    -> retrieve + answer a single question
    evaluate -> faithfulness check over a questions file
"""

import logging

from logger import setup_logging
from ingest import build_index
from query import IndexNotFoundError, main as run_query
from evaluate import run_eval

logger = logging.getLogger(__name__)

DOCS_FOLDER = "data/sample_docs"
QUESTION = "When does the JR pass expire on my Japan trip?"
QUESTIONS_FILE = "eval_questions.txt"


def main() -> None:
    setup_logging()

    build_index(DOCS_FOLDER)

    print("\n" + "=" * 60 + "\n")
    run_query(QUESTION)

    print("\n" + "=" * 60 + "\n")
    run_eval(QUESTIONS_FILE)


if __name__ == "__main__":
    try:
        main()
    except IndexNotFoundError as e:
        # Retrieval before the index exists — run the ingest stage first.
        print(f"Error: {e}")
    except Exception:
        logger.exception("Pipeline failed")
        print("Pipeline failed. See logs for details.")
