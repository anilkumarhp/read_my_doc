"""
query.py — Ask a question against the indexed documents.

This is the "retrieve, generate" half of the pipeline. Retrieval is plain
dense search, no reranking, no query rewriting, no hybrid search, the
naive baseline this whole series builds on top of. Generation includes a
lightweight citation requirement: the model is instructed to only make
claims it can attach a source to, and to say so plainly when the
retrieved chunks don't actually answer the question.

Usage:
    python query.py "When does the JR pass expire on my Japan trip?"
"""

import argparse
import logging
import sys

import chromadb
import ollama
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "personal_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_MODEL = "llama3.2"
TOP_K = 4


class IndexNotFoundError(RuntimeError):
    """Raised when the Chroma collection cannot be opened for querying."""

SYSTEM_PROMPT = """You are answering questions using only the numbered source \
passages provided below. Follow these rules strictly:

1. Only make a claim if it is directly supported by one of the numbered \
sources. After each claim, cite the source number in brackets, like [2].
2. If the sources do not contain enough information to answer the question, \
say so plainly instead of guessing. Do not fill gaps with general knowledge.
3. Keep the answer concise and directly responsive to the question."""


def retrieve(question: str, top_k: int = TOP_K):
    """Dense-search the index for the top_k chunks most relevant to question.

    Raises ValueError on bad arguments and IndexNotFoundError when the
    collection cannot be opened, leaving the exit decision to the caller.
    """
    if not question or not question.strip():
        logger.error("Empty question passed to retrieve")
        raise ValueError("question must not be empty")

    if top_k <= 0:
        logger.error("Invalid top_k: %r", top_k)
        raise ValueError(f"top_k must be positive, got {top_k}")

    logger.info("Retrieving top %d chunk(s) for: %s", top_k, question)

    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
    except Exception:
        logger.exception("Failed to load embedding model: %s", EMBEDDING_MODEL_NAME)
        raise

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    except Exception:
        logger.exception("Failed to open Chroma store at %s", CHROMA_DIR)
        raise

    try:
        collection = client.get_collection(
            COLLECTION_NAME, embedding_function=embedding_fn
        )
    except Exception as e:
        logger.exception("Could not open collection %s", COLLECTION_NAME)
        raise IndexNotFoundError(
            f"No index found at {CHROMA_DIR} (collection {COLLECTION_NAME!r}). "
            "Run ingest.py first."
        ) from e

    try:
        results = collection.query(query_texts=[question], n_results=top_k)
    except Exception:
        logger.exception("Query failed for collection %s", COLLECTION_NAME)
        raise

    # Chroma returns list-of-lists keyed per query; guard against empty/None.
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []

    logger.info("Retrieved %d chunk(s)", len(documents))
    return list(zip(documents, metadatas))


def build_prompt(question: str, retrieved: list[tuple[str, dict]]) -> str:
    numbered_sources = "\n\n".join(
        f"[{i + 1}] (from {meta.get('source', 'unknown')}) {chunk}"
        for i, (chunk, meta) in enumerate(retrieved)
    )
    return (
        f"Sources:\n{numbered_sources}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above, with citations."
    )


def generate_answer(question: str, retrieved: list[tuple[str, dict]]) -> str:
    prompt = build_prompt(question, retrieved)
    logger.info("Generating answer with %s over %d source(s)", OLLAMA_MODEL, len(retrieved))
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        # Generation failure is reported to the user rather than raised, so the
        # CLI still exits cleanly — but log it so the cause is not lost.
        logger.exception("Generation failed with model %s", OLLAMA_MODEL)
        return (
            f"[Generation failed: {e}]\n"
            f"Is Ollama running locally with the '{OLLAMA_MODEL}' model pulled? "
            f"Try: ollama pull {OLLAMA_MODEL}"
        )


def main(question: str) -> None:
    retrieved = retrieve(question)

    if not retrieved:
        logger.warning("Nothing retrieved for question: %s", question)
        print("Nothing retrieved, the index may be empty. Run ingest.py first.")
        return

    print("Retrieved sources:")
    for i, (chunk, meta) in enumerate(retrieved):
        preview = chunk[:90].replace("\n", " ")
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "?")
        print(f"  [{i + 1}] {source} (chunk {chunk_index}): {preview}...")
    print()

    answer = generate_answer(question, retrieved)
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    from logger import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Ask a question against your indexed documents.")
    parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args()

    # Top-level boundary: turn expected failures into a clean message + exit
    # code instead of an uncaught traceback.
    try:
        main(args.question)
    except IndexNotFoundError as e:
        logger.error("%s", e)
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid input: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception:
        logger.exception("Unexpected error while answering the question")
        print("Unexpected error. See logs for details.", file=sys.stderr)
        sys.exit(1)