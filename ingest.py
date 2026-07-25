"""
ingest.py — Load, chunk, embed, and store documents in a local Chroma index.

Usage:
    python ingest.py                    # ingests everything in data/sample_docs
    python ingest.py --docs my_folder   # ingests everything in my_folder
"""

import argparse
import logging
import os
import re
import sys
import uuid

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "personal_docs"

# A modest, well-understood embedding model. Runs locally via
# sentence-transformers once downloaded — no per-call API cost.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Target chunk size in characters and how much consecutive chunks overlap.
# See Chapter 6 of the RAG Patterns series for why these numbers are a
# starting point to tune, not a universal constant.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def load_documents(folder: str) -> list[dict]:
    """Read every .txt file in folder. Returns a list of {source, text} dicts.

    A folder that cannot be listed (missing, not a directory, no permission) is
    fatal and re-raised for the caller to handle. An individual file that cannot
    be read is logged and skipped so one bad file never aborts the whole batch.
    """
    logger.info("Loading documents from %s", folder)

    if not folder or not folder.strip():
        logger.error("Folder path is empty")
        raise ValueError("folder must not be empty")

    documents = []
    skipped = 0

    try:
        filenames = sorted(os.listdir(folder))
    except FileNotFoundError:
        logger.exception("Folder not found: %s", folder)
        raise
    except NotADirectoryError:
        logger.exception("Not a directory: %s", folder)
        raise
    except PermissionError:
        logger.exception("Permission denied for folder: %s", folder)
        raise
    except OSError:
        logger.exception("Failed to list folder: %s", folder)
        raise

    for filename in filenames:
        if not filename.endswith(".txt"):
            logger.debug("Skipping non-txt entry: %s", filename)
            continue

        path = os.path.join(folder, filename)

        if not os.path.isfile(path):
            logger.warning("Skipping non-file entry: %s", path)
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            skipped += 1
            logger.exception("Invalid UTF-8, skipping: %s", filename)
            continue
        except PermissionError:
            skipped += 1
            logger.exception("Permission denied, skipping: %s", filename)
            continue
        except OSError:
            skipped += 1
            logger.exception("Failed to read, skipping: %s", filename)
            continue

        if not text.strip():
            logger.warning("File is empty: %s", filename)

        documents.append({"source": filename, "text": text})
        logger.info("Loaded %s (%d char(s))", filename, len(text))

    if skipped:
        logger.warning("Skipped %d unreadable file(s)", skipped)

    if not documents:
        logger.warning("No documents loaded from %s", folder)

    logger.info("Loaded %d document(s)", len(documents))
    return documents


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    A lightweight version of recursive splitting (Chapter 4.1): try to break
    on paragraph boundaries first, falling back to sentence boundaries for any
    paragraph that's still too long, and only hard-cutting as a last resort.

    Raises ValueError/TypeError on invalid arguments — these signal a caller
    bug, not bad input data, so failing fast is intended.
    """
    if not isinstance(text, str):
        logger.error("Expected text to be str, got %s", type(text).__name__)
        raise TypeError(f"text must be str, got {type(text).__name__}")

    if chunk_size <= 0:
        logger.error("Invalid chunk_size: %r", chunk_size)
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if overlap < 0:
        logger.error("Invalid overlap: %r", overlap)
        raise ValueError(f"overlap must not be negative, got {overlap}")

    if overlap >= chunk_size:
        logger.error("Invalid overlap %r for chunk_size %r", overlap, chunk_size)
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    logger.debug(
        "Splitting %d char(s) (chunk_size=%d, overlap=%d)",
        len(text), chunk_size, overlap,
    )

    if not text.strip():
        logger.warning("Text is empty or whitespace only, nothing to split")
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    logger.debug("Found %d paragraph(s)", len(paragraphs))

    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # Current paragraph alone doesn't fit alongside what's buffered.
        flush()

        if len(para) <= chunk_size:
            current = para
            continue

        # Paragraph itself is too long — fall back to sentence-level splitting.
        logger.debug(
            "Paragraph of %d char(s) exceeds chunk_size, splitting by sentence",
            len(para),
        )
        sentences = re.split(r"(?<=[.!?])\s+", para)
        sentence_buffer = ""
        for sentence in sentences:
            candidate = f"{sentence_buffer} {sentence}".strip()
            if len(candidate) <= chunk_size:
                sentence_buffer = candidate
            else:
                if sentence_buffer:
                    chunks.append(sentence_buffer.strip())
                # Last resort: hard cut an unusually long single sentence.
                if len(sentence) > chunk_size:
                    logger.warning(
                        "Sentence of %d char(s) exceeds chunk_size %d; "
                        "hard-cutting drops the remainder",
                        len(sentence), chunk_size,
                    )
                    sentence_buffer = sentence[:chunk_size]
                else:
                    sentence_buffer = sentence
        if sentence_buffer:
            current = sentence_buffer

    flush()

    # Apply overlap: prepend the tail of the previous chunk to each chunk
    # after the first, softening the boundary problem from Chapter 3.2.
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0 or overlap <= 0:
            overlapped.append(chunk)
        else:
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail} {chunk}")

    logger.info("Split text into %d chunk(s)", len(overlapped))
    return overlapped


def build_index(folder: str) -> None:
    logger.info("Building index from %s", folder)

    documents = load_documents(folder)
    if not documents:
        logger.warning("No .txt files found in %s, nothing to ingest", folder)
        print(f"No .txt files found in {folder}. Nothing to ingest.")
        return

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

    # Fresh collection each run, so re-ingesting doesn't duplicate chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Expected on the first run when the collection does not exist yet.
        logger.debug("No existing collection %s to delete", COLLECTION_NAME)

    try:
        collection = client.create_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )
    except Exception:
        logger.exception("Failed to create collection %s", COLLECTION_NAME)
        raise

    all_chunks, all_ids, all_metadatas = [], [], []
    for doc in documents:
        chunks = recursive_split(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(str(uuid.uuid4()))
            all_metadatas.append({"source": doc["source"], "chunk_index": i})

    if not all_chunks:
        logger.warning("Documents produced no chunks, nothing to add")
        print("Documents contained no usable text. Nothing to ingest.")
        return

    try:
        collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
    except Exception:
        logger.exception("Failed to add %d chunk(s) to the index", len(all_chunks))
        raise

    logger.info(
        "Ingested %d document(s) into %d chunk(s)", len(documents), len(all_chunks)
    )
    print(f"Ingested {len(documents)} document(s) into {len(all_chunks)} chunks.")
    for doc in documents:
        n = sum(1 for m in all_metadatas if m["source"] == doc["source"])
        print(f"  - {doc['source']}: {n} chunks")
    print(f"Index stored at {CHROMA_DIR} (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    from logger import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Ingest documents into the local RAG index.")
    parser.add_argument(
        "--docs", default="data/sample_docs", help="Folder of .txt files to ingest."
    )
    args = parser.parse_args()

    # Top-level boundary: turn expected failures into a clean message + exit
    # code instead of an uncaught traceback.
    try:
        build_index(args.docs)
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as e:
        logger.error("Ingestion failed: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error during ingestion")
        print("Unexpected error during ingestion. See logs for details.", file=sys.stderr)
        sys.exit(1)