import logging
import os

logger = logging.getLogger(__name__)


def load_documents(folder: str) -> list[dict]:
    logger.info("Loading documents from %s", folder)

    documents = []

    try:
        filenames = sorted(os.listdir(folder))
    except FileNotFoundError:
        logger.exception("Folder not found: %s", folder)
        raise
    except PermissionError:
        logger.exception("Permission denied for folder: %s", folder)
        raise

    for filename in filenames:
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)

        if not os.path.isfile(path):
            logger.warning("Skipping directory: %s", path)
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "source": filename,
                "text": text,
            })

            logger.info("Loaded %s", filename)

        except UnicodeDecodeError:
            logger.exception("Invalid UTF-8 encoding: %s", filename)

        except PermissionError:
            logger.exception("Permission denied: %s", filename)

        except OSError:
            logger.exception("Failed to read %s", filename)

    logger.info("Loaded %d document(s)", len(documents))

    return documents