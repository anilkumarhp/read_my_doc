import logging
import sys
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    )

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    # File
    file = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)

    return logger