import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging():
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(
                "logs/app.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            ),
        ],
        force=True,
    )