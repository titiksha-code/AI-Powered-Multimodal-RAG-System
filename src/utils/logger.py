"""
src/utils/logger.py
=====================
Centralized loguru configuration used across the ingestion, retrieval,
and model layers so log formatting/output is consistent everywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logger(log_file: str = "logs/app.log", level: str = "INFO") -> None:
    logger.remove()  # drop default handler to avoid duplicate console logs

    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, level=level, rotation="5 MB", retention="7 days", enqueue=True)


configure_logger()
