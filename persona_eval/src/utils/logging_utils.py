import logging
import os
import sys
from typing import Optional


def setup_logger(
    log_dir: str,
    name: str = "persona_eval",
    console_output: bool = True,
) -> logging.Logger:
    """Set up a logger with file and optional console output."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    log_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "persona_eval") -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)
