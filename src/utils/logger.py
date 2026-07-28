"""
Logging Utility Module.
Configures structured, clean logging across all components of the application.
"""

import sys
import logging
from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger instance.

    Args:
        name (str): Name of the logger, typically __name__ of the module.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

        # Console Log Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

        # Formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
