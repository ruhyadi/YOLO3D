"""Python logging utils."""

import rootutils

ROOT = rootutils.autosetup()

import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Dict

from colorlog import ColoredFormatter


def get_logger(level: str = "INFO") -> logging:
    """
    Get python logger function.

    Args:
        level (int): Logging level. Defaults to 20.

    Returns:
        logging: Python logger function.
    """
    # custom log levels
    custom_levels = {
        "API": 21,
        "MONGODB": 22,
        "REDIS": 23,
        "SQL": 24,
        "MILVUS": 25,
    }
    custom_logging_level(custom_levels)

    # custom log color
    custom_color = {
        "API": "blue",
        "MONGODB": "green",
        "REDIS": "green",
        "SQL": "green",
        "MILVUS": "green",
    }

    # rotating file handler to save logging file
    rfh = TimedRotatingFileHandler(
        filename="logs/logs.log", when="midnight", backupCount=3
    )
    rfh.setLevel(level)
    rfh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s | %(filename)s:%(lineno)d"
        )
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        ColoredFormatter(
            "%(cyan)s%(asctime)s%(reset)s %(log_color)s[%(levelname)s] %(message)s | %(filename)s:%(lineno)d",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold,red",
                **custom_color,
            },
        )
    )

    logging.basicConfig(
        level=level,
        handlers=[rfh, console_handler],
    )

    return logging


def custom_logging_level(levels: Dict[str, int]) -> None:
    """
    Add custom logging levels to logging module.

    Args:
        levels (dict): Dictionary of custom logging levels.
    """
    for key, value in levels.items():
        logging.addLevelName(value, key.upper())
