"""Utils module."""

import rootutils

ROOT = rootutils.autosetup()

import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Dict

from colorlog import ColoredFormatter


def get_logger(level: str = "INFO") -> logging:
    """Logger setup."""
    custom_levels = {
        "MODEL": 21,
        "DATA": 22,
    }
    custom_logging_level(custom_levels)

    # custom log color
    custom_color = {
        "MODEL": "blue",
        "DATA": "green",
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
    """Add custom logging levels to logging module."""
    for key, value in levels.items():
        logging.addLevelName(value, key.upper())
