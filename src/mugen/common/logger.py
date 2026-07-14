"""Logging module."""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name="mugen", log_file=None, level=logging.INFO, console=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


_loggers = {}

def get_logger(name="mugen"):
    if name not in _loggers:
        _loggers[name] = setup_logger(name)
    return _loggers[name]
