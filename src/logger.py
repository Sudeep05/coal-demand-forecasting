"""
logger.py — Centralized logging setup for Coal Demand Forecasting project.
Provides module-specific loggers with rotating file handlers and console output.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# Import config – handle both package and standalone execution
try:
    from src.config import LOGS_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_FORMAT, LOG_DATE_FORMAT
except ImportError:
    from config import LOGS_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_FORMAT, LOG_DATE_FORMAT

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Module-specific log file names
_LOG_FILES: dict = {
    "data": "data.log",
    "training": "training.log",
    "evaluation": "evaluation.log",
    "api": "api.log",
    "monitoring": "monitoring.log",
    "eda": "eda.log",
    "preprocessing": "preprocessing.log",
    "pipeline": "pipeline.log",
}

# Cache for created loggers
_loggers: dict = {}


def get_logger(module_name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger for the specified module.

    Args:
        module_name: Name of the module (e.g., 'data', 'training', 'evaluation')
        log_file: Optional custom log file name. If not provided, uses
                  predefined mapping or defaults to '<module_name>.log'.

    Returns:
        Configured logging.Logger instance.
    """
    if module_name in _loggers:
        return _loggers[module_name]

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Determine log file path
    if log_file is None:
        log_file = _LOG_FILES.get(module_name, f"{module_name}.log")
    log_path = os.path.join(LOGS_DIR, log_file)

    # Formatter
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers (avoid duplicates)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    _loggers[module_name] = logger
    return logger
