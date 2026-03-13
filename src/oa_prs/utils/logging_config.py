"""
Logging configuration using structlog.
"""

import logging
from pathlib import Path
from typing import Optional

import structlog


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    format: str = "json",
) -> None:
    """
    Configure structlog with console and optional file output.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[str | Path]
        If provided, also log to this file
    format : str
        Output format ("json" or "text")

    Examples
    --------
    >>> setup_logging(level="DEBUG", log_file="app.log")
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )

    # Set up structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
            if format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured structlog logger.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of calling module)

    Returns
    -------
    structlog.BoundLogger
        Configured logger instance

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.info("message", key="value")
    """
    return structlog.get_logger(name)
