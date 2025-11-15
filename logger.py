"""Logging configuration module."""

import json
import logging
import sys
from typing import Any, Dict

from config import settings, LogFormat, LogLevel


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


def setup_logging() -> logging.Logger:
    """
    Setup logging configuration based on settings.

    Returns:
        Configured logger instance
    """
    # Get log level from settings
    log_level = getattr(logging, settings.log_level.value, logging.INFO)

    # Create formatter based on settings
    if settings.log_format == LogFormat.JSON:
        formatter = JSONFormatter()
    else:
        # Simple formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # Get application logger
    app_logger = logging.getLogger("ml_api_wrapper")
    app_logger.setLevel(log_level)

    return app_logger


# Create logger instance
logger = setup_logging()

