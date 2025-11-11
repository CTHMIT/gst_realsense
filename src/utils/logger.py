"""A module for creating and managing loggers."""

from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
from collections.abc import Callable
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, TypeVar, cast


class ThreadSafeStream:
    """A thread-safe stream for logging."""

    _lock = threading.Lock()

    def __init__(self, stream):
        """Initialize the thread-safe stream.

        Args:
            stream: The stream to write to.
        """
        self.stream = stream

    def write(self, data):
        """Write data to the stream.

        Args:
            data: The data to write.
        """
        with self._lock:
            self.stream.write(data)
            self.stream.flush()

    def flush(self):
        """Flush the stream."""
        with self._lock:
            self.stream.flush()


class LogLevel(Enum):
    """Enumeration of logging levels."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LogContext(TypedDict, total=False):
    """TypedDict for structured logging contexts."""

    operation: str
    component: str
    user_id: str
    request_id: str
    extra: dict[str, Any]


class SupportsFormat(Protocol):
    """Protocol defining objects that support formatting."""

    def format(self, record: logging.LogRecord) -> str: ...


T = TypeVar("T")


def singleton(cls: type[T]) -> Callable[..., T]:
    """Singleton decorator to ensure only one instance of a class exists."""
    instances: dict[type[Any], Any] = {}

    def get_instance(*args: Any, **kwargs: Any) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return cast(T, instances[cls])

    return get_instance


def get_environment() -> str:
    """Get the current environment name from environment variables.

    Returns:
        str: Current environment name, defaults to "dev" if not set.
    """
    return os.getenv("ENVIRONMENT", "dev")


def get_log_file_path(log_dir: Path) -> Path:
    """Generate log file path using date and environment name.

    Args:
        log_dir: Directory where logs will be stored

    Returns:
        Path: Complete path to the log file
    """
    date_str = datetime.now().strftime("%Y%m%d")
    env = get_environment()
    return log_dir / f"{date_str}_{env}.log"


class ErrorContextFilter(logging.Filter):
    """Custom filter to add error context information to log records."""

    def __init__(self) -> None:
        """Initialize the error context filter."""
        super().__init__()
        self.error_context: str | None = None

    def set_error_context(self, context: str) -> None:
        """Set the error context.

        Args:
            context: Context information to add to error logs
        """
        self.error_context = context

    def filter(self, record: logging.LogRecord) -> bool:
        """Add error context to log records if available.

        Args:
            record: Log record to be filtered

        Returns:
            bool: Always True as we want to include all records
        """
        if self.error_context and record.levelno >= logging.ERROR:
            record.error_context = self.error_context
        elif record.levelno >= logging.ERROR:
            record.error_context = ""
        return True


class CustomFormatter(logging.Formatter):
    """Custom formatter for different log levels with enhanced error tracking."""

    def __init__(self) -> None:
        """Initialize the custom formatter with specific formats per log level."""
        super().__init__()
        # Add function name to the basic format
        self.log_fmt: str = "%(asctime)s [%(levelname)s] - %(filename)s:%(funcName)s - %(message)s"
        self.error_fmt: str = (
            "%(asctime)s [%(levelname)s] - %(filename)s:%(funcName)s - "
            "%(message)s %(error_context)s"
        )

        self.FORMATS: dict[int, str] = {
            logging.DEBUG: self.log_fmt,
            logging.INFO: self.log_fmt,
            logging.WARNING: self.log_fmt,
            logging.ERROR: self.error_fmt,
            logging.CRITICAL: self.error_fmt,
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record.

        Args:
            record: The log record to format.

        Returns:
            The formatted log record.
        """
        log_fmt = self.FORMATS.get(record.levelno, self.log_fmt)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")

        # For error levels, get the actual error location
        if record.levelno >= logging.ERROR:
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                tb = traceback.extract_tb(exc_tb)

                # Get the last non-decorator frame
                for frame in reversed(tb):
                    if frame.filename.endswith(".py") and not frame.filename.endswith("logger.py"):
                        # Don't modify record.message as it doesn't exist yet
                        # Instead, modify record.msg which is the raw message
                        record.msg = f"[{frame.filename}:{frame.lineno}] {record.msg}"
                        break

            if not hasattr(record, "error_context"):
                record.error_context = ""

        return formatter.format(record)


@singleton
class LoggerFactory:
    """Factory class for creating and managing loggers."""

    def __init__(self) -> None:
        """Initialize the logger factory."""
        self.loggers: dict[str, logging.Logger] = {}
        self.error_filters: dict[str, ErrorContextFilter] = {}
        self.base_log_dir: Path = Path("logs")

    def configure_log_directory(self, log_dir: Path) -> None:
        """Configure the base log directory.

        Args:
            log_dir: Path to the log directory
        """
        self.base_log_dir = log_dir

    def get_logger(self, name: str | None = None, log_dir: Path | None = None) -> logging.Logger:
        """Get or create a logger with the specified name.

        Args:
            name: Name of the logger, defaults to environment name
            log_dir: Directory for log files, defaults to factory's base directory

        Returns:
            logging.Logger: Configured logger instance
        """
        logger_name = name if name is not None else get_environment()

        if logger_name in self.loggers:
            return self.loggers[logger_name]

        directory = log_dir if log_dir is not None else self.base_log_dir
        logger = self._setup_logger(logger_name, directory)
        self.loggers[logger_name] = logger
        return logger

    def _setup_logger(self, name: str, log_dir: Path) -> logging.Logger:
        """Set up a new logger with file and stream handlers.

        Args:
            name: Name of the logger
            log_dir: Directory for log files

        Returns:
            logging.Logger: Newly configured logger
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Error context filter
        error_filter = ErrorContextFilter()
        self.error_filters[name] = error_filter

        # File handler
        log_file = get_log_file_path(log_dir)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.addFilter(error_filter)
        fh.setFormatter(CustomFormatter())

        # Terminal handler (StreamHandler)
        sh = logging.StreamHandler(
            ThreadSafeStream(sys.stdout)
        )  # Output to standard output (terminal)
        sh.setLevel(logging.INFO)
        sh.addFilter(error_filter)
        sh.setFormatter(CustomFormatter())

        # Add handlers to logger
        logger.addHandler(fh)
        logger.addHandler(sh)

        return logger

    def set_error_context(self, logger_name: str, context: str) -> None:
        """Set error context for a specific logger.

        Args:
            logger_name: Name of the logger
            context: Error context to set
        """
        if logger_name in self.error_filters:
            self.error_filters[logger_name].set_error_context(context)

    def set_level(self, logger_name: str, level: int | str | LogLevel) -> None:
        """Set the logging level for a specific logger.

        Args:
            logger_name: Name of the logger
            level: Logging level to set
        """
        if logger_name not in self.loggers:
            return

        log_level: int
        if isinstance(level, int):
            log_level = level
        elif isinstance(level, str):
            log_level = getattr(logging, level.upper(), logging.INFO)
        else:  # LogLevel enum
            log_level = getattr(logging, level.name, logging.INFO)

        self.loggers[logger_name].setLevel(log_level)

        for handler in self.loggers[logger_name].handlers:
            handler.setLevel(log_level)


logger_factory = LoggerFactory()
LOGGER = logger_factory.get_logger(log_dir=Path(".logs"))


def get_logger(name: str | None = None, log_dir: Path | None = None) -> logging.Logger:
    """Convenience function to get a logger from the factory.

    Args:
        name: Name of the logger
        log_dir: Directory for log files

    Returns:
        logging.Logger: Configured logger instance
    """
    return logger_factory.get_logger(name, log_dir)


def set_error_context(context: str, logger_name: str | None = None) -> None:
    """Set error context for a logger.

    Args:
        context: Error context to set
        logger_name: Name of the logger, uses default if None
    """
    name = logger_name if logger_name is not None else get_environment()
    logger_factory.set_error_context(name, context)


def set_log_level(level: int | str | LogLevel, logger_name: str | None = None) -> None:
    """Set the logging level for a logger.

    Args:
        level: Logging level to set
        logger_name: Name of the logger, uses default if None
    """
    name = logger_name if logger_name is not None else get_environment()
    logger_factory.set_level(name, level)
