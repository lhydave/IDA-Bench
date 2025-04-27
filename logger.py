import logging
import os
import sys


def setup_logger(
    name: str = "DataSciBench",
    level: int = logging.INFO,
    log_file: str | None = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    mode: str = "a",
) -> logging.Logger:
    """
    Configure and get a logger with the specified settings.

    Args:
        name: The name of the logger
        level: The minimum logging level
        log_file: Optional file path to write logs to
        fmt: The log message format string
        datefmt: The format string for dates/times
        mode: The mode for the log file ('a' for append, 'w' for overwrite)

    Returns:
        A configured logger instance
    """
    # Create logger and set level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if specified
    if log_file:
        # Create parent directories if they don't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Add file handler
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger(name="DataSciBench")  # type: ignore

def configure_global_logger(
    level: int | str = logging.INFO,
    log_file: str | None = None,
    mode: str = "a",
    log_filename: str | None = None,
) -> None:
    """
    Configure the global logger.

    Args:
        level: The minimum logging level (can be int or string level name)
        log_file: Optional file path to write logs to
        mode: The mode for the log file ('a' for append, 'w' for overwrite)
        log_filename: Optional custom log filename to use instead of default
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # If log_filename is specified and log_file is not, use log_filename
    final_log_file = log_file
    if log_filename and not log_file:
        final_log_file = log_filename

    global logger
    logger = setup_logger(name="DataSciBench", level=level, log_file=final_log_file, mode=mode)  # type: ignore
