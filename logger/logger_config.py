import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def get_logger(
    name: str = "docling_pipeline",
    log_dir: str = "logs",
    level: int = logging.INFO,
    max_mb: int = 5,
    backup_count: int = 5,
    console: bool = True,
):
    """
    Production-grade logger with:
    - Rotating log files
    - Optional console output
    - Timestamp formatting
    - Error stack trace logging

    Usage:
        from logger_config import get_logger
        logger = get_logger("INGEST")
        logger.info("hello")
        logger.error("error happened", exc_info=True)
    """

    # Create folder if not exist
    os.makedirs(log_dir, exist_ok=True)

    # File name -> logs/docling_2025-11-29.log
    log_filename = os.path.join(
        log_dir, f"docling_{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logs

    # Format for logs
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] → %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -----------------------
    # Rotating File Handler
    # -----------------------
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=max_mb * 1024 * 1024,  # convert MB → bytes
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # -----------------------
    # Console Handler (optional)
    # -----------------------
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    return logger


# # Quick self-test
# if __name__ == "__main__":
#     log = get_logger("TEST_LOGGER")
#     log.info("Logger started successfully.")
#     log.warning("Warning example.")
#     try:
#         1/0
#     except Exception as e:
#         log.error("Exception captured!", exc_info=True)
