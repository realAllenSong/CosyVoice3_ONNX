"""
Logging utilities for CosyVoice3 ONNX.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler


_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "cosyvoice_onnx",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """Setup and return the logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (None = no file logging)
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "cosyvoice_onnx.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the existing logger or create a default one."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger
