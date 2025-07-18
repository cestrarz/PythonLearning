"""
Logging utilities for Python scripts.

This module provides a simple way to set up logging that captures both print statements
and logging calls to a file, while still printing to the console.

Usage:
    from utils.logging_utils import setup_logging
    
    # Basic setup
    logger = setup_logging(log_file='my_script.log')
    
    # Now all print statements and logging calls will be captured in the log file
    print("This will go to both console and log file")
    logger.info("This will also be in both places")
"""
import os
import sys
import logging
from typing import Optional

class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup_logging(log_file: str, 
                 log_level: int = logging.INFO,
                 log_dir: str = 'output/logs',
                 capture_print: bool = True) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Name of the log file
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_dir: Directory to store log files (will be created if it doesn't exist)
        capture_print: Whether to redirect print statements to the log
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler with detailed format
    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, log_file),
        mode='w'
    )
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler with simple format
    console = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console.setFormatter(console_formatter)
    
    # Add both handlers to the root logger
    logger.addHandler(file_handler)
    logger.addHandler(console)
    
    # Optionally redirect stdout and stderr to the logger
    if capture_print:
        sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), log_level)
        sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)
    
    return logger
