"""
Enhanced logging utility for the Wikipedia Agent.
Provides structured logging with different levels and outputs.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


class AgentLogger:
    """Enhanced logger for the Wikipedia Agent."""
    
    def __init__(self, name: str = "enhanced_wikipedia_agent"):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
        """
        self.name = name
        self.logger = None
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Try to load logging configuration from YAML
        logging_config_path = Path("config/logging.yaml")
        
        if logging_config_path.exists():
            try:
                with open(logging_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Ensure log directory exists
                for handler in config.get('handlers', {}).values():
                    if 'filename' in handler:
                        log_file = Path(handler['filename'])
                        log_file.parent.mkdir(parents=True, exist_ok=True)
                        
                logging.config.dictConfig(config)
                self.logger = logging.getLogger(self.name)
                return
            except Exception as e:
                print(f"Failed to load logging configuration: {e}")
                
        # Fallback to basic configuration
        self._setup_basic_logging()
        
    def _setup_basic_logging(self):
        """Setup basic logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if sys.stdout.isatty():  # Only use colors for interactive terminals
            console_formatter = ColoredFormatter(log_format)
        else:
            console_formatter = logging.Formatter(log_format)
            
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler("logs/agent.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        if self.logger:
            self.logger.debug(message, extra=extra or {})
            
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message."""
        if self.logger:
            self.logger.info(message, extra=extra or {})
            
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        if self.logger:
            self.logger.warning(message, extra=extra or {})
            
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message."""
        if self.logger:
            self.logger.error(message, extra=extra or {})
            
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message."""
        if self.logger:
            self.logger.critical(message, extra=extra or {})
            
    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log exception with traceback."""
        if self.logger:
            self.logger.exception(message, extra=extra or {})
            
    def log_agent_action(self, action: str, details: Dict[str, Any]):
        """Log agent action with structured data."""
        log_data = {
            'action': action,
            'timestamp': datetime.now().isoformat(),
            **details
        }
        self.info(f"Agent Action: {action}", extra=log_data)
        
    def log_tool_usage(self, tool_name: str, query: str, result_summary: str):
        """Log tool usage."""
        log_data = {
            'tool': tool_name,
            'query': query,
            'result_summary': result_summary,
            'timestamp': datetime.now().isoformat()
        }
        self.info(f"Tool Used: {tool_name}", extra=log_data)
        
    def log_performance(self, operation: str, duration: float, metadata: Dict[str, Any]):
        """Log performance metrics."""
        log_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            **metadata
        }
        self.info(f"Performance: {operation} took {duration:.2f}s", extra=log_data)
        
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context."""
        log_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            **context
        }
        self.error(f"Error: {type(error).__name__}: {error}", extra=log_data)
        
    def set_level(self, level: str):
        """Set logging level."""
        if self.logger:
            numeric_level = getattr(logging, level.upper(), None)
            if isinstance(numeric_level, int):
                self.logger.setLevel(numeric_level)
                
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger


# Global logger instance
logger = AgentLogger()

# Convenience functions
def get_logger(name: str = "enhanced_wikipedia_agent") -> AgentLogger:
    """Get a logger instance."""
    return AgentLogger(name)

def log_debug(message: str, extra: Optional[Dict[str, Any]] = None):
    """Log debug message."""
    logger.debug(message, extra)

def log_info(message: str, extra: Optional[Dict[str, Any]] = None):
    """Log info message."""
    logger.info(message, extra)

def log_warning(message: str, extra: Optional[Dict[str, Any]] = None):
    """Log warning message."""
    logger.warning(message, extra)

def log_error(message: str, extra: Optional[Dict[str, Any]] = None):
    """Log error message."""
    logger.error(message, extra)

def log_critical(message: str, extra: Optional[Dict[str, Any]] = None):
    """Log critical message."""
    logger.critical(message, extra)

def log_exception(message: str, extra: Optional[Dict[str, Any]] = None):
    """Log exception with traceback."""
    logger.exception(message, extra)

def log_agent_action(action: str, details: Dict[str, Any]):
    """Log agent action."""
    logger.log_agent_action(action, details)

def log_tool_usage(tool_name: str, query: str, result_summary: str):
    """Log tool usage."""
    logger.log_tool_usage(tool_name, query, result_summary)

def log_performance(operation: str, duration: float, metadata: Dict[str, Any]):
    """Log performance metrics."""
    logger.log_performance(operation, duration, metadata)

def log_error_with_context(error: Exception, context: Dict[str, Any]):
    """Log error with context."""
    logger.log_error_with_context(error, context)