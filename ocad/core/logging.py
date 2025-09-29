"""Structured logging setup for OCAD system."""

import logging
import logging.config
import sys
from typing import Any, Dict

import structlog
from structlog.types import Processor


def configure_logging(log_level: str = "INFO", enable_json: bool = True) -> None:
    """Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to use JSON formatting
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Disable some noisy loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("kafka").setLevel(logging.WARNING)
    
    # Configure structlog
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog instance
    """
    return structlog.get_logger(name)


def log_function_call(func_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Create a log context for function calls.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function arguments to log
        
    Returns:
        Log context dictionary
    """
    return {
        "function": func_name,
        "args": {k: str(v) for k, v in kwargs.items()},
    }


def log_metric(metric_name: str, value: float, labels: Dict[str, str] = None) -> Dict[str, Any]:
    """Create a log context for metrics.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        labels: Optional metric labels
        
    Returns:
        Log context dictionary
    """
    context = {
        "metric": metric_name,
        "value": value,
    }
    if labels:
        context["labels"] = labels
    
    return context
