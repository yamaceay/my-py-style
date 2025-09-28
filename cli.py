"""Command-line interface and logging configuration."""

from __future__ import annotations
from typing import TYPE_CHECKING
import argparse
import logging
import sys

if TYPE_CHECKING:
    from config import RuntimeConfig

from config import canonicalize_config


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser following the standard pattern."""
    parser = argparse.ArgumentParser(
        prog="py-style-simulation",
        description="Python simulation demonstrating the Go-ish coding standard"
    )
    

    subparsers = parser.add_subparsers(dest="cmd", required=True, help="Available commands")
    

    train_parser = subparsers.add_parser("train", help="Run training simulation")
    train_parser.add_argument("--model-name", required=True, help="Model name to use")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--task", choices=["detector", "classifier", "sentiment"], default="classifier", 
                             help="ML task type to run")
    train_parser.add_argument("--processor", choices=["simple", "detailed"], default="simple",
                             help="Data processor type")
    train_parser.add_argument("--data-size", type=int, default=100, help="Size of simulation dataset")
    

    eval_parser = subparsers.add_parser("eval", help="Run evaluation simulation")
    eval_parser.add_argument("--model-name", required=True, help="Model name to evaluate")
    eval_parser.add_argument("--task", choices=["detector", "classifier", "sentiment"], default="classifier",
                            help="ML task type to run")
    eval_parser.add_argument("--data-size", type=int, default=50, help="Size of simulation dataset")
    

    task_parser = subparsers.add_parser("task", help="Demonstrate polymorphic ML tasks")
    task_parser.add_argument("--type", choices=["detector", "classifier", "sentiment"], required=True,
                            help="ML task type to demonstrate")
    task_parser.add_argument("--threshold", type=float, help="Threshold for detector task")
    task_parser.add_argument("--num-classes", type=int, help="Number of classes for classifier task") 
    task_parser.add_argument("--confidence", type=float, help="Confidence threshold for sentiment task")
    task_parser.add_argument("--data-size", type=int, default=10, help="Size of simulation dataset")
    

    parser.add_argument("--device", default=None, help="Device to use (default: auto-detect)")
    parser.add_argument("--out-dir", help="Output directory")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length")
    
    return parser


def configure_logging(level: str, format_json: bool = False) -> None:
    """Configure structured logging following the standard."""
    log_level = getattr(logging, level.upper())
    
    if format_json:

        import json
        import datetime
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                

                if hasattr(record, "extra_fields"):
                    log_entry.update(record.extra_fields)
                

                for key, value in record.__dict__.items():
                    if key not in {"name", "msg", "args", "levelname", "levelno", "pathname", 
                                  "filename", "module", "lineno", "funcName", "created", 
                                  "msecs", "relativeCreated", "thread", "threadName", 
                                  "processName", "process", "getMessage", "exc_info", 
                                  "exc_text", "stack_info", "extra_fields"}:
                        log_entry[key] = value
                
                return json.dumps(log_entry)
        
        formatter = JSONFormatter()
    else:

        class ContextFormatter(logging.Formatter):
            def format(self, record):

                base_msg = super().format(record)
                

                extra_parts = []
                for key, value in record.__dict__.items():
                    if key not in {"name", "msg", "args", "levelname", "levelno", "pathname",
                                  "filename", "module", "lineno", "funcName", "created",
                                  "msecs", "relativeCreated", "thread", "threadName", 
                                  "processName", "process", "getMessage", "exc_info",
                                  "exc_text", "stack_info"}:
                        extra_parts.append(f"{key}={value}")
                
                if extra_parts:
                    return f"{base_msg} [{' '.join(extra_parts)}]"
                return base_msg
        
        formatter = ContextFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_args_to_config(args: argparse.Namespace) -> "RuntimeConfig":
    """Convert parsed arguments to RuntimeConfig following canonicalization pattern."""
    return canonicalize_config(
        device=args.device,
        out_dir=args.out_dir, 
        log_level=args.log_level,
        seed=args.seed,
        max_length=args.max_length,

    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations."""
    logger = logging.getLogger(__name__)
    

    if args.cmd == "task":
        if args.type == "detector" and args.threshold is None:
            logger.warning("No threshold specified for detector, using default")
        elif args.type == "classifier" and args.num_classes is None:
            logger.warning("No num_classes specified for classifier, using default")
        elif args.type == "sentiment" and args.confidence is None:
            logger.warning("No confidence threshold specified for sentiment, using default")
    

    logger.info("arguments_validated", extra={
        "command": args.cmd,
        "model_name": getattr(args, "model_name", None),
        "task_type": getattr(args, "task", None) or getattr(args, "type", None),
        "data_size": args.data_size
    })