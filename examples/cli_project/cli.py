from __future__ import annotations
from typing import TYPE_CHECKING
import argparse
import logging
import sys

if TYPE_CHECKING:
    from config import AppConfig

from config import create_config

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli_project",
        description="Test CLI program for Python"
    )
    
    parser.add_argument(
        "command",
        choices=["run", "process", "status"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory"
    )
    
    return parser

def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_args(args: argparse.Namespace) -> None:
    if args.command not in ["run", "process", "status"]:
        raise ValueError(f"Invalid command: {args.command}")

def parse_args_to_config(args: argparse.Namespace) -> "AppConfig":
    return create_config(
        app_name="cli_project",
        output_dir=args.output_dir,
        verbose=args.verbose,
        config_file=args.config_file
    )
