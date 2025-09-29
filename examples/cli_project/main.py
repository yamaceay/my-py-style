from __future__ import annotations
from typing import TYPE_CHECKING, Any
import logging
import sys

if TYPE_CHECKING:
    from config import AppConfig

from core import DefaultService, DefaultHandler, DefaultProcessor, DefaultRepository
from contexts import ServiceContext, DataContext
from cli import build_parser, configure_logging, parse_args_to_config, validate_args

logger = logging.getLogger(__name__)

def run_application(args, config: "AppConfig") -> dict[str, Any]:
    logger.info('application_starting', extra={'command': args.command})
    
    with DataContext("default_source") as data_ctx:
        with ServiceContext(config) as service:
            if args.command == "run":
                result = service.execute(data_ctx.get_data())
            elif args.command == "process":
                result = service.execute({"action": "process"})
            elif args.command == "status":
                result = {"status": "running", "app": config.app_name}
            else:
                raise ValueError(f"Unknown command: {args.command}")
            
            logger.info('application_complete', extra={'success': True})
            return result

if __name__ == "__main__":
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    validate_args(args)
    config = parse_args_to_config(args)
    
    try:
        result = run_application(args, config)
        print("Application completed successfully!")
        print(f"Result: {result}")
    except Exception as e:
        logger.error('application_failed', extra={'error': str(e)})
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
