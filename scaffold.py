#!/usr/bin/env python3
"""
Python Project Scaffolder - Go-ish Python Patterns Generator

This module generates complete Python projects following Go-inspired patterns:
- Interface-based design with ABC contracts
- Constructor dependency injection
- Context manager resource lifecycle
- TYPE_CHECKING compliance
- Immutable configuration objects
- Structured logging patterns

Usage:
    python scaffold.py project_name --type cli_tool --author "Your Name"
    
Or via Makefile:
    make build my_app type=api_server author="Developer Name"

Generated projects include:
- Complete module structure (interfaces, core, contexts, cli, config, main)
- Modern Python packaging (pyproject.toml)
- Comprehensive documentation
- Git integration (.gitignore)
- Test directory structure

Modification Guidelines:
- Add new project types by extending ProjectType enum
- Customize templates in _get_*_template() methods
- Add feature flags via ScaffoldConfig dataclass
- Extend CLI options in main() argument parser

Author: Yamaç Eren Ay (@yamaceay)
Repository: https://github.com/yamaceay/py-kit
License: MIT
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from typing import Optional

class ProjectType(Enum):
    CLI_TOOL = "cli_tool"

@dataclass(frozen=True)
class ScaffoldConfig:
    project_name: str
    project_type: ProjectType
    target_dir: Path
    author: str
    description: str
    use_logging: bool = True
    use_cli: bool = True
    use_context_managers: bool = True

class ProjectScaffolder:
    def __init__(self, config: ScaffoldConfig):
        self.config = config
        self.project_path = self.config.target_dir / self.config.project_name
    
    def create_project(self) -> None:
        self._create_directory_structure()
        self._generate_core_files()
        self._generate_project_specific_files()
        self._generate_config_files()
        print(f"Project '{self.config.project_name}' created successfully at {self.project_path}")
    
    def _create_directory_structure(self) -> None:
        self.project_path.mkdir(parents=True, exist_ok=True)
        tests_dir = self.project_path / "tests"
        tests_dir.mkdir(exist_ok=True)
        # Create __init__.py for tests directory
        (tests_dir / "__init__.py").write_text(
            '"""Test package for {}."""\n'.format(self.config.project_name),
            encoding="utf-8"
        )
    
    def _generate_core_files(self) -> None:
        self._create_interfaces_file()
        self._create_core_file()
        if self.config.use_context_managers:
            self._create_contexts_file()
        if self.config.use_cli:
            self._create_cli_file()
        self._create_config_file()
        self._create_main_file()
        self._create_import_utils_file()  # Add import utility
        self._create_init_file()
        self._create_py_typed()
        self._create_readme()
        self._create_gitignore()
    
    def _generate_project_specific_files(self) -> None:
        if self.config.project_type == ProjectType.CLI_TOOL:
            self._create_cli_specific_files()
    
    def _generate_config_files(self) -> None:
        self._create_pyproject_toml()
        self._create_requirements_txt()
    
    def _create_interfaces_file(self) -> None:
        content = self._get_interfaces_template()
        self._write_file("interfaces.py", content)
    
    def _create_core_file(self) -> None:
        content = self._get_core_template()
        self._write_file("core.py", content)
    
    def _create_contexts_file(self) -> None:
        content = self._get_contexts_template()
        self._write_file("contexts.py", content)
    
    def _create_cli_file(self) -> None:
        content = self._get_cli_template()
        self._write_file("cli.py", content)
    
    def _create_config_file(self) -> None:
        content = self._get_config_template()
        self._write_file("config.py", content)
    
    def _create_main_file(self) -> None:
        content = self._get_main_template()
        self._write_file("main.py", content)
    
    def _create_import_utils_file(self) -> None:
        content = self._get_import_utils_template()
        self._write_file("import_utils.py", content)
    
    def _create_init_file(self) -> None:
        content = self._get_init_template()
        self._write_file("__init__.py", content)
    
    def _create_py_typed(self) -> None:
        # Create py.typed file to mark package as typed
        self._write_file("py.typed", "")
    
    def _create_readme(self) -> None:
        content = self._get_readme_template()
        self._write_file("README.md", content)
    
    def _create_gitignore(self) -> None:
        content = self._get_gitignore_template()
        self._write_file(".gitignore", content)
    
    def _create_pyproject_toml(self) -> None:
        content = self._get_pyproject_template()
        self._write_file("pyproject.toml", content)
    
    def _create_requirements_txt(self) -> None:
        content = self._get_requirements_template()
        self._write_file("requirements.txt", content)
    
    def _create_cli_specific_files(self) -> None:
        pass
    
    def _write_file(self, filename: str, content: str) -> None:
        filepath = self.project_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    
    def _get_interfaces_template(self) -> str:
        return '''#!/usr/bin/env python3
"""
Abstract Base Classes (Interface Definitions)

This module defines the core contracts for the application using ABC patterns.
All interfaces follow the principle of minimal, focused responsibilities.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}
Modification Guidelines:
- Add new interfaces following ABC pattern
- Keep interfaces minimal and focused
- Use @abstractmethod for all required methods
- Avoid implementation details in interfaces
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class Processor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

class Handler(ABC):
    @abstractmethod
    def handle(self, request: Any) -> Any:
        pass

class Repository(ABC):
    @abstractmethod
    def save(self, entity: Any) -> bool:
        pass
    
    @abstractmethod
    def find(self, identifier: str) -> Any:
        pass

class Service(ABC):
    @abstractmethod
    def execute(self, params: Any) -> Any:
        pass
'''
    
    def _get_core_template(self) -> str:
        type_checking_imports = ""
        runtime_imports = ""
        
        if self.config.use_logging:
            runtime_imports += "import logging\n"
            type_checking_imports += "import logging\n"
        
        return f'''#!/usr/bin/env python3
"""
Core Business Logic Implementations

This module contains concrete implementations of all interfaces defined
in interfaces.py. Uses constructor injection and immutable patterns.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}
Modification Guidelines:
- Use @dataclass(frozen=True) for immutable objects with static deps
- Use regular __init__ for dynamic initialization
- Keep methods focused on single responsibilities
- Inject dependencies through constructor only
- Log with structured context when enabled
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import os
import sys
{runtime_imports}

if TYPE_CHECKING:
    from .config import AppConfig

# Smart import for interfaces
try:
    from .interfaces import Processor, Handler, Repository, Service
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from interfaces import Processor, Handler, Repository, Service

{"logger = logging.getLogger(__name__)" if self.config.use_logging else ""}

@dataclass(frozen=True)
class DefaultProcessor(Processor):
    name: str
    
    def process(self, data: Any) -> Any:
        {"logger.info('processing_data', extra={'processor': self.name})" if self.config.use_logging else "pass"}
        return {{"processed": True, "data": data, "processor": self.name}}

@dataclass(frozen=True)
class DefaultHandler(Handler):
    service: Service
    
    def handle(self, request: Any) -> Any:
        {"logger.info('handling_request', extra={'request_type': type(request).__name__})" if self.config.use_logging else "pass"}
        return self.service.execute(request)

class DefaultRepository(Repository):
    def __init__(self):
        self._storage: dict[str, Any] = {{}}
    
    def save(self, entity: Any) -> bool:
        entity_id = getattr(entity, "id", str(hash(str(entity))))
        self._storage[entity_id] = entity
        {"logger.debug('entity_saved', extra={'entity_id': entity_id})" if self.config.use_logging else "pass"}
        return True
    
    def find(self, identifier: str) -> Any:
        result = self._storage.get(identifier)
        {"logger.debug('entity_found', extra={'entity_id': identifier, 'found': result is not None})" if self.config.use_logging else "pass"}
        return result

@dataclass(frozen=True)
class DefaultService(Service):
    repository: Repository
    processor: Processor
    
    def execute(self, params: Any) -> Any:
        {"logger.info('service_execution_start', extra={'params_type': type(params).__name__})" if self.config.use_logging else "pass"}
        processed = self.processor.process(params)
        self.repository.save(processed)
        {"logger.info('service_execution_complete')" if self.config.use_logging else "pass"}
        return processed
'''
    
    def _get_contexts_template(self) -> str:
        if not self.config.use_context_managers:
            return ""
        
        return f'''#!/usr/bin/env python3
"""
Resource Lifecycle Management (Context Managers)

This module provides context managers for automatic resource setup
and cleanup following the RAII (Resource Acquisition Is Initialization) pattern.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}
Modification Guidelines:
- Use AbstractContextManager for type safety
- Initialize resources in __enter__, cleanup in __exit__
- Return the managed resource from __enter__
- Handle exceptions appropriately in __exit__
- Log lifecycle events when logging is enabled
"""

from __future__ import annotations
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any
import os
import sys
{"import logging" if self.config.use_logging else ""}

if TYPE_CHECKING:
    from .config import AppConfig

# Smart import for core implementations
try:
    from .core import DefaultService, DefaultRepository, DefaultProcessor
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core import DefaultService, DefaultRepository, DefaultProcessor

{"logger = logging.getLogger(__name__)" if self.config.use_logging else ""}

class ServiceContext(AbstractContextManager[DefaultService]):
    def __init__(self, config: "AppConfig"):
        self.config = config
        self.service: DefaultService | None = None
        self.repository: DefaultRepository | None = None
        self.processor: DefaultProcessor | None = None
    
    def __enter__(self) -> DefaultService:
        {"logger.info('initializing_service_context')" if self.config.use_logging else "pass"}
        
        self.repository = DefaultRepository()
        self.processor = DefaultProcessor(name=self.config.app_name)
        self.service = DefaultService(
            repository=self.repository,
            processor=self.processor
        )
        
        {"logger.info('service_context_ready')" if self.config.use_logging else "pass"}
        return self.service
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        {"logger.info('releasing_service_resources')" if self.config.use_logging else "pass"}
        self.service = None
        self.repository = None
        self.processor = None
        return False

class DataContext(AbstractContextManager["DataContext"]):
    def __init__(self, data_source: str):
        self.data_source = data_source
        self.data: list[Any] | None = None
    
    def __enter__(self) -> "DataContext":
        {"logger.info('loading_data', extra={'source': self.data_source})" if self.config.use_logging else "pass"}
        self.data = []
        {"logger.info('data_loaded', extra={'count': len(self.data)})" if self.config.use_logging else "pass"}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        {"logger.info('releasing_data')" if self.config.use_logging else "pass"}
        self.data = None
        return False
    
    def get_data(self) -> list[Any]:
        if self.data is None:
            raise RuntimeError("Data not initialized - use within context manager")
        return self.data
'''
    
    def _get_cli_template(self) -> str:
        if not self.config.use_cli:
            return ""
        
        return f'''#!/usr/bin/env python3
"""
Command-Line Interface and Argument Parsing

This module handles all command-line argument parsing, validation,
and logging configuration for the application.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}
Modification Guidelines:
- Add new arguments to build_parser()
- Implement validation in validate_args()
- Update parse_args_to_config() for new config fields
- Use argparse for consistent CLI patterns
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import argparse
import os
import sys
{"import logging" if self.config.use_logging else ""}

if TYPE_CHECKING:
    from .config import AppConfig

# Smart import for config
try:
    from .config import create_config
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import create_config

def build_parser() -> argparse.ArgumentParser:
    """Build and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="{self.config.project_name}",
        description="{self.config.description}"
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

{"def configure_logging(verbose: bool = False) -> None:" if self.config.use_logging else ""}
{"    \"\"\"Configure application logging.\"\"\"" if self.config.use_logging else ""}
{"    level = logging.DEBUG if verbose else logging.INFO" if self.config.use_logging else ""}
{"    logging.basicConfig(" if self.config.use_logging else ""}
{"        level=level," if self.config.use_logging else ""}
{"        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'" if self.config.use_logging else ""}
{"    )" if self.config.use_logging else ""}

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    valid_commands = ["run", "process", "status"]
    if args.command not in valid_commands:
        raise ValueError(f"Invalid command: {{args.command}}. Must be one of: {{valid_commands}}")

def parse_args_to_config(args: argparse.Namespace) -> "AppConfig":
    """Convert parsed arguments to application configuration."""
    return create_config(
        app_name="{self.config.project_name}",
        output_dir=args.output_dir,
        verbose=args.verbose,
        config_file=args.config_file
    )
'''
    
    def _get_config_template(self) -> str:
        return f'''#!/usr/bin/env python3
"""
Configuration Management

This module defines immutable configuration objects and canonicalization
functions. Follows the principle of single source of truth for all settings.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}
Modification Guidelines:
- Use @dataclass(frozen=True) for immutable config
- Add validation in create_config() function
- Keep configuration flat and explicit
- No global state or hidden defaults
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    app_name: str
    output_dir: Path
    verbose: bool
    config_file: str | None

def create_config(
    app_name: str,
    output_dir: str,
    verbose: bool = False,
    config_file: str | None = None
) -> AppConfig:
    return AppConfig(
        app_name=app_name,
        output_dir=Path(output_dir),
        verbose=verbose,
        config_file=config_file
    )
'''
    
    def _get_main_template(self) -> str:
        return f'''#!/usr/bin/env python3
"""
Application Entry Point and Orchestration

This module wires together all components and provides the main
application logic. Follows dependency injection patterns.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}
Modification Guidelines:
- Keep main() focused on orchestration
- Use context managers for resource management
- Handle errors at the application boundary
- Configure logging before any business logic
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import sys
import os

if TYPE_CHECKING:
    from .config import AppConfig

# Import using the smart import utility
try:
    from .import_utils import smart_import
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from import_utils import smart_import

# Use smart import for all module dependencies
{f"context_imports = smart_import('contexts', ['ServiceContext'])" if self.config.use_context_managers else ""}
{f"cli_imports = smart_import('cli', ['build_parser', 'configure_logging', 'validate_args', 'parse_args_to_config'])" if self.config.use_cli else ""}

# Extract imports for clean usage
{f"ServiceContext = context_imports['ServiceContext']" if self.config.use_context_managers else ""}
{f"build_parser = cli_imports['build_parser']" if self.config.use_cli else ""}
{f"configure_logging = cli_imports['configure_logging']" if self.config.use_cli else ""}
{f"validate_args = cli_imports['validate_args']" if self.config.use_cli else ""}
{f"parse_args_to_config = cli_imports['parse_args_to_config']" if self.config.use_cli else ""}

def main() -> int:
    """Main application entry point."""
    try:
        {f"# Parse command line arguments" if self.config.use_cli else "# Application logic"}
        {f"parser = build_parser()" if self.config.use_cli else ""}
        {f"args = parser.parse_args()" if self.config.use_cli else ""}
        
        {f"# Validate arguments" if self.config.use_cli else ""}
        {f"validate_args(args)" if self.config.use_cli else ""}
        
        {f"# Configure logging" if self.config.use_cli and self.config.use_logging else ""}
        {f"configure_logging(args.verbose)" if self.config.use_cli and self.config.use_logging else ""}
        
        {f"# Create configuration" if self.config.use_cli else ""}
        {f"config = parse_args_to_config(args)" if self.config.use_cli else ""}
        
        {f"# Execute command" if self.config.use_cli else ""}
        {f'''if args.command == "run":
            return run_command(config)
        elif args.command == "process":
            return process_command(config)
        elif args.command == "status":
            return status_command(config)
        else:
            print(f"Unknown command: {{args.command}}", file=sys.stderr)
            return 1''' if self.config.use_cli else 'print("Application started successfully!")\\n        return 0'}
            
    except (ValueError, RuntimeError) as e:
        print(f"Error: {{e}}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user.", file=sys.stderr)
        return 130

{f'''def run_command(config: "AppConfig") -> int:
    """Execute the main application logic."""
    print(f"Running {{config.app_name}}...")
    
    with ServiceContext(config) as service:
        # Example data processing
        test_data = {{"message": "Hello from CLI!", "timestamp": "2024-01-01"}}
        result = service.execute(test_data)
        
        if config.verbose:
            print(f"Processing result: {{result}}")
        
        print("✅ Run completed successfully!")
        return 0

def process_command(config: "AppConfig") -> int:
    """Process data with additional context."""
    print(f"Processing data for {{config.app_name}}...")
    
    with ServiceContext(config) as service:
        # Simulate data processing workflow
        sample_data = [
            {{"id": 1, "value": "first"}},
            {{"id": 2, "value": "second"}},
            {{"id": 3, "value": "third"}}
        ]
        
        results = []
        for item in sample_data:
            result = service.execute(item)
            results.append(result)
            
            if config.verbose:
                print(f"Processed item {{item['id']}}: {{result['processed']}}")
        
        print(f"✅ Processed {{len(results)}} items successfully!")
        return 0

def status_command(config: "AppConfig") -> int:
    """Show application status and configuration."""
    print(f"Status for {{config.app_name}}")
    print(f"Output directory: {{config.output_dir}}")
    print(f"Verbose mode: {{config.verbose}}")
    print(f"Config file: {{config.config_file or 'None'}}")
    print("✅ Application is ready!")
    return 0''' if self.config.use_cli and self.config.use_context_managers else ""}

if __name__ == "__main__":
    sys.exit(main())
'''
    
    def _get_import_utils_template(self) -> str:
        return f'''#!/usr/bin/env python3
"""
Import Utilities for Package/Script Dual Execution

This module provides utilities for handling both relative imports (when run as module)
and absolute imports (when run as script) seamlessly.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}
"""

from __future__ import annotations
import os
import sys
import importlib
from typing import Any, Dict


def smart_import(module_name: str, items: list[str], package_name: str | None = None) -> Dict[str, Any]:
    """
    Import items from a module with automatic fallback between relative and absolute imports.
    
    Args:
        module_name: Name of the module to import from (e.g., 'cli', 'contexts')
        items: List of items to import from the module
        package_name: Optional package name for relative imports (currently unused)
        
    Returns:
        Dictionary mapping item names to imported objects
        
    Usage:
        # Import from cli module
        imports = smart_import('cli', ['build_parser', 'configure_logging'])
        build_parser = imports['build_parser']
        
        # Or unpack directly  
        build_parser, configure_logging = smart_import('cli', ['build_parser', 'configure_logging']).values()
    """
    imported_items = {{}}
    
    # Try relative imports first (when run as module)
    try:
        # Determine if we're in a package context
        frame = sys._getframe(1)  # noqa: SLF001
        calling_module = frame.f_globals.get('__name__', '')
        
        if '.' in calling_module:  # We're in a package
            # Extract package name
            package_parts = calling_module.split('.')
            if package_parts[-1] == '__main__':
                package_parts = package_parts[:-1]
            
            package = '.'.join(package_parts[:-1]) if len(package_parts) > 1 else package_parts[0]
            relative_module = f".{{module_name}}"
            
            module = importlib.import_module(relative_module, package=package)
            for item in items:
                imported_items[item] = getattr(module, item)
        else:
            # Direct execution - use absolute imports
            raise ImportError("Not in package context")
            
    except (ImportError, ValueError, AttributeError):
        # Fallback to absolute imports (when run directly)
        # Add current directory to path if not already there
        current_dir = os.path.dirname(os.path.abspath(sys._getframe(1).f_globals['__file__']))  # noqa: SLF001
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        try:
            module = importlib.import_module(module_name)
            for item in items:
                imported_items[item] = getattr(module, item)
        except ImportError as e:
            raise ImportError(f"Could not import {{items}} from {{module_name}}: {{e}}") from e
    
    return imported_items


def setup_dual_execution_imports(current_file: str) -> None:
    """
    Setup sys.path for dual execution (both module and script).
    
    Args:
        current_file: __file__ from the calling module
    """
    current_dir = os.path.dirname(os.path.abspath(current_file))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
'''
    
    def _get_init_template(self) -> str:
        """Generate __init__.py content with proper package exports."""
        exports = []
        
        # Always export core interfaces and implementations
        exports.extend([
            "# Core interfaces",
            "from .interfaces import Processor, Handler, Repository, Service",
            "",
            "# Core implementations", 
            "from .core import DefaultProcessor, DefaultHandler, DefaultRepository, DefaultService",
        ])
        
        # Add context managers if enabled
        if self.config.use_context_managers:
            exports.extend([
                "",
                "# Context managers",
                "from .contexts import ServiceContext, DataContext",
            ])
        
        # Add CLI components if enabled
        if self.config.use_cli:
            exports.extend([
                "",
                "# CLI components",
                "from .cli import build_parser, parse_args_to_config, validate_args",
            ])
            if self.config.use_logging:
                exports.append("from .cli import configure_logging")
        
        # Add configuration
        exports.extend([
            "",
            "# Configuration",
            "from .config import AppConfig, create_config",
        ])
        
        # Create __all__ list
        all_exports = [
            "Processor", "Handler", "Repository", "Service",
            "DefaultProcessor", "DefaultHandler", "DefaultRepository", "DefaultService",
            "AppConfig", "create_config"
        ]
        
        if self.config.use_context_managers:
            all_exports.extend(["ServiceContext", "DataContext"])
        
        if self.config.use_cli:
            all_exports.extend(["build_parser", "parse_args_to_config", "validate_args"])
            if self.config.use_logging:
                all_exports.append("configure_logging")
        
        return f'''#!/usr/bin/env python3
"""
{self.config.project_name.replace('_', ' ').title()} Package

{self.config.description}

This package follows Go-ish Python patterns with clean modular organization.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}

Usage:
    from {self.config.project_name} import DefaultService, ServiceContext, AppConfig
    
    config = AppConfig(app_name="my_app", output_dir="./output", verbose=True, config_file=None)
    with ServiceContext(config) as service:
        result = service.execute({{"data": "example"}})
"""

from __future__ import annotations

# Package metadata
__version__ = "0.1.0"
__author__ = "{self.config.author}"
__description__ = "{self.config.description}"

{chr(10).join(exports)}

# Public API
__all__ = {all_exports}
'''
    
    def _get_readme_template(self) -> str:
        return f'''# {self.config.project_name}

{self.config.description}

## Installation

```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Usage

### As a Package
```python
from {self.config.project_name} import DefaultService, AppConfig
{'from ' + self.config.project_name + ' import ServiceContext, DataContext' if self.config.use_context_managers else ''}

# Create configuration
config = AppConfig(
    app_name="{self.config.project_name}",
    output_dir="./output",
    verbose=True,
    config_file=None
)

{'# Use context managers for resource management' if self.config.use_context_managers else ''}
{'with ServiceContext(config) as service:' if self.config.use_context_managers else '# Direct usage'}
{'    result = service.execute({"data": "example"})' if self.config.use_context_managers else 'service = DefaultService(DefaultRepository(), DefaultProcessor(name="example"))'}
{'    print(result)' if self.config.use_context_managers else 'result = service.execute({"data": "example"})'}
```

### Command Line Interface
{f"```bash\\npython -m {self.config.project_name} run --verbose\\n# or\\npython main.py run --verbose\\n```" if self.config.use_cli else "```bash\\npython main.py\\n```"}

## Architecture

This project follows Go-ish Python patterns with clean modular organization:

### Modules

- `__init__.py` - Package exports and public API
- `interfaces.py` - Abstract base classes defining contracts
- `core.py` - Concrete implementations of business logic
{f"- `contexts.py` - Context managers for resource lifecycle" if self.config.use_context_managers else ""}
{f"- `cli.py` - Command-line interface" if self.config.use_cli else ""}
- `config.py` - Configuration management
- `main.py` - Application entry point

## Key Principles

- **Interface-based design** with minimal ABC contracts
{f"- **Context managers** for automatic resource management" if self.config.use_context_managers else ""}
- **Explicit dependency injection** through constructor parameters
{f"- **Structured logging** with consistent extra fields" if self.config.use_logging else ""}
- **TYPE_CHECKING** for clean forward references
- **Composition over inheritance**
- **Package-based imports** for clean API surface

## API Reference

### Core Interfaces
- `Processor` - Data processing contract
- `Handler` - Request handling contract  
- `Repository` - Data persistence contract
- `Service` - Business logic execution contract

### Implementations
- `DefaultProcessor` - Basic data processor
- `DefaultHandler` - Request handler with service delegation
- `DefaultRepository` - In-memory storage implementation
- `DefaultService` - Service orchestrating processor and repository

{'### Context Managers' if self.config.use_context_managers else ''}
{'- `ServiceContext` - Manages service lifecycle and dependencies' if self.config.use_context_managers else ''}
{'- `DataContext` - Manages data loading and cleanup' if self.config.use_context_managers else ''}

## Development

Created with Python Project Scaffolder following Go-ish Python guidelines.

Author: {self.config.author}

---
*Generated with Go-ish Python Scaffolder by Yamaç Eren Ay (@yamaceay)*
'''
    
    def _get_gitignore_template(self) -> str:
        return '''__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
*.manifest
*.spec
pip-log.txt
pip-delete-this-directory.txt
.tox/
.nox/
.coverage
.pytest_cache/
cover/
*.cover
*.py,cover
.hypothesis/
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
'''
    
    def _get_pyproject_template(self) -> str:
        return f'''[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{self.config.project_name}"
version = "0.1.0"
description = "{self.config.description}"
authors = [
    {{name = "{self.config.author}", email = "yamaceay@users.noreply.github.com"}}
]
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    {"'logging'," if self.config.use_logging else ""}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "mypy>=0.950",
    "flake8>=4.0.0",
]

[project.scripts]
{self.config.project_name} = "{self.config.project_name}.main:main"

[project.urls]
Homepage = "https://github.com/yamaceay/{self.config.project_name}"
Repository = "https://github.com/yamaceay/{self.config.project_name}"
Issues = "https://github.com/yamaceay/{self.config.project_name}/issues"

[tool.setuptools]
packages = ["{self.config.project_name}"]

[tool.setuptools.package-data]
{self.config.project_name} = ["py.typed"]

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
known_first_party = ["{self.config.project_name}"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
show_error_codes = true
strict = true

[[tool.mypy.overrides]]
module = "{self.config.project_name}.*"
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
'''
    
    def _get_requirements_template(self) -> str:
        return '''# Core dependencies
# Add your project dependencies here
'''

def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Python Project Scaffolder")
    parser.add_argument("project_name", help="Name of the project to create")
    parser.add_argument("--type", choices=["cli_tool"], 
                       default="cli_tool", help="Type of project to create")
    parser.add_argument("--target-dir", type=Path, default=Path.cwd(), 
                       help="Target directory to create project in")
    parser.add_argument("--author", default="Unknown author", help="Project author name")
    parser.add_argument("--description", default="A Go-ish Python project", 
                       help="Project description")
    parser.add_argument("--no-logging", action="store_true", help="Disable logging setup")
    parser.add_argument("--no-cli", action="store_true", help="Disable CLI setup")
    parser.add_argument("--no-context-managers", action="store_true", 
                       help="Disable context managers")
    
    args = parser.parse_args()
    
    config = ScaffoldConfig(
        project_name=args.project_name,
        project_type=ProjectType(args.type),
        target_dir=args.target_dir,
        author=args.author,
        description=args.description,
        use_logging=not args.no_logging,
        use_cli=not args.no_cli,
        use_context_managers=not args.no_context_managers
    )
    
    scaffolder = ProjectScaffolder(config)
    scaffolder.create_project()

if __name__ == "__main__":
    main()