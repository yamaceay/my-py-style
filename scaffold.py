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
        return f'''#!/usr/bin/env python3
"""
Abstract Base Classes (Interface Definitions)

This module defines the core contracts for the application using ABC patterns.
All interfaces follow Rule #1: minimal, focused responsibilities with clear contracts.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}

Architecture Guidelines from README.md:
- Keep interfaces minimal and focused (single responsibility)
- Use @abstractmethod for all required methods
- Avoid implementation details in interfaces
- Define clear contracts that multiple implementations can follow
- Support plug-and-play polymorphism (Rule #2)

Interface Design Principles:
- Methods should accept only essential parameters
- Static dependencies go in implementing class __init__ (Rule #1)
- Dynamic inputs only in method signatures
- Return types should be clear and consistent
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class Processor(ABC):
    """
    Core data processing contract.
    
    Implementations handle data transformation, analysis, or manipulation.
    Examples: TextProcessor, ImageProcessor, DataNormalizer
    """
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process input data and return result.
        
        Args:
            data: Input data to process (dynamic input only)
            
        Returns:
            Processed result
            
        Note: All configuration should be in implementing class __init__
        """
        pass

class Handler(ABC):
    """
    Request/command handling contract.
    
    Implementations handle different types of requests or commands.
    Examples: APIHandler, FileHandler, EventHandler
    """
    
    @abstractmethod
    def handle(self, request: Any) -> Any:
        """
        Handle a request and return response.
        
        Args:
            request: Request to handle (dynamic input only)
            
        Returns:
            Response or result of handling
            
        Note: Handler configuration (routes, auth, etc.) goes in __init__
        """
        pass

class Repository(ABC):
    """
    Data persistence contract.
    
    Implementations provide data storage and retrieval.
    Examples: FileRepository, DatabaseRepository, APIRepository
    """
    
    @abstractmethod
    def save(self, entity: Any) -> bool:
        """
        Persist an entity.
        
        Args:
            entity: Entity to save (dynamic input only)
            
        Returns:
            True if successful, False otherwise
            
        Note: Connection details, credentials go in __init__
        """
        pass
    
    @abstractmethod
    def find(self, identifier: str) -> Any:
        """
        Retrieve an entity by identifier.
        
        Args:
            identifier: Unique identifier (dynamic input only)
            
        Returns:
            Found entity or None if not found
            
        Note: Query configuration goes in implementing class __init__
        """
        pass

class Service(ABC):
    """
    Business logic orchestration contract.
    
    Implementations coordinate between processors, repositories, and handlers.
    Examples: UserService, OrderService, AnalyticsService
    """
    
    @abstractmethod
    def execute(self, params: Any) -> Any:
        """
        Execute business logic with given parameters.
        
        Args:
            params: Execution parameters (dynamic input only)
            
        Returns:
            Result of business logic execution
            
        Note: Service dependencies (processor, repo) injected in __init__
        """
        pass

# Optional: Additional domain-specific interfaces can be added here
# Following the same patterns:

class Validator(ABC):
    """
    Data validation contract.
    
    Implementations provide validation logic for different data types.
    Examples: SchemaValidator, BusinessRuleValidator, FormatValidator
    """
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate input data.
        
        Args:
            data: Data to validate (dynamic input only)
            
        Returns:
            True if valid, False otherwise
            
        Note: Validation rules configured in implementing class __init__
        """
        pass
    
    @abstractmethod
    def get_errors(self) -> list[str]:
        """
        Get validation error messages from last validation.
        
        Returns:
            List of error messages, empty if no errors
        """
        pass

class Transformer(ABC):
    """
    Data transformation contract.
    
    Implementations convert data between different formats or structures.
    Examples: JsonTransformer, XMLTransformer, FormatConverter
    """
    
    @abstractmethod
    def transform(self, source: Any, target_format: str) -> Any:
        """
        Transform data to target format.
        
        Args:
            source: Source data (dynamic input only)
            target_format: Desired output format (dynamic input only)
            
        Returns:
            Transformed data in target format
            
        Note: Transformation rules configured in implementing class __init__
        """
        pass

# Interface composition example - showing how to combine contracts
class ProcessorService(Service, Processor):
    """
    Example of interface composition for services that also process data.
    
    This demonstrates how interfaces can be combined while maintaining
    clear responsibilities and contracts.
    """
    
    # Inherits both Service.execute() and Processor.process() methods
    # Implementing classes must provide both
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
in interfaces.py. Follows Rule #1: static dependencies in constructor,
methods accept only dynamic inputs.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}

Modification Guidelines:
- Use @dataclass(frozen=True) for static dependencies (passed in, never change)
- Use regular __init__ for dynamic initialization or computed dependencies  
- Static deps go in constructor: tokenizer, labels, device, paths, config
- Methods only accept dynamic inputs: data to process, requests to handle
- Keep methods focused on single responsibilities
- Log with structured context when enabled

Rule #1 Examples:
  GOOD: processor = DefaultProcessor(name="task_processor", config=app_config)
        result = processor.process(data)  # Only dynamic input
  
  BAD:  result = processor.process(data, name="task_processor")  # Static in method
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
    """Example of @dataclass(frozen=True) with static dependencies."""
    name: str                    # Static dependency - passed once, never changes
    processing_mode: str = "default"  # Static configuration
    
    def process(self, data: Any) -> Any:
        """Method accepts only dynamic input - data to process."""
        {"logger.info('processing_data', extra={'processor': self.name, 'mode': self.processing_mode})" if self.config.use_logging else "pass"}
        return {{
            "processed": True, 
            "data": data, 
            "processor": self.name,
            "mode": self.processing_mode
        }}

@dataclass(frozen=True)
class DefaultHandler(Handler):
    """Example of dependency injection - service is static dependency."""
    service: Service            # Static dependency - injected once
    handler_name: str = "default"
    
    def handle(self, request: Any) -> Any:
        """Method accepts only dynamic input - request to handle."""
        {"logger.info('handling_request', extra={'handler': self.handler_name, 'request_type': type(request).__name__})" if self.config.use_logging else "pass"}
        return self.service.execute(request)

class DefaultRepository(Repository):
    """Example of regular __init__ for dynamic initialization."""
    
    def __init__(self, storage_config: str = "memory"):
        """Dynamic initialization - can compute storage based on config."""
        self.storage_config = storage_config
        self._storage: dict[str, Any] = {{}}
        
        # Example of computed initialization
        if storage_config == "persistent":
            # Would initialize file/db storage here
            pass
    
    def save(self, entity: Any) -> bool:
        """Method accepts only dynamic input - entity to save."""
        entity_id = getattr(entity, "id", str(hash(str(entity))))
        self._storage[entity_id] = entity
        {"logger.debug('entity_saved', extra={'entity_id': entity_id, 'storage': self.storage_config})" if self.config.use_logging else "pass"}
        return True
    
    def find(self, identifier: str) -> Any:
        """Method accepts only dynamic input - identifier to find."""
        result = self._storage.get(identifier)
        {"logger.debug('entity_found', extra={'entity_id': identifier, 'found': result is not None})" if self.config.use_logging else "pass"}
        return result

@dataclass(frozen=True) 
class DefaultService(Service):
    """Example of composition with static dependency injection."""
    repository: Repository      # Static dependency - injected once
    processor: Processor       # Static dependency - injected once
    service_name: str = "default"
    
    def execute(self, params: Any) -> Any:
        """Method accepts only dynamic input - params to execute on."""
        {"logger.info('service_execution_start', extra={'service': self.service_name, 'params_type': type(params).__name__})" if self.config.use_logging else "pass"}
        
        # Pure business logic - no branching, no re-configuration
        processed = self.processor.process(params)
        self.repository.save(processed)
        
        {"logger.info('service_execution_complete', extra={'service': self.service_name})" if self.config.use_logging else "pass"}
        return processed

# Example of config-driven initialization (follows Rule #3 & #4)
class ConfigDrivenProcessor(Processor):
    """Example using config for dynamic initialization."""
    
    def __init__(self, config: "AppConfig"):
        """Takes config for computed dependencies."""
        self.app_name = config.app_name           # Extract what we need
        self.output_dir = config.output_dir       # No hidden globals
        self.verbose = config.verbose             # Single source of truth
        
        # Computed initialization based on config
        self.processor_id = f"{{self.app_name}}_processor"
        
        # Set up output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, data: Any) -> Any:
        """Only dynamic input - no config re-detection."""
        if self.verbose:
            {"logger.debug('config_driven_processing', extra={'processor_id': self.processor_id})" if self.config.use_logging else "pass"}
        
        return {{
            "processed": True,
            "data": data,
            "processor_id": self.processor_id,
            "output_path": str(self.output_dir)
        }}
'''
    
    def _get_contexts_template(self) -> str:
        if not self.config.use_context_managers:
            return ""
        
        return f'''#!/usr/bin/env python3
"""
Resource Lifecycle Management (Context Managers)

This module implements Rule #5: Context managers for lifecycle.
Heavy resources must be scoped with `with ...:` for explicit construction and cleanup.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}

Modification Guidelines:
- Use AbstractContextManager for type safety
- Initialize resources in __enter__, cleanup in __exit__
- Return the managed resource from __enter__ (not self)
- Handle exceptions appropriately in __exit__
- Log lifecycle events when logging is enabled
- Follow Rule #1: static deps in __init__, dynamic inputs in methods

Rule #5 Examples:
  GOOD: Heavy resource management
        with ServiceContext(config) as service:
            result = service.execute(data)  # Automatic cleanup
  
  BAD:  Manual resource management
        service = create_service(config)
        try:
            result = service.execute(data)
        finally:
            service.cleanup()  # Manual cleanup, might be forgotten
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
    """
    Context manager for service lifecycle (Rule #5).
    
    Manages the complete lifecycle of service and its dependencies:
    - Repository creation and connection
    - Processor initialization 
    - Service composition
    - Automatic cleanup on exit
    
    Static dependencies go in __init__ (Rule #1).
    """
    
    def __init__(self, config: "AppConfig"):
        """Static dependency: config is passed once and never changes (Rule #1)."""
        self.config = config
        self.service: DefaultService | None = None
        self.repository: DefaultRepository | None = None
        self.processor: DefaultProcessor | None = None
    
    def __enter__(self) -> DefaultService:
        """Initialize all resources - explicit construction (Rule #5)."""
        {"logger.info('initializing_service_context', extra={'app_name': self.config.app_name})" if self.config.use_logging else "pass"}
        
        # Initialize components with proper dependency injection (Rule #1)
        # Static deps from config, no re-detection (Rule #3 & #4)
        self.repository = DefaultRepository(storage_config="memory")
        self.processor = DefaultProcessor(
            name=self.config.processor_name,  # Use computed property
            processing_mode="default"
        )
        
        # Compose service with injected dependencies
        self.service = DefaultService(
            repository=self.repository,
            processor=self.processor,
            service_name=self.config.app_name
        )
        
        {"logger.info('service_context_ready', extra={'service_name': self.config.app_name, 'verbose': self.config.verbose})" if self.config.use_logging else "pass"}
        return self.service
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Explicit cleanup - guaranteed to run (Rule #5)."""
        {"logger.info('releasing_service_resources', extra={'had_exception': exc_type is not None})" if self.config.use_logging else "pass"}
        
        # Clean up in reverse order of initialization
        if self.service:
            # Service might have cleanup logic
            pass
            
        if self.repository:
            # Repository might need to close connections, flush data, etc.
            pass
            
        if self.processor:
            # Processor might need to free GPU memory, close files, etc.
            pass
        
        # Clear references
        self.service = None
        self.repository = None  
        self.processor = None
        
        {"logger.info('service_resources_released')" if self.config.use_logging else "pass"}
        return False  # Don't suppress exceptions

class DataContext(AbstractContextManager["DataContext"]):
    """
    Context manager for data loading and cleanup (Rule #5).
    
    Example of resource management for data sources:
    - File handles
    - Database connections  
    - Network streams
    - Large datasets in memory
    """
    
    def __init__(self, data_source: str, config: "AppConfig"):
        """Static dependencies: data_source and config (Rule #1)."""
        self.data_source = data_source
        self.config = config
        self.data: list[Any] | None = None
        self._file_handle = None
        self._connection = None
    
    def __enter__(self) -> "DataContext":
        """Load data resources - explicit acquisition (Rule #5)."""
        {"logger.info('loading_data', extra={'source': self.data_source, 'output_dir': str(self.config.output_dir)})" if self.config.use_logging else "pass"}
        
        # Example: Load data based on source type
        if self.data_source.startswith("file://"):
            # Would open file handle here
            # self._file_handle = open(self.data_source[7:], 'r')
            self.data = []  # Simulate file data
        elif self.data_source.startswith("db://"):
            # Would open database connection here
            # self._connection = create_connection(self.data_source)
            self.data = []  # Simulate database data
        else:
            # Default in-memory data
            self.data = []
        
        {"logger.info('data_loaded', extra={'count': len(self.data), 'source_type': self.data_source.split('://')[0] if '://' in self.data_source else 'default'})" if self.config.use_logging else "pass"}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Explicit cleanup - free resources (Rule #5)."""
        {"logger.info('releasing_data_resources', extra={'had_exception': exc_type is not None})" if self.config.use_logging else "pass"}
        
        # Clean up resources in reverse order
        if self._connection:
            # Would close database connection
            # self._connection.close()
            self._connection = None
            
        if self._file_handle:
            # Would close file handle
            # self._file_handle.close()
            self._file_handle = None
        
        # Clear large data from memory
        if self.data:
            self.data.clear()
        self.data = None
        
        {"logger.info('data_resources_released')" if self.config.use_logging else "pass"}
        return False
    
    def get_data(self) -> list[Any]:
        """Get loaded data - only accepts dynamic input: none needed here."""
        if self.data is None:
            raise RuntimeError("Data not initialized - use within context manager")
        return self.data
    
    def add_data(self, item: Any) -> None:
        """Add data item - dynamic input only (Rule #1)."""
        if self.data is None:
            raise RuntimeError("Data not initialized - use within context manager") 
        self.data.append(item)

class ModelContext(AbstractContextManager["ModelContext"]):
    """
    Example context manager for ML model lifecycle (Rule #5).
    
    Demonstrates managing:
    - GPU memory allocation
    - Model loading from disk
    - Tokenizer initialization
    - Automatic cleanup
    """
    
    def __init__(self, model_path: str, device: str, config: "AppConfig"):
        """Static dependencies: model_path, device, config (Rule #1)."""
        self.model_path = model_path
        self.device = device
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def __enter__(self) -> "ModelContext":
        """Load model resources - heavy resource acquisition (Rule #5)."""
        {"logger.info('loading_model', extra={'model_path': self.model_path, 'device': self.device})" if self.config.use_logging else "pass"}
        
        # Simulate model loading (would use actual ML libraries)
        # self.model = torch.load(self.model_path, map_location=self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = f"model_on_{{self.device}}"
        self.tokenizer = f"tokenizer_for_{{self.model_path}}"
        
        {"logger.info('model_loaded', extra={'device': self.device, 'verbose': self.config.verbose})" if self.config.use_logging else "pass"}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Explicit cleanup - free GPU memory (Rule #5)."""
        {"logger.info('releasing_model_resources', extra={'device': self.device})" if self.config.use_logging else "pass"}
        
        # Free GPU memory and model resources
        if self.model:
            # Would call model.cpu(), del model, torch.cuda.empty_cache(), etc.
            self.model = None
            
        if self.tokenizer:
            # Would clear tokenizer cache if needed
            self.tokenizer = None
        
        {"logger.info('model_resources_released')" if self.config.use_logging else "pass"}
        return False
    
    def predict(self, text: str) -> str:
        """Make prediction - dynamic input only (Rule #1)."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized - use within context manager")
        
        # Would do actual prediction here
        return f"prediction_for_{{text}}_using_{{self.model}}"

# Example usage patterns (following all rules):
def example_proper_context_usage(config: "AppConfig") -> dict[str, Any]:
    """
    Example showing proper context manager usage following all rules.
    
    Demonstrates:
    - Rule #1: Static deps in function signature, dynamic inputs in methods
    - Rule #2: Early selection, no branching
    - Rule #3 & #4: Single config source, passed down
    - Rule #5: Context managers for heavy resources
    - Rule #6: Structured logging over print
    """
    results = []
    
    # Multiple context managers can be nested or chained
    with ServiceContext(config) as service:
        with DataContext("memory://default", config) as data_ctx:
            
            # Add some test data (dynamic input)
            data_ctx.add_data({{"type": "test", "value": 1}})
            data_ctx.add_data({{"type": "test", "value": 2}})
            
            # Process all data
            for item in data_ctx.get_data():
                result = service.execute(item)  # Only dynamic input
                results.append(result)
    
    # Both contexts automatically cleaned up here
    return {{"processed_items": len(results), "results": results}}

# Anti-patterns (DO NOT DO THIS):
#
# WRONG - Manual resource management (violates Rule #5):
# def bad_manual_cleanup():
#     service = DefaultService(...)
#     try:
#         result = service.execute(data)
#     finally:
#         service.cleanup()  # Might be forgotten!
#
# WRONG - No context manager for heavy resources (violates Rule #5):
# def bad_no_context():
#     model = load_gpu_model()  # GPU memory allocated
#     result = model.predict(text)
#     # GPU memory never freed!
#     return result
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

This module follows Rules #3 & #4: One config to rule them all + Passing config downwards.
Single source of truth for all settings with proper canonicalization.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}

Modification Guidelines:
- Use @dataclass(frozen=True) for immutable config (Rule #3)
- Build configuration once at application start 
- Pass config down, never re-detect or re-parse (Rule #4)
- Keep configuration flat and explicit
- No global state or hidden defaults
- Use canonicalization functions for computed values

Rules #3 & #4 Examples:
  GOOD: config = create_config(...); processor = Processor(config.name, config.output_dir)
  BAD:  processor.method() -> config = load_config()  # Re-detection forbidden!

  GOOD: Explicit config passing
        service = Service(config.timeout, config.verbose)
  BAD:  Hidden globals
        SERVICE_TIMEOUT = 30  # Global state forbidden!
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration - single source of truth (Rule #3)."""
    
    # Core application settings
    app_name: str
    output_dir: Path
    verbose: bool
    
    # Optional configuration  
    config_file: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    
    # Computed properties can be methods (since dataclass is frozen)
    @property 
    def log_level(self) -> str:
        """Computed from verbose flag - no re-detection needed."""
        return "DEBUG" if self.verbose else "INFO"
    
    @property
    def processor_name(self) -> str:
        """Computed from app_name - consistent across application."""
        return f"{{self.app_name}}_processor"

def create_config(
    app_name: str,
    output_dir: str,
    verbose: bool = False,
    config_file: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None
) -> AppConfig:
    """
    Create and canonicalize configuration - the ONLY place where defaults are applied.
    This function implements Rule #3: single source of truth.
    
    Args:
        app_name: Application name (required)
        output_dir: Output directory path (required)
        verbose: Enable verbose logging (default: False)
        config_file: Optional configuration file path
        timeout: Operation timeout in seconds (default: 30)
        max_retries: Maximum retry attempts (default: 3)
    
    Returns:
        Immutable AppConfig object
        
    Note: This is the ONLY place where defaults are resolved.
    Never re-detect these values in other parts of the application.
    """
    
    # Canonicalize and validate inputs (Rule #3: single source of truth)
    canonical_output_dir = Path(output_dir).resolve()
    canonical_timeout = timeout if timeout is not None else 30
    canonical_max_retries = max_retries if max_retries is not None else 3
    
    # Ensure output directory exists
    canonical_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate settings
    if canonical_timeout <= 0:
        raise ValueError(f"Timeout must be positive, got: {{canonical_timeout}}")
    if canonical_max_retries < 0:
        raise ValueError(f"Max retries must be non-negative, got: {{canonical_max_retries}}")
    
    return AppConfig(
        app_name=app_name,
        output_dir=canonical_output_dir,
        verbose=verbose,
        config_file=config_file,
        timeout=canonical_timeout,
        max_retries=canonical_max_retries
    )

def extract_config_kwargs(config: AppConfig, *exclude_keys: str) -> dict[str, any]:
    """
    Extract config as kwargs for passing down (Rule #4).
    
    This helper implements Rule #4: passing config downwards.
    Use this to "weaken" config as it goes deeper into the call stack.
    
    Args:
        config: The source configuration
        exclude_keys: Keys to exclude from the result (already extracted)
        
    Returns:
        Dictionary of config values for **kwargs passing
        
    Example:
        config = create_config(...)
        
        # Extract specific values, pass rest as kwargs
        app_name = config.app_name
        output_dir = config.output_dir
        
        # Pass remaining config down
        kwargs = extract_config_kwargs(config, "app_name", "output_dir")
        processor = Processor(app_name, output_dir, **kwargs)
    """
    config_dict = config.__dict__.copy()
    
    # Remove excluded keys (already extracted)
    for key in exclude_keys:
        config_dict.pop(key, None)
        
    return config_dict

# Example of config-driven component setup (follows Rule #4)
def setup_components_from_config(config: AppConfig) -> dict[str, any]:
    """
    Example showing proper config usage following Rule #4.
    
    This demonstrates:
    - No re-detection of config values
    - Explicit extraction of needed parameters  
    - Passing config downwards appropriately
    - No hidden globals or re-parsing
    """
    
    # Extract specific values we need (Rule #4: explicit extraction)
    app_name = config.app_name
    output_dir = config.output_dir
    verbose = config.verbose
    
    # Get remaining config for passing down
    remaining_config = extract_config_kwargs(config, "app_name", "output_dir", "verbose")
    
    # Example component setup with proper config passing
    components = {{
        "app_name": app_name,
        "output_dir": output_dir,
        "verbose": verbose,
        "log_level": config.log_level,  # Use computed property
        "processor_name": config.processor_name,  # Use computed property
        "remaining_config": remaining_config
    }}
    
    return components

# Anti-pattern examples (DO NOT DO THIS):
# 
# WRONG - Global config (violates Rule #3):
# GLOBAL_CONFIG = {{"timeout": 30}}  # Hidden global state
# 
# WRONG - Re-detection (violates Rule #4):
# def some_function():
#     config = load_config()  # Re-parsing configuration
#     timeout = os.getenv("TIMEOUT", 30)  # Re-detecting environment
# 
# WRONG - Hidden defaults in deep functions (violates Rule #3):
# def deep_function(data, output_dir="/tmp/default"):  # Should be explicit
'''
    
    def _get_main_template(self) -> str:
        return f'''#!/usr/bin/env python3
"""
Application Entry Point and Orchestration

This module wires together all components and provides the main
application logic. Follows Rule #2: Choose concrete class once (early selection),
then call common method. No if/elif/else ladders.

Generated by: Go-ish Python Scaffolder
Author: {self.config.author}

Modification Guidelines:
- Keep main() focused on orchestration
- Use early selection for polymorphism (Rule #2) 
- Use context managers for resource management (Rule #5)
- Handle errors at the application boundary
- Configure logging before any business logic
- No runtime branching - choose concrete class once

Rule #2 Examples:
  GOOD: Choose once, execute with common interface
        command_handler = RunCommandHandler() if args.command == "run" else ProcessCommandHandler()
        return command_handler.execute(config)
  
  BAD:  Runtime branching everywhere
        if args.command == "run": return run_logic(config)
        elif args.command == "process": return process_logic(config)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
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

# Command Handler Protocol (Rule #1: Contract as ABC)
class CommandHandler(Protocol):
    def execute(self, config: "AppConfig") -> int:
        """Execute command with given configuration."""
        ...

{f'''# Concrete Command Handlers (Rule #1: Small classes, static deps in __init__)
class RunCommandHandler:
    """Handler for run command - no static dependencies needed."""
    
    def execute(self, config: "AppConfig") -> int:
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

class ProcessCommandHandler:
    """Handler for process command - no static dependencies needed."""
    
    def execute(self, config: "AppConfig") -> int:
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

class StatusCommandHandler:
    """Handler for status command - no static dependencies needed."""
    
    def execute(self, config: "AppConfig") -> int:
        """Show application status and configuration."""
        print(f"Status for {{config.app_name}}")
        print(f"Output directory: {{config.output_dir}}")
        print(f"Verbose mode: {{config.verbose}}")
        print(f"Config file: {{config.config_file or 'None'}}")
        print("✅ Application is ready!")
        return 0''' if self.config.use_cli and self.config.use_context_managers else ""}

def main() -> int:
    """Main application entry point - demonstrates Rule #2 early selection."""
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
        
        {f'''if args.command == "run":
            command_handler: CommandHandler = RunCommandHandler()
        elif args.command == "process":
            command_handler = ProcessCommandHandler()
        elif args.command == "status":
            command_handler = StatusCommandHandler()
        else:
            print(f"Unknown command: {{args.command}}", file=sys.stderr)
            return 1
        
        # Now execute with common interface - no more branching!
        return command_handler.execute(config)''' if self.config.use_cli else '''print("Application started successfully!")
        return 0'''}
            
    except (ValueError, RuntimeError) as e:
        print(f"Error: {{e}}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user.", file=sys.stderr)
        return 130

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