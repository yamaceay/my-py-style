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
    ML_FRAMEWORK = "ml_framework"
    API_SERVER = "api_server" 
    CLI_TOOL = "cli_tool"
    DATA_PIPELINE = "data_pipeline"

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
        (self.project_path / "tests").mkdir(exist_ok=True)
    
    def _generate_core_files(self) -> None:
        self._create_interfaces_file()
        self._create_core_file()
        if self.config.use_context_managers:
            self._create_contexts_file()
        if self.config.use_cli:
            self._create_cli_file()
        self._create_config_file()
        self._create_main_file()
        self._create_readme()
        self._create_gitignore()
    
    def _generate_project_specific_files(self) -> None:
        if self.config.project_type == ProjectType.ML_FRAMEWORK:
            self._create_ml_specific_files()
        elif self.config.project_type == ProjectType.API_SERVER:
            self._create_api_specific_files()
        elif self.config.project_type == ProjectType.CLI_TOOL:
            self._create_cli_specific_files()
        elif self.config.project_type == ProjectType.DATA_PIPELINE:
            self._create_pipeline_specific_files()
    
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
    
    def _create_ml_specific_files(self) -> None:
        pass
    
    def _create_api_specific_files(self) -> None:
        pass
    
    def _create_cli_specific_files(self) -> None:
        pass
    
    def _create_pipeline_specific_files(self) -> None:
        pass
    
    def _write_file(self, filename: str, content: str) -> None:
        filepath = self.project_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    
    def _get_interfaces_template(self) -> str:
        return f'''from __future__ import annotations
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
        
        return f'''from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
{runtime_imports}
if TYPE_CHECKING:
    from config import AppConfig

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
        
        return f'''from __future__ import annotations
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING
{"import logging" if self.config.use_logging else ""}

if TYPE_CHECKING:
    from config import AppConfig

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
        
        return f'''from __future__ import annotations
from typing import TYPE_CHECKING
import argparse
{"import logging" if self.config.use_logging else ""}
import sys

if TYPE_CHECKING:
    from config import AppConfig

from config import create_config

def build_parser() -> argparse.ArgumentParser:
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
{"    level = logging.DEBUG if verbose else logging.INFO" if self.config.use_logging else ""}
{"    logging.basicConfig(" if self.config.use_logging else ""}
{"        level=level," if self.config.use_logging else ""}
{"        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'" if self.config.use_logging else ""}
{"    )" if self.config.use_logging else ""}

def validate_args(args: argparse.Namespace) -> None:
    if args.command not in ["run", "process", "status"]:
        raise ValueError(f"Invalid command: {{args.command}}")

def parse_args_to_config(args: argparse.Namespace) -> "AppConfig":
    return create_config(
        app_name="{self.config.project_name}",
        output_dir=args.output_dir,
        verbose=args.verbose,
        config_file=args.config_file
    )
'''
    
    def _get_config_template(self) -> str:
        return f'''from __future__ import annotations
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
        cli_import = "from cli import build_parser, configure_logging, parse_args_to_config, validate_args" if self.config.use_cli else ""
        context_import = "from contexts import ServiceContext, DataContext" if self.config.use_context_managers else ""
        
        main_logic = ""
        if self.config.use_context_managers and self.config.use_cli:
            main_logic = f'''def run_application(args, config: "AppConfig") -> dict[str, Any]:
    {"logger.info('application_starting', extra={'command': args.command})" if self.config.use_logging else ""}
    
    with DataContext("default_source") as data_ctx:
        with ServiceContext(config) as service:
            if args.command == "run":
                result = service.execute(data_ctx.get_data())
            elif args.command == "process":
                result = service.execute({{"action": "process"}})
            elif args.command == "status":
                result = {{"status": "running", "app": config.app_name}}
            else:
                raise ValueError(f"Unknown command: {{args.command}}")
            
            {"logger.info('application_complete', extra={'success': True})" if self.config.use_logging else ""}
            return result

if __name__ == "__main__":
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    validate_args(args)
    config = parse_args_to_config(args)
    
    try:
        result = run_application(args, config)
        print("Application completed successfully!")
        print(f"Result: {{result}}")
    except Exception as e:
        {"logger.error('application_failed', extra={'error': str(e)})" if self.config.use_logging else ""}
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)'''
        else:
            main_logic = '''def main() -> None:
    print(f"Welcome to {self.config.project_name}!")
    
if __name__ == "__main__":
    main()'''
        
        return f'''from __future__ import annotations
from typing import TYPE_CHECKING, Any
{"import logging" if self.config.use_logging else ""}
import sys

if TYPE_CHECKING:
    from config import AppConfig

from core import DefaultService, DefaultHandler, DefaultProcessor, DefaultRepository
{context_import}
{cli_import}

{"logger = logging.getLogger(__name__)" if self.config.use_logging else ""}

{main_logic}
'''
    
    def _get_readme_template(self) -> str:
        return f'''# {self.config.project_name}

{self.config.description}

## Architecture

This project follows Go-ish Python patterns with clean modular organization:

### Modules

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

## Usage

{f"```bash\\npython main.py run --verbose\\n```" if self.config.use_cli else "```bash\\npython main.py\\n```"}

## Installation

```bash
pip install -r requirements.txt
```

## Development

Created with Python Project Scaffolder following Go-ish Python guidelines.

Author: {self.config.author}
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
    {{name = "{self.config.author}"}}
]
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    {"'logging'," if self.config.use_logging else ""}
]

[project.scripts]
{self.config.project_name} = "{self.config.project_name}.main:main"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
'''
    
    def _get_requirements_template(self) -> str:
        return '''# Core dependencies
# Add your project dependencies here
'''

def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Python Project Scaffolder")
    parser.add_argument("project_name", help="Name of the project to create")
    parser.add_argument("--type", choices=["ml_framework", "api_server", "cli_tool", "data_pipeline"], 
                       default="cli_tool", help="Type of project to create")
    parser.add_argument("--target-dir", type=Path, default=Path.cwd(), 
                       help="Target directory to create project in")
    parser.add_argument("--author", default="Unknown Author", help="Project author name")
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