# Go-ish Python: Patterns, Guidelines & Project Scaffolder

> **A comprehensive guide to writing maintainable, scalable Python code using Go-inspired patterns**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository demonstrates class-centric, plug-and-play Python design: **class contracts + constructor injection**, **no runtime branching**, **CLI as single source of truth**, **context-scoped resources**, **structured logging**, and **DRY config**.

The code should read like prose—types and method names tell the story. Comments are optional because design is explicit.

## TL;DR (the rules I live by)

- **Contracts as ABCs; implementations as small classes.** Static deps (tokenizer, labels, device, paths) go into `__init__`; methods only take **dynamic input**.
- **Plug‑and‑play polymorphism > branching.** Choose the concrete class once (CLI/factory/context) and then call the common method. No `if/elif/else` ladders.
- **One config to rule them all.** Build `RuntimeConfig` once at the top; pass it down. No config files. No hidden globals. No re‑detecting device in submodules.
- **Context managers for lifecycle.** Heavy resources must be used with `with ...:`; construction and cleanup are explicit.
- **Top‑level defaults only.** Deep helpers are positional and explicit; if you expose a default, it's `None` and you canonicalize immediately.
- **Logging over `print`.** Library code logs with context; CLI may print final user‑facing summaries.

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Quick Start](#quick-start)
- [The Rules](#the-rules)
- [Architecture Guide](#architecture-guide)
- [Project Scaffolder](#project-scaffolder)
- [Repository Transformation](#repository-transformation)
- [Detailed Examples](#detailed-examples)
- [Checklist & Best Practices](#checklist--best-practices)

---

## Project Map

```
interfaces.py  # ABCs: Processor, Handler, Repository, Service
config.py      # AppConfig + create_config + canonicalization
core.py        # Concrete classes: DefaultProcessor, DefaultHandler, DefaultRepository, DefaultService
contexts.py    # Context managers: ServiceContext, DataContext
cli.py         # Argument parser + logging setup + validation
main.py        # Wiring: parse → config → contexts → process → execute
pyproject.toml # Modern Python packaging configuration
requirements.txt # Dependencies
.gitignore     # Comprehensive ignore patterns
README.md      # Project documentation
```

---

## Core Philosophy

This approach prioritizes **explicitness**, **composition**, and **predictable code paths** over clever abstractions. The code should read like prose—types and method names tell the story, comments become optional because design is explicit.

### Why Go-ish Python?

- **Maintainable**: Clear module boundaries and explicit dependencies
- **Testable**: Interface-based design enables easy mocking
- **Scalable**: Modular structure supports project growth
- **Predictable**: No hidden state or surprising behavior
- **Observable**: Built-in structured logging and monitoring

### Architecture at a Glance

```
CLI (args) ──▶ create_config() ──▶ AppConfig
                              └─▶ DataContext ─────▶ data loading/cleanup
                              └─▶ ServiceContext ──▶ DefaultService (repository, processor)
                                                   └─▶ DefaultProcessor(name).process(data)
                                                   └─▶ DefaultRepository().save(entity)
                                                   └─▶ DefaultHandler(service).handle(request)
```

- **ABCs define the contracts** (`interfaces.py`).
- **Classes bind static state at init** (e.g., `DefaultProcessor(name)`, `DefaultService(repository, processor)`).
- **Methods take only the changing parts** (`process(data)`, `handle(request)`, `execute(params)`).
- **Contexts make lifecycles obvious** (`with ServiceContext(...) as service:`, `with DataContext(...) as data:`).

---

## Quick Start

### Step-by-Step Project Creation

**1. Clone this repository**
```bash
git clone https://github.com/yamaceay/my-py-style.git
cd my-py-style
```

**2. Create your first project using the Makefile**
```bash
# Basic CLI tool project
make build project=my_awesome_tool

# API server with custom author
make build project=my_api type=api_server author="Your Name"

# Data pipeline with description
make build project=data_processor type=data_pipeline description="Advanced data processing pipeline"
```

**3. Navigate to your new project and test it**
```bash
cd my_awesome_tool
python main.py run --verbose
```

### Makefile Reference

The included Makefile provides convenient commands for project management:

#### `make build` - Project Scaffolding
Generate new projects with Go-ish Python patterns:

```bash
# Required parameter
make build project=<project_name>

# Optional parameters
make build project=my_app type=cli_tool              # Project type
make build project=my_app author="Jane Doe"           # Author name
make build project=my_app description="My awesome app" # Description
make build project=my_app target_dir=/path/to/dir     # Custom target directory

# Feature flags
make build project=simple_app no_logging=1           # Disable logging
make build project=simple_app no_cli=1               # Disable CLI
make build project=simple_app no_context_managers=1  # Disable contexts
```

**Project types available:**
- `cli_tool` (default) - Command-line applications
- `api_server` - REST API servers
- `data_pipeline` - Data processing workflows
- `ml_framework` - Machine learning applications

#### `make install` - Development Setup
Installs GitHub CLI and Copilot CLI extension for AI assistance:

```bash
make install
```

This command:
- Installs GitHub CLI if not present
- Authenticates with GitHub (interactive)
- Installs GitHub Copilot CLI extension
- Sets up AI-powered development tools

#### `make prompt` - AI Transformation Guide
Displays the Go-ish Python transformation prompt:

```bash
make prompt  # Shows prompt and copies to clipboard
```

Use this prompt with Claude, ChatGPT, or other AI tools to transform existing Python projects.

#### `make sync-docs` - Documentation Sync
Synchronizes documentation between prompt.md and README.md:

```bash
make sync-docs
```

Runs automatically via GitHub Actions when files change.

### Design Choices (why classes, not Protocols)

- `Protocol` is nice for duck typing, but I want **constructors** to lock in static deps (`tokenizer`, `labels`, `device`). That makes call sites clean and prevents "parameter soup" on every call.
- I use **ABCs** to document contracts when there are multiple implementations. The code remains pluggable without runtime type checks.

```python
# interfaces.py (excerpt)
class Collator(ABC):
    @abstractmethod
    def collate(self, batch: list[dict]) -> dict[str, Any]: ...

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, predictions: list[list[int]], targets: list[int]) -> "EvaluationResult": ...
```

### Run the Example

```bash
# Test the reference implementation
python3 main.py train --model-name demo-classifier --data-size 50
```

---

## The Rules

### Rule #1: Contracts as ABCs; implementations as small classes

Static dependencies (tokenizer, labels, device, paths) go into `__init__`. Methods only take **dynamic input**.

**When to use `@dataclass(frozen=True)` vs regular `__init__`:**
- **Use `@dataclass(frozen=True)`** when all dependencies are **static** (passed in and never change)
- **Use regular `__init__`** when you need **dynamic initialization** or dependencies that depend on other dependencies

```python
# ✅ Good: Pure static deps → use dataclass
@dataclass(frozen=True)
class DefaultProcessor(Processor):
    name: str               # Static dependency
    timeout: int           # Static configuration
    
    def process(self, data: Any) -> Any:  # Dynamic input
        return {"processed": True, "data": data, "processor": self.name}

# ✅ Good: Dynamic initialization → use regular __init__
class DefaultService(Service):
    def __init__(self, repository: Repository, processor: Processor, config: AppConfig):
        self.repository = repository
        self.processor = processor
        self.config = config
        # Dynamic initialization based on dependencies
        self.cache = self._setup_cache(config.cache_size)  # Depends on config
        self.validator = self._create_validator(processor.name)  # Depends on processor
    
    def execute(self, params: Any) -> Any:  # Dynamic input only
        processed = self.processor.process(params)
        return self.repository.save(processed)

# ❌ Bad: Dependencies passed to every method call
class BadProcessor:
    def process(self, data: Any, name: str, timeout: int) -> Any:
        return {"processed": True, "data": data, "processor": name}
```

### Rule #2: Plug-and-play polymorphism over branching

Choose the concrete class once (CLI/factory/context), then call the common method. No `if/elif/else` ladders.

```python
# ✅ Good: Polymorphic selection
if args.command == "process":
    context_mgr = ProcessingContext(config)
elif args.command == "serve":
    context_mgr = ServerContext(config)

with context_mgr as service:
    result = service.execute(data)  # Same interface, different behavior

# ❌ Bad: Runtime branching everywhere
def handle_request(data, command, config):
    if command == "process":
        # processing logic here
    elif command == "serve":
        # server logic here
```

### Rule #3: One config to rule them all

Build `RuntimeConfig` once at the top; pass it down. No config files. No hidden globals. No re-detecting device in submodules.

```python
# ✅ Good: Single config source
@dataclass(frozen=True)
class AppConfig:
    app_name: str
    output_dir: Path
    verbose: bool
    timeout: int

def main():
    cfg = parse_args_to_config(args)  # Build once
    with ServiceContext(cfg) as service:
        processor = DefaultProcessor(name=cfg.app_name)  # Pass down
```

### Rule #4: Context managers for lifecycle

Heavy resources must be used with `with ...:`; construction and cleanup are explicit.

```python
# ✅ Good: Explicit lifecycle management
class ServiceContext(AbstractContextManager):
    def __enter__(self) -> DefaultService:
        self.repository = DefaultRepository()
        self.processor = DefaultProcessor(name=self.config.app_name)
        self.service = DefaultService(self.repository, self.processor)
        return self.service
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.service.cleanup()  # Explicit cleanup
        return False
```

### Rule #5: Logging over print

Library code logs with context; CLI may print final user-facing summaries.

```python
# ✅ Good: Structured logging with context
logger.info("processing_complete", extra={
    "records_processed": count,
    "duration_ms": elapsed,
    "processor": processor_name
})

# CLI summary (user-facing)
print(f"Processing completed! Records processed: {count}")
```

---

## Architecture Guide

### Project Structure

```
project_name/
├── interfaces.py     # Abstract base classes (ABCs)
├── core.py          # Concrete implementations
├── contexts.py      # Context managers for resource lifecycle
├── cli.py           # Command-line interface and argument parsing
├── config.py        # Configuration management and canonicalization
├── main.py          # Application orchestration and entry point
├── README.md        # Documentation
└── tests/           # Unit tests
```

### Module Responsibilities

#### `interfaces.py` - Contracts Only

```python
from abc import ABC, abstractmethod

class Processor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

class Service(ABC):
    @abstractmethod
    def execute(self, params: Any) -> Any:
        pass
```

#### `core.py` - Business Logic

```python
from __future__ import annotations
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from config import RuntimeConfig

from interfaces import Processor, Service

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DefaultProcessor(Processor):
    config: "RuntimeConfig"
    name: str
    
    def process(self, data: Any) -> Any:
        logger.info("processing_data", extra={"data_size": len(data), "processor": self.name})
        return {"processed": True, "data": data}
```

#### `contexts.py` - Resource Management

```python
class ServiceContext(AbstractContextManager):
    def __init__(self, config: "RuntimeConfig"):
        self.config = config
        self.service: Service | None = None
    
    def __enter__(self) -> Service:
        logger.info("initializing_service_context")
        self.service = DefaultService(self.config)
        return self.service
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.info("releasing_service_resources")
        if self.service:
            self.service.cleanup()
        return False
```

### TYPE_CHECKING Pattern

Consistent across all modules for clean imports:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import RuntimeConfig    # Type-only imports
    from other_module import SomeType

from actual_module import real_function  # Runtime imports
```

---

## Project Scaffolder Tutorial

The included scaffolder generates complete, production-ready projects following Go-ish Python patterns automatically.

### Complete Walkthrough

#### 1. Create Your First CLI Tool

```bash
# Generate a CLI tool project
make build project=task_manager type=cli_tool author="Your Name"
cd task_manager

# Explore the generated structure
ls -la
# interfaces.py  core.py  contexts.py  cli.py  config.py  main.py
# pyproject.toml  requirements.txt  .gitignore  README.md  tests/

# Test the generated application
python main.py run --verbose
# Application completed successfully!
# Result: {'processed': True, 'data': [], 'processor': 'task_manager'}
```

#### 2. API Server Example

```bash
# Generate an API server project
make build project=user_api type=api_server description="User management API"
cd user_api

# The generated structure includes API-specific components
python main.py process --output-dir ./api_output
```

#### 3. Data Pipeline Example

```bash
# Generate a data processing pipeline
make build project=log_processor type=data_pipeline \
  author="Data Team" description="Log analysis pipeline"
cd log_processor

# Test data processing
python main.py run --verbose --output-dir ./processed_logs
```

#### 4. Minimal Project (No Extras)

```bash
# Generate minimal project without logging/contexts
make build project=simple_calc no_logging=1 no_context_managers=1
cd simple_calc

# Check the simpler structure
cat main.py  # Much simpler without contexts and logging
```

### Generated Architecture Deep Dive

Every scaffolded project follows the same proven structure:

#### Module Organization

```
project_name/
├── interfaces.py     # ABC contracts (Processor, Handler, Repository, Service)
├── core.py          # Concrete implementations with constructor injection
├── contexts.py      # Resource lifecycle management (if enabled)
├── cli.py           # Argument parsing and validation (if enabled)  
├── config.py        # Immutable configuration objects
├── main.py          # Application orchestration and entry point
├── pyproject.toml   # Modern Python packaging
├── requirements.txt # Dependency management
├── .gitignore       # Comprehensive ignore patterns
├── README.md        # Project-specific documentation
└── tests/           # Test directory structure
```

#### Generated Components

**Core Interfaces** (`interfaces.py`):
```python
class Processor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any: ...

class Service(ABC):
    @abstractmethod  
    def execute(self, params: Any) -> Any: ...

class Repository(ABC):
    @abstractmethod
    def save(self, entity: Any) -> bool: ...
    
    @abstractmethod
    def find(self, identifier: str) -> Any: ...
```

**Concrete Implementations** (`core.py`):
```python
@dataclass(frozen=True)
class DefaultProcessor(Processor):
    name: str  # Static dependency
    
    def process(self, data: Any) -> Any:  # Dynamic input only
        return {"processed": True, "data": data, "processor": self.name}

@dataclass(frozen=True)  
class DefaultService(Service):
    repository: Repository  # Injected dependency
    processor: Processor   # Injected dependency
    
    def execute(self, params: Any) -> Any:
        processed = self.processor.process(params)
        self.repository.save(processed)
        return processed
```

**Context Management** (`contexts.py`):
```python
class ServiceContext(AbstractContextManager[DefaultService]):
    def __enter__(self) -> DefaultService:
        self.repository = DefaultRepository()
        self.processor = DefaultProcessor(name=self.config.app_name)
        self.service = DefaultService(self.repository, self.processor)
        return self.service
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Automatic cleanup
        return False
```

### Customization Options

#### Project Types

Each type generates appropriate components:

- **`cli_tool`** (default): Command-line applications with argument parsing
- **`api_server`**: REST API structure with request handling
- **`data_pipeline`**: Data processing workflows with batch operations  
- **`ml_framework`**: Machine learning applications with model management

#### Feature Flags

Control generated features:

```bash
# Disable specific features
make build project=minimal_app no_logging=1      # No structured logging
make build project=simple_app no_cli=1           # No argument parsing
make build project=basic_app no_context_managers=1  # No resource management

# Combine flags
make build project=bare_app no_logging=1 no_cli=1 no_context_managers=1
```

#### Development Ready Features

Every project includes:

- **Modern packaging** with `pyproject.toml`
- **Type checking** with proper forward references
- **Testing structure** ready for pytest
- **Git integration** with comprehensive `.gitignore`
- **Documentation** with usage examples
- **CLI integration** (if enabled) with validation

### Advanced Usage Patterns

#### Custom Project Location

```bash
# Generate in specific directory
make build project=my_app target_dir=/path/to/workspace

# Organize multiple projects
mkdir -p ~/projects/go-ish-python
make build project=project1 target_dir=~/projects/go-ish-python
make build project=project2 target_dir=~/projects/go-ish-python
```

#### Development Workflow

```bash
# 1. Generate project
make build project=analytics_tool type=data_pipeline

# 2. Set up development environment
cd analytics_tool
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run tests
python -m pytest tests/

# 4. Start development
python main.py run --verbose
```

#### Integration with AI Tools

```bash
# Generate project for AI transformation
make build project=legacy_converter type=cli_tool

# Get transformation prompt
make prompt

# Use the prompt with Claude/ChatGPT to transform existing code
# into the generated project structure
```

---

## Repository Transformation

### AI Transformation Prompt

Use the comprehensive transformation prompt with Claude, GPT-4, or similar AI:

**Quick Access:**
- Copy from: [`prompt.md`](./prompt.md)
- Terminal: `make prompt` (copies to clipboard)
- GitHub Pages: [Transformation Guide](https://yamaceay.github.io/my-py-style/prompt)

The prompt includes detailed architecture requirements, implementation patterns, and a complete transformation checklist for converting existing Python projects to Go-ish patterns.

### Manual Transformation Steps

For manual transformation of existing codebases:

1. **Analyze Current Structure**
   ```bash
   # Identify main components and their dependencies
   find . -name "*.py" -exec grep -l "class\\|def\\|import" {} \;
   ```

2. **Extract Interfaces**
   - Find all major classes that could have multiple implementations
   - Create ABCs in `interfaces.py`
   - Define minimal method signatures

3. **Reorganize Implementation**
   - Move concrete classes to `core.py`
   - Apply constructor injection pattern
   - Use `@dataclass(frozen=True)` for immutability

4. **Add Context Managers**
   - Identify resources that need lifecycle management
   - Create context managers in `contexts.py`
   - Ensure proper cleanup

5. **Consolidate Configuration**
   - Create single `RuntimeConfig` dataclass
   - Build canonicalization functions
   - Remove global variables and config files

6. **Apply TYPE_CHECKING**
   - Add proper import organization
   - Use forward references for type hints
   - Separate runtime vs type-only imports

---

## Detailed Examples

### Example: Data Processing Application

The reference implementation demonstrates a complete data processing workflow:

```python
# CLI args → Context setup → Processing loop → Results
def run_application(args, config: AppConfig):
    logger.info("application_starting", extra={"command": args.command})
    
    with DataContext("default_source") as data_ctx:
        with ServiceContext(config) as service:
            # Components use constructor injection
            processor = DefaultProcessor(name=config.app_name)
            repository = DefaultRepository()
            handler = DefaultHandler(service)
            
            # Polymorphic execution - no branching
            if args.command == "run":
                result = service.execute(data_ctx.get_data())
            elif args.command == "process":
                result = processor.process({"action": "batch_process"})
            elif args.command == "status":
                result = {"status": "running", "app": config.app_name}
            
            logger.info("application_complete", extra={"success": True})
            return result
```

### Key Benefits Demonstrated

**Testability**: Each component can be mocked easily via interfaces
```python
# Easy to test with mock implementations
mock_repository = MockRepository()
mock_processor = MockProcessor()
service = DefaultService(mock_repository, mock_processor)
result = service.execute(test_data)
```

**Maintainability**: Clear boundaries and explicit dependencies
```python
# Easy to understand what each component needs
@dataclass(frozen=True) 
class DefaultProcessor(Processor):
    name: str           # Only static dependency needed
    
    def process(self, data: Any) -> Any:
        return {"processed": True, "data": data, "processor": self.name}
```

**Extensibility**: New implementations follow same patterns
```python
# Add new processor type without changing existing code
class AdvancedProcessor(Processor):
    def process(self, data: Any) -> Any:
        # New implementation with advanced logic
        return {"advanced_processed": True, "data": data}

# Plug into existing system seamlessly  
with ServiceContext(config) as service:
    # Service automatically uses new processor type
    result = service.execute(data)
```

---

## Checklist & Best Practices

### Architecture Checklist

- [ ] **Static deps in `__init__`; methods accept only dynamic inputs**
- [ ] **No `if/elif/else` dispatch in core logic—use classes and early selection**  
- [ ] **Only CLI parses input; no config files or global state**
- [ ] **One `RuntimeConfig` per run; no re-detecting device/paths**
- [ ] **Library code logs; CLI prints final summaries**
- [ ] **Context managers around heavy resources**
- [ ] **Deep helpers are positional with no defaults**

### Code Quality Guidelines

**Good Patterns:**
```python
# Constructor injection (see Rule #1 for dataclass vs __init__ guidance)
@dataclass(frozen=True)
class DefaultService(Service):
    repository: Repository
    processor: Processor
    timeout: int
    
    def execute(self, params: Any) -> Any:
        processed = self.processor.process(params)
        self.repository.save(processed)
        return processed

# Context-managed resources
with ServiceContext(config) as service:
    result = service.execute(data)

# Structured logging
logger.info("operation_complete", extra={
    "duration_ms": elapsed,
    "records_processed": count
})
```

**❌ Anti-Patterns:**
```python
# Runtime branching (avoid)
def handle_request(data, handler_type):
    if handler_type == "fast":
        return FastHandler().handle(data)
    elif handler_type == "secure": 
        return SecureHandler().handle(data)

# Global configuration (avoid)
GLOBAL_CONFIG = {...}

def some_function(data):
    timeout = GLOBAL_CONFIG["timeout"]  # Hidden dependency
```

### Performance Considerations

- **Frozen dataclasses** are faster and safer than mutable ones
- **Context managers** prevent resource leaks and improve memory usage
- **Constructor injection** enables better caching and optimization
- **Early polymorphic selection** avoids repeated branching overhead

### Testing Strategy

```python
# Easy mocking with interfaces
def test_service():
    mock_repo = MockRepository()
    mock_processor = MockProcessor() 
    service = DefaultService(mock_repo, mock_processor, timeout=30)
    
    result = service.execute(test_data)
    
    assert mock_processor.process.called_with(test_data)
    assert mock_repo.save.called_with(result)

# Context manager testing
def test_resource_lifecycle():
    with ServiceContext(test_config) as service:
        assert service is not None
        result = service.execute(test_data)
    
    # Resource automatically cleaned up
    assert service.is_closed()
```

---

## Why This Approach Works

### Maintainability
- **Explicit dependencies**: No hidden global state or surprise imports
- **Clear boundaries**: Each module has a single, obvious responsibility  
- **Predictable structure**: Same patterns across all projects

### Testability
- **Interface-based**: Easy mocking and dependency injection
- **Isolated components**: Units can be tested independently
- **Deterministic**: No hidden state means reproducible tests

### Scalability
- **Modular design**: Add new features without breaking existing code
- **Pluggable architecture**: Swap implementations without changing consumers
- **Resource management**: Proper cleanup prevents memory leaks

### Developer Experience
- **Self-documenting**: Code structure tells the story
- **IDE-friendly**: Strong typing and clear interfaces
- **Debugging**: Structured logging provides clear execution traces

---

## License

MIT License - Build great software, share knowledge freely.

---

*Created for maintainable, scalable Python development*