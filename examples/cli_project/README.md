# cli_project

CLI Tool

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
from cli_project import DefaultService, AppConfig
from cli_project import ServiceContext, DataContext

# Create configuration
config = AppConfig(
    app_name="cli_project",
    output_dir="./output",
    verbose=True,
    config_file=None
)

# Use context managers for resource management
with ServiceContext(config) as service:
    result = service.execute({"data": "example"})
    print(result)
```

### Command Line Interface
```bash\npython -m cli_project run --verbose\n# or\npython main.py run --verbose\n```

## Architecture

This project follows Go-ish Python patterns with clean modular organization:

### Modules

- `__init__.py` - Package exports and public API
- `interfaces.py` - Abstract base classes defining contracts
- `core.py` - Concrete implementations of business logic
- `contexts.py` - Context managers for resource lifecycle
- `cli.py` - Command-line interface
- `config.py` - Configuration management
- `main.py` - Application entry point

## Key Principles

- **Interface-based design** with minimal ABC contracts
- **Context managers** for automatic resource management
- **Explicit dependency injection** through constructor parameters
- **Structured logging** with consistent extra fields
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

### Context Managers
- `ServiceContext` - Manages service lifecycle and dependencies
- `DataContext` - Manages data loading and cleanup

## Development

Created with Python Project Scaffolder following Go-ish Python guidelines.

Author: Yamaç Eren Ay

---
*Generated with Go-ish Python Scaffolder by Yamaç Eren Ay (@yamaceay)*
