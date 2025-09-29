# cli_project

Test CLI program for Python

## Architecture

This project follows Go-ish Python patterns with clean modular organization:

### Modules

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

## Usage

```bash\npython main.py run --verbose\n```

## Installation

```bash
pip install -r requirements.txt
```

## Development

Created with Python Project Scaffolder following Go-ish Python guidelines.

Author: Yamaç Eren Ay

---
*Generated with Go-ish Python Scaffolder by Yamaç Eren Ay (@yamaceay)*
