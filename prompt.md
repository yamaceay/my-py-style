You are an expert Python developer tasked with transforming a Python codebase to follow "Go-ish Python" patterns. Apply these specific guidelines:

## CORE PHILOSOPHY & BEST PRACTICES

**The rules to live by:**
- **Contracts as ABCs; implementations as small classes.** Static deps (tokenizer, labels, device, paths) go into `__init__`; methods only take **dynamic input**.
- **Plug‑and‑play polymorphism > branching.** Choose the concrete class once (CLI/factory/context) and then call the common method. No `if/elif/else` ladders.
- **One config to rule them all.** Build `AppConfig` once at the top; pass it down. No config files. No hidden globals. No re‑detecting device in submodules.
- **Context managers for lifecycle.** Heavy resources must be used with `with ...:`; construction and cleanup are explicit.
- **Top‑level defaults only.** Deep helpers are positional and explicit; if you expose a default, it's `None` and you canonicalize immediately.
- **Logging over `print`.** Library code logs with context; CLI may print final user‑facing summaries.

**When to use patterns:**
- **Use `@dataclass(frozen=True)`** when all dependencies are **static** (passed in and never change)
- **Use regular `__init__`** when you need **dynamic initialization** or dependencies that depend on other dependencies

## ARCHITECTURE REQUIREMENTS

1. **Module Organization**:
   - `interfaces.py` - All abstract base classes (ABCs) only
   - `core.py` - All concrete implementations 
   - `contexts.py` - Context managers for resource lifecycle
   - `cli.py` - Command-line interface and argument parsing (optional)
   - `config.py` - Configuration management with immutable dataclasses
   - `main.py` - Application orchestration and entry point
   - `import_utils.py` - Smart import utilities for dual execution
   - `__init__.py` - Package exports and public API
   - `py.typed` - Type checking marker file

2. **Smart Import Pattern** (import_utils.py):
   ```python
   # Automatic relative/absolute import handling
   from import_utils import smart_import
   
   # Use smart import for all module dependencies
   context_imports = smart_import('contexts', ['ServiceContext'])
   cli_imports = smart_import('cli', ['build_parser', 'configure_logging'])
   
   # Extract imports for clean usage
   ServiceContext = context_imports['ServiceContext']
   build_parser = cli_imports['build_parser']
   ```

3. **Dual Execution Support**:
   - **Direct execution**: `python3 main.py --help` ✅
   - **Module execution**: `python3 -m package.main --help` ✅  
   - **Package imports**: `from package import Service` ✅
   - Smart import utility handles both contexts automatically

4. **TYPE_CHECKING Pattern**:
   ```python
   from __future__ import annotations
   from typing import TYPE_CHECKING
   
   if TYPE_CHECKING:
       from .config import AppConfig  # Type-only imports
   
   # Use smart_import for runtime imports
   config_imports = smart_import('config', ['AppConfig'])
   AppConfig = config_imports['AppConfig']
   ```

5. **Interface Design**:
   - Every major component must implement an ABC
   - Static dependencies go in `__init__`
   - Methods take only dynamic inputs
   - Use `@dataclass(frozen=True)` for immutable implementations

6. **Context Manager Pattern**:
   - Heavy resources (models, DB connections, file handles) use context managers
   - Context managers return the resource, not self
   - Explicit cleanup in `__exit__`
   - Smart imports handle contexts module loading

7. **Configuration Management**:
   - Single `AppConfig` dataclass for all configuration
   - Built once at application start, passed down
   - No global variables or config files
   - Use canonicalization functions for defaults

8. **Logging Requirements** (optional):
   - Structured logging with `extra` fields
   - Library code logs, CLI prints user summaries
   - Consistent naming: `operation_complete`, `resource_loaded`
   - Configure logging in CLI module

9. **CLI Integration** (optional):
   - Argument parsing in dedicated `cli.py` module
   - Validation functions for arguments
   - Configuration builder from parsed args
   - Support for verbose logging, config files, output directories

10. **Package Structure**:
    - Professional Python packaging with `pyproject.toml`
    - Proper `__init__.py` with public API exports
    - Type checking support with `py.typed`
    - Complete project metadata and documentation

11. **Eliminate Branching**:
    - Replace `if/elif/else` chains with polymorphism
    - Choose concrete class once at startup
    - Use common interface for all execution paths

## GENERATED PROJECT STRUCTURE

```
project_name/
├── __init__.py           # Package exports and public API
├── interfaces.py         # Abstract base classes
├── core.py              # Concrete implementations
├── contexts.py          # Context managers (optional)
├── cli.py               # Command-line interface (optional)
├── config.py            # Configuration management
├── main.py              # Application entry point
├── import_utils.py      # Smart import utilities
├── py.typed             # Type checking marker
├── pyproject.toml       # Python packaging configuration
├── requirements.txt     # Dependencies
├── README.md           # Project documentation
├── .gitignore          # Git ignore patterns
└── tests/              # Test directory
    └── __init__.py
```

## SMART IMPORT UTILITIES

Create `import_utils.py` with automatic relative/absolute import handling:

```python
def smart_import(module_name: str, items: list[str]) -> Dict[str, Any]:
    """
    Import items with automatic fallback between relative and absolute imports.
    
    - Detects execution context (module vs script)
    - Uses relative imports for module execution
    - Falls back to absolute imports for direct execution
    - Handles sys.path modifications transparently
    """
    # Implementation handles both contexts automatically
```

## MAIN APPLICATION PATTERNS

```python
# main.py - Clean import pattern
try:
    from .import_utils import smart_import
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from import_utils import smart_import

# Use smart import for all dependencies
context_imports = smart_import('contexts', ['ServiceContext'])
cli_imports = smart_import('cli', ['build_parser', 'configure_logging'])

# Extract for clean usage
ServiceContext = context_imports['ServiceContext']
build_parser = cli_imports['build_parser']

def main() -> int:
    """Application entry point supporting both execution methods."""
    # CLI parsing, validation, configuration
    # Context manager usage
    # Error handling with specific exceptions
```

## PACKAGE FEATURES

**CLI Options**:
- `--type cli_tool` - Project type
- `--author "Name"` - Author attribution
- `--description "Description"` - Project description
- `--no-logging` - Disable logging setup
- `--no-cli` - Disable CLI interface
- `--no-context-managers` - Disable context managers

**Generated Features**:
- ✅ Professional Python packaging
- ✅ Dual execution support (script + module)
- ✅ Type checking with py.typed
- ✅ Comprehensive documentation
- ✅ CLI with argument parsing
- ✅ Context manager resource lifecycle
- ✅ Structured logging patterns
- ✅ Git integration

## TRANSFORMATION CHECKLIST

- [ ] Create `import_utils.py` with smart import utilities
- [ ] Extract all interfaces to `interfaces.py`
- [ ] Move implementations to `core.py` with smart imports
- [ ] Create context managers in `contexts.py` with smart imports
- [ ] Consolidate CLI logic in `cli.py` with smart imports
- [ ] Build single config system in `config.py`
- [ ] Orchestrate in `main.py` with smart import pattern
- [ ] Create `__init__.py` with public API exports
- [ ] Add `py.typed` for type checking support
- [ ] Apply TYPE_CHECKING pattern consistently
- [ ] Replace branching with polymorphism
- [ ] Add structured logging (optional)
- [ ] Ensure dual execution support (script + module)
- [ ] Create professional packaging files

Transform the codebase while preserving all functionality. The result should be a professional Python package that works in all execution contexts with clean Go-ish patterns.