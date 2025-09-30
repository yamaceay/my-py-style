You are an expert Python developer tasked with transforming a Python codebase to follow "Go-ish Python" patterns. Apply these specific guidelines:

## CORE PHILOSOPHY & BEST PRACTICES

**The six rules to live by:**

### Rule #1: Contracts as ABCs; implementations as small classes

First and foremost, the code should look like plug-and-play. **Static dependencies go in `__init__`; methods accept only dynamic inputs.**

**Bad example:**
```python
class MyProcessor:
    def process(self, data: Any, mode: str, timestamp: str) -> Any: ...

# Usage - passing static config every time
result = processor.process(data, mode="test", timestamp="2023-01-01T00:00:00Z")
```

**Good example:**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

class Processor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

@dataclass(frozen=True)
class MyProcessor(Processor):
    mode: str       # Static dependency in __init__
    timestamp: str  # Static dependency in __init__
    
    def process(self, data: Any) -> Any:  # Only dynamic input
        return {"processed": True, "data": data, "mode": self.mode}

# Usage - static deps set once
processor = MyProcessor(mode="test", timestamp="2023-01-01T00:00:00Z")
result = processor.process(data)  # Only dynamic input
```

**When to use `@dataclass(frozen=True)` vs regular `__init__`:**
- **Use `@dataclass(frozen=True)`** when all dependencies are **static** (passed in and never change)
- **Use regular `__init__`** when you need **dynamic initialization** or dependencies that depend on other dependencies

### Rule #2: Plug-and-play polymorphism over branching

**Choose the concrete class once (CLI/factory/context), then call the common method. No `if/elif/else` ladders.**

**Bad example:**
```python
class SuperService(Service):
    def execute(self, command: str, data: Any) -> Any:
        if command == "process":
            return call_processing_logic(data)
        elif command == "serve":
            return call_server_logic(data)
        raise ValueError(f"Unknown command: {command}")
```

**Good example:**
```python
class ProcessingService(Service): ...
class ServerService(Service): ...

# Early selection - choose once
if command == "process":
    service = ProcessingService()
elif command == "serve":
    service = ServerService()

# No more branching - use common interface
result = service.execute(data)
```

### Rule #3: One config to rule them all

**Build `AppConfig` once at the top; pass it down. No config files. No hidden globals. No re-detecting device in submodules.**

**Bad example:**
```python
GLOBAL_CONFIG = {"app_name": "my_app", "output_dir": "/tmp/output"}

class Processor:
    def __init__(self, app_name: str, output_dir: Path = Path("/tmp/default")):
        self.app_name = app_name
        self.output_dir = output_dir

processor = Processor(GLOBAL_CONFIG["app_name"])  # Hidden global
```

**Good example:**
```python
@dataclass(frozen=True)
class AppConfig:
    app_name: str
    output_dir: Path
    verbose: bool
    timeout: int

def main():
    cfg = parse_args_to_config(args)  # Build once
    processor = Processor(cfg.app_name, cfg.output_dir)  # Pass down
```

### Rule #4: Passing config downwards

**No part of the application should re-detect or re-parse configuration.** Pass config down explicitly, extract what you need.

**Bad example:**
```python
class Processor:
    def __init__(self, app_name: str, sub_output_dir: Path):
        output_dir = self._detect_output_dir()  # Re-detecting forbidden!
```

**Good example:**
```python
# Extract what you need, pass rest down
config_kwargs = app_config.__dict__
app_name = config_kwargs.pop("app_name")
output_dir = config_kwargs.pop("output_dir")
processor = Processor(app_name, output_dir / "processor", **config_kwargs)
```

### Rule #5: Context managers for lifecycle

**Heavy resources must be scoped with `with ...:`; construction and cleanup are explicit.**

**Bad example:**
```python
repository = DefaultRepository()
processor = DefaultProcessor(name=config.app_name)
service = DefaultService(repository, processor)
# ... manual cleanup, might be forgotten
service.cleanup()
```

**Good example:**
```python
class ServiceContext(AbstractContextManager):
    def __enter__(self) -> DefaultService:
        self.repository = DefaultRepository()
        self.processor = DefaultProcessor(name=self.config.app_name)
        self.service = DefaultService(self.repository, self.processor)
        return self.service
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.service.cleanup()  # Explicit cleanup guaranteed
        return False

# Usage
with ServiceContext(config) as service:
    result = service.execute(data)  # Automatic cleanup
```

### Rule #6: Logging over print

**Library code logs with context; CLI may print final user-facing summaries.**

**Bad example:**
```python
print(f"Processing completed! Records processed: {count}")
```

**Good example:**
```python
logger.info("processing_complete", extra={
    "records_processed": count,
    "duration_ms": elapsed,
    "processor": processor_name
})
```

---

## ARCHITECTURE REQUIREMENTS

1. **Module Organization** (from README.md Architecture Guide):
   - `interfaces.py` - Abstract base classes (ABCs) only - minimal, focused contracts
   - `core.py` - Concrete implementations with constructor injection
   - `contexts.py` - Context managers for resource lifecycle (optional)
   - `cli.py` - Command-line interface and argument parsing (optional)
   - `config.py` - Immutable configuration objects with canonicalization
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
   
   # Use smart_import for runtime imports when needed
   config_imports = smart_import('config', ['AppConfig'])
   AppConfig = config_imports['AppConfig']
   ```

5. **Interface Design** (Rule #1 compliance):
   ```python
   # interfaces.py - Contracts only
   class Processor(ABC):
       @abstractmethod
       def process(self, data: Any) -> Any:  # Only dynamic input
           pass

   # core.py - Implementations with static deps in __init__
   @dataclass(frozen=True)
   class DefaultProcessor(Processor):
       name: str                    # Static dependency
       processing_mode: str = "default"  # Static configuration
       
       def process(self, data: Any) -> Any:  # Only dynamic input
           return {"processed": True, "data": data, "processor": self.name}
   ```

6. **Context Manager Pattern** (Rule #5 compliance):
   ```python
   class ServiceContext(AbstractContextManager[DefaultService]):
       def __init__(self, config: AppConfig):
           self.config = config  # Static dependency
           
       def __enter__(self) -> DefaultService:
           # Explicit resource acquisition
           self.repository = DefaultRepository()
           self.processor = DefaultProcessor(name=self.config.app_name)
           self.service = DefaultService(self.repository, self.processor)
           return self.service
       
       def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
           # Explicit cleanup guaranteed
           return False
   ```

7. **Configuration Management** (Rules #3 & #4 compliance):
   ```python
   @dataclass(frozen=True)
   class AppConfig:
       app_name: str
       output_dir: Path
       verbose: bool
       timeout: int = 30
       
       @property 
       def processor_name(self) -> str:
           return f"{self.app_name}_processor"  # Computed property

   def create_config(...) -> AppConfig:
       # The ONLY place where defaults are applied
       return AppConfig(...)
   ```

8. **Polymorphism Over Branching** (Rule #2 compliance):
   ```python
   # main.py - Early selection, no runtime branching
   if args.command == "run":
       command_handler: CommandHandler = RunCommandHandler()
   elif args.command == "process":
       command_handler = ProcessCommandHandler()
   
   # No more branching - use common interface
   return command_handler.execute(config)
   ```

9. **Logging Requirements** (Rule #6, optional):
   ```python
   logger.info("processing_complete", extra={
       "records_processed": count,
       "duration_ms": elapsed,
       "processor": processor_name
   })
   ```

10. **Package Structure**:
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
    ```



## TRANSFORMATION CHECKLIST

Apply these transformations systematically to follow all 6 rules:

### Rule #1: Contracts as ABCs; implementations as small classes
- [ ] Extract all interfaces to `interfaces.py` with minimal ABC contracts
- [ ] Move static dependencies to `__init__` (tokenizer, labels, device, paths, config)
- [ ] Ensure methods only accept dynamic inputs (data to process, requests to handle)
- [ ] Use `@dataclass(frozen=True)` for static dependencies
- [ ] Use regular `__init__` for dynamic/computed initialization
- [ ] Add proper docstrings explaining static vs dynamic inputs

### Rule #2: Plug-and-play polymorphism over branching
- [ ] Replace `if/elif/else` chains with polymorphism
- [ ] Implement early selection pattern (choose class once at startup)
- [ ] Create common interfaces for branching logic
- [ ] Use Protocol/ABC for command handlers, processors, services
- [ ] Eliminate runtime branching in business logic

### Rule #3: One config to rule them all
- [ ] Create single `AppConfig` dataclass with all settings
- [ ] Build configuration once at application start (CLI/main)
- [ ] Remove global variables and config files
- [ ] Add canonicalization function `create_config()` for defaults
- [ ] Use computed properties for derived values

### Rule #4: Passing config downwards
- [ ] Pass config explicitly, never re-detect or re-parse
- [ ] Extract specific values, pass remaining as `**kwargs`
- [ ] Use `extract_config_kwargs()` helper for weakening config
- [ ] Remove environment variable lookups in deep functions
- [ ] Eliminate hidden globals and scattered configuration

### Rule #5: Context managers for lifecycle
- [ ] Create context managers in `contexts.py` with smart imports
- [ ] Wrap heavy resources (GPU models, DB connections, file handles)
- [ ] Use `AbstractContextManager` for type safety
- [ ] Implement explicit cleanup in `__exit__`
- [ ] Return managed resource from `__enter__`, not self

### Rule #6: Logging over print
- [ ] Replace `print()` statements with structured logging
- [ ] Use `logger.info("operation_name", extra={...})` pattern
- [ ] Configure logging in CLI module (if enabled)
- [ ] Library code logs with context, CLI prints summaries
- [ ] Consistent event naming: `processing_complete`, `resource_loaded`

### Architecture Implementation
- [ ] Create `import_utils.py` with smart import utilities
- [ ] Apply TYPE_CHECKING pattern consistently across modules
- [ ] Ensure dual execution support (script + module)
- [ ] Create professional packaging files (`pyproject.toml`, `__init__.py`)
- [ ] Add `py.typed` for type checking support
- [ ] Update documentation and examples

## IMPLEMENTATION EXAMPLE

```python
# main.py demonstrating all 6 rules
def main() -> int:
    try:
        # Rule #3: Build config once
        config = parse_args_to_config(args)
        
        # Rule #2: Early selection - choose class once
        if args.command == "run":
            handler = RunCommandHandler()
        elif args.command == "process":
            handler = ProcessCommandHandler()
        
        # Rule #5: Context managers for resources
        with ServiceContext(config) as service:
            # Rule #1: Only dynamic input to execute
            return handler.execute(config)
            
    except (ValueError, RuntimeError) as e:
        logger.error("application_error", extra={"error": str(e)})
        return 1
```

Transform the codebase while preserving all functionality. The result should be a professional Python package that works in all execution contexts with clean Go-ish patterns following all 6 rules consistently.