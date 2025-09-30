# Go-ish Python: Patterns, Guidelines & Project Scaffolder

*Created by Yamaç Eren Ay ([@yamaceay](https://github.com/yamaceay)) for maintainable, scalable Python development*

> **A comprehensive guide to writing maintainable, scalable Python code using Go-inspired patterns**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Abstract*: This repository demonstrates class-centric, plug-and-play Python design: **class contracts + constructor injection**, **no runtime branching**, **CLI as single source of truth**, **context-scoped resources**, **structured logging**, and **DRY config**. Additionally, it provides a **Makefile-based project scaffolder** to generate new projects following these patterns automatically, and a **comprehensive AI transformation prompt** to convert existing codebases.

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Quick Start](#quick-start)
    - [Step-by-Step Project Creation](#step-by-step-project-creation)
    - [Makefile Reference](#makefile-reference)
- [The Rules](#the-rules)
    - [Rule #1: Contracts as ABCs; implementations as small classes](#rule-1-contracts-as-abcs-implementations-as-small-classes)
    - [Rule #2: Plug-and-play polymorphism over branching](#rule-2-plug-and-play-polymorphism-over-branching)
    - [Rule #3: One config to rule them all](#rule-3-one-config-to-rule-them-all)
    - [Rule #4: Passing config downwards](#rule-4-passing-config-downwards)
    - [Rule #5: Context managers for lifecycle](#rule-5-context-managers-for-lifecycle)
    - [Rule #6 [BONUS]: Logging over print](#rule-6-bonus-logging-over-print)
- [Architecture Guide](#architecture-guide)
    - [Project Structure](#project-structure)
    - [Module Responsibilities](#module-responsibilities)
    - [TYPE_CHECKING Pattern](#type_checking-pattern)
    - [Project Types](#project-types)
    - [Development Ready Features](#development-ready-features)
- [Repository Transformation](#repository-transformation)
    - [AI Transformation Prompt](#ai-transformation-prompt)
- [License](#license)

---


## Core Philosophy

> Would it be too much to say that the code should speak for itself? What if we just throw all comments away and let the design shine?

Python is a very generous language, which allows for many (many) different styles of programming. For example, some confusion exists around whether Python is object-oriented or functional. The truth is, it can be both (and more). You can write functional code, procedural code, stick to OOP, or mix and stay in between.

This liberty is a double-edged sword. While it allows for great flexibility, it can also lead to **inconsistent codebases**, **hard-to-maintain projects**, and **unpredictable behavior**. This is especially true in projects that grow over time, involve multiple contributors, or sub-projects. *Trust me, I've been working with multiple projects very recently, each seemingly has its own style and conventions, and I got headaches constantly while switching contexts.*

To mitigate these issues, this repository advocates for a **Go-inspired approach** to Python development. Go is known for its simplicity, strictness and clarity. I believe, by adopting similar principles in Python, we can achieve a codebase that is:
- **Maintainable**: Clear module boundaries and explicit dependencies
- **Testable**: Interface-based design enables easy mocking
- **Scalable**: Modular structure supports project growth
- **Predictable**: No hidden state or surprising behavior
- **Observable**: Built-in structured logging and monitoring

---

## Quick Start

### Step-by-Step Project Creation

**1. Clone this repository**
```bash
git clone https://github.com/yamaceay/py-kit.git
cd py-kit
```

**2. Create your first project using the Makefile**
```bash
# CLI Tool example
make build my_project type=cli_tool author="Yamaç Eren Ay" description="Advanced data processing pipeline" target_dir=examples
```

**3. Navigate to your new project and test it**
```bash
cd my_project

# Explore the generated structure
ls -la
# interfaces.py  core.py  contexts.py  cli.py  config.py  main.py
# pyproject.toml  requirements.txt  .gitignore  README.md  tests/

# Test the generated application
python main.py run --verbose
# Application completed successfully!
# Result: {'processed': True, 'data': [], 'processor': 'task_manager'}
```

**4. Customize and extend your project**
- Implement your own classes, interfaces etc. in the appropriate files. More on [Architecture Guide](#architecture-guide).

### Makefile Reference

The included Makefile provides convenient commands for project management:

#### `make build` - Project Scaffolding
Generate new projects with Go-ish Python patterns:

```bash
# Required parameter
make build <project_name>

# Optional parameters
make build my_app type=cli_tool              # Project type
make build my_app author="Jane Doe"           # Author name
make build my_app description="My awesome app" # Description
make build my_app target_dir=/path/to/dir     # Custom target directory

# Feature flags
make build simple_app no_logging=1           # Disable logging
make build simple_app no_cli=1               # Disable CLI
make build simple_app no_context_managers=1  # Disable contexts
```

**Project types available:**
- `cli_tool` (default) - Command-line applications

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

## The Rules

Let's take the five + one core rules one by one, with examples. 

### Rule #1: Contracts as ABCs; implementations as small classes

First and foremost, the code should look like plug-and-play. The code below is **not** a good example:

```py
...
from typing import Any

class MyProcessor:
    def process(self, data: Any, *args) -> Any: ...

def wrapper(data: Any, processor: MyProcessor, *args) -> Any:
    return processor.process(data, *args)

...

# Usage
processor = MyProcessor()
result = wrapper(data, processor, mode="test", timestamp="2023-01-01T00:00:00Z")
```

The above code is bad because: `wrapper` takes a concrete class as a parameter. This makes it hard to use different implementations without changing the function signature. For example, if we want to use a mock processor for testing, we would have to change the `wrapper` function to accept a different type. This leads to unnecessary **runtime branching** and **code duplication**.

The correct way is to use **interfaces (ABCs)** and **constructor injection**. This way, we can define a contract that multiple implementations can adhere to, and we can inject the desired implementation at runtime without changing the function signature.

```py
...
from abc import ABC, abstractmethod
from dataclasses import dataclass

class Processor(ABC):
    @abstractmethod
    def process(self, data: Any, *args) -> Any:
        pass

@dataclass(frozen=True)
class MyProcessor(Processor):
    def process(self, data: Any, *args) -> Any: ...

def wrapper(data: Any, processor: Processor, *args) -> Any:
    return processor.process(data, *args)

...

# Usage
processor = MyProcessor()
result = wrapper(data, processor)
```

The code below is a good example which allows dependency injection and polymorphism. Now, consider two more small adjustments to make it even better. Firstly, not every dependency should be passed at every method call, some attributes are better suited as class attributes.

```py
...
@dataclass(frozen=True)
class MyProcessor(Processor):
    mode: str       # Static dependency
    timestamp: str  # Static dependency
    def process(self, data: Any) -> Any: ...

...

# Usage
processor = MyProcessor(mode="test", timestamp="2023-01-01T00:00:00Z")
result = wrapper(data, processor)  # No need to pass mode/timestamp every time
```

Furthermore, if the class attributes have to be set dynamically based on other dependencies, we can use a regular `__init__` method instead of `@dataclass(frozen=True)`.

```py
...
class MyProcessor(Processor):
    def __init__(self, config: AppConfig):
        self.mode = config.mode          # Dynamic initialization
        self.timestamp = config.timestamp  # Dynamic initialization
        self._setup()                    # Additional setup if needed

    def _setup(self) -> None:
        # Additional setup logic if needed
        pass

    def process(self, data: Any) -> Any: ...
...
# Usage
config = AppConfig(mode="test", timestamp="2023-01-01T00:00:00Z")
processor = MyProcessor(config)
result = wrapper(data, processor)  # No need to pass mode/timestamp every time
```

**When to use `@dataclass(frozen=True)` vs regular `__init__`:**
- **Use `@dataclass(frozen=True)`** when all dependencies are **static** (passed in and never change)
- **Use regular `__init__`** when you need **dynamic initialization** or dependencies that depend on other dependencies

### Rule #2: Plug-and-play polymorphism over branching

Following a similar example, consider a scenario where we have multiple implementations of a service, and we want to select one based on user input (e.g., CLI arguments). The wrong way would be to use `if/elif/else` statements everywhere in the code to select the appropriate implementation. For example, the code below is bad because the class is stressed with runtime branching:

```python
class SuperService(Service):
    def execute(self, command: str, data: Any) -> Any: ...
        if command == "process":
            return call_processing_logic(data)
        elif command == "serve":
            return call_server_logic(data)
        raise ValueError(f"Unknown command: {command}")

# Usage
service = SuperService()
result = service.execute(args.command, data)
```

Or look at a slightly better but still bad example, where the branching is moved to a factory function, but the service getter still has to deal with branching:

```python
class ProcessingService(Service):
    def execute(self, data: Any) -> Any: ...
        return call_processing_logic(data)

class ServerService(Service):
    def execute(self, data: Any) -> Any: ...
        return call_server_logic(data)

def get_service(command: str) -> Service:
    if command == "process":
        return ProcessingService()
    elif command == "serve":
        return ServerService()
    raise ValueError(f"Unknown command: {command}") 

# Usage
service = get_service(args.command)
result = service.execute(data)
```

A much more better approach is to use **polymorphism** and **early selection** at the very beginning.

```python
class ProcessingService(Service): ...
class ServerService(Service): ...

if command == "process":
    service = ProcessingService()
elif command == "serve":
    service = ServerService()

# Usage
result = service.execute(data)
```

If this selection logic is repeated in multiple places, it will yield flattened code with good reading flow, allowing you to focus on the business logic instead of the branching.

**Choose the concrete class once (CLI/factory/context), then call the common method. No `if/elif/else` ladders.**

### Rule #3: One config to rule them all

Nowadays, there are many ways to manage configuration in Python: environment variables, config files (YAML, JSON, INI), command-line arguments, and even hardcoded values. This often leads to **scattered configuration sources**, **hidden globals**, and **inconsistent state**, and without any doubt this is the most confusing topic of this guide. Not only you have to look in multiple places to understand the configuration, but also it becomes hard to ensure that all parts of the application are using the same configuration. For example, the code below is bad because it uses three different sources of configuration, but still cannot change the output directory dynamically:

```python
GLOBAL_CONFIG = {
    "app_name": "my_app",
    "output_dir": "/tmp/output",
    "verbose": True,
    "timeout": 30,
}

cfg = GLOBAL_CONFIG
cfg = overwrite_with_env_vars(cfg)
cfg = overwrite_with_cli_args(cfg, args)

...
class Processor:
    def __init__(self, app_name: str, output_dir: Path = Path("/tmp/default-output")) -> None:
        self.app_name = app_name
        self.output_dir = output_dir

processor = Processor(cfg.app_name)
```

The correct way is to have a **single source of truth** for configuration, typically built from command-line arguments (CLI) at the start of the application. This configuration object should be **immutable** and **passed down** to all parts of the application that need it. It is similar to React.js-style props drilling, but for Python.

```python
@dataclass(frozen=True)
class AppConfig:
    app_name: str
    output_dir: Path
    verbose: bool
    timeout: int

class Processor:
    def __init__(self, app_name: str, output_dir: Path) -> None:
        self.app_name = app_name
        self.output_dir = output_dir

def main():
    cfg = parse_args_to_config(args)  # Build once
    processor = Processor(cfg.app_name, cfg.output_dir)  # Pass down
```

As you may notice, the only source of configuration is the `AppConfig` dataclass, which is built once at the start of the application. It is better to remove defaults from deep helpers and make them explicit in the config. However, if you prefer to keep the high-level logic cleaner and are sure about the defaults, you can define the optional parameters in a standard way:

```python
...
from typing import Optional

class Processor:
    def __init__(self, app_name: str, output_dir: Optional[Path] = None) -> None:
        self.app_name = app_name
        self.output_dir = output_dir if output_dir is not None else Path("/tmp/default-output")

def main():
    cfg = parse_args_to_config(args)
    processor_kwargs = {}
    if cfg.output_dir is not None:
        processor_kwargs["output_dir"] = cfg.output_dir
    processor = Processor(cfg.app_name, **processor_kwargs)  # Pass down
```

Or, if you still want to deal with them in the top-level logic, you can use a canonicalization function:

```python
def canonicalize_config(cfg: AppConfig) -> AppConfig:
    output_dir = cfg.output_dir or Path("/tmp/default-output")
    return AppConfig(
        app_name=cfg.app_name,
        output_dir=output_dir,
        verbose=cfg.verbose,
        timeout=cfg.timeout,
    )

...
def main():
    cfg = parse_args_to_config(args)
    cfg = canonicalize_config(cfg)  # Apply defaults
    processor = Processor(cfg.app_name, cfg.output_dir)
```

This is a complicated topic, and depending on your project, you may choose different strategies. *It's funny that I still have no idea which one is the absolute best.* 

### Rule #4: Passing config downwards

Continuing the previous rule, **no part of the application should re-detect or re-parse configuration**. For example, if you have a CLI argument for `--output-dir`, no part of the application should try to read an environment variable or a config file to determine the output directory again. The configuration should be built once at the start and passed down. This avoids inconsistencies and makes the code easier to reason about. For example, this is bad:

```python
def something_with_output_dir(output_dir: Path) -> None: ...

class AppConfig:
    app_name: str
    output_dir: Path
    sub_output_dir: Path
    verbose: bool
    timeout: int

class Processor:
    def __init__(self, app_name: str, sub_output_dir: Path) -> None:
        self.app_name = app_name
        self.sub_output_dir = sub_output_dir
        ...
        output_dir = self._detect_output_dir()  # Re-detecting from sub-dir
        something_with_output_dir(output_dir)

    def _detect_output_dir(self) -> Path:
        return self.sub_output_dir.parent

processor = Processor(cfg.app_name, cfg.output_dir / "processor")  # Implicit sub-dir
```

Following the example above, a slightly more explicit way is to pass `output_dir` from the config:

```python
...
class Processor:
    def __init__(self, app_name: str, output_dir: Path, sub_output_dir: Path) -> None:
        self.app_name = app_name
        self.output_dir = output_dir
        self.sub_output_dir = sub_output_dir
        ...
        output_dir = self.output_dir
        something_with_output_dir(output_dir)

processor = Processor(cfg.app_name, cfg.output_dir, cfg.output_dir / "processor")
```

However, the problem is that now the caller has to know about the sub-directory structure, which is leaking implementation details. A better way is to let the `Processor` use the config directly:

```python
def something_with_output_dir(config: AppConfig) -> None:
    output_dir = config.output_dir
    ...

class Processor:
    def __init__(self, app_name: str, sub_output_dir: Path, config: AppConfig) -> None:
        self.app_name = app_name
        self.sub_output_dir = sub_output_dir
        self.config = config
        ...
        something_with_output_dir(self.config) # Use from config directly

processor = Processor(cfg.app_name, cfg.output_dir / "processor", cfg)
```

However, this approach brings back the problem of hidden globals, and hides the arguments of the bottom functions by passing the whole config object. We should be able to pop out the fully used parameters to save the bottom-level classes and functions from dealing with the whole config. The final solution is to **extract the fully used parameters** and pass them down:

```python
def something_with_output_dir(output_dir: Path) -> None: ...

class AppConfig:
    app_name: str
    output_dir: Path
    verbose: bool
    timeout: int

class Processor:
    def __init__(self, app_name: str, sub_output_dir: Path, **config_kwargs) -> None:
        self.app_name = app_name
        self.sub_output_dir = sub_output_dir
        self.config = config
        ...
        something_with_output_dir(**config_kwargs) # Use from config directly

config_kwargs = app_config.__dict__
app_name = config_kwargs.pop("app_name")
output_dir = config_kwargs.pop("output_dir")
processor = Processor(app_name, output_dir / "processor", **config_kwargs)
```

If kwargs should be used across multiple calls (arbitrarily nested), you can also use a copy of the config object as follows:

```python

def anything_with_suboutput_dir(sub_output_dir: Path, **config_kwargs) -> None: ...

def something_with_output_dir(output_dir: Path, **config_kwargs) -> None: ...

def another_thing_with_timeout(timeout: int, **config_kwargs) -> None: ...

def nested_thing_with_verbose(verbose: bool, **config_kwargs) -> None:
    timeout = config_kwargs.pop("timeout")
    another_thing_with_timeout(timeout, **config_kwargs)

class Processor:
    def __init__(self, app_name: str, **config_kwargs) -> None:
        self.app_name = app_name
        self.sub_output_dir = config_kwargs.pop("sub_output_dir")
        ...
        sub_output_dir_for_anything = self.sub_output_dir
        anything_with_suboutput_dir(sub_output_dir_for_anything, **config_kwargs)

        something_config_kwargs = config_kwargs.copy()
        output_dir_for_something = something_config_kwargs.pop("output_dir")
        something_with_output_dir(output_dir_for_something, **something_config_kwargs)

        nested_config_kwargs = config_kwargs.copy()
        verbose = nested_config_kwargs.pop("verbose")
        nested_thing_with_verbose(verbose, **nested_config_kwargs)
        ...
```

In short, configuration keyword arguments are weakened as they go deeper into the call stack, and the fully used parameters are popped out and passed explicitly. This means, if any parameter has to be used at least one more time in a deeper level, it should not be popped out from the config kwargs. This way, the bottom-level functions and classes only deal with the parameters they actually use, without hidden globals or unnecessary coupling to the whole config object.

**In line with the previous rule, the key takeaway is to have one immutable config object that is built once and passed down, avoiding hidden globals and scattered configuration sources.**

### Rule #5: Context managers for lifecycle

Some resources are garbage collected automatically (e.g., lists, dicts, strings), while others require explicit cleanup (e.g., file handles, network connections, database connections, ML models persisted into disk). In Python, a recommended way to manage such resources is to use **context managers**. This ensures that resources are properly acquired and released, even in the presence of exceptions.

For example, the code below is bad because it doesn't use a context manager, and the resource may not be released properly:

```python
repository = DefaultRepository()
processor = DefaultProcessor(name=self.config.app_name)
service = DefaultService(self.repository, self.processor)
...
service.cleanup()  # Manual cleanup, may be forgotten
```

Instead, we can define a context manager for the service, which lets the service die automatically when the context is exited:

```python
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

**Heavy resources must be scoped with `with ...:`; so that the construction and cleanup are handled explicitly.**

### Rule #6 [BONUS]: Logging over print

Logging is a crucial aspect of any application, as it provides insights into the application's behavior and helps with debugging and monitoring. In Python, the built-in `logging` module is a powerful tool for structured logging.

This is ugly to read within standard output:
```python
print(f"Processing completed! Records processed: {count}")
```

This is much more readable and structured:
```python
logger.info("processing_complete", extra={
    "records_processed": count,
    "duration_ms": elapsed,
    "processor": processor_name
})
```

**Library code logs with context; CLI may print final user-facing summaries.**

---

## Architecture Guide

If you have a chance to create your own repository from scratch, you can easily follow the guidelines and patterns above. Below is a reference architecture that you can follow, or you can auto-generate using the make based project scaffolder (detailed in the [Quick Start](#quick-start)).

### Project Structure

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

### Project Types

Each type generates appropriate components:

- **`cli_tool`** (default): Command-line applications with argument parsing

You can extend the Makefile to add more types as needed. Then, you need to extend the scaffolder logic in `Makefile` to generate the appropriate files and structure if the type is different.

### Development Ready Features

Every project includes:

- **Modern packaging** with `pyproject.toml`
- **Type checking** with proper forward references
- **Testing structure** ready for pytest
- **Git integration** with comprehensive `.gitignore`
- **Documentation** with usage examples
- **CLI integration** (if enabled) with validation

---

## Repository Transformation

If it's already too late and you have an existing codebase, don't worry! You can still transform it to follow these patterns. Below are two approaches: using an AI transformation prompt.

### AI Transformation Prompt

Use the comprehensive transformation prompt with Claude, GPT-4, or similar AI:

**Quick Access:**
- Copy from: [`prompt.md`](./prompt.md)
- Terminal: `make prompt` (copies to clipboard)
- GitHub Pages: [Transformation Guide](https://yamaceay.github.io/py-kit/prompt)

The prompt includes detailed architecture requirements, implementation patterns, and a complete transformation checklist for converting existing Python projects to Go-ish patterns.

---

## License

MIT License - Build great software, share knowledge freely.