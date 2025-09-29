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
interfaces.py  # ABCs: Collator, Evaluator, EvaluationResult, Classifier, Trainer
config.py      # RuntimeConfig + canonicalize_config + seed utils + mock tokenizer/model
core.py        # Concrete classes: MyCollator, MyEvaluator, MyEvaluationResult, MyTrainer, MyClassifier
contexts.py    # Context managers for classifier + data
cli.py         # Single argument parser + logging setup + arg validation
main.py        # Wiring: parse → config → contexts → collate/classify/evaluate/train
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
CLI (args) ──▶ canonicalize_config() ──▶ RuntimeConfig
                                  └─▶ MyDataContext ─────▶ train/val batches
                                  └─▶ MyClassifierContext ─▶ MyClassifier (model, tokenizer)
                                                           └─▶ MyCollator(tokenizer, max_length).collate(batch)
                                                           └─▶ MyEvaluator(labels).evaluate(preds, targets)
                                                           └─▶ MyTrainer(config).train(..., epochs)
```

- **ABCs define the contracts** (`interfaces.py`).
- **Classes bind static state at init** (e.g., `MyCollator(tokenizer, max_length)`).
- **Methods take only the changing parts** (`collate(batch)`, `evaluate(preds, targets)`, `classify(input_ids)`).
- **Contexts make lifecycles obvious** (`with MyClassifierContext(...) as classifier:`).

---

## Quick Start

### Generate a New Project

```bash
# Create a new CLI tool
python3 scaffold.py my_tool --type cli_tool --author "Your Name"

# Create an ML framework
python3 scaffold.py ml_project --type ml_framework --description "ML Pipeline"

# Minimal project (no logging/contexts)
python3 scaffold.py simple_app --no-logging --no-context-managers
```

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
class MyCollator(Collator):
    tokenizer: MockTokenizer  # Static dependency
    max_length: int          # Static configuration
    
    def collate(self, batch: list[dict]) -> dict[str, Any]:  # Dynamic input
        return self.tokenizer.tokenize(batch, max_length=self.max_length)

# ✅ Good: Dynamic initialization → use regular __init__
class MyClassifier(Classifier):
    def __init__(self, model_name: str, labels: list[str], config: RuntimeConfig):
        self.model_name = model_name
        self.labels = labels
        self.config = config
        # Dynamic initialization based on dependencies
        self.tokenizer = self._load_tokenizer(model_name)  # Depends on model_name
        self.model = self._load_model(model_name, len(labels))  # Depends on both
    
    def classify(self, input_ids: list[int]) -> int:  # Dynamic input only
        return self.model.predict(input_ids)

# ❌ Bad: Dependencies passed to every method call
class BadCollator:
    def collate(self, batch: list[dict], tokenizer, max_length) -> dict:
        return tokenizer.tokenize(batch, max_length=max_length)
```

### Rule #2: Plug-and-play polymorphism over branching

Choose the concrete class once (CLI/factory/context), then call the common method. No `if/elif/else` ladders.

```python
# ✅ Good: Polymorphic selection
if args.task_type == "classifier":
    context_mgr = ClassifierContext(model_name, labels, cfg)
elif args.task_type == "sentiment":
    context_mgr = SentimentContext(model_name, labels, cfg)

with context_mgr as task:
    result = task.run(data)  # Same interface, different behavior

# ❌ Bad: Runtime branching everywhere
def process_data(data, task_type, model_name, labels):
    if task_type == "classifier":
        # classifier logic here
    elif task_type == "sentiment":
        # sentiment logic here
```

### Rule #3: One config to rule them all

Build `RuntimeConfig` once at the top; pass it down. No config files. No hidden globals. No re-detecting device in submodules.

```python
# ✅ Good: Single config source
@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    model_name: str
    max_length: int
    epochs: int

def main():
    cfg = parse_args_to_config(args)  # Build once
    with ClassifierContext(cfg.model_name, labels, cfg) as classifier:
        trainer = MyTrainer(config=cfg)  # Pass down
```

### Rule #4: Context managers for lifecycle

Heavy resources must be used with `with ...:`; construction and cleanup are explicit.

```python
# ✅ Good: Explicit lifecycle management
class ClassifierContext(AbstractContextManager):
    def __enter__(self) -> MyClassifier:
        self.classifier = MyClassifier(self.model_name, self.labels, self.cfg)
        return self.classifier
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.classifier.cleanup()  # Explicit cleanup
        return False
```

### Rule #5: Logging over print

Library code logs with context; CLI may print final user-facing summaries.

```python
# ✅ Good: Structured logging with context
logger.info("training_complete", extra={
    "final_accuracy": accuracy,
    "epochs": epochs,
    "model": model_name
})

# CLI summary (user-facing)
print(f"Training completed! Final accuracy: {accuracy:.3f}")
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

## Project Scaffolder

The included `scaffold.py` generates new projects following all Go-ish patterns automatically.

### Features

**Complete Project Generation**
- Modular architecture with proper separation of concerns
- TYPE_CHECKING compliance and forward references
- Interface-based design with ABC contracts
- Context manager support for resource lifecycle
- Structured logging with consistent patterns
- CLI integration with argument parsing

**Flexible Configuration**
- **Project Types**: ML framework, API server, CLI tool, data pipeline
- **Optional Features**: Logging, CLI interface, context managers
- **Customizable**: Author, description, target directory

### Usage

```bash
# Basic project
python3 scaffold.py my_project --type cli_tool

# ML framework with full features
python3 scaffold.py ml_framework --type ml_framework --author "ML Engineer" \
  --description "Advanced ML pipeline with Go-ish patterns"

# Minimal project
python3 scaffold.py simple_tool --no-logging --no-context-managers

# API server
python3 scaffold.py my_api --type api_server --target-dir /path/to/projects
```

### Generated Project Features

**Architecture**
- Clean module boundaries (`interfaces` → `core` → `contexts` → `main`)
- Proper dependency injection through constructors
- Context managers for resource lifecycle
- Configuration management with immutable objects

**Development Ready**
- `pyproject.toml` for modern Python packaging
- `requirements.txt` for dependencies
- `.gitignore` with comprehensive patterns
- Test directory structure
- Comprehensive README with usage examples

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

### Example: ML Training Pipeline

The reference implementation demonstrates a complete ML training workflow:

```python
# Generated data → Context setup → Training loop → Evaluation
def run_training_simulation(args, cfg: RuntimeConfig):
    train_data, val_data = generate_simulation_data(args.data_size)
    labels = ["positive", "negative", "neutral"]
    
    with MyDataContext(train_data, val_data) as data_ctx:
        with MyClassifierContext(args.model_name, labels, cfg) as classifier:
            # Components use constructor injection
            collator = MyCollator(classifier.tokenizer, max_length=cfg.max_length)
            evaluator = MyEvaluator(labels)  
            trainer = MyTrainer(config=cfg)
            
            # Polymorphic execution - no branching
            results = trainer.train(
                classifier=classifier,
                train_data=data_ctx.get_train_data(),
                val_data=data_ctx.get_val_data(), 
                collator=collator,
                evaluator=evaluator,
                epochs=cfg.epochs
            )
            
            return results
```

### Key Benefits Demonstrated

**Testability**: Each component can be mocked easily via interfaces
```python
# Easy to test with mock implementations
mock_classifier = MockClassifier(labels=["A", "B"])
trainer = MyTrainer(config=test_config)
result = trainer.train(mock_classifier, test_data, ...)
```

**Maintainability**: Clear boundaries and explicit dependencies
```python
# Easy to understand what each component needs
@dataclass(frozen=True) 
class MyTrainer(Trainer):
    config: RuntimeConfig  # Only dependency needed
```

**Extensibility**: New implementations follow same patterns
```python
# Add new classifier type without changing existing code
class AdvancedClassifier(Classifier):
    def classify(self, inputs): 
        # New implementation
        pass

# Plug into existing system seamlessly  
with AdvancedClassifierContext(...) as classifier:
    trainer.train(classifier, ...)
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
class MyService(Service):
    repository: Repository
    processor: Processor
    timeout: int
    
    def execute(self, data: Any) -> Any:
        processed = self.processor.process(data)
        return self.repository.save(processed)

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
def process(data, processor_type):
    if processor_type == "fast":
        return FastProcessor().process(data)
    elif processor_type == "accurate": 
        return AccurateProcessor().process(data)

# Global configuration (avoid)
GLOBAL_CONFIG = {...}

def some_function(data):
    device = GLOBAL_CONFIG["device"]  # Hidden dependency
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
    service = MyService(mock_repo, mock_processor)
    
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