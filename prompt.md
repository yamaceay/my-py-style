Transform this Python codebase to follow Go-ish Python patterns. Apply these specific guidelines:

## ARCHITECTURE REQUIREMENTS

1. **Module Organization**:
   - `interfaces.py` - All abstract base classes (ABCs) only
   - `core.py` - All concrete implementations 
   - `contexts.py` - Context managers for resource lifecycle
   - `cli.py` - Command-line interface and argument parsing
   - `config.py` - Configuration management with immutable dataclasses
   - `main.py` - Application orchestration and entry point

2. **TYPE_CHECKING Pattern**:
   ```python
   from __future__ import annotations
   from typing import TYPE_CHECKING
   
   if TYPE_CHECKING:
       from config import RuntimeConfig  # Type-only imports
   
   from other_module import actual_function  # Runtime imports
   ```

3. **Interface Design**:
   - Every major component must implement an ABC
   - Static dependencies go in `__init__`
   - Methods take only dynamic inputs
   - Use `@dataclass(frozen=True)` for immutable implementations

4. **Context Manager Pattern**:
   - Heavy resources (models, DB connections, file handles) use context managers
   - Context managers return the resource, not self
   - Explicit cleanup in `__exit__`

5. **Configuration Management**:
   - Single `RuntimeConfig` dataclass for all configuration
   - Built once at application start, passed down
   - No global variables or config files
   - Use canonicalization functions for defaults

6. **Logging Requirements**:
   - Structured logging with `extra` fields
   - Library code logs, CLI prints user summaries
   - Consistent naming: `operation_complete`, `resource_loaded`

7. **Eliminate Branching**:
   - Replace `if/elif/else` chains with polymorphism
   - Choose concrete class once at startup
   - Use common interface for all execution paths

## TRANSFORMATION CHECKLIST

- [ ] Extract all interfaces to `interfaces.py`
- [ ] Move implementations to `core.py`
- [ ] Create context managers in `contexts.py`
- [ ] Consolidate CLI logic in `cli.py`
- [ ] Build single config system in `config.py`
- [ ] Orchestrate in `main.py`
- [ ] Apply TYPE_CHECKING pattern consistently
- [ ] Replace branching with polymorphism
- [ ] Add structured logging
- [ ] Ensure proper resource lifecycle management

Transform the codebase while preserving all functionality. Show the complete refactored structure with explanations of major changes.