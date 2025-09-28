# Class‑Centric, Go‑ish Plug‑and‑Play Python
> A tiny reference skeleton showing how I design Python programs: **class contracts + constructor injection**, **no runtime branching**, **CLI as the single source of truth**, **context‑scoped resources**, **real logging**, and **DRY config**.

This repo demonstrates the standard on a toy ML workflow (collate → classify → evaluate → train). The code should read like prose; the types and method names tell the story. Comments are optional because design is explicit.

---

## TL;DR (the rules I live by)

- **Contracts as ABCs; implementations as small classes.** Static deps (tokenizer, labels, device, paths) go into `__init__`; methods only take **dynamic input**.
- **Plug‑and‑play polymorphism > branching.** Choose the concrete class once (CLI/factory/context) and then call the common method. No `if/elif/else` ladders.
- **One config to rule them all.** Build `RuntimeConfig` once at the top; pass it down. No config files. No hidden globals. No re‑detecting device in submodules.
- **Context managers for lifecycle.** Heavy resources must be used with `with ...:`; construction and cleanup are explicit.
- **Top‑level defaults only.** Deep helpers are positional and explicit; if you expose a default, it’s `None` and you canonicalize immediately.
- **Logging over `print`.** Library code logs with context; CLI may print final user‑facing summaries.

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

## Architecture at a Glance

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

## Design Choices (why classes, not Protocols)

- `Protocol` is nice for duck typing, but I want **constructors** to lock in static deps (`tokenizer`, `labels`, `device`). That makes call sites clean and prevents “parameter soup” on every call.
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

---

## Walkthrough (end‑to‑end)

### 1) CLI → Config (single source of truth)
```python
# cli.py
parser = build_parser()
args = parser.parse_args()
configure_logging(args.log_level)
cfg = parse_args_to_config(args)  # canonicalize once
```

```python
# config.py
@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    out_dir: str
    log_level: str
    seed: int | None = None
    max_length: int = 128
    epochs: int = 3
```

### 2) Context‑scoped resources
```python
# contexts.py
with MyClassifierContext(args.model_name, labels, cfg) as classifier:
    # classifier.model + classifier.tokenizer are ready
    ...
```

### 3) Static deps in __init__, dynamic in methods
```python
# core.py
@dataclass(frozen=True)
class MyCollator(Collator):
    tokenizer: MockTokenizer       # static
    max_length: int                # static

    def collate(self, batch: list[dict]) -> dict[str, Any]:  # dynamic
        texts = [x["text"] for x in batch]
        out = self.tokenizer.tokenize(texts, max_length=self.max_length, truncation=True, padding=True)
        out["labels"] = [x["label"] for x in batch]
        return out
```

```python
@dataclass(frozen=True)
class MyEvaluator(Evaluator):
    labels: list[str]              # static

    def evaluate(self, predictions: list[list[int]], targets: list[int]) -> EvaluationResult:  # dynamic
        # compute metrics, return a value object
        return MyEvaluationResult(accuracy=..., total_samples=..., correct_predictions=...)
```

### 4) Orchestration is high‑level and obvious
```python
# main.py
with MyDataContext(train_data, val_data) as data_ctx:
    with MyClassifierContext(args.model_name, labels, cfg) as classifier:
        collator  = MyCollator(classifier.tokenizer, max_length=cfg.max_length)
        evaluator = MyEvaluator(labels)
        trainer   = MyTrainer(classifier=classifier, config=cfg)

        results = trainer.train(
            train_data=data_ctx.get_train_data(),
            val_data=data_ctx.get_val_data(),
            collator=collator,
            evaluator=evaluator,
            epochs=cfg.epochs,
        )
```

- No branching: the choice of concrete class happens **before** execution (by CLI + context + constructor).

---

## Logging > print

- Every module uses `logger = logging.getLogger(__name__)`.
- Logging calls include context via `extra={...}` (e.g., epoch, sizes, accuracy).
- The CLI configures the root logger once (`configure_logging`).
- The library layer doesn’t `print`; only the CLI may print final summaries for humans.

> In the demo, `MyEvaluationResult.visualize()` prints for simplicity. In production, prefer logging or return a structured object and let the caller decide how to render.

---

## Extending the System (zero drama)

> New collator? New evaluator? New model? Follow this checklist.

1. **Define/confirm the contract** in `interfaces.py` (usually already there).
2. **Implement a class** binding static deps in `__init__` and exposing a tiny method:
   - `Collator`: `__init__(tokenizer, max_length)` → `collate(batch)`  
   - `Evaluator`: `__init__(labels)` → `evaluate(preds, targets)` → returns `EvaluationResult`
   - `Classifier`: `__init__(model_name, labels, cfg)` → `classify(input_ids)`
   - `Trainer`: `__init__(config)` → `train(classifier, train, val, collator, evaluator, epochs)`
3. **Wire it** in `main.py` (or add a subcommand in `cli.py`). No runtime branching inside core logic.
4. **Log context** at each major step (`start`, `complete`, `metrics`), no prints.
5. **Keep config DRY**: do not re‑detect device/paths; use `RuntimeConfig`.

---

## CLI Examples

```bash
# run a training simulation
python main.py train --model-name demo-model --epochs 3 --data-size 120 --log-level INFO
```

You’ll see structured logs (and a brief human‑readable summary at the end).

---

## Testing the Contracts

Test the **interfaces** so every implementation is held to the same bar.

```python
# tests/test_trainer_contract.py
from interfaces import Trainer, Collator, Evaluator

def test_trainer_runs(trainer: Trainer, coll: Collator, eval_: Evaluator, data):
    history = trainer.train(data["train"], data["val"], coll, eval_, epochs=2)
    assert "final_accuracy" in history
```

---

## Why this works

- **Constructor injection** makes call sites minimal and obvious.
- **ABCs + small classes** keep contracts crisp and pluggable.
- **Contexts** make resource lifecycles explicit.
- **One config** prevents drift and duplicate logic.
- **No branching** means code paths are predictable and easy to refactor.
- **Logging** gives you observability without littering the code with prints.

---

## Guardrails (quick checklist)

- [ ] Static deps live in `__init__`; methods accept only dynamic inputs.
- [ ] No `if/elif/else` dispatch inside core logic—use classes and early selection.
- [ ] Only the CLI parses input; no config files.
- [ ] One `RuntimeConfig` per run; no re‑detecting device/paths.
- [ ] Library code logs; CLI can print a final summary.
- [ ] Context managers around heavy resources.
- [ ] Deep helpers are positional with no defaults.

---

## License

MIT. Clean code, clean conscience.