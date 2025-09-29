"""Core configuration and runtime management for the simulation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    """Canonical runtime configuration following DRY principle."""

    device: str
    out_dir: str
    log_level: str
    seed: int | None = None
    max_length: int = 128
    epochs: int = 3


def canonicalize_config(
    device: str | None = None,
    out_dir: str | None = None,
    log_level: str | None = None,
    seed: int | None = None,
    max_length: int | None = None,
    epochs: int | None = None,
) -> RuntimeConfig:
    """Canonicalize configuration with top-level defaults only."""
    return RuntimeConfig(
        device=device or "cpu",
        out_dir=out_dir or "./simulation_output",
        log_level=log_level or "INFO",
        seed=seed,
        max_length=max_length or 128,
        epochs=epochs or 3,
    )


def set_seed(seed: int) -> None:
    """Simple seed setting for reproducibility."""
    import random
    random.seed(seed)




class MockTokenizer:
    """Simulated tokenizer for demonstration."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def tokenize(self, texts: list[str], max_length: int, truncation: bool = True, padding: bool = True) -> dict[str, Any]:
        """Simulate tokenization by converting texts to simple numeric representation."""

        tokens = []
        attention_masks = []
        
        for text in texts:
            token_count = min(len(text.split()), max_length) if truncation else len(text.split())

            padded_tokens = [1] * token_count + [0] * (max_length - token_count) if padding else [1] * token_count
            attention_mask = [1] * token_count + [0] * (max_length - token_count) if padding else [1] * token_count
            
            tokens.append(padded_tokens[:max_length])
            attention_masks.append(attention_mask[:max_length])
        
        return {"input_ids": tokens, "attention_mask": attention_masks}


class MockModel:
    """Simulated model for demonstration."""
    
    def __init__(self, model_name: str, device: str, labels: list[str]):
        self.model_name = model_name
        self.device = device
        self.labels = labels
        self.num_labels = len(labels)
    
    def predict(self, input_ids: list[list[int]]) -> list[list[int]]:
        """Simulate model predictions with simple logic."""
        import random
        predictions = []
        for seq in input_ids:


            predictions.append(seq_preds)
        return predictions
    
    def release(self) -> None:
        """Simulate model cleanup."""
        pass


def load_model(model_name: str, device: str, labels: list[str]) -> MockModel:
    """Factory function for loading models."""
    return MockModel(model_name, device, labels)


def load_tokenizer(model_name: str) -> MockTokenizer:
    """Factory function for loading tokenizers."""
    return MockTokenizer(model_name)