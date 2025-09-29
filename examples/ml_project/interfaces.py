from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class Collator(ABC):
    @abstractmethod
    def collate(self, batch: list[dict]) -> dict[str, Any]:
        pass

class Evaluator(ABC):
    @abstractmethod  
    def evaluate(self, predictions: list[list[int]], targets: list[int]) -> "EvaluationResult":
        pass

class EvaluationResult(ABC):
    @abstractmethod
    def visualize(self) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

class Classifier(ABC):
    @abstractmethod
    def classify(self, inputs: list[list[int]]) -> list[list[int]]:
        pass

class Trainer(ABC):
    @abstractmethod
    def train(
        self, 
        classifier: Classifier,
        train_data: list[dict], 
        val_data: list[dict], 
        collator: Collator, 
        evaluator: Evaluator, 
        epochs: int
    ) -> dict[str, Any]:
        pass