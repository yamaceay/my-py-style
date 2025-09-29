from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class Processor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

class Handler(ABC):
    @abstractmethod
    def handle(self, request: Any) -> Any:
        pass

class Repository(ABC):
    @abstractmethod
    def save(self, entity: Any) -> bool:
        pass
    
    @abstractmethod
    def find(self, identifier: str) -> Any:
        pass

class Service(ABC):
    @abstractmethod
    def execute(self, params: Any) -> Any:
        pass
