from __future__ import annotations
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from config import RuntimeConfig

from core import MyClassifier

logger = logging.getLogger(__name__)

class MyClassifierContext(AbstractContextManager["MyClassifier"]):
    def __init__(self, model_name: str, labels: list[str], cfg: "RuntimeConfig"):
        self.model_name = model_name
        self.labels = labels
        self.cfg = cfg
        self.classifier: "MyClassifier" | None = None
    
    def __enter__(self) -> "MyClassifier":
        logger.info("initializing_classification_context", extra={
            "model_name": self.model_name,
            "device": self.cfg.device,
            "num_labels": len(self.labels)
        })
        
        self.classifier = MyClassifier(
            model_name=self.model_name,
            labels=self.labels,
            cfg=self.cfg
        )
        
        logger.info("classification_context_ready")
        return self.classifier
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.info("releasing_classification_resources")
        if self.classifier:
            self.classifier.cleanup()
        self.classifier = None
        return False

class MyDataContext(AbstractContextManager["MyDataContext"]):
    def __init__(self, train_data: list[dict], val_data: list[dict]):
        self.train_data: list[dict] | None = None
        self.val_data: list[dict] | None = None
        self._source_train_data = train_data
        self._source_val_data = val_data
    
    def __enter__(self) -> "MyDataContext":
        logger.info("loading_data_into_memory", extra={
            "train_size": len(self._source_train_data),
            "val_size": len(self._source_val_data)
        })
        
        self.train_data = self._source_train_data.copy()
        self.val_data = self._source_val_data.copy()
        
        logger.info("data_loaded_into_memory", extra={
            "train_size": len(self.train_data),
            "val_size": len(self.val_data)
        })
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.info("releasing_data_from_memory")
        self.train_data = None
        self.val_data = None
        return False
    
    def get_train_data(self) -> list[dict]:
        if self.train_data is None:
            raise RuntimeError("Data not initialized - use within context manager")
        return self.train_data
    
    def get_val_data(self) -> list[dict]:
        if self.val_data is None:
            raise RuntimeError("Data not initialized - use within context manager")
        return self.val_data