from __future__ import annotations
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from config import AppConfig

from core import DefaultService, DefaultRepository, DefaultProcessor

logger = logging.getLogger(__name__)

class ServiceContext(AbstractContextManager[DefaultService]):
    def __init__(self, config: "AppConfig"):
        self.config = config
        self.service: DefaultService | None = None
        self.repository: DefaultRepository | None = None
        self.processor: DefaultProcessor | None = None
    
    def __enter__(self) -> DefaultService:
        logger.info('initializing_service_context')
        
        self.repository = DefaultRepository()
        self.processor = DefaultProcessor(name=self.config.app_name)
        self.service = DefaultService(
            repository=self.repository,
            processor=self.processor
        )
        
        logger.info('service_context_ready')
        return self.service
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.info('releasing_service_resources')
        self.service = None
        self.repository = None
        self.processor = None
        return False

class DataContext(AbstractContextManager["DataContext"]):
    def __init__(self, data_source: str):
        self.data_source = data_source
        self.data: list[Any] | None = None
    
    def __enter__(self) -> "DataContext":
        logger.info('loading_data', extra={'source': self.data_source})
        self.data = []
        logger.info('data_loaded', extra={'count': len(self.data)})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.info('releasing_data')
        self.data = None
        return False
    
    def get_data(self) -> list[Any]:
        if self.data is None:
            raise RuntimeError("Data not initialized - use within context manager")
        return self.data
