from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import logging

if TYPE_CHECKING:
    from config import AppConfig

from interfaces import Processor, Handler, Repository, Service

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DefaultProcessor(Processor):
    name: str
    
    def process(self, data: Any) -> Any:
        logger.info('processing_data', extra={'processor': self.name})
        return {"processed": True, "data": data, "processor": self.name}

@dataclass(frozen=True)
class DefaultHandler(Handler):
    service: Service
    
    def handle(self, request: Any) -> Any:
        logger.info('handling_request', extra={'request_type': type(request).__name__})
        return self.service.execute(request)

class DefaultRepository(Repository):
    def __init__(self):
        self._storage: dict[str, Any] = {}
    
    def save(self, entity: Any) -> bool:
        entity_id = getattr(entity, "id", str(hash(str(entity))))
        self._storage[entity_id] = entity
        logger.debug('entity_saved', extra={'entity_id': entity_id})
        return True
    
    def find(self, identifier: str) -> Any:
        result = self._storage.get(identifier)
        logger.debug('entity_found', extra={'entity_id': identifier, 'found': result is not None})
        return result

@dataclass(frozen=True)
class DefaultService(Service):
    repository: Repository
    processor: Processor
    
    def execute(self, params: Any) -> Any:
        logger.info('service_execution_start', extra={'params_type': type(params).__name__})
        processed = self.processor.process(params)
        self.repository.save(processed)
        logger.info('service_execution_complete')
        return processed
