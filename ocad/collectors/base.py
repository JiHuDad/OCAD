"""Base classes for data collectors."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

import structlog

from ..core.config import NetconfConfig
from ..core.logging import get_logger
from ..core.models import Capabilities, Endpoint, MetricSample


logger = get_logger(__name__)


class BaseCollector(ABC):
    """Base class for all metric collectors."""
    
    def __init__(self, config: NetconfConfig):
        """Initialize base collector.
        
        Args:
            config: NETCONF configuration
        """
        self.config = config
        self.logger = logger.bind(component=self.__class__.__name__.lower())
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    @abstractmethod
    async def collect(self, endpoint: Endpoint, capabilities: Capabilities) -> Optional[MetricSample]:
        """Collect metrics from an endpoint.
        
        Args:
            endpoint: Endpoint to collect from
            capabilities: Endpoint capabilities
            
        Returns:
            Metric sample or None if collection failed
        """
        pass
    
    @abstractmethod
    def can_collect(self, capabilities: Capabilities) -> bool:
        """Check if this collector can work with given capabilities.
        
        Args:
            capabilities: Endpoint capabilities
            
        Returns:
            True if collector can work with these capabilities
        """
        pass
    
    async def start_collection(
        self, 
        endpoints: List[Endpoint], 
        capabilities_map: Dict[str, Capabilities],
        interval_seconds: int = 60
    ) -> AsyncGenerator[MetricSample, None]:
        """Start continuous collection from multiple endpoints.
        
        Args:
            endpoints: List of endpoints to collect from
            capabilities_map: Mapping of endpoint ID to capabilities
            interval_seconds: Collection interval
            
        Yields:
            Metric samples as they are collected
        """
        self._running = True
        
        # Filter endpoints that this collector can handle
        compatible_endpoints = [
            ep for ep in endpoints 
            if ep.id in capabilities_map and self.can_collect(capabilities_map[ep.id])
        ]
        
        self.logger.info(
            "Starting collection",
            total_endpoints=len(endpoints),
            compatible_endpoints=len(compatible_endpoints),
            interval=interval_seconds,
        )
        
        # Create collection tasks
        for endpoint in compatible_endpoints:
            task = asyncio.create_task(
                self._collect_loop(endpoint, capabilities_map[endpoint.id], interval_seconds)
            )
            self._tasks.append(task)
        
        # Yield samples as they come in
        try:
            while self._running:
                # Check for completed tasks and restart them
                completed_tasks = [task for task in self._tasks if task.done()]
                for task in completed_tasks:
                    try:
                        result = task.result()
                        if result:
                            yield result
                    except Exception as e:
                        self.logger.error("Collection task failed", error=str(e))
                    
                    # Remove completed task
                    self._tasks.remove(task)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                
        finally:
            await self.stop_collection()
    
    async def stop_collection(self) -> None:
        """Stop all collection tasks."""
        self._running = False
        
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        self.logger.info("Collection stopped")
    
    async def _collect_loop(
        self, 
        endpoint: Endpoint, 
        capabilities: Capabilities, 
        interval_seconds: int
    ) -> None:
        """Collection loop for a single endpoint.
        
        Args:
            endpoint: Endpoint to collect from
            capabilities: Endpoint capabilities
            interval_seconds: Collection interval
        """
        while self._running:
            try:
                sample = await self.collect(endpoint, capabilities)
                if sample:
                    # In a real implementation, this would be sent to a message queue
                    # For now, we'll just log it
                    self.logger.debug(
                        "Sample collected",
                        endpoint_id=endpoint.id,
                        sample=sample.dict(exclude_none=True),
                    )
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Collection failed",
                    endpoint_id=endpoint.id,
                    error=str(e),
                )
                # Wait before retrying
                await asyncio.sleep(min(interval_seconds, 30))


class CollectorManager:
    """Manages multiple collectors."""
    
    def __init__(self, collectors: List[BaseCollector]):
        """Initialize collector manager.
        
        Args:
            collectors: List of collectors to manage
        """
        self.collectors = collectors
        self.logger = get_logger(__name__).bind(component="collector_manager")
        self._running = False
    
    async def start(
        self, 
        endpoints: List[Endpoint], 
        capabilities_map: Dict[str, Capabilities],
        interval_seconds: int = 60
    ) -> AsyncGenerator[MetricSample, None]:
        """Start all collectors.
        
        Args:
            endpoints: List of endpoints
            capabilities_map: Capabilities mapping
            interval_seconds: Collection interval
            
        Yields:
            Metric samples from all collectors
        """
        self._running = True
        
        # Start all collectors
        collector_generators = []
        for collector in self.collectors:
            gen = collector.start_collection(endpoints, capabilities_map, interval_seconds)
            collector_generators.append(gen)
        
        self.logger.info(
            "All collectors started",
            collector_count=len(self.collectors),
            endpoint_count=len(endpoints),
        )
        
        # Merge samples from all collectors
        try:
            while self._running:
                for gen in collector_generators:
                    try:
                        sample = await gen.__anext__()
                        yield sample
                    except StopAsyncIteration:
                        pass
                    except Exception as e:
                        self.logger.error("Collector error", error=str(e))
                
                await asyncio.sleep(0.1)
        
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop all collectors."""
        self._running = False
        
        for collector in self.collectors:
            await collector.stop_collection()
        
        self.logger.info("All collectors stopped")
