"""모든 컴포넌트를 조정하는 메인 시스템 오케스트레이터.
OCAD 시스템의 핵심 조정자로 모든 구성요소 간의 데이터 흐름을 관리합니다.
"""

import asyncio
import time
from typing import Dict, List, Optional

import structlog

from ..alerts.manager import AlertManager
from ..capability.detector import CapabilityDetector, CapabilityRegistry
from ..collectors.base import CollectorManager
from ..collectors.ecpri_delay import EcpriDelayCollector
from ..collectors.lbm import LbmCollector
from ..collectors.udp_echo import UdpEchoCollector
from ..core.config import Settings
from ..core.logging import get_logger
from ..core.models import Alert, Endpoint, KPIMetrics, Severity
from ..detectors.base import CompositeDetector
from ..detectors.changepoint import ChangePointDetector
from ..detectors.residual import ResidualDetector
from ..detectors.rule_based import RuleBasedDetector
from ..features.engine import FeatureEngine


logger = get_logger(__name__)


class SystemOrchestrator:
    """모든 시스템 컴포넌트를 조정하는 메인 오케스트레이터.
    
    OCAD 시스템의 핵심 구성 요소들을 관리하고 조정합니다:
    - Capability 감지 및 등록
    - 데이터 수집 (UDP Echo, eCPRI, LBM)
    - 피처 추출 및 변환
    - 이상 탐지 (룰, 변화점, 잔차, 다변량)
    - 알람 관리 및 통지
    """
    
    def __init__(self, config: Settings):
        """시스템 오케스트레이터를 초기화합니다.
        
        Args:
            config: 시스템 설정
        """
        self.config = config
        self.logger = logger.bind(component="orchestrator")
        
        # 핵심 컴포넌트 초기화
        self.capability_detector = CapabilityDetector(config.netconf)
        self.capability_registry = CapabilityRegistry()
        
        # 데이터 수집기 초기화
        self.collectors = [
            UdpEchoCollector(config.netconf),      # UDP Echo RTT 수집
            EcpriDelayCollector(config.netconf),   # eCPRI 지연 수집
            LbmCollector(config.netconf),          # LBM 루프백 수집
        ]
        self.collector_manager = CollectorManager(self.collectors)
        
        # 피처 엔진 초기화
        self.feature_engine = FeatureEngine(config.feature)
        
        # 이상 탐지기들 초기화
        detectors = [
            RuleBasedDetector(config.detection),    # 룰 기반 탐지
            ChangePointDetector(config.detection),  # 변화점 탐지 (CUSUM)
            ResidualDetector(config.detection),     # 잔차 기반 탐지 (TCN/LSTM)
        ]
        
        # 다변량 탐지기 추가 (선택사항)
        try:
            from ..detectors.multivariate import MultivariateDetector
            detectors.append(MultivariateDetector(config.detection))
        except ImportError:
            self.logger.warning("다변량 탐지기 사용 불가 (sklearn 누락)")
        except Exception as e:
            self.logger.warning("다변량 탐지기 초기화 실패", error=str(e))
        
        self.composite_detector = CompositeDetector(config.detection, detectors)
        
        # 알람 관리자 초기화
        self.alert_manager = AlertManager(config.alert, config.detection)
        
        # 시스템 상태
        self.is_running = False
        self.endpoints: Dict[str, Endpoint] = {}  # 등록된 엔드포인트들
        self.processing_tasks: List[asyncio.Task] = []  # 처리 태스크들
        
        # 시스템 메트릭
        self.metrics = {
            "samples_processed": 0,    # 처리된 샘플 수
            "features_extracted": 0,   # 추출된 피처 수
            "alerts_generated": 0,     # 생성된 알람 수
            "processing_times": [],    # 처리 시간 기록
        }
    
    async def start(self) -> None:
        """Start the orchestration system."""
        if self.is_running:
            return
        
        self.logger.info("Starting system orchestrator")
        
        try:
            # Start the main processing loop
            processing_task = asyncio.create_task(self._main_processing_loop())
            self.processing_tasks.append(processing_task)
            
            self.is_running = True
            self.logger.info("System orchestrator started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start orchestrator", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the orchestration system."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping system orchestrator")
        
        self.is_running = False
        
        # Stop collector manager
        await self.collector_manager.stop()
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        self.processing_tasks.clear()
        
        self.logger.info("System orchestrator stopped")
    
    async def add_endpoint(self, endpoint: Endpoint) -> bool:
        """Add an endpoint for monitoring.
        
        Args:
            endpoint: Endpoint to add
            
        Returns:
            True if successful
        """
        try:
            # Detect capabilities
            capabilities = await self.capability_detector.detect_capabilities(endpoint)
            
            # Register endpoint and capabilities
            self.capability_registry.register_endpoint(endpoint, capabilities)
            self.endpoints[endpoint.id] = endpoint
            
            self.logger.info(
                "Endpoint added",
                endpoint_id=endpoint.id,
                host=endpoint.host,
                capabilities=capabilities.dict(),
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to add endpoint",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return False
    
    async def remove_endpoint(self, endpoint_id: str) -> bool:
        """Remove an endpoint from monitoring.
        
        Args:
            endpoint_id: Endpoint identifier
            
        Returns:
            True if successful
        """
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            self.logger.info("Endpoint removed", endpoint_id=endpoint_id)
            return True
        
        return False
    
    async def list_endpoints(self) -> List[Endpoint]:
        """List all registered endpoints.
        
        Returns:
            List of endpoints
        """
        return list(self.endpoints.values())
    
    async def list_alerts(
        self,
        severity: Optional[Severity] = None,
        endpoint_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Alert]:
        """List alerts with optional filtering.
        
        Args:
            severity: Filter by severity
            endpoint_id: Filter by endpoint
            limit: Maximum number of alerts
            offset: Offset for pagination
            
        Returns:
            List of alerts
        """
        alerts = self.alert_manager.get_active_alerts()
        
        # Apply filters
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if endpoint_id:
            alerts = [a for a in alerts if a.endpoint_id == endpoint_id]
        
        # Apply pagination
        alerts = alerts[offset:offset + limit]
        
        return alerts
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if successful
        """
        return self.alert_manager.acknowledge_alert(alert_id)
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if successful
        """
        return self.alert_manager.resolve_alert(alert_id)
    
    async def suppress_endpoint(self, endpoint_id: str, duration_seconds: int) -> None:
        """Suppress alerts for an endpoint.
        
        Args:
            endpoint_id: Endpoint identifier
            duration_seconds: Suppression duration
        """
        self.alert_manager.suppress_endpoint(endpoint_id, duration_seconds)
    
    async def get_statistics(self) -> Dict:
        """Get system statistics.
        
        Returns:
            Dictionary with statistics
        """
        alert_stats = self.alert_manager.get_alert_stats()
        capability_coverage = self.capability_registry.get_capability_coverage()
        
        # Calculate processing latency p95
        processing_latency_p95 = None
        if self.metrics["processing_times"]:
            import numpy as np
            processing_latency_p95 = np.percentile(self.metrics["processing_times"], 95)
        
        return {
            "endpoints": len(self.endpoints),
            "active_alerts": alert_stats["active_alerts"],
            "suppressed_endpoints": alert_stats["suppressed_endpoints"],
            "capability_coverage": capability_coverage,
            "samples_processed": self.metrics["samples_processed"],
            "features_extracted": self.metrics["features_extracted"],
            "alerts_generated": self.metrics["alerts_generated"],
            "processing_latency_p95": processing_latency_p95,
        }
    
    async def get_kpi_metrics(self) -> KPIMetrics:
        """Get KPI metrics.
        
        Returns:
            KPI metrics
        """
        # This would typically query a database for historical data
        # For now, return current metrics
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        period_start = now - timedelta(hours=24)
        
        return KPIMetrics(
            period_start=period_start,
            period_end=now,
            total_alerts=self.metrics["alerts_generated"],
            true_positives=0,  # Would need feedback system
            false_positives=0,  # Would need feedback system
            false_negatives=0,  # Would need feedback system
            capability_coverage=self.capability_registry.get_capability_coverage(),
        )
    
    async def get_health_status(self) -> Dict:
        """Get system health status.
        
        Returns:
            Health status dictionary
        """
        return {
            "healthy": self.is_running,
            "components": {
                "orchestrator": self.is_running,
                "collectors": len(self.collectors),
                "endpoints": len(self.endpoints),
                "active_alerts": len(self.alert_manager.get_active_alerts()),
            },
            "uptime_seconds": time.time() - getattr(self, "_start_time", time.time()),
        }
    
    async def get_debug_info(self) -> Dict:
        """Get debug information.
        
        Returns:
            Debug information
        """
        window_stats = self.feature_engine.get_window_stats()
        
        return {
            "feature_windows": window_stats,
            "collector_status": {
                collector.__class__.__name__: "active" 
                for collector in self.collectors
            },
            "processing_metrics": self.metrics,
        }
    
    async def _main_processing_loop(self) -> None:
        """Main processing loop that coordinates data flow."""
        self._start_time = time.time()
        
        # Get capabilities mapping
        capabilities_map = {
            ep_id: self.capability_registry.get_capabilities(ep_id)
            for ep_id in self.endpoints.keys()
        }
        
        # Start collector manager
        async for sample in self.collector_manager.start(
            list(self.endpoints.values()),
            capabilities_map,
            interval_seconds=60  # Collection interval
        ):
            if not self.is_running:
                break
            
            await self._process_sample(sample)
    
    async def _process_sample(self, sample) -> None:
        """Process a single metric sample.
        
        Args:
            sample: Metric sample to process
        """
        start_time = time.time()
        
        try:
            # Extract features
            features = self.feature_engine.process_sample(sample)
            self.metrics["samples_processed"] += 1
            
            if features is None:
                return  # Not enough data yet
            
            self.metrics["features_extracted"] += 1
            
            # Get endpoint capabilities
            capabilities = self.capability_registry.get_capabilities(sample.endpoint_id)
            if capabilities is None:
                self.logger.warning(
                    "No capabilities found for endpoint",
                    endpoint_id=sample.endpoint_id,
                )
                return
            
            # Run detection
            detection_score = self.composite_detector.detect(features, capabilities)
            
            # Process alert
            alert = self.alert_manager.process_detection(
                detection_score, features, capabilities
            )
            
            if alert:
                self.metrics["alerts_generated"] += 1
                self.logger.info(
                    "Alert generated",
                    alert_id=alert.id,
                    endpoint_id=alert.endpoint_id,
                    severity=alert.severity,
                )
            
            # Record processing time
            processing_time = time.time() - start_time
            self.metrics["processing_times"].append(processing_time)
            
            # Keep only recent processing times (for memory efficiency)
            if len(self.metrics["processing_times"]) > 1000:
                self.metrics["processing_times"] = self.metrics["processing_times"][-500:]
                
        except Exception as e:
            self.logger.error(
                "Failed to process sample",
                endpoint_id=sample.endpoint_id,
                error=str(e),
            )
