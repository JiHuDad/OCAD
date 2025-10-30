# PADIT 서비스 구현 가이드

**Part 2**: Microservices 구현 및 전체 연동

---

## 🚀 Phase 3: 핵심 서비스 구현 (Week 5-8)

### 3.1 Ingestion Service (수집 서비스)

```python
# services/ingestion/main.py
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import yaml

from padit.shared.messaging import EventBus, KafkaBroker
from padit.shared.events import MetricCollected
from padit.services.ingestion.plugin_registry import registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionService:
    """메트릭 수집 서비스"""

    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.running = False

    async def start(self):
        """서비스 시작"""
        logger.info("Starting Ingestion Service...")

        # 플러그인 발견 및 등록
        plugin_dir = Path(self.config.get("plugin_dir", "./plugins"))
        registry.discover_plugins(plugin_dir)

        logger.info(f"Loaded adapters: {list(registry.list().keys())}")

        # 이벤트 버스 연결
        await self.event_bus.start()

        # 설정된 어댑터 시작
        self.running = True
        tasks = []

        for adapter_config in self.config.get("adapters", []):
            adapter_name = adapter_config["name"]
            adapter = registry.get(adapter_name)

            if not adapter:
                logger.warning(f"Adapter not found: {adapter_name}")
                continue

            if not adapter.validate_config(adapter_config):
                logger.error(f"Invalid config for adapter: {adapter_name}")
                continue

            # 어댑터별 수집 태스크 생성
            task = asyncio.create_task(
                self._run_adapter(adapter, adapter_config)
            )
            tasks.append(task)

        logger.info(f"Started {len(tasks)} adapters")

        # 서비스 실행 (Ctrl+C까지)
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.running = False

        await self.event_bus.stop()

    async def _run_adapter(self, adapter, config: Dict):
        """어댑터 실행 (백그라운드)"""
        adapter_name = adapter.name
        logger.info(f"Starting adapter: {adapter_name}")

        try:
            async for metric in adapter.collect(config):
                if not self.running:
                    break

                # 메트릭 수집 이벤트 발행
                event = MetricCollected(metric=metric)
                await self.event_bus.publish(event)

                logger.debug(
                    f"Metric collected: {metric.source_id} - "
                    f"{metric.metric_name} = {metric.value}"
                )

        except Exception as e:
            logger.error(f"Adapter {adapter_name} failed: {e}")


async def main():
    """메인 함수"""
    # 설정 로드
    with open("config/ingestion.yaml") as f:
        config = yaml.safe_load(f)

    # Kafka 브로커 생성
    broker = KafkaBroker(config["kafka"]["bootstrap_servers"])
    event_bus = EventBus(broker)

    # 서비스 시작
    service = IngestionService(config, event_bus)
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())
```

**설정 파일**:
```yaml
# config/ingestion.yaml
kafka:
  bootstrap_servers: "localhost:9092"

plugin_dir: "./plugins"

adapters:
  - name: "oran-cfm"
    interval_sec: 10
    endpoints:
      - id: "o-ru-001"
        host: "192.168.1.100"
        port: 830
        udp_echo: true
        ecpri: true
        lbm: true

  - name: "http"
    interval_sec: 30
    timeout_sec: 10
    endpoints:
      - id: "api-server-1"
        url: "https://api.example.com/health"
        method: "GET"
      - id: "api-server-2"
        url: "https://api2.example.com/status"
```

### 3.2 Feature Engineering Service (피처 추출 서비스)

```python
# services/feature_engineering/engine.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from collections import deque
import numpy as np
from scipy import stats

from padit.shared.domain import Metric, Features

logger = logging.getLogger(__name__)

class FeatureEngine:
    """피처 엔지니어링 엔진"""

    def __init__(self, window_size: int = 300):
        """
        Args:
            window_size: 윈도우 크기 (초)
        """
        self.window_size = window_size
        # 각 source_id별 슬라이딩 윈도우
        self.windows: Dict[str, deque] = {}

    def add_metric(self, metric: Metric):
        """메트릭 추가"""
        key = f"{metric.source_id}:{metric.metric_name}"

        if key not in self.windows:
            self.windows[key] = deque(maxlen=1000)  # 최대 1000개

        self.windows[key].append(metric)

    def extract_features(self, source_id: str, metric_name: str) -> Features:
        """피처 추출"""
        key = f"{source_id}:{metric_name}"
        window = self.windows.get(key, deque())

        if not window:
            raise ValueError(f"No data for {key}")

        # 윈도우 내 데이터 필터링
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_size)

        values = [
            m.value for m in window
            if m.timestamp >= window_start
        ]

        if not values:
            raise ValueError(f"No data in window for {key}")

        return self._compute_features(
            source_id=source_id,
            values=values,
            window_start=window_start,
            window_end=now
        )

    def _compute_features(
        self,
        source_id: str,
        values: List[float],
        window_start: datetime,
        window_end: datetime
    ) -> Features:
        """피처 계산"""
        arr = np.array(values)

        # 기본 통계
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        median_val = float(np.median(arr))

        # 분위수
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))

        # 추세 (선형 회귀 기울기)
        if len(values) > 1:
            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, arr)
            trend = float(slope)
        else:
            trend = 0.0

        # EWMA (지수가중이동평균)
        alpha = 0.3
        ewma = mean_val
        for val in values:
            ewma = alpha * val + (1 - alpha) * ewma
        ewma = float(ewma)

        # 기울기 (마지막 값 - 평균)
        gradient = float(values[-1] - mean_val)

        # CUSUM
        cusum = 0.0
        threshold = mean_val
        for val in values:
            cusum = max(0, cusum + (val - threshold))
        cusum = float(cusum)

        return Features(
            timestamp=window_end,
            source_id=source_id,
            window_start=window_start,
            window_end=window_end,
            mean=mean_val,
            std=std_val,
            min=min_val,
            max=max_val,
            median=median_val,
            p25=p25,
            p75=p75,
            p95=p95,
            p99=p99,
            trend=trend,
            ewma=ewma,
            gradient=gradient,
            cusum=cusum,
            raw_values=values,
        )


# services/feature_engineering/main.py
import asyncio
import logging
import yaml
import json
from datetime import datetime

from padit.shared.messaging import EventBus, KafkaBroker
from padit.shared.events import MetricCollected, FeaturesExtracted
from padit.shared.domain import Metric
from padit.services.feature_engineering.engine import FeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineeringService:
    """피처 엔지니어링 서비스"""

    def __init__(self, config: dict, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.engine = FeatureEngine(
            window_size=config.get("window_size_sec", 300)
        )
        self.extract_interval = config.get("extract_interval_sec", 60)

    async def start(self):
        """서비스 시작"""
        logger.info("Starting Feature Engineering Service...")

        await self.event_bus.start()

        # 메트릭 수집 이벤트 구독
        @self.event_bus.subscribe("metriccollected")
        async def handle_metric_collected(message: str):
            event = MetricCollected.from_dict(json.loads(message))
            self.engine.add_metric(event.metric)
            logger.debug(f"Metric added: {event.metric.source_id}")

        await self.event_bus.start_consuming()

        # 주기적 피처 추출
        await self._periodic_extraction()

    async def _periodic_extraction(self):
        """주기적 피처 추출"""
        while True:
            await asyncio.sleep(self.extract_interval)

            # 모든 윈도우에 대해 피처 추출
            for key in list(self.engine.windows.keys()):
                source_id, metric_name = key.split(":", 1)

                try:
                    features = self.engine.extract_features(source_id, metric_name)

                    # 피처 추출 이벤트 발행
                    event = FeaturesExtracted(
                        features={
                            "source_id": features.source_id,
                            "timestamp": features.timestamp.isoformat(),
                            "mean": features.mean,
                            "std": features.std,
                            "p95": features.p95,
                            "p99": features.p99,
                            "trend": features.trend,
                            "ewma": features.ewma,
                            "gradient": features.gradient,
                            "cusum": features.cusum,
                        }
                    )
                    await self.event_bus.publish(event)

                    logger.info(f"Features extracted: {source_id} - {metric_name}")

                except Exception as e:
                    logger.error(f"Feature extraction failed for {key}: {e}")


async def main():
    with open("config/feature_engineering.yaml") as f:
        config = yaml.safe_load(f)

    broker = KafkaBroker(config["kafka"]["bootstrap_servers"])
    event_bus = EventBus(broker)

    service = FeatureEngineeringService(config, event_bus)
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())
```

### 3.3 Detection Service (탐지 서비스)

```python
# services/detection/detectors/rule_based.py
from padit.shared.domain import Features, DetectionScore

class RuleBasedDetector:
    """룰 기반 탐지기"""

    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

    def detect(self, features: Features) -> DetectionScore:
        """이상 탐지"""
        score = 0.0
        evidence = {}

        # P99 임계값 체크
        p99_threshold = self.thresholds.get("p99", 10.0)
        if features.p99 > p99_threshold:
            score += 0.5
            evidence["p99_violation"] = {
                "value": features.p99,
                "threshold": p99_threshold
            }

        # CUSUM 임계값 체크
        cusum_threshold = self.thresholds.get("cusum", 50.0)
        if features.cusum > cusum_threshold:
            score += 0.5
            evidence["cusum_violation"] = {
                "value": features.cusum,
                "threshold": cusum_threshold
            }

        is_anomaly = score > 0.5

        return DetectionScore(
            timestamp=features.timestamp,
            source_id=features.source_id,
            rule_score=min(1.0, score),
            composite_score=min(1.0, score),
            is_anomaly=is_anomaly,
            confidence=0.8 if is_anomaly else 0.9,
            evidence=evidence,
            detector_name="rule_based"
        )


# services/detection/main.py
import asyncio
import logging
import yaml
import json

from padit.shared.messaging import EventBus, KafkaBroker
from padit.shared.events import FeaturesExtracted, AnomalyDetected
from padit.shared.domain import Features
from padit.services.detection.detectors.rule_based import RuleBasedDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionService:
    """탐지 서비스"""

    def __init__(self, config: dict, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus

        # 탐지기 초기화
        self.detectors = {
            "rule_based": RuleBasedDetector(
                thresholds=config.get("thresholds", {})
            )
        }

    async def start(self):
        """서비스 시작"""
        logger.info("Starting Detection Service...")

        await self.event_bus.start()

        # 피처 추출 이벤트 구독
        @self.event_bus.subscribe("featuresextracted")
        async def handle_features_extracted(message: str):
            event = FeaturesExtracted.from_dict(json.loads(message))
            await self._detect(event.features)

        await self.event_bus.start_consuming()

        # Keep running
        while True:
            await asyncio.sleep(1)

    async def _detect(self, features_dict: dict):
        """이상 탐지 수행"""
        # Features 복원
        features = Features(
            timestamp=datetime.fromisoformat(features_dict["timestamp"]),
            source_id=features_dict["source_id"],
            window_start=datetime.utcnow(),  # placeholder
            window_end=datetime.utcnow(),
            mean=features_dict["mean"],
            std=features_dict["std"],
            min=0.0,  # placeholder
            max=0.0,
            median=0.0,
            p25=0.0,
            p75=0.0,
            p95=features_dict["p95"],
            p99=features_dict["p99"],
            trend=features_dict["trend"],
            ewma=features_dict["ewma"],
            gradient=features_dict["gradient"],
            cusum=features_dict["cusum"],
        )

        # 모든 탐지기 실행
        for detector_name, detector in self.detectors.items():
            try:
                detection = detector.detect(features)

                if detection.is_anomaly:
                    # 이상 탐지 이벤트 발행
                    event = AnomalyDetected(
                        detection={
                            "timestamp": detection.timestamp.isoformat(),
                            "source_id": detection.source_id,
                            "score": detection.composite_score,
                            "confidence": detection.confidence,
                            "evidence": detection.evidence,
                            "detector": detector_name,
                        }
                    )
                    await self.event_bus.publish(event)

                    logger.warning(
                        f"Anomaly detected: {detection.source_id} - "
                        f"score={detection.composite_score:.2f}"
                    )

            except Exception as e:
                logger.error(f"Detector {detector_name} failed: {e}")


async def main():
    from datetime import datetime

    with open("config/detection.yaml") as f:
        config = yaml.safe_load(f)

    broker = KafkaBroker(config["kafka"]["bootstrap_servers"])
    event_bus = EventBus(broker)

    service = DetectionService(config, event_bus)
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())
```

### 3.4 Alert Service (알림 서비스)

```python
# services/alert/main.py
import asyncio
import logging
import yaml
import json
from datetime import datetime

from padit.shared.messaging import EventBus, KafkaBroker
from padit.shared.events import AnomalyDetected, AlertCreated
from padit.shared.domain import Alert, Severity
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertService:
    """알림 서비스"""

    def __init__(self, config: dict, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.alert_threshold = config.get("alert_threshold", 0.7)

    async def start(self):
        """서비스 시작"""
        logger.info("Starting Alert Service...")

        await self.event_bus.start()

        # 이상 탐지 이벤트 구독
        @self.event_bus.subscribe("anomalydetected")
        async def handle_anomaly_detected(message: str):
            event = AnomalyDetected.from_dict(json.loads(message))
            await self._create_alert(event.detection)

        await self.event_bus.start_consuming()

        # Keep running
        while True:
            await asyncio.sleep(1)

    async def _create_alert(self, detection: dict):
        """알림 생성"""
        score = detection["score"]

        if score < self.alert_threshold:
            logger.debug(f"Score {score} below threshold, skipping alert")
            return

        # 심각도 결정
        if score >= 0.9:
            severity = Severity.CRITICAL
        elif score >= 0.7:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO

        # 알림 생성
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.fromisoformat(detection["timestamp"]),
            source_id=detection["source_id"],
            severity=severity,
            title=f"Anomaly detected on {detection['source_id']}",
            description=f"Anomaly score: {score:.2f}\nConfidence: {detection['confidence']:.2f}",
            detection_score=None,  # placeholder
            metadata={
                "evidence": detection.get("evidence", {}),
                "detector": detection.get("detector"),
            }
        )

        # 알림 이벤트 발행
        event = AlertCreated(
            alert={
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "source_id": alert.source_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
            }
        )
        await self.event_bus.publish(event)

        logger.warning(f"Alert created: {alert.title}")

        # 실제 알림 전송 (Slack, Email, etc.)
        await self._send_notification(alert)

    async def _send_notification(self, alert: Alert):
        """알림 전송"""
        # TODO: Slack/Email/PagerDuty 통합
        logger.info(f"Notification sent: {alert.title}")


async def main():
    with open("config/alert.yaml") as f:
        config = yaml.safe_load(f)

    broker = KafkaBroker(config["kafka"]["bootstrap_servers"])
    event_bus = EventBus(broker)

    service = AlertService(config, event_bus)
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 Phase 4: 전체 연동 및 테스트 (Week 9-10)

### 4.1 통합 실행 스크립트

```bash
#!/bin/bash
# scripts/start_all.sh

echo "Starting PADIT platform..."

# 인프라 시작 (Docker Compose)
echo "Starting infrastructure..."
docker-compose up -d

# 서비스 시작 대기
echo "Waiting for services to be ready..."
sleep 10

# Ingestion Service
echo "Starting Ingestion Service..."
python -m services.ingestion.main &
INGESTION_PID=$!

# Feature Engineering Service
echo "Starting Feature Engineering Service..."
python -m services.feature_engineering.main &
FEATURE_PID=$!

# Detection Service
echo "Starting Detection Service..."
python -m services.detection.main &
DETECTION_PID=$!

# Alert Service
echo "Starting Alert Service..."
python -m services.alert.main &
ALERT_PID=$!

echo "All services started!"
echo "Ingestion PID: $INGESTION_PID"
echo "Feature PID: $FEATURE_PID"
echo "Detection PID: $DETECTION_PID"
echo "Alert PID: $ALERT_PID"

# Ctrl+C 핸들러
trap "echo 'Stopping services...'; kill $INGESTION_PID $FEATURE_PID $DETECTION_PID $ALERT_PID; docker-compose down" EXIT

wait
```

### 4.2 End-to-End 테스트

```python
# tests/e2e/test_complete_pipeline.py
import pytest
import asyncio
from datetime import datetime, timedelta
import time

from padit.shared.messaging import EventBus, RedisBroker
from padit.shared.domain import Metric, MetricType
from padit.shared.events import MetricCollected, AnomalyDetected

@pytest.mark.asyncio
async def test_complete_detection_pipeline():
    """전체 파이프라인 E2E 테스트"""

    # Redis broker 사용 (테스트용)
    broker = RedisBroker("redis://localhost:6379")
    event_bus = EventBus(broker)

    await event_bus.start()

    # 알림 수신 대기
    alerts = []

    @event_bus.subscribe("anomalydetected")
    async def collect_alerts(message: str):
        import json
        event = AnomalyDetected.from_dict(json.loads(message))
        alerts.append(event)

    await event_bus.start_consuming()

    # 정상 메트릭 10개 발행
    for i in range(10):
        metric = Metric(
            timestamp=datetime.utcnow(),
            source_id="test-endpoint-1",
            metric_name="test_metric",
            value=5.0 + (i * 0.1),  # 5.0 ~ 5.9
            unit="ms",
            metric_type=MetricType.LATENCY,
            protocol="test",
        )

        event = MetricCollected(metric=metric)
        await event_bus.publish(event)

        await asyncio.sleep(0.1)

    # 이상 메트릭 발행
    for i in range(5):
        metric = Metric(
            timestamp=datetime.utcnow(),
            source_id="test-endpoint-1",
            metric_name="test_metric",
            value=50.0 + (i * 10),  # 50 ~ 90 (이상!)
            unit="ms",
            metric_type=MetricType.LATENCY,
            protocol="test",
        )

        event = MetricCollected(metric=metric)
        await event_bus.publish(event)

        await asyncio.sleep(0.1)

    # 알림 대기
    await asyncio.sleep(5)

    await event_bus.stop()

    # 검증
    assert len(alerts) > 0, "No alerts detected"
    assert alerts[0].detection["source_id"] == "test-endpoint-1"
    print(f"✅ Detected {len(alerts)} alerts")
```

---

## 📊 Phase 5: Kubernetes 배포 (Week 11-12)

### 5.1 Dockerfile

```dockerfile
# services/ingestion/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY shared/ ./shared/
COPY services/ingestion/ ./services/ingestion/
COPY plugins/ ./plugins/
COPY config/ ./config/

# 서비스 실행
CMD ["python", "-m", "services.ingestion.main"]
```

### 5.2 Kubernetes Manifests

```yaml
# kubernetes/base/ingestion-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: padit-ingestion
  labels:
    app: padit
    component: ingestion
spec:
  replicas: 3
  selector:
    matchLabels:
      app: padit
      component: ingestion
  template:
    metadata:
      labels:
        app: padit
        component: ingestion
    spec:
      containers:
      - name: ingestion
        image: padit/ingestion:latest
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: padit-ingestion
spec:
  selector:
    app: padit
    component: ingestion
  ports:
  - port: 8080
    targetPort: 8080
```

### 5.3 Helm Chart

```yaml
# helm/padit/Chart.yaml
apiVersion: v2
name: padit
description: Protocol Anomaly Detection & Intelligence Toolkit
version: 0.1.0
appVersion: "1.0.0"

# helm/padit/values.yaml
global:
  kafka:
    bootstrapServers: "kafka:9092"

ingestion:
  replicas: 3
  image:
    repository: padit/ingestion
    tag: "latest"
  resources:
    requests:
      memory: 256Mi
      cpu: 200m

featureEngineering:
  replicas: 2
  image:
    repository: padit/feature-engineering
    tag: "latest"

detection:
  replicas: 5
  image:
    repository: padit/detection
    tag: "latest"

alert:
  replicas: 2
  image:
    repository: padit/alert
    tag: "latest"
```

---

## 🎓 요약

### 완성된 구조

```
padit/
├── shared/                    # 공유 라이브러리
│   ├── domain/                # 도메인 모델 ✅
│   ├── events/                # 이벤트 정의 ✅
│   ├── messaging/             # 메시지 버스 ✅
│   └── storage/               # 저장소 (TODO)
│
├── services/                  # Microservices
│   ├── ingestion/             # 수집 서비스 ✅
│   ├── feature_engineering/   # 피처 엔지니어링 ✅
│   ├── detection/             # 탐지 서비스 ✅
│   ├── alert/                 # 알림 서비스 ✅
│   └── api/                   # API Gateway (TODO)
│
├── plugins/                   # 플러그인
│   └── protocol_adapters/
│       ├── oran_cfm/          # ORAN 어댑터 ✅
│       └── http/              # HTTP 어댑터 ✅
│
├── tests/
│   └── e2e/                   # E2E 테스트 ✅
│
├── kubernetes/                # K8s manifests ✅
└── docker-compose.yml         # Local development ✅
```

### 다음 단계

1. **API Gateway**: REST/GraphQL API 구현
2. **Web UI**: React/Vue 대시보드
3. **AutoML**: 자동 모델 학습 파이프라인
4. **More Adapters**: MQTT, Modbus, CAN 등

---

**작성자**: Claude Code
**작성일**: 2025-10-27
**버전**: 2.0.0 (Services Implementation)
