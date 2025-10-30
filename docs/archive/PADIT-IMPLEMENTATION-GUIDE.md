# PADIT 구현 가이드

**프로젝트명**: PADIT (Protocol Anomaly Detection & Intelligence Toolkit)
**목적**: 실제 구현을 위한 상세 가이드

---

## 🎯 Phase 1: 프로젝트 초기화 (Week 1-2)

### 1.1 Git 저장소 생성

```bash
# 새 프로젝트 디렉토리 생성
mkdir -p ~/projects/padit
cd ~/projects/padit

# Git 초기화
git init
git branch -M main

# 기본 구조 생성
mkdir -p {services,shared,training,web,plugins,config,tests,docs}
mkdir -p services/{ingestion,feature_engineering,detection,alert,api}
mkdir -p shared/{domain,events,messaging,storage}
mkdir -p training/{pipelines,experiments,models}
mkdir -p plugins/{protocol_adapters,detectors,notifiers}
mkdir -p config/{base,dev,prod}
mkdir -p tests/{unit,integration,e2e}
mkdir -p docs/{architecture,api,deployment,tutorials}

# .gitignore 생성
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local
*.local.yaml

# Data
data/
logs/
*.log

# Models
models/*.pth
models/*.pkl
!models/.gitkeep

# Docker
docker-compose.override.yml

# OS
.DS_Store
Thumbs.db
EOF

# README 생성
cat > README.md << 'EOF'
# PADIT - Protocol Anomaly Detection & Intelligence Toolkit

Universal protocol anomaly detection platform powered by ML/AI.

## Quick Start

```bash
# Clone
git clone https://github.com/your-org/padit.git
cd padit

# Setup
docker-compose up -d

# Access
http://localhost:8080  # API
http://localhost:3000  # Web UI
http://localhost:3100  # Grafana
```

## Features

- 🔌 **Protocol Agnostic**: Support any protocol via plugins
- 🤖 **AI-Powered**: AutoML, hyperparameter tuning
- 📊 **Real-time**: Streaming detection with low latency
- 🏗️ **Production-Ready**: Kubernetes-native, HA, auto-scaling
- 🔧 **Developer-Friendly**: Easy plugin development

## Documentation

- [Architecture](docs/architecture/README.md)
- [API Reference](docs/api/README.md)
- [Plugin Development](docs/tutorials/plugin-development.md)
- [Deployment](docs/deployment/README.md)

## License

Apache 2.0
EOF

git add .
git commit -m "Initial commit: project structure"
```

### 1.2 핵심 도메인 모델 정의

```python
# shared/domain/metric.py
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum

class MetricType(Enum):
    """메트릭 타입"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    CUSTOM = "custom"

@dataclass(frozen=True)
class Metric:
    """프로토콜 독립적 메트릭"""

    # 필수 필드
    timestamp: datetime
    source_id: str              # 엔드포인트/디바이스 ID
    metric_name: str            # 메트릭 이름
    value: float                # 메트릭 값

    # 선택 필드
    unit: Optional[str] = None  # 단위 (ms, bytes, %, ...)
    metric_type: MetricType = MetricType.CUSTOM
    protocol: Optional[str] = None  # 프로토콜 이름
    metadata: Dict[str, Any] = None  # 추가 메타데이터

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "source_id": self.source_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "metric_type": self.metric_type.value,
            "protocol": self.protocol,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """딕셔너리에서 생성"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_id=data["source_id"],
            metric_name=data["metric_name"],
            value=float(data["value"]),
            unit=data.get("unit"),
            metric_type=MetricType(data.get("metric_type", "custom")),
            protocol=data.get("protocol"),
            metadata=data.get("metadata", {}),
        )


# shared/domain/features.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

@dataclass
class Features:
    """피처 벡터"""

    timestamp: datetime
    source_id: str
    window_start: datetime
    window_end: datetime

    # 통계 피처
    mean: float
    std: float
    min: float
    max: float
    median: float

    # 분위수
    p25: float
    p75: float
    p95: float
    p99: float

    # 시계열 피처
    trend: Optional[float] = None       # 추세
    seasonality: Optional[float] = None  # 계절성
    ewma: Optional[float] = None        # 지수가중이동평균

    # 변화량
    gradient: Optional[float] = None    # 기울기
    cusum: Optional[float] = None       # CUSUM

    # 원본 데이터 (선택)
    raw_values: Optional[List[float]] = None
    metadata: Dict[str, Any] = None

    def to_array(self) -> np.ndarray:
        """NumPy 배열로 변환 (ML 모델 입력용)"""
        return np.array([
            self.mean, self.std, self.min, self.max, self.median,
            self.p25, self.p75, self.p95, self.p99,
            self.trend or 0.0,
            self.seasonality or 0.0,
            self.ewma or 0.0,
            self.gradient or 0.0,
            self.cusum or 0.0,
        ])


# shared/domain/detection.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

class Severity(Enum):
    """심각도"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class DetectionScore:
    """탐지 점수"""

    timestamp: datetime
    source_id: str

    # 개별 탐지기 점수
    rule_score: float = 0.0
    changepoint_score: float = 0.0
    residual_score: float = 0.0
    multivariate_score: float = 0.0

    # 종합 점수
    composite_score: float = 0.0

    # 이상 여부
    is_anomaly: bool = False
    confidence: float = 0.0

    # 근거
    evidence: Dict[str, Any] = None
    detector_name: Optional[str] = None

    def __post_init__(self):
        if self.evidence is None:
            object.__setattr__(self, 'evidence', {})

@dataclass
class Alert:
    """알림"""

    alert_id: str
    timestamp: datetime
    source_id: str

    severity: Severity
    title: str
    description: str

    # 탐지 정보
    detection_score: DetectionScore

    # 상태
    is_acknowledged: bool = False
    is_resolved: bool = False

    # 메타데이터
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            object.__setattr__(self, 'tags', [])
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


# shared/domain/__init__.py
from .metric import Metric, MetricType
from .features import Features
from .detection import DetectionScore, Alert, Severity

__all__ = [
    "Metric",
    "MetricType",
    "Features",
    "DetectionScore",
    "Alert",
    "Severity",
]
```

### 1.3 이벤트 정의

```python
# shared/events/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict
import uuid
import json

@dataclass
class Event(ABC):
    """Base Event"""

    event_id: str = None
    event_type: str = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.event_id is None:
            object.__setattr__(self, 'event_id', str(uuid.uuid4()))
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.utcnow())
        if self.event_type is None:
            object.__setattr__(self, 'event_type', self.__class__.__name__)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict())

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """딕셔너리에서 생성"""
        pass

    @property
    def topic(self) -> str:
        """이벤트 토픽 (Kafka topic)"""
        return f"{self.event_type.lower().replace('_', '.')}"


# shared/events/metric_events.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any
from .base import Event
from ..domain import Metric

@dataclass
class MetricCollected(Event):
    """메트릭 수집 이벤트"""

    metric: Metric = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data['metric'] = self.metric.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricCollected':
        return cls(
            event_id=data['event_id'],
            event_type=data['event_type'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metric=Metric.from_dict(data['metric']),
        )


@dataclass
class FeaturesExtracted(Event):
    """피처 추출 이벤트"""

    features: Dict[str, Any] = None  # Features를 dict로 직렬화

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeaturesExtracted':
        return cls(
            event_id=data['event_id'],
            event_type=data['event_type'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            features=data['features'],
        )


@dataclass
class AnomalyDetected(Event):
    """이상 탐지 이벤트"""

    detection: Dict[str, Any] = None  # DetectionScore를 dict로 직렬화

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnomalyDetected':
        return cls(
            event_id=data['event_id'],
            event_type=data['event_type'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            detection=data['detection'],
        )


@dataclass
class AlertCreated(Event):
    """알림 생성 이벤트"""

    alert: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertCreated':
        return cls(
            event_id=data['event_id'],
            event_type=data['event_type'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            alert=data['alert'],
        )
```

### 1.4 메시지 버스 추상화

```python
# shared/messaging/base.py
from abc import ABC, abstractmethod
from typing import Callable, Any
from ..events.base import Event

class MessageBroker(ABC):
    """메시지 브로커 인터페이스"""

    @abstractmethod
    async def connect(self):
        """연결"""
        pass

    @abstractmethod
    async def disconnect(self):
        """연결 해제"""
        pass

    @abstractmethod
    async def publish(self, topic: str, message: str):
        """메시지 발행"""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[str], Any]):
        """메시지 구독"""
        pass


# shared/messaging/kafka_broker.py
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from typing import Callable, Any
import asyncio
import logging
from .base import MessageBroker

logger = logging.getLogger(__name__)

class KafkaBroker(MessageBroker):
    """Kafka 브로커 구현"""

    def __init__(self, bootstrap_servers: str):
        self.bootstrap_servers = bootstrap_servers
        self.producer: AIOKafkaProducer = None
        self.consumers: dict = {}

    async def connect(self):
        """Kafka 연결"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: v.encode('utf-8')
        )
        await self.producer.start()
        logger.info(f"Kafka producer connected: {self.bootstrap_servers}")

    async def disconnect(self):
        """Kafka 연결 해제"""
        if self.producer:
            await self.producer.stop()

        for consumer in self.consumers.values():
            await consumer.stop()

        logger.info("Kafka disconnected")

    async def publish(self, topic: str, message: str):
        """메시지 발행"""
        if not self.producer:
            raise RuntimeError("Producer not connected")

        await self.producer.send_and_wait(topic, message)
        logger.debug(f"Published to {topic}: {message[:100]}...")

    async def subscribe(self, topic: str, handler: Callable[[str], Any]):
        """메시지 구독"""
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=f"padit-{topic}-consumer",
            value_deserializer=lambda v: v.decode('utf-8')
        )

        await consumer.start()
        self.consumers[topic] = consumer

        logger.info(f"Subscribed to topic: {topic}")

        # 백그라운드에서 메시지 소비
        asyncio.create_task(self._consume(consumer, handler))

    async def _consume(self, consumer: AIOKafkaConsumer, handler: Callable):
        """메시지 소비 (백그라운드)"""
        try:
            async for message in consumer:
                try:
                    await handler(message.value)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
        except Exception as e:
            logger.error(f"Consumer error: {e}")


# shared/messaging/redis_broker.py (간단한 대안)
import aioredis
from typing import Callable, Any
import asyncio
import logging
from .base import MessageBroker

logger = logging.getLogger(__name__)

class RedisBroker(MessageBroker):
    """Redis Streams 브로커 구현 (경량 대안)"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: aioredis.Redis = None
        self.running = True

    async def connect(self):
        """Redis 연결"""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        logger.info(f"Redis connected: {self.redis_url}")

    async def disconnect(self):
        """Redis 연결 해제"""
        self.running = False
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
        logger.info("Redis disconnected")

    async def publish(self, topic: str, message: str):
        """메시지 발행 (Redis Streams XADD)"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        await self.redis.xadd(topic, {'data': message})
        logger.debug(f"Published to {topic}")

    async def subscribe(self, topic: str, handler: Callable[[str], Any]):
        """메시지 구독 (Redis Streams XREAD)"""
        if not self.redis:
            raise RuntimeError("Redis not connected")

        logger.info(f"Subscribed to topic: {topic}")

        # 백그라운드에서 메시지 소비
        asyncio.create_task(self._consume(topic, handler))

    async def _consume(self, topic: str, handler: Callable):
        """메시지 소비 (폴링)"""
        last_id = '0-0'

        while self.running:
            try:
                messages = await self.redis.xread(
                    [topic],
                    latest_ids=[last_id],
                    timeout=1000
                )

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        last_id = msg_id
                        message = fields[b'data'].decode('utf-8')

                        try:
                            await handler(message)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")

            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(1)


# shared/messaging/event_bus.py
from typing import Callable, Dict, List
import asyncio
import logging
from .base import MessageBroker
from ..events.base import Event

logger = logging.getLogger(__name__)

class EventBus:
    """이벤트 버스 (Facade)"""

    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.handlers: Dict[str, List[Callable]] = {}

    async def start(self):
        """시작"""
        await self.broker.connect()
        logger.info("EventBus started")

    async def stop(self):
        """종료"""
        await self.broker.disconnect()
        logger.info("EventBus stopped")

    async def publish(self, event: Event):
        """이벤트 발행"""
        topic = event.topic
        message = event.to_json()
        await self.broker.publish(topic, message)
        logger.debug(f"Event published: {event.event_type}")

    def subscribe(self, event_type: str, handler: Callable):
        """이벤트 구독 (데코레이터로 사용 가능)"""
        def decorator(func: Callable):
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(func)
            logger.info(f"Handler registered: {event_type} -> {func.__name__}")
            return func

        if callable(handler):
            # 직접 호출: event_bus.subscribe("metric.collected", my_handler)
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            return handler
        else:
            # 데코레이터: @event_bus.subscribe("metric.collected")
            return decorator

    async def start_consuming(self):
        """모든 구독된 이벤트 소비 시작"""
        for event_type, handlers in self.handlers.items():
            topic = event_type.lower().replace('_', '.')

            async def handle_message(message: str):
                for handler in handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")

            await self.broker.subscribe(topic, handle_message)
```

### 1.5 Docker Compose 설정

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Message Broker
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: 'CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT'
      KAFKA_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092'
      KAFKA_PROCESS_ROLES: 'broker,controller'
      KAFKA_CONTROLLER_QUORUM_VOTERS: '1@kafka:29093'
      KAFKA_LISTENERS: 'PLAINTEXT://kafka:29092,CONTROLLER://kafka:29093,PLAINTEXT_HOST://0.0.0.0:9092'
      KAFKA_INTER_BROKER_LISTENER_NAME: 'PLAINTEXT'
      KAFKA_CONTROLLER_LISTENER_NAMES: 'CONTROLLER'
      KAFKA_LOG_DIRS: '/tmp/kraft-combined-logs'
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
    volumes:
      - kafka_data:/var/lib/kafka/data

  # Time Series Database
  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: adminadmin
      DOCKER_INFLUXDB_INIT_ORG: padit
      DOCKER_INFLUXDB_INIT_BUCKET: metrics
      DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: my-super-secret-token
    volumes:
      - influxdb_data:/var/lib/influxdb2

  # PostgreSQL (Metadata)
  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: padit
      POSTGRES_PASSWORD: padit123
      POSTGRES_DB: padit
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Redis (Cache & Light messaging)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # MinIO (Object Storage for models)
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

  # Grafana (Visualization)
  grafana:
    image: grafana/grafana:10.0.0
    ports:
      - "3100:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_INSTALL_PLUGINS: grafana-influxdb-datasource
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  kafka_data:
  influxdb_data:
  postgres_data:
  redis_data:
  minio_data:
  grafana_data:
```

---

## 🔌 Phase 2: 첫 번째 프로토콜 어댑터 구현 (Week 3-4)

### 2.1 ORAN CFM 어댑터 (OCAD에서 마이그레이션)

```python
# plugins/protocol_adapters/oran_cfm/adapter.py
from typing import AsyncIterator, Dict, Any
from datetime import datetime
import asyncio

from padit.shared.domain import Metric, MetricType

class ORANCFMAdapter:
    """ORAN CFM 프로토콜 어댑터"""

    @property
    def name(self) -> str:
        return "oran-cfm"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "ORAN CFM-Lite protocol adapter (UDP Echo, eCPRI, LBM)"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """설정 검증"""
        required = ["endpoints"]
        return all(k in config for k in required)

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Metric]:
        """메트릭 수집"""
        endpoints = config["endpoints"]
        interval = config.get("interval_sec", 10)

        while True:
            for endpoint in endpoints:
                # UDP Echo 수집
                if endpoint.get("udp_echo", True):
                    rtt = await self._collect_udp_echo(endpoint)
                    yield Metric(
                        timestamp=datetime.utcnow(),
                        source_id=endpoint["id"],
                        metric_name="udp_echo_rtt",
                        value=rtt,
                        unit="ms",
                        metric_type=MetricType.LATENCY,
                        protocol="oran-cfm",
                        metadata={"endpoint": endpoint["host"]}
                    )

                # eCPRI delay 수집
                if endpoint.get("ecpri", True):
                    delay = await self._collect_ecpri_delay(endpoint)
                    yield Metric(
                        timestamp=datetime.utcnow(),
                        source_id=endpoint["id"],
                        metric_name="ecpri_delay",
                        value=delay,
                        unit="us",
                        metric_type=MetricType.LATENCY,
                        protocol="oran-cfm",
                        metadata={"endpoint": endpoint["host"]}
                    )

                # LBM 수집
                if endpoint.get("lbm", True):
                    lbm_rtt = await self._collect_lbm(endpoint)
                    yield Metric(
                        timestamp=datetime.utcnow(),
                        source_id=endpoint["id"],
                        metric_name="lbm_rtt",
                        value=lbm_rtt,
                        unit="ms",
                        metric_type=MetricType.LATENCY,
                        protocol="oran-cfm",
                        metadata={"endpoint": endpoint["host"]}
                    )

            await asyncio.sleep(interval)

    async def _collect_udp_echo(self, endpoint: Dict) -> float:
        """UDP Echo RTT 측정"""
        # OCAD 코드에서 마이그레이션
        # ocad/collectors/udp_echo.py 로직 사용
        pass

    async def _collect_ecpri_delay(self, endpoint: Dict) -> float:
        """eCPRI delay 측정"""
        pass

    async def _collect_lbm(self, endpoint: Dict) -> float:
        """LBM RTT 측정"""
        pass


# plugins/protocol_adapters/oran_cfm/__init__.py
from .adapter import ORANCFMAdapter

def create_adapter():
    """플러그인 엔트리포인트"""
    return ORANCFMAdapter()
```

### 2.2 HTTP 어댑터 (범용성 검증)

```python
# plugins/protocol_adapters/http/adapter.py
from typing import AsyncIterator, Dict, Any
from datetime import datetime
import asyncio
import aiohttp
import time

from padit.shared.domain import Metric, MetricType

class HTTPAdapter:
    """HTTP/HTTPS 프로토콜 어댑터"""

    @property
    def name(self) -> str:
        return "http"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        required = ["endpoints"]
        return all(k in config for k in required)

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Metric]:
        """HTTP 엔드포인트 모니터링"""
        endpoints = config["endpoints"]
        interval = config.get("interval_sec", 30)
        timeout = config.get("timeout_sec", 10)

        async with aiohttp.ClientSession() as session:
            while True:
                for endpoint in endpoints:
                    url = endpoint["url"]
                    method = endpoint.get("method", "GET")

                    # Response time 측정
                    start = time.time()
                    try:
                        async with session.request(
                            method, url, timeout=timeout
                        ) as response:
                            elapsed_ms = (time.time() - start) * 1000
                            status = response.status

                            # Latency metric
                            yield Metric(
                                timestamp=datetime.utcnow(),
                                source_id=endpoint.get("id", url),
                                metric_name="http_response_time",
                                value=elapsed_ms,
                                unit="ms",
                                metric_type=MetricType.LATENCY,
                                protocol="http",
                                metadata={
                                    "url": url,
                                    "method": method,
                                    "status_code": status
                                }
                            )

                            # Availability metric
                            is_success = 200 <= status < 300
                            yield Metric(
                                timestamp=datetime.utcnow(),
                                source_id=endpoint.get("id", url),
                                metric_name="http_availability",
                                value=1.0 if is_success else 0.0,
                                unit="%",
                                metric_type=MetricType.AVAILABILITY,
                                protocol="http",
                                metadata={
                                    "url": url,
                                    "status_code": status
                                }
                            )

                    except asyncio.TimeoutError:
                        # Timeout
                        yield Metric(
                            timestamp=datetime.utcnow(),
                            source_id=endpoint.get("id", url),
                            metric_name="http_availability",
                            value=0.0,
                            unit="%",
                            metric_type=MetricType.AVAILABILITY,
                            protocol="http",
                            metadata={
                                "url": url,
                                "error": "timeout"
                            }
                        )

                    except Exception as e:
                        # Error
                        yield Metric(
                            timestamp=datetime.utcnow(),
                            source_id=endpoint.get("id", url),
                            metric_name="http_availability",
                            value=0.0,
                            unit="%",
                            metric_type=MetricType.AVAILABILITY,
                            protocol="http",
                            metadata={
                                "url": url,
                                "error": str(e)
                            }
                        )

                await asyncio.sleep(interval)


# plugins/protocol_adapters/http/__init__.py
from .adapter import HTTPAdapter

def create_adapter():
    return HTTPAdapter()
```

### 2.3 플러그인 레지스트리

```python
# services/ingestion/plugin_registry.py
from typing import Dict, Optional
from pathlib import Path
import importlib.util
import logging

logger = logging.getLogger(__name__)

class PluginRegistry:
    """프로토콜 어댑터 플러그인 레지스트리"""

    def __init__(self):
        self.adapters: Dict[str, Any] = {}

    def register(self, adapter):
        """어댑터 등록"""
        name = adapter.name
        if name in self.adapters:
            logger.warning(f"Adapter {name} already registered, overwriting")

        self.adapters[name] = adapter
        logger.info(f"Adapter registered: {name} v{adapter.version}")

    def get(self, name: str) -> Optional[Any]:
        """어댑터 조회"""
        return self.adapters.get(name)

    def list(self) -> Dict[str, Any]:
        """모든 어댑터 조회"""
        return {
            name: {
                "version": adapter.version,
                "description": getattr(adapter, "description", "")
            }
            for name, adapter in self.adapters.items()
        }

    def discover_plugins(self, plugin_dir: Path):
        """플러그인 자동 발견 및 등록"""
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return

        # protocol_adapters 디렉토리 탐색
        adapters_dir = plugin_dir / "protocol_adapters"
        if not adapters_dir.exists():
            return

        for adapter_dir in adapters_dir.iterdir():
            if not adapter_dir.is_dir():
                continue

            # __init__.py에서 create_adapter 함수 찾기
            init_file = adapter_dir / "__init__.py"
            if not init_file.exists():
                continue

            try:
                # 동적 import
                spec = importlib.util.spec_from_file_location(
                    f"plugin_{adapter_dir.name}",
                    init_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # create_adapter() 호출
                if hasattr(module, "create_adapter"):
                    adapter = module.create_adapter()
                    self.register(adapter)
                else:
                    logger.warning(
                        f"Plugin {adapter_dir.name} missing create_adapter()"
                    )

            except Exception as e:
                logger.error(f"Failed to load plugin {adapter_dir.name}: {e}")


# 전역 레지스트리
registry = PluginRegistry()
```

계속해서 더 구현할까요? 다음 내용:
- Ingestion Service 구현
- Feature Engineering Service
- Detection Service
- 전체 연동 예제

원하시는 방향을 알려주시면 계속 진행하겠습니다!