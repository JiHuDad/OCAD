# 범용 프로토콜 이상탐지 플랫폼 설계안

**프로젝트명**: **PADIT** (Protocol Anomaly Detection & Intelligence Toolkit)
**기반 프로젝트**: OCAD (ORAN CFM-Lite AI Anomaly Detection System)
**목표**: CFM-ORAN을 넘어 모든 프로토콜의 이상탐지를 가능케 하는 플랫폼

---

## 🎯 핵심 개념

### 현재 (OCAD)
```
ORAN 특화 시스템
├── CFM 프로토콜 전용
├── UDP Echo, eCPRI, LBM 하드코딩
└── 단일 도메인 (통신 네트워크)
```

### 목표 (PADIT)
```
범용 프로토콜 이상탐지 플랫폼
├── 프로토콜 Agnostic (HTTP, MQTT, Modbus, CAN, ...)
├── Plugin 아키텍처 (새 프로토콜 추가 용이)
├── Multi-domain (통신, IoT, 산업제어, 금융, ...)
└── 학습-추론 분리 + AutoML
```

---

## 🏗️ 아키텍처 설계

### 1. 전체 아키텍처 (Hexagonal + Microservices + Event-Driven)

```
┌──────────────────────────────────────────────────────────────────┐
│                     PADIT Platform                                │
│                  (Protocol Anomaly Detection)                     │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Ingestion Layer (Input Adapters)               │
├─────────────────────────────────────────────────────────────────┤
│  Protocol Adapters (Pluggable)                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ ORAN-CFM │ │   HTTP   │ │   MQTT   │ │  Modbus  │ ...      │
│  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  - NETCONF/YANG  - REST API    - Pub/Sub    - TCP/RTU         │
│  - UDP Echo      - gRPC        - QoS         - Industrial      │
│  - eCPRI         - GraphQL                                      │
├─────────────────────────────────────────────────────────────────┤
│  Data Normalization Layer                                       │
│  ┌────────────────────────────────────────────────────┐        │
│  │ Protocol-agnostic metric format                    │        │
│  │ {timestamp, source, metric_name, value, metadata}  │        │
│  └────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Message Bus (Event-Driven)                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│  │  Kafka   │ │ RabbitMQ │ │  Redis   │  (선택 가능)          │
│  │ Streams  │ │  AMQP    │ │ Streams  │                       │
│  └──────────┘ └──────────┘ └──────────┘                       │
│  Topics: raw-metrics, features, detections, alerts             │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Engine (Microservices)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────┐        │
│  │  Feature Engineering Service                       │        │
│  │  - Streaming feature extraction (Flink/Spark)      │        │
│  │  - Time windows, aggregations, statistics          │        │
│  │  - Feature store (Feast/Tecton)                    │        │
│  └────────────────────────────────────────────────────┘        │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────┐        │
│  │  Detection Service                                 │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐          │        │
│  │  │  Rule    │ │Changepoint│ │Residual │          │        │
│  │  │ Engine   │ │ Detector  │ │Detector │          │        │
│  │  └──────────┘ └──────────┘ └──────────┘          │        │
│  │  - Pluggable detector interface                   │        │
│  │  - Model registry (MLflow/BentoML)                │        │
│  │  - A/B testing, canary deployment                 │        │
│  └────────────────────────────────────────────────────┘        │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────┐        │
│  │  Alert Management Service                          │        │
│  │  - Evidence-based filtering                        │        │
│  │  - Correlation engine                              │        │
│  │  - Notification routing (Email/Slack/PagerDuty)    │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Training Platform (MLOps)                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐        │
│  │  AutoML Pipeline (Kubeflow/MLflow)                 │        │
│  │  - Automated feature selection                     │        │
│  │  - Hyperparameter tuning (Optuna/Ray Tune)         │        │
│  │  - Model selection (TCN/LSTM/Transformer/...)      │        │
│  │  - Distributed training (Ray/Dask)                 │        │
│  └────────────────────────────────────────────────────┘        │
│  ┌────────────────────────────────────────────────────┐        │
│  │  Model Registry & Versioning                       │        │
│  │  - MLflow Model Registry                           │        │
│  │  - Model lineage tracking                          │        │
│  │  - A/B testing experiments                         │        │
│  └────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Storage Layer (Polyglot Persistence)          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │TimeSeries│ │Document  │ │  Graph   │ │  Object  │          │
│  │InfluxDB  │ │MongoDB   │ │  Neo4j   │ │  MinIO   │          │
│  │Prometheus│ │Cassandra │ │  (관계)   │ │ (모델)   │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  - 메트릭      - 메타데이터  - 엔티티관계   - 학습데이터      │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐     │
│  │  Web UI        │ │  REST API      │ │  GraphQL API   │     │
│  │  React/Vue     │ │  FastAPI       │ │  (Flexible)    │     │
│  │  + Grafana     │ │  + OpenAPI     │ │                │     │
│  └────────────────┘ └────────────────┘ └────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 핵심 설계 패턴

### 1. Hexagonal Architecture (Ports & Adapters)

```python
# 핵심 도메인 (프로토콜 독립적)
class AnomalyDetectionService:
    def __init__(self, detector_port: DetectorPort):
        self.detector = detector_port

    def detect(self, features: Features) -> Detection:
        return self.detector.detect(features)

# Port (인터페이스)
class DetectorPort(ABC):
    @abstractmethod
    def detect(self, features: Features) -> Detection:
        pass

# Adapters (구현체)
class TCNDetectorAdapter(DetectorPort):
    def detect(self, features: Features) -> Detection:
        # TCN 모델로 탐지
        pass

class RuleBasedDetectorAdapter(DetectorPort):
    def detect(self, features: Features) -> Detection:
        # 룰 기반 탐지
        pass
```

**장점**:
- 프로토콜 어댑터 추가/제거 용이
- 핵심 비즈니스 로직과 인프라 분리
- 테스트 용이성

### 2. Plugin Architecture (동적 로딩)

```python
# Plugin 인터페이스
class ProtocolAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """프로토콜 이름"""
        pass

    @abstractmethod
    def collect(self, config: dict) -> Iterator[Metric]:
        """메트릭 수집"""
        pass

    @abstractmethod
    def normalize(self, raw_data: bytes) -> Metric:
        """정규화"""
        pass

# Plugin 레지스트리
class PluginRegistry:
    def __init__(self):
        self.adapters: Dict[str, ProtocolAdapter] = {}

    def register(self, adapter: ProtocolAdapter):
        self.adapters[adapter.name] = adapter

    def discover_plugins(self, plugin_dir: Path):
        """플러그인 자동 발견"""
        for module in plugin_dir.glob("*_adapter.py"):
            # 동적 import
            spec = importlib.util.spec_from_file_location(module.stem, module)
            plugin = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin)

            # 플러그인 등록
            adapter = plugin.create_adapter()
            self.register(adapter)

# 사용 예제
registry = PluginRegistry()
registry.discover_plugins(Path("plugins/"))
oran_adapter = registry.get("oran-cfm")
http_adapter = registry.get("http")
```

**디렉토리 구조**:
```
padit/
├── core/                    # 핵심 도메인
├── plugins/                 # 플러그인
│   ├── oran_cfm_adapter.py
│   ├── http_adapter.py
│   ├── mqtt_adapter.py
│   └── modbus_adapter.py
└── adapters/                # 외부 통합
```

### 3. Event-Driven Architecture (CQRS + Event Sourcing)

```python
# Event
@dataclass
class MetricCollected(Event):
    metric_id: str
    timestamp: datetime
    source: str
    value: float
    metadata: dict

@dataclass
class AnomalyDetected(Event):
    detection_id: str
    timestamp: datetime
    score: float
    evidence: dict

# Event Bus
class EventBus:
    def __init__(self, broker: MessageBroker):
        self.broker = broker

    def publish(self, event: Event):
        self.broker.send(event.topic, event.to_json())

    def subscribe(self, topic: str, handler: Callable):
        self.broker.consume(topic, handler)

# Event Handlers (느슨한 결합)
class FeatureEngineeringHandler:
    def handle(self, event: MetricCollected):
        features = self.extract_features(event)
        self.event_bus.publish(FeaturesExtracted(features))

class DetectionHandler:
    def handle(self, event: FeaturesExtracted):
        detection = self.detector.detect(event.features)
        if detection.is_anomaly:
            self.event_bus.publish(AnomalyDetected(detection))

class AlertHandler:
    def handle(self, event: AnomalyDetected):
        alert = self.alert_manager.create_alert(event)
        self.notification_service.send(alert)
```

**장점**:
- 서비스 간 느슨한 결합
- 확장성 (서비스 독립 스케일링)
- 이벤트 재생 가능 (디버깅/재학습)

### 4. CQRS (Command Query Responsibility Segregation)

```python
# Command (쓰기)
class TrainModelCommand:
    model_type: str
    training_data: Path
    hyperparameters: dict

class CommandHandler:
    def handle(self, cmd: TrainModelCommand):
        # 학습 실행
        model = self.trainer.train(
            cmd.model_type,
            cmd.training_data,
            cmd.hyperparameters
        )
        # 이벤트 발행
        self.event_bus.publish(ModelTrained(model.id))

# Query (읽기)
class GetAnomalyScoreQuery:
    endpoint_id: str
    time_range: TimeRange

class QueryHandler:
    def handle(self, query: GetAnomalyScoreQuery):
        # 읽기 전용 저장소에서 조회 (최적화된 인덱스)
        return self.read_store.get_scores(
            query.endpoint_id,
            query.time_range
        )
```

**장점**:
- 읽기/쓰기 최적화 분리
- 복잡한 쿼리 성능 향상
- 이벤트 소싱과 조합 가능

### 5. Strategy Pattern (탐지 알고리즘)

```python
class DetectionStrategy(ABC):
    @abstractmethod
    def detect(self, features: Features) -> DetectionScore:
        pass

class RuleBasedStrategy(DetectionStrategy):
    def detect(self, features: Features) -> DetectionScore:
        # 룰 기반 로직
        pass

class MLBasedStrategy(DetectionStrategy):
    def __init__(self, model: Model):
        self.model = model

    def detect(self, features: Features) -> DetectionScore:
        # ML 모델 추론
        pass

class CompositeStrategy(DetectionStrategy):
    def __init__(self, strategies: List[DetectionStrategy], weights: List[float]):
        self.strategies = strategies
        self.weights = weights

    def detect(self, features: Features) -> DetectionScore:
        scores = [s.detect(features) for s in self.strategies]
        return self.combine(scores, self.weights)

# 사용
detector = CompositeStrategy([
    RuleBasedStrategy(),
    MLBasedStrategy(tcn_model),
    MLBasedStrategy(lstm_model),
], weights=[0.3, 0.4, 0.3])
```

### 6. Circuit Breaker Pattern (장애 격리)

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None

    def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

# 사용
circuit_breaker = CircuitBreaker()

def collect_metrics():
    return circuit_breaker.call(oran_adapter.collect)
```

---

## 🚀 기술 스택

### Backend
```yaml
Core:
  - Python 3.11+ (Type hints, asyncio)
  - FastAPI (API framework)
  - Pydantic v2 (Validation)

Message Bus:
  - Kafka (High throughput, event sourcing)
  - Redis Streams (Low latency, simple setup)

Feature Engineering:
  - Apache Flink (Streaming)
  - Pandas/Polars (Batch)
  - Feast (Feature store)

ML/AI:
  - PyTorch (Deep learning)
  - Scikit-learn (Classical ML)
  - MLflow (Experiment tracking, model registry)
  - Optuna (Hyperparameter tuning)
  - Ray (Distributed training)

Storage:
  - InfluxDB (Time series metrics)
  - PostgreSQL (Metadata, configuration)
  - MongoDB (Flexible schema)
  - MinIO (Object storage for models)
  - Neo4j (Entity relationships - optional)

Observability:
  - Prometheus (Metrics)
  - Jaeger (Distributed tracing)
  - ELK Stack (Logging)
  - Grafana (Visualization)
```

### Frontend
```yaml
Web UI:
  - React 18+ (or Vue 3)
  - TypeScript
  - TanStack Query (Data fetching)
  - Zustand (State management)
  - Recharts/D3.js (Visualization)
  - Tailwind CSS (Styling)

Dashboarding:
  - Grafana (Built-in support)
  - Customizable panels
```

### Infrastructure
```yaml
Containerization:
  - Docker
  - Docker Compose (Development)

Orchestration:
  - Kubernetes (Production)
  - Helm Charts (Package management)

CI/CD:
  - GitHub Actions
  - ArgoCD (GitOps)

IaC:
  - Terraform (Cloud resources)
  - Ansible (Configuration)
```

---

## 📦 프로젝트 구조

```
padit/                                    # Protocol Anomaly Detection & Intelligence Toolkit
├── README.md
├── ARCHITECTURE.md
├── docker-compose.yml
├── kubernetes/                           # K8s manifests
│   ├── base/
│   └── overlays/
│
├── services/                             # Microservices
│   ├── ingestion/                        # 수집 서비스
│   │   ├── adapters/                     # 프로토콜 어댑터
│   │   │   ├── oran_cfm/
│   │   │   ├── http/
│   │   │   ├── mqtt/
│   │   │   └── modbus/
│   │   ├── normalizer.py
│   │   └── main.py
│   │
│   ├── feature_engineering/              # 피처 엔지니어링
│   │   ├── streaming.py (Flink job)
│   │   ├── batch.py
│   │   └── feature_store.py (Feast)
│   │
│   ├── detection/                        # 탐지 서비스
│   │   ├── detectors/
│   │   │   ├── rule_based.py
│   │   │   ├── changepoint.py
│   │   │   ├── residual.py
│   │   │   └── multivariate.py
│   │   ├── model_registry.py (MLflow)
│   │   └── main.py
│   │
│   ├── alert/                            # 알림 서비스
│   │   ├── correlation.py
│   │   ├── notification.py
│   │   └── main.py
│   │
│   └── api/                              # API Gateway
│       ├── rest/
│       ├── graphql/
│       └── websocket/
│
├── training/                             # MLOps 플랫폼
│   ├── pipelines/                        # Kubeflow pipelines
│   │   ├── automl.py
│   │   ├── feature_selection.py
│   │   └── model_training.py
│   ├── experiments/                      # MLflow experiments
│   └── models/                           # Model definitions
│       ├── tcn.py
│       ├── lstm.py
│       └── transformer.py
│
├── shared/                               # 공유 라이브러리
│   ├── domain/                           # 도메인 모델
│   │   ├── metric.py
│   │   ├── feature.py
│   │   ├── detection.py
│   │   └── alert.py
│   ├── events/                           # 이벤트 정의
│   ├── messaging/                        # 메시지 버스 추상화
│   └── storage/                          # 저장소 추상화
│
├── web/                                  # Frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   └── api/
│   └── package.json
│
├── plugins/                              # 외부 플러그인
│   ├── protocol_adapters/
│   ├── detectors/
│   └── notifiers/
│
├── config/                               # 설정
│   ├── base.yaml
│   ├── dev.yaml
│   └── prod.yaml
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── docs/
    ├── architecture/
    ├── api/
    ├── deployment/
    └── plugin-development/
```

---

## 🔌 플러그인 개발 가이드

### 새 프로토콜 어댑터 추가

```python
# plugins/protocol_adapters/my_protocol_adapter.py

from padit.shared.domain import Metric
from padit.services.ingestion.adapters.base import ProtocolAdapter

class MyProtocolAdapter(ProtocolAdapter):
    """My Protocol 어댑터 예제"""

    @property
    def name(self) -> str:
        return "my-protocol"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate_config(self, config: dict) -> bool:
        """설정 검증"""
        required = ["host", "port"]
        return all(k in config for k in required)

    async def collect(self, config: dict) -> AsyncIterator[Metric]:
        """메트릭 수집"""
        client = MyProtocolClient(config["host"], config["port"])

        async for raw_data in client.stream():
            metric = self.normalize(raw_data)
            yield metric

    def normalize(self, raw_data: bytes) -> Metric:
        """프로토콜 독립적 형식으로 정규화"""
        parsed = self.parse(raw_data)

        return Metric(
            timestamp=parsed.timestamp,
            source=parsed.source_id,
            metric_name=parsed.metric_type,
            value=parsed.value,
            metadata={
                "protocol": "my-protocol",
                "raw": parsed.to_dict()
            }
        )

# 플러그인 등록
def create_adapter() -> ProtocolAdapter:
    return MyProtocolAdapter()
```

### 새 탐지기 추가

```python
# plugins/detectors/my_detector.py

from padit.shared.domain import Features, DetectionScore
from padit.services.detection.base import DetectorPlugin

class MyDetector(DetectorPlugin):
    """커스텀 탐지 알고리즘"""

    @property
    def name(self) -> str:
        return "my-detector"

    def train(self, training_data: Dataset) -> Model:
        """모델 학습"""
        # 커스텀 학습 로직
        pass

    def detect(self, features: Features) -> DetectionScore:
        """이상 탐지"""
        # 커스텀 탐지 로직
        score = self.model.predict(features)

        return DetectionScore(
            timestamp=features.timestamp,
            score=score,
            confidence=self.calculate_confidence(features),
            evidence=self.extract_evidence(features)
        )
```

---

## 🎯 OCAD에서 PADIT로 마이그레이션

### Phase 1: Core 추출
```bash
# OCAD의 핵심 로직을 PADIT core로 이동
ocad/detectors/      → padit/services/detection/detectors/
ocad/features/       → padit/services/feature_engineering/
ocad/alerts/         → padit/services/alert/
ocad/core/models.py  → padit/shared/domain/
```

### Phase 2: ORAN 어댑터화
```bash
# ORAN 특화 로직을 플러그인으로 변환
ocad/collectors/     → padit/plugins/protocol_adapters/oran_cfm/
ocad/capability/     → padit/plugins/protocol_adapters/oran_cfm/capability.py
```

### Phase 3: 아키텍처 전환
```python
# Before (OCAD - Monolithic)
class SystemOrchestrator:
    def __init__(self):
        self.collectors = [UDPEchoCollector(), eCPRICollector()]
        self.detector = CompositeDetector()
        self.alert_manager = AlertManager()

    def run(self):
        while True:
            metrics = self.collect()
            features = self.extract_features(metrics)
            detection = self.detector.detect(features)
            if detection.is_anomaly:
                self.alert_manager.alert(detection)

# After (PADIT - Event-Driven Microservices)
# Ingestion Service
async def ingestion_service():
    adapter = registry.get_adapter("oran-cfm")
    async for metric in adapter.collect(config):
        await event_bus.publish(MetricCollected(metric))

# Feature Engineering Service
@event_handler("metric.collected")
async def handle_metric_collected(event: MetricCollected):
    features = await feature_engine.extract(event.metric)
    await event_bus.publish(FeaturesExtracted(features))

# Detection Service
@event_handler("features.extracted")
async def handle_features_extracted(event: FeaturesExtracted):
    detection = await detector.detect(event.features)
    if detection.is_anomaly:
        await event_bus.publish(AnomalyDetected(detection))
```

---

## 📊 배포 시나리오

### Development (로컬)
```bash
# Docker Compose로 전체 스택 실행
docker-compose up

# 접속
http://localhost:8080  # API
http://localhost:3000  # Web UI
http://localhost:3100  # Grafana
```

### Production (Kubernetes)
```yaml
# Helm chart 설치
helm install padit ./helm/padit \
  --namespace padit \
  --create-namespace \
  --values values-prod.yaml

# 서비스별 스케일링
kubectl scale deployment ingestion --replicas=5
kubectl scale deployment detection --replicas=10
```

### Multi-tenant (SaaS)
```
tenant-a.padit.io  → Namespace: padit-tenant-a
tenant-b.padit.io  → Namespace: padit-tenant-b

각 테넌트별 격리:
- 데이터베이스 분리
- 리소스 쿼터
- RBAC 정책
```

---

## 🎓 학습 곡선 & 문서

### 개발자 온보딩
1. **Quick Start** (30분) - Docker Compose로 로컬 실행
2. **Architecture Overview** (1시간) - 전체 아키텍처 이해
3. **Plugin Development** (2시간) - 첫 번째 어댑터 작성
4. **Core Contribution** (1일) - 핵심 로직 기여

### 사용자 온보딩
1. **Installation** (10분) - Helm chart 설치
2. **First Protocol** (20분) - ORAN 어댑터 설정
3. **Training** (30분) - 첫 번째 모델 학습
4. **Inference** (10분) - 실시간 탐지 시작

---

## 💡 핵심 차별화 요소

### 1. Protocol Agnostic
- 통신, IoT, 산업제어, 금융 등 모든 도메인
- 플러그인으로 새 프로토콜 추가 (코어 수정 불필요)

### 2. Production-Ready
- Kubernetes-native
- High availability (HA)
- Auto-scaling
- Observability (Prometheus, Jaeger, ELK)

### 3. MLOps Built-in
- AutoML (자동 모델 선택, 하이퍼파라미터 튜닝)
- Model versioning & registry
- A/B testing
- Continuous training

### 4. Developer-Friendly
- 명확한 플러그인 인터페이스
- 풍부한 문서 및 예제
- CLI 도구 (padit-cli)
- Local development 용이 (Docker Compose)

### 5. Enterprise Features
- Multi-tenancy
- RBAC (Role-Based Access Control)
- Audit logging
- SLA monitoring

---

## 🚦 다음 단계

### Immediate (새 Git 프로젝트 생성)
```bash
# 1. 새 프로젝트 생성
git init padit
cd padit

# 2. 기본 구조 생성
mkdir -p services/{ingestion,feature_engineering,detection,alert,api}
mkdir -p shared/{domain,events,messaging,storage}
mkdir -p training/{pipelines,experiments,models}
mkdir -p web plugins config tests docs

# 3. OCAD 코드 마이그레이션 시작
# - Core 로직 추출
# - ORAN 어댑터 플러그인화
# - 아키텍처 전환 (Monolithic → Microservices)
```

### Short-term (1-2개월)
- [ ] 핵심 도메인 모델 정의
- [ ] 이벤트 버스 구현 (Kafka)
- [ ] ORAN 어댑터 플러그인 완성
- [ ] HTTP 어댑터 구현 (범용성 검증)
- [ ] 기본 Web UI

### Mid-term (3-6개월)
- [ ] AutoML 파이프라인 (Kubeflow)
- [ ] Feature store (Feast)
- [ ] MQTT, Modbus 어댑터
- [ ] Multi-tenancy 지원
- [ ] 상용 문서 완성

### Long-term (6-12개월)
- [ ] Enterprise features (RBAC, Audit)
- [ ] Marketplace (플러그인 생태계)
- [ ] SaaS 버전 출시
- [ ] 커뮤니티 빌드업

---

## 📝 결론

### OCAD의 성공을 PADIT로 확장
- ✅ OCAD에서 검증된 탐지 알고리즘
- ✅ 학습-추론 분리 아키텍처
- ✅ 실전 경험 (ORAN 도메인)

### 최신 SW 아키텍처 적용
- ✅ Hexagonal Architecture (확장성)
- ✅ Microservices (독립 배포)
- ✅ Event-Driven (느슨한 결합)
- ✅ Plugin Architecture (플러그 & 플레이)
- ✅ MLOps (지속적 개선)

### 범용 플랫폼으로의 진화
- ✅ Protocol Agnostic
- ✅ Multi-domain
- ✅ Production-Ready
- ✅ Developer-Friendly

**다음 단계**: 새 Git 프로젝트 `padit` 생성 후 점진적 마이그레이션 시작

---

**작성자**: Claude Code
**작성일**: 2025-10-27
**버전**: 1.0.0 (Design Proposal)
