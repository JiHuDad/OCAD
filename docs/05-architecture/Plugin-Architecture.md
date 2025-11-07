# OCAD 플러그인 아키텍처

> **최종 업데이트**: 2025-11-05 (Phase 4)
> **대상 독자**: 시스템 아키텍트, 플러그인 개발자

## 개요

OCAD 플러그인 시스템은 **동적 플러그인 로딩**을 통해 프로토콜 어댑터와 탐지기를 런타임에 추가/제거할 수 있는 확장 가능한 아키텍처를 제공합니다.

## 설계 원칙

1. **확장성 (Extensibility)**: 새로운 프로토콜과 탐지 알고리즘을 코드 수정 없이 추가
2. **격리성 (Isolation)**: 각 플러그인은 독립적으로 동작하며 다른 플러그인에 영향을 주지 않음
3. **타입 안정성 (Type Safety)**: 추상 클래스를 통한 명확한 인터페이스 정의
4. **성능 (Performance)**: 비동기 처리 및 배치 최적화

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                      OCAD Core System                        │
│  ┌───────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ Orchestrator  │  │ Feature Engine │  │ Alert Manager │  │
│  └───────┬───────┘  └────────────────┘  └───────────────┘  │
└──────────┼──────────────────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Plugin Registry                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Dynamic Plugin Discovery & Loading                    │ │
│  │  - Scan plugin directories                             │ │
│  │  - Import Python modules                               │ │
│  │  - Instantiate plugin classes                          │ │
│  │  - Validate interfaces                                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────┐     ┌──────────────────────────┐  │
│  │ Protocol Adapters   │     │ Detector Plugins         │  │
│  │ Registry            │     │ Registry                 │  │
│  │                     │     │                          │  │
│  │ {name → adapter}    │     │ {name → detector}        │  │
│  │ {protocol → [adpts]}│     │ {protocol → [detectors]} │  │
│  └─────────────────────┘     └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
           │                              │
           ↓                              ↓
┌──────────────────────┐      ┌─────────────────────────────┐
│ Protocol Adapters    │      │ Detectors                   │
├──────────────────────┤      ├─────────────────────────────┤
│ - CFM Adapter        │      │ - LSTM Detector             │
│ - BFD Adapter        │      │ - HMM Detector              │
│ - BGP Adapter        │      │ - GNN Detector              │
│ - PTP Adapter        │      │ - TCN Detector              │
│ - Custom Adapters... │      │ - Custom Detectors...       │
└──────────────────────┘      └─────────────────────────────┘
```

## 핵심 컴포넌트

### 1. Plugin Base Interface

**파일**: `ocad/plugins/base.py`

두 가지 추상 클래스를 정의:

```python
class ProtocolAdapter(ABC):
    """프로토콜 어댑터 기본 인터페이스"""
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]: ...

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool: ...

    @abstractmethod
    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]: ...

    @abstractmethod
    def get_recommended_models(self) -> List[str]: ...


class DetectorPlugin(ABC):
    """탐지기 플러그인 기본 인터페이스"""
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    @abstractmethod
    def supported_protocols(self) -> List[str]: ...

    @abstractmethod
    def train(self, data: Any, **kwargs) -> None: ...

    @abstractmethod
    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def save_model(self, path: str) -> None: ...

    @abstractmethod
    def load_model(self, path: str) -> None: ...
```

**설계 장점**:
- 명확한 계약(Contract) 정의
- 타입 체킹 지원
- IDE 자동 완성
- 런타임 검증

### 2. Plugin Registry

**파일**: `ocad/plugins/registry.py`

**책임**:
- 플러그인 디렉토리 스캔
- Python 모듈 동적 임포트
- 플러그인 인스턴스 관리
- 쿼리 API 제공

**주요 메서드**:

```python
class PluginRegistry:
    def discover_plugins(self, plugin_dir: Path) -> None:
        """플러그인 디렉토리에서 모든 플러그인 발견 및 로드"""

    def get_protocol_adapter(self, name: str) -> Optional[ProtocolAdapter]:
        """이름으로 프로토콜 어댑터 조회"""

    def get_detector(self, name: str) -> Optional[DetectorPlugin]:
        """이름으로 탐지기 조회"""

    def get_detectors_for_protocol(self, protocol: str) -> List[DetectorPlugin]:
        """특정 프로토콜을 지원하는 모든 탐지기 조회"""

    def list_protocol_adapters(self) -> Dict[str, Dict]:
        """모든 프로토콜 어댑터 목록"""

    def list_detectors(self) -> Dict[str, Dict]:
        """모든 탐지기 목록"""
```

**동적 로딩 메커니즘**:

```python
def discover_plugins(self, plugin_dir: Path):
    # 1. protocol_adapters 디렉토리 스캔
    adapters_dir = plugin_dir / "protocol_adapters"
    for subdir in adapters_dir.iterdir():
        if subdir.is_dir() and (subdir / "__init__.py").exists():
            # 2. 동적 import
            module = importlib.import_module(f"ocad.plugins.protocol_adapters.{subdir.name}")

            # 3. 'adapter' 변수 찾기
            if hasattr(module, "adapter"):
                adapter = getattr(module, "adapter")

                # 4. 인터페이스 검증
                if isinstance(adapter, ProtocolAdapter):
                    # 5. 레지스트리에 등록
                    self._protocol_adapters[adapter.name] = adapter

    # 동일하게 detectors 디렉토리 스캔
    # ...
```

### 3. Plugin Implementation

**디렉토리 구조**:
```
ocad/plugins/
├── protocol_adapters/
│   └── bfd/
│       ├── __init__.py       # adapter = BFDAdapter()
│       ├── collector.py      # 메트릭 수집 로직
│       └── simulator.py      # 테스트용 시뮬레이터
└── detectors/
    └── lstm/
        ├── __init__.py       # detector = LSTMDetector()
        ├── model.py          # PyTorch 모델 정의
        └── trainer.py        # 학습 로직
```

**등록 규칙**:
1. `__init__.py`에 `adapter` 또는 `detector` 변수 정의
2. 해당 변수는 `ProtocolAdapter` 또는 `DetectorPlugin` 인스턴스여야 함
3. 레지스트리가 자동으로 스캔하여 등록

## 데이터 흐름

### 1. 메트릭 수집 흐름

```
1. Orchestrator
   ↓ (config)
2. ProtocolAdapter.collect()
   ↓ (async generator)
3. Yield metric dictionaries
   ↓
4. Feature Engine
   ↓ (features)
5. DetectorPlugin.detect()
   ↓ (anomaly score)
6. Alert Manager
```

**메트릭 딕셔너리 형식**:
```python
{
    "timestamp": datetime.utcnow(),
    "source_id": "bfd-session-1",
    "metric_name": "session_state",
    "value": 2.0,  # float
    "metadata": {
        "local_ip": "192.168.1.1",
        "remote_ip": "192.168.1.2",
        "protocol": "bfd"
    }
}
```

### 2. 탐지 흐름

```
1. Feature Engine extracts features
   ↓
2. For each compatible detector:
   ↓
3. detector.detect(features)
   ↓
4. Return anomaly score
   ↓
5. Composite scoring (weighted ensemble)
   ↓
6. Alert if score > threshold
```

**탐지 결과 형식**:
```python
{
    "score": 0.85,        # 0.0-1.0 (1.0 = highly anomalous)
    "is_anomaly": True,   # Binary classification
    "confidence": 0.92,   # Model confidence
    "detector_name": "lstm",
    "evidence": {         # Optional details
        "predicted_value": 5.2,
        "actual_value": 15.8,
        "deviation": 10.6
    }
}
```

## 확장성 메커니즘

### 1. 새 프로토콜 추가

**요구사항**:
- `ProtocolAdapter` 인터페이스 구현
- `ocad/plugins/protocol_adapters/<name>/` 디렉토리 생성
- `__init__.py`에 `adapter = MyAdapter()` 정의

**자동 통합**:
- 레지스트리가 자동 발견
- CLI에 자동 표시
- 기존 탐지기와 자동 연동

### 2. 새 탐지기 추가

**요구사항**:
- `DetectorPlugin` 인터페이스 구현
- `ocad/plugins/detectors/<name>/` 디렉토리 생성
- `__init__.py`에 `detector = MyDetector()` 정의

**자동 통합**:
- 지원 프로토콜에 자동 매칭
- 학습/추론 파이프라인 자동 연동

### 3. 의존성 관리

**플러그인별 의존성**:
```python
# ocad/plugins/detectors/lstm/__init__.py

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. LSTM detector disabled.")

if PYTORCH_AVAILABLE:
    class LSTMDetector(DetectorPlugin):
        # ...

    detector = LSTMDetector()
else:
    detector = None  # 레지스트리가 None 무시
```

## 성능 고려사항

### 1. 메모리 관리

**문제**: 여러 탐지기가 메모리를 많이 사용
**해결**:
```python
# config/plugins.yaml
global:
  model_cache_size: 50  # 최대 50개 모델 캐시
  model_auto_reload: false  # 메모리 부족 시 비활성화
```

### 2. 비동기 처리

**문제**: 동기 코드가 이벤트 루프 블로킹
**해결**:
```python
# ✅ Good: Async generator
async def collect(self, config) -> AsyncIterator[Dict]:
    while True:
        metrics = await self._fetch_metrics()  # Non-blocking
        for metric in metrics:
            yield metric
        await asyncio.sleep(interval)
```

### 3. 배치 처리

**문제**: 메트릭을 하나씩 처리하면 오버헤드 증가
**해결**:
```python
# config/plugins.yaml
protocol_adapters:
  bfd:
    config:
      batch_size: 100  # 100개씩 배치 처리
```

### 4. 타임아웃 관리

**문제**: 느린 플러그인이 전체 시스템 지연
**해결**:
```python
# config/plugins.yaml
global:
  collection_timeout_sec: 30  # 30초 타임아웃
  detector_timeout_sec: 5     # 5초 탐지 타임아웃
```

## 보안 고려사항

### 1. 플러그인 격리

**문제**: 악의적인 플러그인이 시스템 전체에 영향
**해결**:
- 플러그인은 별도 프로세스에서 실행 가능 (향후)
- 리소스 제한 (CPU, 메모리)

### 2. 설정 검증

**문제**: 잘못된 설정이 보안 취약점 발생
**해결**:
```python
def validate_config(self, config: Dict) -> bool:
    # IP 주소 화이트리스트 검증
    allowed_ips = ["192.168.0.0/16", "10.0.0.0/8"]

    # 포트 범위 검증
    if not (1024 <= config["port"] <= 65535):
        raise ValueError("Port must be in range 1024-65535")

    return True
```

### 3. 모델 무결성

**문제**: 변조된 모델 파일 로드
**해결**:
```python
import hashlib

def load_model(self, path: str):
    # 체크섬 검증
    with open(f"{path}.sha256") as f:
        expected_hash = f.read().strip()

    with open(path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    if actual_hash != expected_hash:
        raise ValueError("Model file integrity check failed")

    # 안전하게 로드
    model = torch.load(path)
```

## 테스트 전략

### 1. 단위 테스트

```python
# tests/plugins/test_bfd_adapter.py
@pytest.mark.asyncio
async def test_bfd_adapter_basic():
    adapter = BFDAdapter()

    # Test properties
    assert adapter.name == "bfd"
    assert len(adapter.supported_metrics) == 7

    # Test collection
    config = {...}
    collected = []
    async for metric in adapter.collect(config):
        collected.append(metric)
        if len(collected) >= 10:
            break

    assert len(collected) == 10
```

### 2. 통합 테스트

```python
# tests/integration/test_plugin_integration.py
def test_adapter_detector_integration():
    registry = PluginRegistry()
    registry.discover_plugins("ocad/plugins")

    adapter = registry.get_protocol_adapter("bfd")
    detector = registry.get_detector("lstm")

    # Collect and detect
    metric = await adapter.collect(config).__anext__()
    result = detector.detect(metric)

    assert "score" in result
```

### 3. 성능 테스트

```python
# scripts/test_all_plugins.py --performance
async def test_performance_100_endpoints():
    # 100개 엔드포인트 동시 수집
    # 처리량, 메모리, CPU 측정
    ...
```

## 향후 확장 계획

### 1. Remote Plugin Loading
- HTTP/gRPC를 통한 원격 플러그인 로드
- 플러그인 마켓플레이스

### 2. Plugin Versioning
- Semantic versioning 강제
- 호환성 체크
- 자동 업그레이드

### 3. Plugin Marketplace
- 공식 플러그인 저장소
- 커뮤니티 플러그인 공유
- 평가 및 리뷰 시스템

### 4. Multi-tenancy
- 테넌트별 플러그인 격리
- 리소스 할당 제어

## 참고 자료

- [Plugin-User-Guide.md](../06-plugins/Plugin-User-Guide.md): 사용자 가이드
- [Plugin-Development-Guide.md](../07-development/Plugin-Development-Guide.md): 개발 가이드
- [PROTOCOL-ANOMALY-DETECTION-PLAN.md](../PROTOCOL-ANOMALY-DETECTION-PLAN.md): 프로토콜 확장 계획
- [ocad/plugins/base.py](../../ocad/plugins/base.py): 기본 인터페이스 소스
- [ocad/plugins/registry.py](../../ocad/plugins/registry.py): 레지스트리 소스
