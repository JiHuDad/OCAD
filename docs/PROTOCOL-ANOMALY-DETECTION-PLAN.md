# 프로토콜 이상 탐지 확장 계획

**프로젝트명**: OCAD → PADIT 확장
**목표**: CFM을 넘어 BFD, BGP, PTP 등 다양한 프로토콜 이상 탐지 지원
**작성일**: 2025-11-05
**버전**: 1.0.0

---

## 🎯 목표 및 비전

### 현재 (OCAD)
- **단일 프로토콜**: CFM (Connectivity Fault Management) 전용
- **고정 메트릭**: UDP Echo, eCPRI, LBM, CCM
- **특화 시스템**: 초기 CFM 중심 설계

### 목표 (PADIT)
- **다중 프로토콜**: CFM, BFD, BGP, PTP, 기타 커스텀 프로토콜
- **플러그인 아키텍처**: 프로토콜별 독립적인 어댑터
- **범용 플랫폼**: 통신, 라우팅, 시간 동기화 등 모든 네트워크 프로토콜

---

## 📊 프로토콜 특성 분석

### 1. CFM (Connectivity Fault Management) ✅ **구현 완료**

#### 프로토콜 특성
- **계층**: Ethernet Layer 2 (802.1ag)
- **목적**: 네트워크 연결성 및 성능 모니터링
- **주요 메트릭**:
  - UDP Echo RTT (Round-Trip Time)
  - eCPRI One-way Delay
  - LBM (Loopback Message) RTT
  - CCM (Continuity Check Message) 미수신 횟수
- **데이터 특성**:
  - 시계열 데이터
  - 연속적 값 (ms, μs 단위)
  - 주기적 수집 (10초 간격)
  - 이상 패턴: 점진적 증가 (drift), 급격한 증가 (spike), 패킷 손실

#### 적합한 AI 모델 (현재 사용 중)
| 모델 | 목적 | 이유 |
|------|------|------|
| **TCN (Temporal Convolutional Network)** | 예측-잔차 탐지 | 시계열 예측에 강점, 긴 시퀀스 처리, LSTM보다 빠름 |
| **Isolation Forest** | 다변량 이상 탐지 | 여러 메트릭 간 상관관계 탐지, 비지도 학습 |
| **CUSUM** | 변화점 탐지 | 점진적 변화 (drift) 조기 탐지 |
| **Rule-based** | 임계값 위반 | 빠른 이상 탐지 (지연 < 1초) |

#### 구현 상태
- ✅ 데이터 수집 (실시간 + 파일 기반)
- ✅ 학습-추론 분리 아키텍처
- ✅ 4개 모델 학습 완료 (TCN 3개 + Isolation Forest 1개)
- ✅ 추론 파이프라인 통합
- ✅ 알람 시스템

---

### 2. BFD (Bidirectional Forwarding Detection) 🆕

#### 프로토콜 특성
- **계층**: Network Layer (Layer 3) / Link Layer
- **목적**: 네트워크 경로 장애 빠른 탐지 (sub-second)
- **주요 메트릭**:
  - **Session State**: Up, Down, Admin Down, Init
  - **Detection Time**: 실제 장애 탐지 시간 (ms)
  - **Echo Interval**: Echo 패킷 전송 간격
  - **Remote State**: 원격 노드 상태
  - **Diagnostic Code**: 장애 원인 코드
  - **Multiplier**: 패킷 손실 허용 배수
- **데이터 특성**:
  - **상태 전이 기반**: Up ↔ Down 상태 변화
  - **빠른 주기**: 10-50ms 간격 (CFM보다 100배 빠름)
  - **이진 상태**: 정상(Up) vs 비정상(Down)
  - **이상 패턴**:
    - **Flapping**: 빈번한 Up/Down 전환 (링크 불안정)
    - **False Positive**: 일시적 패킷 손실로 인한 오탐
    - **Slow Detection**: 예상보다 느린 장애 탐지

#### 적합한 AI 모델

| 모델 | 목적 | 이유 | 우선순위 |
|------|------|------|----------|
| **LSTM** | 상태 전이 패턴 학습 | 순차적 상태 변화 예측, 메모리 셀로 장기 의존성 학습 | 🔴 High |
| **Hidden Markov Model (HMM)** | 상태 전이 모델링 | 이산적 상태 전이 확률 계산, Flapping 패턴 탐지 | 🟡 Medium |
| **CUSUM / PELT** | 변화점 탐지 | 상태 전이 시점 정확히 탐지 | 🔴 High |
| **Rule-based** | 빠른 임계값 탐지 | Multiplier 초과, 비정상 Diagnostic Code | 🔴 High |

#### 데이터 예시
```csv
timestamp,session_id,local_state,remote_state,detection_time_ms,echo_interval_ms,multiplier,diagnostic_code
2025-11-05 10:00:00,session-1,Up,Up,15,50,3,0
2025-11-05 10:00:05,session-1,Down,Up,52,50,3,1
2025-11-05 10:00:10,session-1,Up,Up,18,50,3,0
2025-11-05 10:00:15,session-1,Down,Down,200,50,3,2  # Flapping 징조
```

#### 구현 우선순위
1. **Phase 1**: Rule-based 탐지 (임계값, Diagnostic Code)
2. **Phase 2**: CUSUM 변화점 탐지 (상태 전이 시점)
3. **Phase 3**: LSTM 모델 학습 (Flapping 패턴 예측)

---

### 3. BGP (Border Gateway Protocol) 🆕

#### 프로토콜 특성
- **계층**: Application Layer (TCP 기반)
- **목적**: 인터넷 라우팅 프로토콜 (AS 간 경로 교환)
- **주요 메트릭**:
  - **Peer State**: Idle, Connect, Active, OpenSent, OpenConfirm, Established
  - **Route Updates**: 수신/송신 라우트 업데이트 수
  - **Prefixes Received**: 수신한 프리픽스 수
  - **AS Path Length**: AS 경로 길이
  - **Route Flapping**: 라우트 추가/삭제 빈도
  - **Keepalive / Hold Time**: 연결 유지 타이머
  - **UPDATE Message Rate**: 업데이트 메시지 빈도
- **데이터 특성**:
  - **그래프 구조**: AS 간 연결 관계 (노드 + 엣지)
  - **이벤트 기반**: UPDATE, NOTIFICATION 메시지
  - **대규모 데이터**: 수백만 개 라우트
  - **이상 패턴**:
    - **Route Hijacking**: AS Path 조작
    - **Route Leakage**: 잘못된 라우트 전파
    - **Flapping**: 빈번한 라우트 변경
    - **Prefix Anomaly**: 비정상적인 프리픽스 길이

#### 적합한 AI 모델

| 모델 | 목적 | 이유 | 우선순위 |
|------|------|------|----------|
| **Graph Neural Network (GNN)** | AS 그래프 분석 | AS 간 관계를 그래프로 모델링, Route Hijacking 탐지 | 🔴 High |
| **LSTM / Transformer** | 시퀀스 패턴 분석 | UPDATE 메시지 시퀀스 학습, 비정상적인 패턴 탐지 | 🟡 Medium |
| **Isolation Forest** | 이상 프리픽스 탐지 | 비정상적인 프리픽스 길이, AS Path 길이 탐지 | 🟡 Medium |
| **Rule-based** | 알려진 공격 패턴 | RPKI (Resource Public Key Infrastructure) 검증 | 🔴 High |

#### 데이터 예시
```csv
timestamp,peer_id,peer_state,prefixes_received,as_path_length,updates_received,route_flaps
2025-11-05 10:00:00,peer-1,Established,50000,3,10,0
2025-11-05 10:05:00,peer-1,Established,50005,3,15,2
2025-11-05 10:10:00,peer-1,Idle,0,0,0,0  # Peer Down
2025-11-05 10:15:00,peer-1,Established,50010,5,200,50  # Route Hijacking 의심
```

#### 그래프 데이터 예시
```json
{
  "nodes": [
    {"as": "AS64512", "type": "origin"},
    {"as": "AS64513", "type": "transit"},
    {"as": "AS64514", "type": "destination"}
  ],
  "edges": [
    {"source": "AS64512", "target": "AS64513", "prefix": "192.0.2.0/24"},
    {"source": "AS64513", "target": "AS64514", "prefix": "192.0.2.0/24"}
  ]
}
```

#### 구현 우선순위
1. **Phase 1**: Rule-based 탐지 (Flapping 횟수, Peer State 변화)
2. **Phase 2**: Isolation Forest (이상 AS Path Length, Prefix 수)
3. **Phase 3**: GNN 모델 학습 (Route Hijacking, AS 그래프 이상 탐지)

---

### 4. PTP (Precision Time Protocol) 🆕

#### 프로토콜 특성
- **계층**: Application Layer (UDP/Ethernet)
- **목적**: 네트워크 시간 동기화 (나노초 정밀도)
- **주요 메트릭**:
  - **Offset from Master**: 마스터 시계와의 시간 차이 (ns)
  - **Mean Path Delay**: 평균 경로 지연 (ns)
  - **Jitter**: 시간 변동성 (ns)
  - **Sync Interval**: Sync 메시지 전송 간격
  - **Clock State**: Master, Slave, Passive, Listening
  - **Steps Removed**: 마스터까지의 홉 수
- **데이터 특성**:
  - **고정밀 시계열**: 나노초 단위 측정
  - **주기적 패턴**: Sync, Delay_Req 메시지 주기
  - **마스터-슬레이브 관계**: 계층 구조
  - **이상 패턴**:
    - **Offset Drift**: 시간 차이 점진적 증가
    - **Jitter Spike**: 네트워크 혼잡으로 인한 지연 변동
    - **Master Change**: 마스터 시계 변경 (재동기화)
    - **Sync Loss**: Sync 메시지 미수신

#### 적합한 AI 모델

| 모델 | 목적 | 이유 | 우선순위 |
|------|------|------|----------|
| **TCN** | 시계열 예측-잔차 탐지 | Offset 예측, Drift 조기 탐지 (CFM과 유사) | 🔴 High |
| **Autoencoder** | 정상 패턴 학습 | 정상 Offset/Jitter 패턴 학습, 재구성 오차로 이상 탐지 | 🟡 Medium |
| **CUSUM / PELT** | Drift 변화점 탐지 | 점진적 Offset 증가 탐지 | 🔴 High |
| **Rule-based** | 임계값 위반 | Offset > 1μs, Jitter > 100ns | 🔴 High |

#### 데이터 예시
```csv
timestamp,slave_id,offset_ns,mean_path_delay_ns,jitter_ns,clock_state,steps_removed,sync_interval_ms
2025-11-05 10:00:00,slave-1,50,1000,10,Slave,2,125
2025-11-05 10:00:01,slave-1,52,1000,12,Slave,2,125
2025-11-05 10:00:02,slave-1,150,1000,50,Slave,2,125  # Jitter Spike
2025-11-05 10:00:03,slave-1,200,1050,80,Slave,2,125  # Offset Drift 시작
```

#### 구현 우선순위
1. **Phase 1**: Rule-based 탐지 (Offset, Jitter 임계값)
2. **Phase 2**: CUSUM 변화점 탐지 (Drift 조기 경보)
3. **Phase 3**: TCN 모델 학습 (예측-잔차 기반 이상 탐지)

---

## 🏗️ 플러그인 기반 아키텍처 설계

### 아키텍처 원칙

1. **프로토콜 독립성**: 각 프로토콜은 독립적인 플러그인으로 구현
2. **모델 선택 자유도**: 프로토콜별로 최적 AI 모델 선택 가능
3. **공통 인터페이스**: 모든 플러그인은 동일한 인터페이스 구현
4. **동적 로딩**: 런타임에 플러그인 추가/제거 가능

### 플러그인 인터페이스

```python
# ocad/plugins/base.py (새로 작성)

from abc import ABC, abstractmethod
from typing import Dict, Any, List, AsyncIterator
from datetime import datetime
from ocad.core.schemas import Metric, DetectionScore

class ProtocolAdapter(ABC):
    """프로토콜 어댑터 기본 인터페이스"""

    @property
    @abstractmethod
    def name(self) -> str:
        """프로토콜 이름 (예: 'bfd', 'bgp', 'ptp')"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """플러그인 버전"""
        pass

    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """지원하는 메트릭 목록"""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """설정 검증"""
        pass

    @abstractmethod
    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Metric]:
        """메트릭 수집 (제너레이터)"""
        pass

    @abstractmethod
    def get_recommended_models(self) -> List[str]:
        """권장 AI 모델 목록 (예: ['tcn', 'lstm', 'rule-based'])"""
        pass


class DetectorPlugin(ABC):
    """탐지기 플러그인 기본 인터페이스"""

    @property
    @abstractmethod
    def name(self) -> str:
        """탐지기 이름 (예: 'gnn', 'hmm')"""
        pass

    @property
    @abstractmethod
    def supported_protocols(self) -> List[str]:
        """지원하는 프로토콜 목록"""
        pass

    @abstractmethod
    def train(self, data: Any) -> None:
        """모델 학습"""
        pass

    @abstractmethod
    def detect(self, features: Dict[str, Any]) -> DetectionScore:
        """이상 탐지"""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """모델 저장"""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """모델 로드"""
        pass
```

### 플러그인 레지스트리

```python
# ocad/plugins/registry.py (새로 작성)

from typing import Dict, Optional
from pathlib import Path
import importlib.util
import logging
from .base import ProtocolAdapter, DetectorPlugin

logger = logging.getLogger(__name__)

class PluginRegistry:
    """플러그인 관리 시스템"""

    def __init__(self):
        self.protocol_adapters: Dict[str, ProtocolAdapter] = {}
        self.detectors: Dict[str, DetectorPlugin] = {}

    def register_protocol_adapter(self, adapter: ProtocolAdapter):
        """프로토콜 어댑터 등록"""
        name = adapter.name
        if name in self.protocol_adapters:
            logger.warning(f"Protocol adapter '{name}' already registered, overwriting")

        self.protocol_adapters[name] = adapter
        logger.info(f"Protocol adapter registered: {name} v{adapter.version}")

    def register_detector(self, detector: DetectorPlugin):
        """탐지기 등록"""
        name = detector.name
        if name in self.detectors:
            logger.warning(f"Detector '{name}' already registered, overwriting")

        self.detectors[name] = detector
        logger.info(f"Detector registered: {name}")

    def get_protocol_adapter(self, name: str) -> Optional[ProtocolAdapter]:
        """프로토콜 어댑터 조회"""
        return self.protocol_adapters.get(name)

    def get_detector(self, name: str) -> Optional[DetectorPlugin]:
        """탐지기 조회"""
        return self.detectors.get(name)

    def list_protocol_adapters(self) -> Dict[str, Dict[str, Any]]:
        """등록된 프로토콜 어댑터 목록"""
        return {
            name: {
                "version": adapter.version,
                "supported_metrics": adapter.supported_metrics,
                "recommended_models": adapter.get_recommended_models(),
            }
            for name, adapter in self.protocol_adapters.items()
        }

    def list_detectors(self) -> Dict[str, Dict[str, Any]]:
        """등록된 탐지기 목록"""
        return {
            name: {
                "supported_protocols": detector.supported_protocols,
            }
            for name, detector in self.detectors.items()
        }

    def discover_plugins(self, plugin_dir: Path):
        """플러그인 자동 발견 및 로드"""
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return

        # Protocol Adapters
        adapters_dir = plugin_dir / "protocol_adapters"
        if adapters_dir.exists():
            for adapter_dir in adapters_dir.iterdir():
                if not adapter_dir.is_dir():
                    continue
                self._load_protocol_adapter(adapter_dir)

        # Detectors
        detectors_dir = plugin_dir / "detectors"
        if detectors_dir.exists():
            for detector_dir in detectors_dir.iterdir():
                if not detector_dir.is_dir():
                    continue
                self._load_detector(detector_dir)

    def _load_protocol_adapter(self, adapter_dir: Path):
        """프로토콜 어댑터 로드"""
        init_file = adapter_dir / "__init__.py"
        if not init_file.exists():
            return

        try:
            spec = importlib.util.spec_from_file_location(
                f"plugin_{adapter_dir.name}",
                init_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "create_adapter"):
                adapter = module.create_adapter()
                self.register_protocol_adapter(adapter)
            else:
                logger.warning(f"Adapter {adapter_dir.name} missing create_adapter()")

        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_dir.name}: {e}")

    def _load_detector(self, detector_dir: Path):
        """탐지기 로드"""
        init_file = detector_dir / "__init__.py"
        if not init_file.exists():
            return

        try:
            spec = importlib.util.spec_from_file_location(
                f"detector_{detector_dir.name}",
                init_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "create_detector"):
                detector = module.create_detector()
                self.register_detector(detector)
            else:
                logger.warning(f"Detector {detector_dir.name} missing create_detector()")

        except Exception as e:
            logger.error(f"Failed to load detector {detector_dir.name}: {e}")


# 전역 레지스트리 인스턴스
registry = PluginRegistry()
```

### 디렉토리 구조

```
OCAD/
├── ocad/
│   ├── plugins/                    # 플러그인 시스템 (NEW)
│   │   ├── __init__.py
│   │   ├── base.py                 # 플러그인 인터페이스
│   │   ├── registry.py             # 플러그인 레지스트리
│   │   │
│   │   ├── protocol_adapters/      # 프로토콜 어댑터 플러그인
│   │   │   ├── cfm/                # CFM (기존 코드 마이그레이션)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adapter.py
│   │   │   │   └── collector.py
│   │   │   │
│   │   │   ├── bfd/                # BFD (NEW)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adapter.py
│   │   │   │   └── collector.py
│   │   │   │
│   │   │   ├── bgp/                # BGP (NEW)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adapter.py
│   │   │   │   └── collector.py
│   │   │   │
│   │   │   └── ptp/                # PTP (NEW)
│   │   │       ├── __init__.py
│   │   │       ├── adapter.py
│   │   │       └── collector.py
│   │   │
│   │   └── detectors/              # 탐지기 플러그인
│   │       ├── gnn/                # Graph Neural Network (NEW)
│   │       │   ├── __init__.py
│   │       │   ├── detector.py
│   │       │   └── model.py
│   │       │
│   │       ├── hmm/                # Hidden Markov Model (NEW)
│   │       │   ├── __init__.py
│   │       │   ├── detector.py
│   │       │   └── model.py
│   │       │
│   │       └── autoencoder/        # Autoencoder (NEW)
│   │           ├── __init__.py
│   │           ├── detector.py
│   │           └── model.py
│   │
│   ├── collectors/                 # (기존) CFM 수집기 → plugins/protocol_adapters/cfm/로 이동 예정
│   ├── detectors/                  # (기존) 탐지기 → 플러그인과 공존
│   ├── features/                   # (기존) 피처 엔지니어링
│   ├── models/                     # (기존) 학습된 모델
│   ├── training/                   # (기존) 학습 모듈
│   └── ...
│
├── config/
│   └── plugins.yaml                # 플러그인 설정 (NEW)
│
└── scripts/
    ├── register_plugin.py          # 플러그인 등록 스크립트 (NEW)
    └── list_plugins.py             # 플러그인 목록 조회 (NEW)
```

---

## 📋 구현 계획

### Phase 0: 플러그인 인프라 구축 (Week 1-2)

**목표**: 플러그인 시스템 기반 구현

**작업 내용**:
1. ✅ `ocad/plugins/base.py` 작성 (ProtocolAdapter, DetectorPlugin 인터페이스)
2. ✅ `ocad/plugins/registry.py` 작성 (PluginRegistry 구현)
3. ✅ 디렉토리 구조 생성 (`ocad/plugins/protocol_adapters/`, `ocad/plugins/detectors/`)
4. ✅ 플러그인 자동 발견 (discover_plugins) 기능
5. ✅ 설정 파일 (`config/plugins.yaml`) 추가
6. ✅ 플러그인 관리 CLI 명령어 추가 (`python -m ocad.cli list-plugins`, `register-plugin`)

**검증 방법**:
```bash
# 플러그인 목록 조회
python -m ocad.cli list-plugins

# 출력 예시:
# Protocol Adapters:
#   - cfm (v1.0.0): UDP Echo, eCPRI, LBM, CCM
# Detectors:
#   - tcn: CFM, PTP
#   - isolation_forest: CFM, BFD, PTP
```

---

### Phase 1: BFD 프로토콜 지원 (Week 3-4)

**목표**: BFD 프로토콜 어댑터 및 탐지 모델 구현

#### 1.1 BFD 어댑터 구현
```python
# ocad/plugins/protocol_adapters/bfd/adapter.py

from ocad.plugins.base import ProtocolAdapter
from typing import AsyncIterator, Dict, Any, List
from ocad.core.schemas import Metric

class BFDAdapter(ProtocolAdapter):
    @property
    def name(self) -> str:
        return "bfd"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "session_state",
            "detection_time_ms",
            "echo_interval_ms",
            "remote_state",
            "diagnostic_code",
            "multiplier"
        ]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        required = ["sessions"]
        return all(k in config for k in required)

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Metric]:
        """BFD 세션 모니터링"""
        sessions = config["sessions"]
        interval = config.get("interval_sec", 5)

        while True:
            for session in sessions:
                # BFD 세션 상태 수집 (SNMP, NETCONF, 또는 로그 파싱)
                state_data = await self._collect_session_state(session)

                yield Metric(
                    timestamp=datetime.utcnow(),
                    source_id=session["id"],
                    metric_name="bfd_session_state",
                    value=self._state_to_value(state_data["local_state"]),
                    metadata={
                        "local_state": state_data["local_state"],
                        "remote_state": state_data["remote_state"],
                        "detection_time_ms": state_data["detection_time_ms"],
                        "diagnostic_code": state_data["diagnostic_code"],
                    }
                )

            await asyncio.sleep(interval)

    def get_recommended_models(self) -> List[str]:
        return ["lstm", "hmm", "cusum", "rule-based"]

    async def _collect_session_state(self, session: Dict) -> Dict:
        """BFD 세션 상태 수집 (구현 필요)"""
        # SNMP OID: .1.3.6.1.2.1.222 (BFD-STD-MIB)
        # 또는 NETCONF/YANG 사용
        pass

    def _state_to_value(self, state: str) -> float:
        """상태를 숫자로 변환 (ML 입력용)"""
        state_map = {"AdminDown": 0.0, "Down": 1.0, "Init": 2.0, "Up": 3.0}
        return state_map.get(state, -1.0)
```

#### 1.2 HMM 탐지기 구현
```python
# ocad/plugins/detectors/hmm/detector.py

from ocad.plugins.base import DetectorPlugin
from hmmlearn import hmm
import numpy as np

class HMMDetector(DetectorPlugin):
    @property
    def name(self) -> str:
        return "hmm"

    @property
    def supported_protocols(self) -> List[str]:
        return ["bfd"]

    def train(self, data: np.ndarray):
        """Hidden Markov Model 학습"""
        self.model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        self.model.fit(data)

    def detect(self, features: Dict[str, Any]) -> DetectionScore:
        """상태 전이 확률로 이상 탐지"""
        sequence = np.array([features["state_sequence"]])
        log_prob = self.model.score(sequence)

        # 낮은 확률 = 이상 패턴
        threshold = -10.0
        is_anomaly = log_prob < threshold

        return DetectionScore(
            timestamp=features["timestamp"],
            score=abs(log_prob),
            is_anomaly=is_anomaly,
            confidence=0.8,
            detector_name="hmm",
        )
```

#### 1.3 데이터 수집 및 학습
```bash
# BFD 데이터 수집 (실제 BFD 세션 또는 시뮬레이터)
python scripts/plugins/collect_bfd_data.py --sessions sessions.yaml --duration 24h

# HMM 모델 학습
python scripts/plugins/train_bfd_hmm.py \
    --data data/bfd/training_normal.csv \
    --output ocad/models/bfd/hmm_v1.0.0.pkl

# 추론 테스트
python scripts/plugins/infer_bfd.py \
    --input data/bfd/test_flapping.csv \
    --model ocad/models/bfd/hmm_v1.0.0.pkl
```

---

### Phase 2: BGP 프로토콜 지원 (Week 5-8)

**목표**: BGP 프로토콜 어댑터 및 GNN 모델 구현

#### 2.1 BGP 어댑터 구현
```python
# ocad/plugins/protocol_adapters/bgp/adapter.py

class BGPAdapter(ProtocolAdapter):
    @property
    def name(self) -> str:
        return "bgp"

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "peer_state",
            "prefixes_received",
            "as_path_length",
            "route_updates",
            "route_flaps",
        ]

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Metric]:
        """BGP 피어 모니터링"""
        peers = config["peers"]
        interval = config.get("interval_sec", 60)

        while True:
            for peer in peers:
                # BGP 피어 상태 수집 (SNMP, BGP Monitoring Protocol)
                peer_data = await self._collect_peer_state(peer)

                yield Metric(
                    timestamp=datetime.utcnow(),
                    source_id=peer["id"],
                    metric_name="bgp_peer_state",
                    value=self._state_to_value(peer_data["state"]),
                    metadata=peer_data,
                )

                # AS Graph 수집 (UPDATE 메시지 파싱)
                if peer_data["state"] == "Established":
                    as_graph = await self._collect_as_graph(peer)
                    yield Metric(
                        timestamp=datetime.utcnow(),
                        source_id=peer["id"],
                        metric_name="bgp_as_graph",
                        value=len(as_graph["nodes"]),
                        metadata={"graph": as_graph},
                    )

            await asyncio.sleep(interval)

    def get_recommended_models(self) -> List[str]:
        return ["gnn", "lstm", "isolation_forest", "rule-based"]
```

#### 2.2 GNN 탐지기 구현
```python
# ocad/plugins/detectors/gnn/detector.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNDetector(DetectorPlugin):
    @property
    def name(self) -> str:
        return "gnn"

    @property
    def supported_protocols(self) -> List[str]:
        return ["bgp"]

    def train(self, data: Any):
        """Graph Neural Network 학습"""
        # PyTorch Geometric 사용
        self.model = GNNModel(input_dim=10, hidden_dim=64, output_dim=1)
        # 학습 로직 (AS 그래프 데이터셋)
        pass

    def detect(self, features: Dict[str, Any]) -> DetectionScore:
        """AS 그래프 이상 탐지"""
        graph = features["as_graph"]
        # GNN 추론
        score = self.model(graph)

        return DetectionScore(
            timestamp=features["timestamp"],
            score=score.item(),
            is_anomaly=score > 0.5,
            detector_name="gnn",
        )
```

---

### Phase 3: PTP 프로토콜 지원 (Week 9-10)

**목표**: PTP 프로토콜 어댑터 및 TCN 모델 재사용

#### 3.1 PTP 어댑터 구현
```python
# ocad/plugins/protocol_adapters/ptp/adapter.py

class PTPAdapter(ProtocolAdapter):
    @property
    def name(self) -> str:
        return "ptp"

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "offset_from_master_ns",
            "mean_path_delay_ns",
            "jitter_ns",
            "clock_state",
            "steps_removed",
        ]

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Metric]:
        """PTP 슬레이브 클럭 모니터링"""
        slaves = config["slaves"]
        interval = config.get("interval_sec", 1)

        while True:
            for slave in slaves:
                ptp_data = await self._collect_ptp_state(slave)

                yield Metric(
                    timestamp=datetime.utcnow(),
                    source_id=slave["id"],
                    metric_name="ptp_offset_ns",
                    value=ptp_data["offset_from_master_ns"],
                    metadata=ptp_data,
                )

            await asyncio.sleep(interval)

    def get_recommended_models(self) -> List[str]:
        return ["tcn", "autoencoder", "cusum", "rule-based"]
```

#### 3.2 기존 TCN 모델 재사용
```bash
# PTP 데이터로 TCN 재학습
python scripts/train_tcn_model.py \
    --metric-type ptp_offset \
    --train-data data/ptp/training_normal.parquet \
    --epochs 50 \
    --output ocad/models/tcn/ptp_offset_v1.0.0.pth

# 추론
python scripts/inference_simple.py \
    --input data/ptp/test_drift.csv \
    --protocol ptp
```

---

### Phase 4: 플러그인 통합 및 테스트 (Week 11-12)

**목표**: 모든 프로토콜 통합 실행

#### 4.1 통합 설정 파일
```yaml
# config/plugins.yaml

plugins:
  protocol_adapters:
    cfm:
      enabled: true
      config:
        endpoints:
          - id: oru-1
            host: 192.168.1.100
            udp_echo: true
            ecpri: true
            lbm: true

    bfd:
      enabled: true
      config:
        sessions:
          - id: bfd-session-1
            local_ip: 192.168.1.1
            remote_ip: 192.168.1.2

    bgp:
      enabled: true
      config:
        peers:
          - id: bgp-peer-1
            peer_ip: 10.0.0.1
            peer_as: 64512

    ptp:
      enabled: true
      config:
        slaves:
          - id: ptp-slave-1
            interface: eth0

  detectors:
    tcn:
      enabled: true
      protocols: [cfm, ptp]
      model_path: ocad/models/tcn/

    lstm:
      enabled: true
      protocols: [bfd]
      model_path: ocad/models/lstm/

    gnn:
      enabled: true
      protocols: [bgp]
      model_path: ocad/models/gnn/

    isolation_forest:
      enabled: true
      protocols: [cfm, bgp]
      model_path: ocad/models/isolation_forest/
```

#### 4.2 통합 실행
```bash
# 플러그인 로드 및 실행
python -m ocad.main --config config/plugins.yaml

# 로그 확인
tail -f logs/test_*/summary/summary.log

# 출력 예시:
# [INFO] PluginRegistry: Protocol adapter registered: cfm v1.0.0
# [INFO] PluginRegistry: Protocol adapter registered: bfd v1.0.0
# [INFO] PluginRegistry: Protocol adapter registered: bgp v1.0.0
# [INFO] PluginRegistry: Protocol adapter registered: ptp v1.0.0
# [INFO] DetectorRegistry: Detector registered: tcn (protocols: cfm, ptp)
# [INFO] DetectorRegistry: Detector registered: lstm (protocols: bfd)
# [INFO] DetectorRegistry: Detector registered: gnn (protocols: bgp)
```

---

## 📊 프로토콜별 AI 모델 매칭 요약

| 프로토콜 | 데이터 특성 | 추천 AI 모델 | 우선순위 | 비고 |
|---------|------------|-------------|---------|------|
| **CFM** (✅ 완료) | 시계열, 연속값 | TCN, Isolation Forest, CUSUM | - | OCAD 현재 상태 |
| **BFD** | 상태 전이, 빠른 주기 | LSTM, HMM, CUSUM | 🔴 High | Flapping 탐지 중요 |
| **BGP** | 그래프, 이벤트 | GNN, LSTM, Isolation Forest | 🟡 Medium | Route Hijacking 탐지 |
| **PTP** | 고정밀 시계열 | TCN, Autoencoder, CUSUM | 🟡 Medium | CFM과 유사 |

---

## 🎯 성공 지표

### 기술적 목표
- ✅ 4개 프로토콜 지원 (CFM, BFD, BGP, PTP)
- ✅ 플러그인 시스템 완성 (동적 로딩)
- ✅ 프로토콜별 최적 AI 모델 적용
- ✅ 모델 재사용 가능 (TCN, Isolation Forest)

### 성능 목표
- **BFD**: Flapping 탐지율 > 90%, False Positive < 5%
- **BGP**: Route Hijacking 탐지율 > 85%
- **PTP**: Offset Drift 조기 경보 > 3분

### 사용성 목표
- 플러그인 추가 < 2시간 (새 프로토콜)
- CLI 명령어로 플러그인 관리 가능
- 설정 파일로 프로토콜 활성화/비활성화

---

## 📚 참고 자료

### 프로토콜 스펙
- **BFD**: RFC 5880 - Bidirectional Forwarding Detection (BFD)
- **BGP**: RFC 4271 - Border Gateway Protocol 4 (BGP-4)
- **PTP**: IEEE 1588-2008 - Precision Time Protocol

### AI 모델
- **GNN**: PyTorch Geometric - https://pytorch-geometric.readthedocs.io/
- **HMM**: hmmlearn - https://hmmlearn.readthedocs.io/
- **Autoencoder**: PyTorch - https://pytorch.org/tutorials/beginner/basics/autoencoderyt_tutorial.html

---

## 📈 구현 현황 (2025-11-07 업데이트)

### ✅ 완료된 작업

#### Phase 0-4: 플러그인 시스템 구축 완료 (2025-11-05)
- ✅ 플러그인 인프라 (ProtocolAdapter, DetectorPlugin)
- ✅ BFD 프로토콜 지원 (LSTM, HMM)
- ✅ CLI 명령어 (`list-plugins`, `plugin-info`, `enable-plugin`, `test-plugin`, `detect`)
- ✅ 통합 테스트 및 문서화 (4개 문서 작성)

#### 프로토콜별 통합 쉘 스크립트 구축 (2025-11-07)
- ✅ **`scripts/train.sh`**: 프로토콜별 모델 학습 통합
  - 사용법: `./scripts/train.sh --protocol <protocol> --data <dir> --output <model-dir>`
  - 지원 프로토콜: BFD, BGP, PTP, CFM
  - 자동 모델 타입 탐지, 메타데이터 생성

- ✅ **`scripts/infer.sh`**: 프로토콜별 추론 및 리포트 자동 생성
  - 사용법: `./scripts/infer.sh --protocol <protocol> --model <dir> --data <dir>`
  - 타임스탬프 자동 생성, 리포트 자동 파이프라인

#### CFM 인터페이스 통일 (2025-11-07)
- ✅ **일관된 인터페이스**: 모든 프로토콜과 동일 (`--model`, `--data`, `--output`)
- ✅ **중복 제거**: `--metrics` 옵션 제거 (predictions.csv에서 직접 계산)
- ✅ **모드 자동 감지**: Validation/Production 모드 (is_anomaly 유무로 판단)

### 📊 프로토콜별 구현 상태

| 프로토콜 | 어댑터 | 탐지기 | 학습 스크립트 | 추론 스크립트 | 리포트 | 상태 |
|---------|-------|--------|-------------|-------------|--------|------|
| **CFM** | ✅ | ✅ Isolation Forest | ✅ train.sh | ✅ infer.sh | ✅ | **완료** |
| **BFD** | ✅ | ✅ LSTM, HMM | ✅ train.sh | ✅ infer.sh | ✅ | **완료** |
| **PTP** | ⏳ | ✅ TCN | ✅ train.sh | ✅ infer.sh | ✅ | **진행중** |
| **BGP** | ⏳ | ⏳ GNN | ⏳ | ⏳ | ⏳ | **Phase 2** |

### 🎯 남은 작업

#### Phase 2: BGP 프로토콜 지원 (진행 예정)
- BGP 어댑터 구현 (AS-path 분석)
- GNN 탐지기 구현
- BGP 데이터 생성 스크립트
- 통합 테스트

#### Phase 3: PTP 프로토콜 완성 (진행 예정)
- PTP 어댑터 구현 (시간 동기화)
- TCN 탐지기 통합 (CFM TCN 재사용)
- PTP 데이터 생성 스크립트

---

## 🚀 다음 단계

### 즉시 (이번 주)
1. ~~**Phase 0 시작**: 플러그인 인프라 구축~~ ✅ 완료
2. ~~**문서 업데이트**~~ ✅ 완료
3. **BGP Phase 2 준비**: GNN 모델 연구 및 설계

### 단기 (1-2개월)
- ~~**Phase 1**: BFD 프로토콜 + LSTM/HMM 모델~~ ✅ 완료
- **Phase 2**: BGP 프로토콜 + GNN 모델 (진행 예정)

### 중기 (3-4개월)
- **Phase 3**: PTP 프로토콜 + TCN 재사용 (일부 완료)
- ~~**Phase 4**: 통합 테스트 및 문서화~~ ✅ 완료

---

**작성자**: Claude Code
**최초 작성**: 2025-11-05
**최종 업데이트**: 2025-11-07
**버전**: 1.1.0
**상태**: 🎯 Phase 0-1 완료, Phase 2-3 진행중
