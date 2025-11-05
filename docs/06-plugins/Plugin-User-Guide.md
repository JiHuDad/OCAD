# OCAD 플러그인 사용 가이드

> **최종 업데이트**: 2025-11-05 (Phase 4)
> **대상 독자**: 운영자, 시스템 관리자, 네트워크 엔지니어
> **소요 시간**: 15-20분

## 목차

1. [개요](#개요)
2. [플러그인 시스템 이해하기](#플러그인-시스템-이해하기)
3. [프로토콜 어댑터 사용법](#프로토콜-어댑터-사용법)
4. [탐지기 사용법](#탐지기-사용법)
5. [설정 파일 작성](#설정-파일-작성)
6. [CLI 명령어 레퍼런스](#cli-명령어-레퍼런스)
7. [실전 예제](#실전-예제)
8. [문제 해결](#문제-해결)

---

## 개요

OCAD 플러그인 시스템은 **프로토콜 어댑터**(Protocol Adapter)와 **탐지기**(Detector)를 통해 다양한 네트워크 프로토콜의 이상 탐지를 지원합니다.

### 지원 프로토콜

| 프로토콜 | 설명 | 상태 | 권장 탐지기 |
|---------|------|------|------------|
| **CFM** | Connectivity Fault Management (UDP Echo, eCPRI, LBM, CCM) | ✅ 완료 (Phase 0) | LSTM, TCN |
| **BFD** | Bidirectional Forwarding Detection (세션 모니터링, 플래핑 탐지) | ✅ 완료 (Phase 1) | LSTM, HMM |
| **BGP** | Border Gateway Protocol (AS-path 분석, hijacking 탐지) | ⏳ 진행중 (Phase 2) | LSTM, HMM, GNN |
| **PTP** | Precision Time Protocol (시간 동기화 모니터링) | ⏳ 진행중 (Phase 3) | LSTM, TCN |

### 지원 탐지기 (AI 모델)

| 탐지기 | 유형 | 지원 프로토콜 | 상태 |
|--------|------|---------------|------|
| **LSTM** | Recurrent Neural Network | BFD, BGP, CFM, PTP | ✅ 완료 |
| **HMM** | Hidden Markov Model | BFD, BGP | ✅ 완료 |
| **GNN** | Graph Neural Network | BGP | ⏳ Phase 2 |
| **TCN** | Temporal Convolutional Network | PTP, CFM | ⏳ Phase 3 |
| **Autoencoder** | Unsupervised Learning | All | 선택 사항 |

---

## 플러그인 시스템 이해하기

### 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   OCAD Core System                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Plugin Registry                        │
│  (동적 플러그인 로딩 및 관리)                              │
└─────────────────────────────────────────────────────────┘
         ↙                                    ↘
┌──────────────────────┐          ┌──────────────────────┐
│ Protocol Adapters    │          │    Detectors         │
│  - CFM               │          │  - LSTM              │
│  - BFD               │          │  - HMM               │
│  - BGP               │          │  - GNN               │
│  - PTP               │          │  - TCN               │
└──────────────────────┘          └──────────────────────┘
         ↓                                    ↓
┌──────────────────────┐          ┌──────────────────────┐
│  Metric Collection   │          │ Anomaly Detection    │
└──────────────────────┘          └──────────────────────┘
```

### 주요 개념

#### 1. Protocol Adapter (프로토콜 어댑터)
- **역할**: 특정 프로토콜에서 메트릭을 수집
- **입력**: 프로토콜별 설정 (IP, 포트, 세션 정보 등)
- **출력**: 표준화된 메트릭 데이터

#### 2. Detector (탐지기)
- **역할**: 메트릭을 분석하여 이상 탐지
- **입력**: 수집된 메트릭 데이터
- **출력**: 이상 점수 (0.0 = 정상, 1.0 = 이상)

#### 3. Plugin Registry (플러그인 레지스트리)
- **역할**: 플러그인 동적 로딩 및 관리
- **기능**: 자동 발견, 버전 관리, 의존성 체크

---

## 프로토콜 어댑터 사용법

### 1. 사용 가능한 어댑터 확인

```bash
# 모든 플러그인 목록 보기
python -m ocad.cli list-plugins

# 특정 어댑터 정보 확인
python -m ocad.cli plugin-info cfm
python -m ocad.cli plugin-info bfd
```

**출력 예시**:
```
Protocol Adapter: cfm
Version: 1.0.0
Description: CFM protocol adapter v1.0.0

Supported Metrics:
  • rtt_ms
  • loss_rate
  • jitter_ms
  • ecpri_delay_ms
  • lbm_response_time_ms

Recommended AI Models:
  • lstm
  • tcn
```

### 2. CFM 어댑터 사용

**기능**: UDP Echo, eCPRI 지연, LBM 응답 시간 모니터링

**설정 예제**:
```yaml
protocol_adapters:
  cfm:
    enabled: true
    config:
      udp_echo:
        enabled: true
        port: 50000
        packet_size_bytes: 64
        timeout_ms: 1000

      ecpri:
        enabled: true
        port: 50001

      lbm:
        enabled: true
        interval_sec: 10

      interval_sec: 10  # 10초마다 수집
```

**CLI 테스트**:
```bash
# CFM 어댑터 단독 테스트
python -m ocad.cli test-plugin cfm

# 실시간 모니터링 (60초)
python -m ocad.cli detect cfm --endpoint 192.168.1.100 --duration 60
```

### 3. BFD 어댑터 사용

**기능**: BFD 세션 상태 모니터링, 플래핑 탐지

**설정 예제**:
```yaml
protocol_adapters:
  bfd:
    enabled: true
    config:
      sessions:
        - id: "bfd-session-1"
          local_ip: "192.168.1.1"
          remote_ip: "192.168.1.2"
          interval_ms: 50      # 50ms 탐지 간격
          multiplier: 3        # 3배 타임아웃
          min_echo_interval_ms: 50

        - id: "bfd-session-2"
          local_ip: "192.168.2.1"
          remote_ip: "192.168.2.2"
          interval_ms: 100
          multiplier: 5

      flapping_detection:
        enabled: true
        window_seconds: 60    # 60초 윈도우
        threshold_count: 5    # 5회 변경 시 알람

      interval_sec: 1
```

**모니터링 메트릭**:
- `session_state`: 세션 상태 (0=Down, 1=Init, 2=Up, 3=AdminDown)
- `detection_time_ms`: 탐지 시간
- `echo_interval_ms`: Echo 간격
- `remote_state`: 원격 상태
- `diagnostic_code`: 진단 코드
- `multiplier`: 탐지 배수
- `flap_count`: 플래핑 횟수

**CLI 테스트**:
```bash
# BFD 어댑터 테스트
python -m ocad.cli test-plugin bfd

# 특정 세션 모니터링 (LSTM 탐지기 사용)
python -m ocad.cli detect bfd --endpoint 192.168.1.1 --detector lstm --duration 120
```

### 4. BGP 어댑터 사용 (Phase 2)

**기능**: BGP UPDATE 분석, AS-path 모니터링, hijacking 탐지

**설정 예제**:
```yaml
protocol_adapters:
  bgp:
    enabled: true
    config:
      sessions:
        - id: "bgp-peer-1"
          local_as: 65001
          peer_ip: "192.168.10.1"
          peer_as: 65002

      update_monitoring:
        enabled: true
        track_as_path: true

      hijacking_detection:
        enabled: true
        known_prefixes_file: "config/bgp_known_prefixes.yaml"
```

### 5. PTP 어댑터 사용 (Phase 3)

**기능**: 시간 동기화 정확도, 드리프트 모니터링

**설정 예제**:
```yaml
protocol_adapters:
  ptp:
    enabled: true
    config:
      domain: 0
      offset_threshold_ns: 100      # 100ns 임계값
      drift_threshold_ppb: 50       # 50ppb 드리프트
      interval_sec: 1
```

---

## 탐지기 사용법

### 1. LSTM 탐지기

**지원 프로토콜**: BFD, BGP, CFM, PTP

**특징**:
- 시계열 예측 기반 이상 탐지
- 시퀀스 길이: 30-100 timesteps
- 장기 의존성 학습 가능

**설정 예제**:
```yaml
detectors:
  lstm:
    enabled: true
    protocols: ["bfd", "bgp", "cfm", "ptp"]
    config:
      hidden_size: 64
      num_layers: 2
      sequence_length: 50
      anomaly_threshold: 0.7
      use_pretrained: true
      model_dir: "ocad/models/lstm/"

      protocol_configs:
        bfd:
          sequence_length: 30
          anomaly_threshold: 0.75
        cfm:
          sequence_length: 50
          anomaly_threshold: 0.7
```

**학습 명령어**:
```bash
# BFD 데이터로 LSTM 학습
python -m ocad.cli train-detector lstm \
    --data data/training/bfd_train.parquet \
    --epochs 50 \
    --batch-size 32 \
    --output ocad/models/lstm/bfd_lstm_v1.pth

# 학습된 모델로 실시간 탐지
python -m ocad.cli detect bfd \
    --endpoint 192.168.1.1 \
    --detector lstm \
    --duration 60
```

### 2. HMM 탐지기

**지원 프로토콜**: BFD, BGP

**특징**:
- 상태 전이 기반 이상 탐지
- BFD: 4개 상태 (Up, Down, Init, AdminDown)
- BGP: 6개 상태 (Idle, Connect, Active, OpenSent, OpenConfirm, Established)

**설정 예제**:
```yaml
detectors:
  hmm:
    enabled: true
    protocols: ["bfd", "bgp"]
    config:
      n_states: 3
      anomaly_threshold: 0.6
      use_pretrained: true

      protocol_configs:
        bfd:
          n_states: 4
          anomaly_threshold: 0.7
        bgp:
          n_states: 6
          anomaly_threshold: 0.65
```

**학습 및 탐지**:
```bash
# HMM 학습
python -m ocad.cli train-detector hmm \
    --data data/training/bfd_states.parquet \
    --output ocad/models/hmm/bfd_hmm_v1.pkl

# HMM으로 탐지
python -m ocad.cli detect bfd \
    --endpoint 192.168.1.1 \
    --detector hmm
```

### 3. GNN 탐지기 (Phase 2)

**지원 프로토콜**: BGP

**특징**:
- 그래프 구조 분석
- AS-path 토폴로지 학습
- BGP hijacking 탐지

**설정 예제**:
```yaml
detectors:
  gnn:
    enabled: true
    protocols: ["bgp"]
    config:
      graph_conv_type: "GCN"
      hidden_channels: 128
      num_layers: 3
      anomaly_threshold: 0.75
```

### 4. TCN 탐지기 (Phase 3)

**지원 프로토콜**: PTP, CFM

**특징**:
- 장기 시계열 패턴 학습
- Causal convolution
- Dilation을 통한 receptive field 확장

**설정 예제**:
```yaml
detectors:
  tcn:
    enabled: true
    protocols: ["ptp", "cfm"]
    config:
      num_channels: [64, 128, 256]
      kernel_size: 3
      sequence_length: 100
      anomaly_threshold: 0.7

      protocol_configs:
        ptp:
          sequence_length: 120  # 2분
          anomaly_threshold: 0.75
        cfm:
          sequence_length: 60   # 10분
          anomaly_threshold: 0.7
```

---

## 설정 파일 작성

### 기본 구조

플러그인 설정은 `config/plugins.yaml`에 작성합니다:

```yaml
# Protocol adapters
protocol_adapters:
  <protocol_name>:
    enabled: true/false
    config:
      # Protocol-specific settings

# Detectors
detectors:
  <detector_name>:
    enabled: true/false
    protocols: [list of protocols]
    config:
      # Detector-specific settings

# Global settings
global:
  plugin_dirs: [list of directories]
  log_level: INFO
  max_concurrent_collections: 10
```

### 예제: BFD + LSTM 설정

```yaml
protocol_adapters:
  bfd:
    enabled: true
    config:
      sessions:
        - id: "core-router-1"
          local_ip: "10.0.1.1"
          remote_ip: "10.0.1.2"
          interval_ms: 50
          multiplier: 3

      flapping_detection:
        enabled: true
        window_seconds: 60
        threshold_count: 5

      interval_sec: 1

detectors:
  lstm:
    enabled: true
    protocols: ["bfd"]
    config:
      sequence_length: 30
      anomaly_threshold: 0.75
      use_pretrained: true
      model_dir: "ocad/models/lstm/"

global:
  log_level: INFO
  max_concurrent_collections: 5
  collection_timeout_sec: 30
```

---

## CLI 명령어 레퍼런스

### 플러그인 관리

#### `list-plugins`
사용 가능한 모든 플러그인 목록 표시

```bash
python -m ocad.cli list-plugins [--plugin-dir <path>]
```

**예제**:
```bash
python -m ocad.cli list-plugins
```

#### `plugin-info`
특정 플러그인의 상세 정보 표시

```bash
python -m ocad.cli plugin-info <name> [--plugin-dir <path>]
```

**예제**:
```bash
python -m ocad.cli plugin-info bfd
python -m ocad.cli plugin-info lstm
```

#### `enable-plugin`
플러그인 활성화

```bash
python -m ocad.cli enable-plugin <name> [--config <path>]
```

**예제**:
```bash
python -m ocad.cli enable-plugin bfd
python -m ocad.cli enable-plugin lstm --config config/plugins.yaml
```

#### `disable-plugin`
플러그인 비활성화

```bash
python -m ocad.cli disable-plugin <name> [--config <path>]
```

**예제**:
```bash
python -m ocad.cli disable-plugin gnn
```

#### `test-plugin`
플러그인 단독 테스트

```bash
python -m ocad.cli test-plugin <name> [--plugin-dir <path>]
```

**예제**:
```bash
python -m ocad.cli test-plugin cfm
python -m ocad.cli test-plugin hmm
```

### 탐지기 학습

#### `train-detector`
탐지기 모델 학습

```bash
python -m ocad.cli train-detector <name> \
    --data <path> \
    [--output <path>] \
    [--epochs <n>] \
    [--batch-size <n>]
```

**예제**:
```bash
# LSTM 학습
python -m ocad.cli train-detector lstm \
    --data data/training/bfd_train.parquet \
    --epochs 50 \
    --batch-size 32

# HMM 학습 (커스텀 출력 경로)
python -m ocad.cli train-detector hmm \
    --data data/training/bfd_states.parquet \
    --output models/custom/hmm_bfd.pkl
```

### 실시간 탐지

#### `detect`
실시간 이상 탐지 실행

```bash
python -m ocad.cli detect <protocol> \
    --endpoint <id or ip> \
    [--detector <name>] \
    [--duration <seconds>]
```

**예제**:
```bash
# BFD 모니터링 (모든 호환 탐지기 사용)
python -m ocad.cli detect bfd --endpoint 192.168.1.1 --duration 120

# CFM 모니터링 (LSTM만 사용)
python -m ocad.cli detect cfm --endpoint 192.168.1.100 --detector lstm --duration 60

# PTP 모니터링 (TCN 사용, 5분)
python -m ocad.cli detect ptp --endpoint 10.0.1.1 --detector tcn --duration 300
```

---

## 실전 예제

### 예제 1: BFD 세션 모니터링 (단일 탐지기)

**시나리오**: 코어 라우터 간 BFD 세션을 LSTM으로 모니터링

**1단계: 설정 파일 작성** (`config/plugins.yaml`)
```yaml
protocol_adapters:
  bfd:
    enabled: true
    config:
      sessions:
        - id: "core-r1-r2"
          local_ip: "10.0.1.1"
          remote_ip: "10.0.1.2"
          interval_ms: 50
          multiplier: 3
      interval_sec: 1

detectors:
  lstm:
    enabled: true
    protocols: ["bfd"]
    config:
      sequence_length: 30
      anomaly_threshold: 0.75
      use_pretrained: true
```

**2단계: 학습 데이터 생성**
```bash
python scripts/generate_bfd_data.py \
    --sessions 1 \
    --duration-hours 24 \
    --output data/training/bfd_core_train.parquet
```

**3단계: LSTM 학습**
```bash
python -m ocad.cli train-detector lstm \
    --data data/training/bfd_core_train.parquet \
    --epochs 50 \
    --output ocad/models/lstm/bfd_core_v1.pth
```

**4단계: 실시간 모니터링**
```bash
python -m ocad.cli detect bfd \
    --endpoint 10.0.1.1 \
    --detector lstm \
    --duration 300
```

**예상 출력**:
```
Starting real-time detection
Protocol: bfd
Endpoint: 10.0.1.1
Duration: 300 seconds

Using detectors: ['lstm']

Collecting metrics and detecting anomalies...

✓ [lstm] session_state=2.00 (score=0.12)
✓ [lstm] detection_time_ms=45.20 (score=0.15)
🚨 ANOMALY [lstm] session_state=0.00 (score=0.85)  # BFD Down!
✓ [lstm] session_state=2.00 (score=0.10)
...

===========================================================
Detection Summary
Samples processed: 2100
Anomalies detected: 3
Anomaly rate: 0.1%
```

### 예제 2: 멀티 프로토콜 모니터링

**시나리오**: CFM과 BFD를 동시에 모니터링

**설정**:
```yaml
protocol_adapters:
  cfm:
    enabled: true
    config:
      interval_sec: 10

  bfd:
    enabled: true
    config:
      sessions:
        - id: "bfd-1"
          local_ip: "192.168.1.1"
          remote_ip: "192.168.1.2"
      interval_sec: 1

detectors:
  lstm:
    enabled: true
    protocols: ["cfm", "bfd"]
```

**실행**:
```bash
# Terminal 1: CFM 모니터링
python -m ocad.cli detect cfm --endpoint 192.168.1.100 --detector lstm &

# Terminal 2: BFD 모니터링
python -m ocad.cli detect bfd --endpoint 192.168.1.1 --detector lstm &
```

### 예제 3: 커스텀 임계값 설정

**시나리오**: BFD 플래핑에 민감한 환경 (임계값 낮춤)

**설정**:
```yaml
protocol_adapters:
  bfd:
    enabled: true
    config:
      flapping_detection:
        enabled: true
        window_seconds: 30    # 30초로 단축
        threshold_count: 3    # 3회로 낮춤

detectors:
  lstm:
    enabled: true
    protocols: ["bfd"]
    config:
      protocol_configs:
        bfd:
          anomaly_threshold: 0.6  # 기본 0.75에서 낮춤
```

---

## 문제 해결

### Q1: 플러그인이 발견되지 않음

**증상**:
```
❌ Plugin 'bfd' not found
```

**해결**:
```bash
# 1. 플러그인 디렉토리 확인
ls -la ocad/plugins/protocol_adapters/bfd/

# 2. __init__.py 파일 확인
cat ocad/plugins/protocol_adapters/bfd/__init__.py

# 3. 수동으로 플러그인 발견
python -m ocad.cli list-plugins --plugin-dir ocad/plugins
```

### Q2: 모델 로딩 실패

**증상**:
```
Failed to load model: ocad/models/lstm/bfd_lstm_v1.pth
```

**해결**:
```bash
# 1. 모델 파일 존재 확인
ls -la ocad/models/lstm/

# 2. 모델 재학습
python -m ocad.cli train-detector lstm \
    --data data/training/bfd_train.parquet \
    --output ocad/models/lstm/bfd_lstm_v1.pth

# 3. 설정에서 use_pretrained: false로 변경 (온라인 학습)
```

### Q3: 메트릭 수집 실패

**증상**:
```
Collection failed: Connection timeout
```

**해결**:
```bash
# 1. 엔드포인트 연결 확인
ping 192.168.1.1

# 2. 방화벽 확인
# 3. 설정의 timeout 값 증가
```

**설정 수정**:
```yaml
protocol_adapters:
  bfd:
    config:
      interval_sec: 1
global:
  collection_timeout_sec: 60  # 30초 → 60초로 증가
```

### Q4: 탐지 성능 문제

**증상**: 탐지가 느리거나 메모리 부족

**해결**:
```yaml
global:
  max_concurrent_collections: 5  # 10 → 5로 감소
  model_cache_size: 50           # 100 → 50으로 감소

detectors:
  lstm:
    config:
      batch_size: 16  # 32 → 16으로 감소
```

### Q5: 로그 확인

**위치**:
```bash
# 시스템 로그
tail -f logs/ocad.log

# 플러그인별 로그
tail -f logs/plugins/bfd_adapter.log
tail -f logs/plugins/lstm_detector.log
```

**로그 레벨 변경**:
```yaml
global:
  log_level: DEBUG  # INFO → DEBUG
  log_plugin_lifecycle: true
```

---

## 다음 단계

1. **플러그인 개발**: [Plugin-Development-Guide.md](../07-development/Plugin-Development-Guide.md)
2. **플러그인 아키텍처**: [Plugin-Architecture.md](../05-architecture/Plugin-Architecture.md)
3. **빠른 시작 튜토리얼**: [Plugin-Tutorial.md](../02-user-guides/Plugin-Tutorial.md)

---

## 참고 자료

- [PROTOCOL-ANOMALY-DETECTION-PLAN.md](../PROTOCOL-ANOMALY-DETECTION-PLAN.md): 프로토콜 확장 계획
- [config/plugins.yaml](../../config/plugins.yaml): 전체 설정 예제
- [scripts/test_all_plugins.py](../../scripts/test_all_plugins.py): 통합 테스트 스크립트
