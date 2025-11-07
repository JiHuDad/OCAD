# OCAD 플러그인 튜토리얼 (5분 빠른 시작)

> **소요 시간**: 5분
> **대상**: 플러그인 시스템 처음 사용하는 사용자

## 1단계: 플러그인 확인 (30초)

사용 가능한 플러그인 목록을 확인합니다:

```bash
# 모든 플러그인 보기
python -m ocad.cli list-plugins
```

**예상 출력**:
```
Protocol Adapters:
┌──────┬─────────┬──────────────────┬──────────────────┐
│ Name │ Version │ Supported        │ Recommended      │
│      │         │ Metrics          │ Models           │
├──────┼─────────┼──────────────────┼──────────────────┤
│ cfm  │ 1.0.0   │ rtt_ms,          │ lstm, tcn        │
│      │         │ loss_rate, ...   │                  │
├──────┼─────────┼──────────────────┼──────────────────┤
│ bfd  │ 1.0.0   │ session_state,   │ lstm, hmm        │
│      │         │ detection_time...│                  │
└──────┴─────────┴──────────────────┴──────────────────┘

Detectors:
┌──────┬─────────┬─────────────────────────┐
│ Name │ Version │ Supported Protocols     │
├──────┼─────────┼─────────────────────────┤
│ lstm │ 1.0.0   │ bfd, bgp, cfm, ptp      │
├──────┼─────────┼─────────────────────────┤
│ hmm  │ 1.0.0   │ bfd, bgp                │
└──────┴─────────┴─────────────────────────┘
```

## 2단계: 플러그인 상세 정보 확인 (30초)

특정 플러그인의 상세 정보를 확인합니다:

```bash
# BFD 어댑터 정보
python -m ocad.cli plugin-info bfd

# LSTM 탐지기 정보
python -m ocad.cli plugin-info lstm
```

**예상 출력**:
```
Protocol Adapter: bfd
Version: 1.0.0
Description: BFD protocol adapter v1.0.0

Supported Metrics:
  • session_state
  • detection_time_ms
  • echo_interval_ms
  • remote_state
  • diagnostic_code
  • multiplier
  • flap_count

Recommended AI Models:
  • lstm
  • hmm

Example Configuration:
protocol_adapters:
  bfd:
    enabled: true
    config:
      # Add protocol-specific config here
```

## 3단계: 플러그인 활성화 (1분)

사용할 플러그인을 활성화합니다:

```bash
# BFD 어댑터 활성화
python -m ocad.cli enable-plugin bfd

# LSTM 탐지기 활성화
python -m ocad.cli enable-plugin lstm
```

**결과**: `config/plugins.yaml` 파일이 생성/업데이트됩니다.

## 4단계: 설정 파일 편집 (1분)

`config/plugins.yaml`을 열어 BFD 세션 정보를 입력합니다:

```yaml
protocol_adapters:
  bfd:
    enabled: true
    config:
      sessions:
        - id: "my-first-bfd-session"
          local_ip: "192.168.1.1"
          remote_ip: "192.168.1.2"
          interval_ms: 50
          multiplier: 3
      interval_sec: 1

detectors:
  lstm:
    enabled: true
    protocols: ["bfd"]
    config:
      anomaly_threshold: 0.75
      use_pretrained: false  # 온라인 학습 사용
```

## 5단계: 플러그인 테스트 (1분)

플러그인이 정상 동작하는지 테스트합니다:

```bash
# BFD 어댑터 테스트 (메트릭 수집)
python -m ocad.cli test-plugin bfd
```

**예상 출력**:
```
Testing plugin: bfd

Protocol Adapter: bfd v1.0.0
✓ Configuration validation: True

Collecting test metrics (5 samples)...
  • session_state: 2.00
  • detection_time_ms: 45.20
  • echo_interval_ms: 50.00
  • remote_state: 2.00
  • diagnostic_code: 0.00

✓ Test passed: collected 5 metrics
```

## 6단계: 실시간 이상 탐지 실행 (1분)

BFD 세션을 실시간으로 모니터링하고 이상 탐지를 수행합니다:

```bash
# BFD 모니터링 (60초 동안)
python -m ocad.cli detect bfd \
    --endpoint 192.168.1.1 \
    --detector lstm \
    --duration 60
```

**예상 출력**:
```
Starting real-time detection
Protocol: bfd
Endpoint: 192.168.1.1
Duration: 60 seconds

Using detectors: ['lstm']

Collecting metrics and detecting anomalies...

✓ [lstm] session_state=2.00 (score=0.12)   # 정상
✓ [lstm] detection_time_ms=45.20 (score=0.08)
✓ [lstm] session_state=2.00 (score=0.10)
🚨 ANOMALY [lstm] session_state=0.00 (score=0.92)  # 이상!
✓ [lstm] session_state=2.00 (score=0.15)
...

===========================================================
Detection Summary
Samples processed: 420
Anomalies detected: 1
Anomaly rate: 0.2%
```

## 완료!

축하합니다! 5분 만에 OCAD 플러그인 시스템의 기본 사용법을 익혔습니다.

## 다음 단계

### 초급: 학습 데이터로 모델 훈련

```bash
# 1. 학습 데이터 생성 (24시간 시뮬레이션)
python scripts/generate_bfd_data.py \
    --sessions 10 \
    --duration-hours 24 \
    --output data/training/bfd_train.parquet

# 2. LSTM 모델 학습
python -m ocad.cli train-detector lstm \
    --data data/training/bfd_train.parquet \
    --epochs 50 \
    --batch-size 32

# 3. 학습된 모델로 탐지
# (config/plugins.yaml에서 use_pretrained: true로 변경)
python -m ocad.cli detect bfd --endpoint 192.168.1.1
```

### 중급: 멀티 프로토콜 모니터링

```bash
# CFM과 BFD 동시 활성화
python -m ocad.cli enable-plugin cfm
python -m ocad.cli enable-plugin bfd

# 설정 파일 편집 (config/plugins.yaml)
# 두 프로토콜 모두 enabled: true로 설정

# 통합 테스트 실행
python scripts/test_all_plugins.py
```

### 고급: 커스텀 플러그인 개발

[Plugin-Development-Guide.md](../07-development/Plugin-Development-Guide.md)를 참조하여 자신만의 프로토콜 어댑터나 탐지기를 개발할 수 있습니다.

## 일반적인 문제

### Q: "Plugin 'bfd' not found" 오류

**해결**:
```bash
# 플러그인 디렉토리 확인
ls ocad/plugins/protocol_adapters/bfd/

# __init__.py 파일 있는지 확인
cat ocad/plugins/protocol_adapters/bfd/__init__.py
```

### Q: 모델 로딩 실패

**해결**:
```yaml
# config/plugins.yaml에서 온라인 학습 사용
detectors:
  lstm:
    config:
      use_pretrained: false  # 사전 학습 모델 없이 실행
```

### Q: 메트릭 수집 타임아웃

**해결**:
```yaml
# config/plugins.yaml에서 타임아웃 증가
global:
  collection_timeout_sec: 60  # 30초 → 60초
```

## 참고 자료

- **상세 사용법**: [Plugin-User-Guide.md](../06-plugins/Plugin-User-Guide.md)
- **플러그인 개발**: [Plugin-Development-Guide.md](../07-development/Plugin-Development-Guide.md)
- **아키텍처**: [Plugin-Architecture.md](../05-architecture/Plugin-Architecture.md)
- **프로토콜 확장 계획**: [PROTOCOL-ANOMALY-DETECTION-PLAN.md](../PROTOCOL-ANOMALY-DETECTION-PLAN.md)

---

**도움이 필요하신가요?** 문서를 읽거나 GitHub Issues에 질문을 남겨주세요.
