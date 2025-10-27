# 데이터 소스 추상화 구현 완료 요약

**날짜**: 2025-10-27
**작업 목표**: 학습/추론 시 데이터 소스를 선택할 수 있도록 하고, 향후 실시간 스트리밍 지원을 위한 확장 가능한 설계 구현

---

## 📋 작업 개요

사용자의 요구사항:
> "모델을 학습할때, 데이터를 선택하게 하는게 맞지 않아? 추론 시에도 데이터 선택하도록 했으면 해. 나중을 위해서 적절한 분리 설계를 해줘. 나중에는 실시간으로 데이터가 전달될수 있으니..."

이 요구사항을 충족하기 위해 데이터 소스 추상화 레이어를 설계하고 구현했습니다.

---

## ✅ 완료된 작업

### 1. 핵심 인터페이스 설계 및 구현

**파일**: [ocad/core/data_source.py](../ocad/core/data_source.py) (350+ 줄)

#### 주요 클래스

1. **DataSource (추상 인터페이스)**
   ```python
   class DataSource(ABC):
       def __iter__(self) -> Iterator[DataBatch]
       def get_metadata(self) -> Dict[str, Any]
       def close(self)
   ```
   - 모든 데이터 소스의 공통 인터페이스
   - Iterator 프로토콜로 배치 단위 처리
   - 메타데이터 제공 (총 레코드 수, 시간 범위, 라벨 분포 등)

2. **DataBatch (배치 데이터 구조)**
   ```python
   @dataclass
   class DataBatch:
       metrics: List[Dict[str, Any]]  # 메트릭 레코드
       metadata: Dict[str, Any]       # 배치 메타데이터
   ```
   - 통일된 배치 형식
   - 모든 데이터 소스가 동일한 형식 반환

3. **FileDataSource (파일 기반 구현)**
   - CSV 파일 지원
   - Excel 파일 지원 (openpyxl)
   - Parquet 파일 지원 (pyarrow)
   - Wide/Long 형식 자동 감지
   - 배치 크기 설정 가능 (기본값: 100)

4. **StreamingDataSource (스트리밍 - 인터페이스만)**
   - Kafka, WebSocket, REST API 지원 예정
   - 현재는 인터페이스만 정의 (향후 구현)

5. **DataSourceFactory (팩토리 패턴)**
   ```python
   config = {
       "type": "file",
       "file_path": "data.csv",
       "batch_size": 100
   }
   source = DataSourceFactory.create_from_config(config)
   ```
   - 설정 기반 데이터 소스 생성
   - 타입에 따라 적절한 구현체 반환

### 2. 학습 스크립트 구현

**파일**: [scripts/train_model.py](../scripts/train_model.py) (250+ 줄)

#### 기능
- 데이터 소스 선택 가능 (파일 경로 지정)
- 배치 단위 데이터 로드
- 정상 데이터 검증 (학습은 정상 데이터만 사용)
- 메타데이터 출력 (총 레코드 수, 정상 비율 등)

#### 사용법
```bash
# CSV 파일에서 학습
python scripts/train_model.py \
    --data-source data/training_normal_only.csv \
    --epochs 50 \
    --batch-size 32

# Parquet 파일에서 학습
python scripts/train_model.py \
    --data-source data/training/timeseries_train.parquet \
    --epochs 50
```

#### 테스트 결과
```
✅ 데이터 소스 생성 완료
✅ 데이터 로드 완료: 28,800개
✅ 정상 데이터 비율: 100.0%
```

### 3. 추론 스크립트 구현

**파일**: [scripts/run_inference.py](../scripts/run_inference.py) (280+ 줄)

#### 기능
- 데이터 소스 선택 가능
- 룰 기반 이상 탐지 (UDP Echo RTT, eCPRI delay, LBM RTT)
- 결과 분석 (정확도, Confusion Matrix, 탐지기별 평균 점수)
- 결과 저장 (CSV 파일)

#### 사용법
```bash
# 파일에서 추론
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv \
    --threshold 0.5 \
    --output data/inference_results.csv

# 임계값 조정
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv \
    --rule-threshold 8.0 \
    --threshold 0.3 \
    --output data/inference_results_tuned.csv
```

#### 테스트 결과
```
✅ 총 780개 레코드 처리 완료

예측 분포:
  normal: 565개 (72.4%)
  anomaly: 215개 (27.6%)

정확도: 88.85%

Confusion Matrix:
예측       anomaly  normal
실제
anomaly      215      87
normal         0     478

탐지기별 평균 점수:
  rule_based     : 0.329
  ecpri          : 0.286
  lbm            : 0.254
  composite      : 0.290
```

**성능 분석**:
- True Positive: 215개 (이상 정확히 탐지)
- False Negative: 87개 (이상을 정상으로 오판)
- False Positive: 0개 (정상을 이상으로 오판 없음)
- 정확도: 88.85%

**False Negative 원인**:
- Drift 초기 단계 (점진적 증가 초기)의 미탐지
- 현재 임계값(10ms)보다 낮은 이상값
- 해결 방법: 임계값 조정 또는 변화점 탐지기 추가

### 4. 학습/추론 데이터 생성기

**파일**: [scripts/generate_training_inference_data.py](../scripts/generate_training_inference_data.py) (470+ 줄)

#### 생성 데이터

**학습 데이터** (`data/training_normal_only.csv`):
- 총 28,800개 레코드 (정상 데이터만)
- 8개 엔드포인트 × 1시간 × 60초 간격
- 정상 범위:
  - UDP Echo RTT: 5.0 ± 0.3 ms
  - eCPRI delay: 100.0 ± 5.0 µs
  - LBM RTT: 7.0 ± 0.4 ms

**추론 테스트 데이터** (`data/inference_test_scenarios.csv`):
- 총 780개 레코드
- 정상: 478개 (61.3%)
- 이상: 302개 (38.7%)

**6가지 시나리오**:
1. **Normal (정상 운영)**: 180개
   - 정상 범위 내 변동
2. **Drift (점진적 증가)**: 180개
   - 30분 동안 점진적으로 값 증가
   - UDP RTT: 5ms → 20ms
   - eCPRI: 100µs → 300µs
3. **Spike (급격한 일시적 증가)**: 120개
   - 매 10분마다 spike 발생
   - UDP RTT: 5ms → 20ms (일시적)
4. **Jitter (불규칙 변동)**: 120개
   - 불규칙하게 큰 변동
5. **Multi-metric Failure (다중 메트릭 장애)**: 90개
   - 여러 메트릭 동시 이상
6. **Recovery (복구)**: 90개
   - 이상 → 정상 복구 과정

### 5. 문서 작성

#### 신규 문서
1. **[Data-Source-Abstraction-Design.md](Data-Source-Abstraction-Design.md)** (350+ 줄)
   - 설계 원칙 및 아키텍처
   - 핵심 클래스 설명
   - 사용 예제
   - FAQ

2. **[Data-Source-Abstraction-Summary.md](Data-Source-Abstraction-Summary.md)** (이 문서)
   - 작업 완료 요약
   - 테스트 결과
   - 다음 단계

#### 업데이트된 문서
1. **[CLAUDE.md](../CLAUDE.md)**
   - 최근 작업 내용 업데이트
   - 데이터 소스 추상화 내용 추가

---

## 📊 테스트 결과 요약

### 학습 데이터 로드 테스트

**명령어**:
```bash
python scripts/train_model.py --data-source data/training_normal_only.csv
```

**결과**:
```
✅ 데이터 소스 생성 완료
✅ 데이터 로드 완료: 28,800개
✅ 정상 데이터 비율: 100.0%
   - udp_echo_rtt: 4.1 ~ 5.9 ms
   - ecpri_delay: 79.4 ~ 125.9 µs
   - lbm_rtt: 5.8 ~ 8.3 ms
```

### 추론 테스트

**명령어**:
```bash
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv \
    --output data/inference_results.csv
```

**결과**:
```
총 레코드: 780개
정확도: 88.85%

Confusion Matrix:
         anomaly  normal
anomaly      215      87
normal         0     478
```

**성능 지표**:
- Precision (정밀도): 100% (215/215)
- Recall (재현율): 71.2% (215/302)
- F1 Score: 83.1%

---

## 🏗️ 설계 특징

### 1. 관심사 분리 (Separation of Concerns)

```
┌─────────────────────────────────┐
│  학습/추론 로직 (train/inference) │  ← 비즈니스 로직
└─────────────────────────────────┘
              ▼
┌─────────────────────────────────┐
│  DataSource 인터페이스           │  ← 추상화 레이어
└─────────────────────────────────┘
              ▼
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────┐      ┌─────────────┐
│  File   │      │  Streaming  │  ← 구현체
└─────────┘      └─────────────┘
```

### 2. 확장성 (Extensibility)

- 새로운 데이터 소스 추가 시 기존 코드 수정 불필요
- Factory 패턴으로 손쉬운 확장

```python
# 새 데이터 소스 추가 예제
class KafkaDataSource(DataSource):
    def __iter__(self):
        # Kafka Consumer 로직
        pass

# Factory에 등록만 하면 끝
if source_type == "kafka":
    return KafkaDataSource(config)
```

### 3. 일관성 (Consistency)

- 모든 데이터 소스는 동일한 인터페이스
- 학습/추론 코드는 데이터 소스와 독립적

```python
# 파일이든 스트림이든 동일한 코드
for batch in data_source:
    for metric in batch.metrics:
        process(metric)
```

---

## 🚀 향후 계획

### Phase 2: 스트리밍 데이터 소스 (예정)

**구현 대상**:
1. **Kafka Consumer**
   - Real-time 메트릭 수신
   - 배치 크기만큼 모아서 처리
   - 재연결 로직

2. **WebSocket Client**
   - ORAN 장비에서 실시간 푸시
   - 비동기 처리

3. **REST API Polling**
   - 주기적 폴링
   - Rate limiting 고려

**사용 예제 (향후)**:
```bash
# Kafka에서 실시간 추론
python scripts/run_inference.py \
    --streaming \
    --kafka-broker localhost:9092 \
    --kafka-topic oran-metrics \
    --batch-size 100
```

### Phase 3: 하이브리드 모드

- 파일 기반 학습 + 스트리밍 추론
- 온라인 학습 (Incremental Learning)
- 모델 A/B 테스팅

---

## 📁 파일 구조

```
OCAD/
├── data/
│   ├── training_normal_only.csv              # 학습 데이터 (28,800개)
│   ├── inference_test_scenarios.csv          # 추론 테스트 데이터 (780개)
│   └── inference_results.csv                 # 추론 결과
│
├── ocad/
│   └── core/
│       └── data_source.py                    # 데이터 소스 추상화 (NEW)
│
├── scripts/
│   ├── train_model.py                        # 학습 스크립트 (NEW)
│   ├── run_inference.py                      # 추론 스크립트 (NEW)
│   └── generate_training_inference_data.py   # 데이터 생성기 (NEW)
│
└── docs/
    ├── Data-Source-Abstraction-Design.md     # 설계 문서 (NEW)
    ├── Data-Source-Abstraction-Summary.md    # 요약 문서 (NEW)
    └── Training-Inference-Workflow.md        # 워크플로우 가이드
```

---

## 🎯 핵심 성과

### 사용자 요구사항 충족

✅ **학습 시 데이터 선택 가능**
```bash
python scripts/train_model.py --data-source <파일 경로>
```

✅ **추론 시 데이터 선택 가능**
```bash
python scripts/run_inference.py --data-source <파일 경로>
```

✅ **향후 실시간 스트리밍 지원 준비**
- `StreamingDataSource` 인터페이스 정의
- Factory 패턴으로 확장 용이

✅ **적절한 분리 설계**
- 데이터 소스 계층 분리
- 학습/추론 로직과 독립적
- 새로운 소스 추가 시 기존 코드 수정 불필요

### 기술적 성과

1. **추상화 레벨 적절**
   - 너무 복잡하지 않음
   - 확장 가능
   - 이해하기 쉬움

2. **테스트 완료**
   - 학습 데이터 로드: 28,800개 성공
   - 추론 테스트: 780개 처리, 88.85% 정확도

3. **문서화 완료**
   - 설계 문서
   - 사용 가이드
   - 예제 코드

---

## 📚 참고 문서

### 필수 문서
- [Data-Source-Abstraction-Design.md](Data-Source-Abstraction-Design.md) - 상세 설계 문서
- [Training-Inference-Workflow.md](Training-Inference-Workflow.md) - 학습-추론 워크플로우
- [Quick-Start-Guide.md](Quick-Start-Guide.md) - 빠른 시작 가이드

### 관련 문서
- [File-Based-Input-Implementation-Summary.md](File-Based-Input-Implementation-Summary.md) - 파일 입력 구현
- [Training-Inference-Separation-Design.md](Training-Inference-Separation-Design.md) - 학습-추론 분리 설계

---

## ❓ FAQ

### Q1: 왜 파일 전체를 메모리에 로드하나요?

**A**: 현재 대상 데이터가 중소규모(수만~수십만 건)이므로 전체 로드가 간단하고 빠릅니다. 대용량 데이터는 향후 청크 단위 로딩으로 변경할 수 있습니다.

### Q2: 스트리밍 데이터 소스는 언제 구현되나요?

**A**: CFM 담당자와 협의 후 실시간 데이터 수집 요구사항이 확정되면 구현 예정입니다.

### Q3: 정확도 88.85%가 만족스러운가요?

**A**: False Positive가 0개(정상을 이상으로 오판 없음)이므로 매우 보수적인 탐지입니다. False Negative 87개는 주로 Drift 초기 단계 미탐지로, 임계값 조정이나 변화점 탐지기 추가로 개선 가능합니다.

### Q4: 실제 학습은 언제 통합되나요?

**A**: 현재 `train_model.py`는 데이터 로드까지만 구현되었습니다. TCN/LSTM 모델 학습 로직은 `scripts/train_tcn_model.py`에 있으며, 다음 단계에서 통합 예정입니다.

---

**작성자**: Claude Code
**최종 업데이트**: 2025-10-27
**버전**: 1.0.0
**상태**: ✅ Phase 1 완료
