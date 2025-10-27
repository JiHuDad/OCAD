# 데이터 소스 추상화 설계

**날짜**: 2025-10-27
**버전**: 1.0.0
**목적**: 학습/추론 시 파일 기반 및 실시간 스트리밍 데이터 소스를 통일된 인터페이스로 처리

---

## 개요

OCAD 시스템은 초기에는 파일 기반(CSV/Excel/Parquet) 데이터로 학습 및 추론을 수행하지만, 향후 실시간 스트리밍(Kafka/WebSocket) 데이터 처리도 지원해야 합니다. 이를 위해 데이터 소스를 추상화하여 학습/추론 로직이 데이터 입력 방식과 독립적으로 동작하도록 설계했습니다.

---

## 설계 원칙

### 1. 관심사 분리 (Separation of Concerns)

- **데이터 소스 계층**: 데이터를 어디서 어떻게 읽을 것인가
- **학습/추론 로직 계층**: 데이터를 어떻게 처리할 것인가

### 2. 확장성 (Extensibility)

- 새로운 데이터 소스 추가 시 기존 코드 수정 불필요
- Factory 패턴으로 데이터 소스 생성

### 3. 일관성 (Consistency)

- 모든 데이터 소스는 동일한 인터페이스 제공
- 배치 처리 방식 통일 (Iterator Protocol)

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│              학습/추론 스크립트                            │
│  (train_model.py / run_inference.py)                    │
└─────────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│           DataSourceFactory                             │
│  config → 적절한 DataSource 구현체 생성                   │
└─────────────────────────────────────────────────────────┘
                         ▼
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌───────────────────┐          ┌───────────────────┐
│ FileDataSource    │          │StreamingDataSource│
│ (현재 구현됨)      │          │  (향후 구현)       │
│                   │          │                   │
│ - CSV            │          │ - Kafka           │
│ - Excel          │          │ - WebSocket       │
│ - Parquet        │          │ - REST API        │
└───────────────────┘          └───────────────────┘
        ▼                                 ▼
┌─────────────────────────────────────────────────────────┐
│              DataBatch (통일된 배치 형식)                  │
│  - metrics: List[Dict]                                 │
│  - metadata: Dict                                      │
└─────────────────────────────────────────────────────────┘
```

---

## 핵심 클래스

### DataSource (추상 인터페이스)

```python
class DataSource(ABC):
    """데이터 소스 추상 인터페이스."""

    @abstractmethod
    def __iter__(self) -> Iterator[DataBatch]:
        """배치 단위로 데이터를 반환하는 Iterator."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """데이터 소스 메타데이터 (총 레코드 수, 시간 범위 등)."""
        pass

    @abstractmethod
    def close(self):
        """리소스 정리."""
        pass
```

### DataBatch (배치 데이터 구조)

```python
@dataclass
class DataBatch:
    """배치 데이터 구조."""
    metrics: List[Dict[str, Any]]  # 메트릭 레코드 리스트
    metadata: Dict[str, Any]       # 배치 메타데이터
```

### FileDataSource (파일 기반 구현)

```python
class FileDataSource(DataSource):
    """파일 기반 데이터 소스 (CSV/Excel/Parquet)."""

    def __init__(
        self,
        file_path: Path,
        batch_size: int = 100,
        loader_type: str = "auto"
    ):
        """
        Args:
            file_path: 파일 경로
            batch_size: 배치 크기
            loader_type: "auto", "csv", "excel", "parquet"
        """
        # 파일 확장자에 따라 적절한 로더 선택
        # 전체 데이터를 메모리에 로드 (중소규모 데이터)

    def __iter__(self) -> Iterator[DataBatch]:
        """배치 단위로 데이터 반환."""
        for i in range(0, len(self.data), self.batch_size):
            batch_data = self.data[i:i + self.batch_size]
            yield DataBatch(
                metrics=batch_data,
                metadata={"batch_index": i // self.batch_size}
            )
```

### StreamingDataSource (스트리밍 구현 - 향후)

```python
class StreamingDataSource(DataSource):
    """스트리밍 데이터 소스 (Kafka/WebSocket)."""

    def __init__(
        self,
        source_type: str,  # "kafka", "websocket", "rest"
        connection_config: Dict[str, Any],
        batch_size: int = 100,
        timeout_ms: int = 5000
    ):
        """
        Args:
            source_type: 스트리밍 소스 타입
            connection_config: 연결 설정 (Kafka broker, WebSocket URL 등)
            batch_size: 배치 크기
            timeout_ms: 타임아웃 (밀리초)
        """
        pass

    def __iter__(self) -> Iterator[DataBatch]:
        """실시간 스트림에서 배치 단위로 데이터 수집."""
        # Kafka Consumer Poll
        # WebSocket 메시지 수신
        # 배치 크기만큼 모이면 yield
        pass
```

### DataSourceFactory (Factory 패턴)

```python
class DataSourceFactory:
    """데이터 소스 Factory."""

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> DataSource:
        """설정에서 데이터 소스 생성."""
        source_type = config.get("type")

        if source_type == "file":
            return FileDataSource(
                file_path=Path(config["file_path"]),
                batch_size=config.get("batch_size", 100)
            )
        elif source_type == "kafka":
            return StreamingDataSource(
                source_type="kafka",
                connection_config=config["kafka"],
                batch_size=config.get("batch_size", 100)
            )
        elif source_type == "websocket":
            return StreamingDataSource(
                source_type="websocket",
                connection_config=config["websocket"],
                batch_size=config.get("batch_size", 100)
            )
        else:
            raise ValueError(f"알 수 없는 데이터 소스 타입: {source_type}")
```

---

## 사용 예제

### 1. 학습 스크립트 (`scripts/train_model.py`)

```bash
# 파일에서 학습
python scripts/train_model.py \
    --data-source data/training_normal_only.csv \
    --epochs 50 \
    --batch-size 32

# 또는 Parquet 파일
python scripts/train_model.py \
    --data-source data/training/timeseries_train.parquet \
    --epochs 50
```

**코드**:
```python
# 데이터 소스 생성
data_source_config = {
    "type": "file",
    "file_path": args.data_source,
    "batch_size": args.batch_size
}
data_source = DataSourceFactory.create_from_config(data_source_config)

# 데이터 로드
all_data = []
for batch in data_source:
    all_data.extend(batch.metrics)

# 학습 수행
train_tcn_model(all_data, epochs=args.epochs)
```

### 2. 추론 스크립트 (`scripts/run_inference.py`)

```bash
# 파일에서 추론
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv \
    --threshold 0.5 \
    --output results.csv

# 실시간 Kafka 스트림에서 추론 (향후)
python scripts/run_inference.py \
    --streaming \
    --kafka-broker localhost:9092 \
    --kafka-topic oran-metrics \
    --output results_stream.csv
```

**코드 (파일)**:
```python
data_source_config = {
    "type": "file",
    "file_path": args.data_source,
    "batch_size": args.batch_size
}
data_source = DataSourceFactory.create_from_config(data_source_config)

# 추론 수행
for batch in data_source:
    for metric_data in batch.metrics:
        # 이상 탐지 로직
        score = detect_anomaly(metric_data)
        results.append(score)
```

**코드 (스트리밍 - 향후)**:
```python
data_source_config = {
    "type": "kafka",
    "kafka": {
        "bootstrap_servers": args.kafka_broker,
        "topic": args.kafka_topic,
        "group_id": "ocad-inference"
    },
    "batch_size": 100
}
data_source = DataSourceFactory.create_from_config(data_source_config)

# 실시간 추론
for batch in data_source:  # 무한 루프 (Ctrl+C로 종료)
    for metric_data in batch.metrics:
        score = detect_anomaly(metric_data)
        # 결과 저장 또는 알람 발송
```

---

## 구현 현황

### ✅ 완료 (Phase 1 - 파일 기반)

1. **추상 인터페이스 정의**
   - `DataSource` 추상 클래스
   - `DataBatch` 데이터 구조
   - `DataSourceFactory` 팩토리 클래스

2. **FileDataSource 구현**
   - CSV 파일 지원
   - Excel 파일 지원 (openpyxl)
   - Parquet 파일 지원 (pyarrow)
   - Wide/Long 형식 자동 감지
   - 배치 처리 (Iterator 패턴)

3. **학습 스크립트 통합**
   - `scripts/train_model.py` 생성
   - 데이터 소스 선택 가능
   - 정상 데이터 검증

4. **추론 스크립트 통합**
   - `scripts/run_inference.py` 생성
   - 데이터 소스 선택 가능
   - 결과 분석 및 저장

5. **테스트 완료**
   - 학습 데이터 (28,800개 정상 레코드) 로드 성공
   - 추론 데이터 (780개 레코드, 6가지 시나리오) 처리 성공
   - 정확도: 88.85% (215개 이상 탐지, 87개 False Negative)

### ⏳ 향후 구현 (Phase 2 - 스트리밍)

1. **StreamingDataSource 구현**
   - Kafka Consumer 통합
   - WebSocket 클라이언트 통합
   - REST API 폴링 지원

2. **실시간 추론 파이프라인**
   - 무한 스트림 처리
   - 타임아웃 및 재연결 로직
   - 배압(Backpressure) 관리

3. **하이브리드 모드**
   - 파일 기반 학습 + 스트리밍 추론
   - 온라인 학습 (Incremental Learning)

---

## 설계 결정 사항

### 1. 배치 크기 (Batch Size)

- **기본값**: 100개
- **학습**: 메모리 효율 고려 (전체 데이터를 한번에 로드하지 않음)
- **추론**: 실시간성 고려 (너무 크면 지연 발생)

### 2. Iterator Protocol 선택 이유

- Python의 `for` 루프와 자연스럽게 통합
- 지연 평가(Lazy Evaluation)로 메모리 효율적
- 스트리밍 데이터와 파일 데이터를 동일하게 처리

```python
# 파일이든 스트림이든 동일한 코드
for batch in data_source:
    process(batch)
```

### 3. Metadata 포함 이유

- 데이터 소스 정보 (총 레코드 수, 시간 범위, 라벨 분포)
- 디버깅 및 모니터링 용이
- 결과 재현성 (어떤 데이터로 학습/추론했는지 기록)

---

## 성능 고려사항

### 파일 기반 (FileDataSource)

- **장점**: 간단, 재현 가능, 디버깅 용이
- **단점**: 대용량 파일 시 메모리 부족 가능
- **해결책**:
  - Parquet 파일 사용 (압축 효율 좋음)
  - 청크 단위 로딩 (현재는 전체 로드)

### 스트리밍 (StreamingDataSource - 향후)

- **장점**: 실시간 처리, 무한 데이터 가능
- **단점**: 네트워크 지연, 재연결 필요
- **해결책**:
  - 타임아웃 설정
  - 재연결 로직
  - 로컬 버퍼링

---

## 테스트 방법

### 1. 학습 데이터 로드 테스트

```bash
python scripts/train_model.py \
    --data-source data/training_normal_only.csv \
    --epochs 1
```

**예상 출력**:
```
✅ 데이터 소스 생성 완료
✅ 데이터 로드 완료: 28,800개
✅ 정상 데이터 비율: 100.0%
```

### 2. 추론 테스트

```bash
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv \
    --output data/inference_results.csv
```

**예상 출력**:
```
✅ 총 780개 레코드 처리 완료
정확도: 88.85%
```

### 3. 결과 확인

```bash
# 결과 파일 확인
head -20 data/inference_results.csv

# Python으로 분석
python3 -c "
import pandas as pd
df = pd.read_csv('data/inference_results.csv')
print(df['predicted_label'].value_counts())
print(f'Accuracy: {(df[\"label\"] == df[\"predicted_label\"]).mean() * 100:.2f}%')
"
```

---

## 문서 참조

### 관련 문서

- [Quick-Start-Guide.md](Quick-Start-Guide.md) - 빠른 시작 가이드
- [Training-Inference-Workflow.md](Training-Inference-Workflow.md) - 학습-추론 워크플로우
- [File-Based-Input-Implementation-Summary.md](File-Based-Input-Implementation-Summary.md) - 파일 입력 구현 요약

### 코드 위치

- 데이터 소스: [ocad/core/data_source.py](../ocad/core/data_source.py)
- 학습 스크립트: [scripts/train_model.py](../scripts/train_model.py)
- 추론 스크립트: [scripts/run_inference.py](../scripts/run_inference.py)
- 데이터 생성: [scripts/generate_training_inference_data.py](../scripts/generate_training_inference_data.py)

---

## FAQ

### Q1: 왜 파일 전체를 메모리에 로드하나요?

**A**: 현재는 중소규모 데이터(수만~수십만 건)를 대상으로 하므로 전체 로드가 간단하고 빠릅니다. 향후 대용량 데이터는 청크 단위 로딩으로 변경 예정입니다.

### Q2: 스트리밍 데이터 소스는 언제 구현되나요?

**A**: CFM 담당자와 협의 후 실시간 데이터 수집 요구사항이 확정되면 구현됩니다. 현재는 파일 기반으로 충분합니다.

### Q3: 학습 시 배치 크기와 추론 시 배치 크기가 달라도 되나요?

**A**: 네, 독립적으로 설정 가능합니다. 학습은 메모리 고려해서 작게, 추론은 처리 효율 고려해서 크게 설정할 수 있습니다.

### Q4: Wide 형식과 Long 형식 중 어느 것이 좋나요?

**A**:
- **Wide 형식**: 사람이 읽기 쉬움, Excel 작업 용이
- **Long 형식**: 프로그래밍 처리 용이, 시계열 분석에 유리
- 현재 시스템은 둘 다 지원하며 자동 감지합니다.

---

**작성자**: Claude Code
**최종 업데이트**: 2025-10-27
**버전**: 1.0.0
**상태**: ✅ Phase 1 완료, Phase 2 대기
