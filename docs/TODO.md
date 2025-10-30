# OCAD 작업 TODO 리스트

**작성일**: 2025-10-28
**목적**: 완전한 학습-추론 파이프라인 구축 및 모델 배포

---

## 📋 현재 상태 (2025-10-30 업데이트)

### ✅ 완료된 작업

1. **데이터 파이프라인**
   - ✅ 학습/추론 데이터 생성 스크립트 (`generate_training_inference_data.py`)
   - ✅ 데이터 소스 추상화 (FileDataSource)
   - ✅ 정상 데이터 28,800개, 추론 테스트 데이터 780개 생성
   - ✅ **NEW**: 시계열 시퀀스 변환 스크립트 (`prepare_timeseries_data.py`, `prepare_timeseries_data_v2.py`)
   - ✅ **NEW**: UDP Echo 28,750 시퀀스, eCPRI/LBM 각 1,430 시퀀스 생성

2. **추론 시스템**
   - ✅ 룰 기반 탐지 (Rule-based Detection) 구현
   - ✅ 추론+보고서 통합 스크립트 (`inference_with_report.py`)
   - ✅ 자동 타임스탬프 포함 보고서 생성
   - ✅ 정확도: 정상 100%, 비정상 82.45%

3. **학습된 모델 (Phase 1-2 완료)**
   - ✅ **NEW**: UDP Echo TCN v2.0.0 학습 완료 (17 epochs, R²=0.19)
   - ✅ **NEW**: eCPRI TCN v2.0.0 학습 완료 (7 epochs)
   - ✅ **NEW**: LBM TCN v2.0.0 학습 완료 (6 epochs)
   - ✅ **NEW**: 모델 검증 스크립트 작성 및 테스트 통과

---

## 🔴 다음 작업 (Phase 3-4)

### ✅ 1. **시계열 데이터셋 준비 및 검증** (완료)

**문제**: 현재 TCN 학습 스크립트가 요구하는 데이터 형식 불일치

**해결 방안**:

```bash
# A. 데이터 형식 확인
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/timeseries_train.parquet')
print('컬럼:', df.columns.tolist())
print('샘플:')
print(df.head(3))
"

# B. 필요한 컬럼 확인
# TimeSeriesDataset이 요구하는 컬럼:
# - timestamp
# - endpoint_id
# - metric_type
# - value
# - sequence (시계열 시퀀스, list)
# - target (예측 타겟, float)
# - is_anomaly (라벨, bool)
```

**작업 스텝**:
1. `ocad/training/datasets/timeseries_dataset.py` 요구사항 분석
2. 시계열 시퀀스 생성 스크립트 작성 (`scripts/prepare_timeseries_data.py`)
3. 슬라이딩 윈도우로 sequence 생성 (예: 10개 timestep → 1개 target)
4. 정상 데이터만 사용하여 학습 데이터 준비

**파일 생성**:
- `scripts/prepare_timeseries_data.py` - 시계열 데이터 변환 스크립트

---

### ✅ 2. **TCN 모델 학습** (완료)

**목표**: UDP Echo, eCPRI, LBM 각 메트릭별 TCN 모델 학습

**명령어** (예상):

```bash
# UDP Echo RTT 예측 모델 학습
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --train-data data/processed/timeseries_train.parquet \
    --val-data data/processed/timeseries_val.parquet \
    --test-data data/processed/timeseries_test.parquet \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --early-stopping \
    --patience 10 \
    --output-dir ocad/models/tcn \
    --version v2.0.0

# eCPRI Delay 예측 모델 학습
python scripts/train_tcn_model.py \
    --metric-type ecpri \
    --epochs 100 \
    --batch-size 64 \
    --output-dir ocad/models/tcn \
    --version v2.0.0

# LBM RTT 예측 모델 학습
python scripts/train_tcn_model.py \
    --metric-type lbm \
    --epochs 100 \
    --batch-size 64 \
    --output-dir ocad/models/tcn \
    --version v2.0.0
```

**출력**:
- `ocad/models/tcn/udp_echo_v2.0.0.pth` (학습된 모델)
- `ocad/models/tcn/udp_echo_v2.0.0.json` (메타데이터)
- `ocad/models/metadata/performance_reports/udp_echo_v2.0.0_report.json`

**예상 소요 시간**: 각 메트릭당 30-60분 (GPU 없이 CPU 학습 시)

---

### 3. **Isolation Forest 모델 학습** (30분-1시간)

**목표**: 다변량 이상 탐지 모델 학습

**데이터 준비**:

```bash
# 다변량 데이터 준비 스크립트 작성
python scripts/prepare_multivariate_data.py \
    --input data/training_normal_only.csv \
    --output data/processed/multivariate_train.parquet
```

**학습 명령어** (예상):

```bash
python scripts/train_isolation_forest.py \
    --train-data data/processed/multivariate_train.parquet \
    --test-data data/processed/multivariate_test.parquet \
    --n-estimators 100 \
    --contamination 0.1 \
    --output-dir ocad/models/isolation_forest \
    --version v2.0.0
```

**출력**:
- `ocad/models/isolation_forest/multivariate_v2.0.0.pkl`
- `ocad/models/isolation_forest/multivariate_v2.0.0.json`

---

### 4. **CUSUM (Changepoint Detection) 파라미터 튜닝** (30분)

**목표**: CUSUM 임계값 최적화

**현재 상태**: CUSUM은 통계 기반이므로 학습 불필요, 파라미터 튜닝만 필요

**작업**:

```python
# scripts/tune_cusum_parameters.py
# - 학습 데이터로 최적 임계값 탐색
# - Grid search: threshold, drift
# - 결과를 config 파일로 저장
```

**출력**:
- `config/cusum_params.yaml` (최적화된 파라미터)

---

## 🟡 중요 작업 (우선순위 중간)

### 5. **학습된 모델 통합 및 추론 업데이트** (1-2시간)

**목표**: ResidualDetector와 MultivariateDetector가 학습된 모델 사용

**현재 문제**:
- `run_inference.py`는 룰 기반만 사용
- TCN, Isolation Forest 모델 로딩 필요

**작업**:

```python
# ocad/detectors/residual.py 수정
class ResidualDetector(BaseDetector):
    def __init__(self, model_path: Path):
        # 학습된 TCN 모델 로드
        self.models = {
            'udp_echo': torch.load(model_path / 'udp_echo_v2.0.0.pth'),
            'ecpri': torch.load(model_path / 'ecpri_v2.0.0.pth'),
            'lbm': torch.load(model_path / 'lbm_v2.0.0.pth'),
        }

    def detect(self, features: dict) -> float:
        # 예측 수행 및 잔차 계산
        residual = actual - predicted
        return anomaly_score

# ocad/detectors/multivariate.py 수정
class MultivariateDetector(BaseDetector):
    def __init__(self, model_path: Path):
        # Isolation Forest 로드
        self.model = joblib.load(model_path / 'multivariate_v2.0.0.pkl')

    def detect(self, features: dict) -> float:
        # Isolation Forest 예측
        return self.model.score_samples([features])
```

**업데이트 파일**:
- `ocad/detectors/residual.py`
- `ocad/detectors/multivariate.py`
- `scripts/inference_with_report.py` (모델 로딩 로직 추가)

---

### 6. **ONNX 변환 스크립트 작성** (1시간)

**목표**: PyTorch TCN 모델을 ONNX로 변환

**스크립트 작성**:

```bash
# scripts/convert_to_onnx.py
python scripts/convert_to_onnx.py \
    --model-path ocad/models/tcn/udp_echo_v2.0.0.pth \
    --output ocad/models/onnx/udp_echo_v2.0.0.onnx \
    --input-shape 1,10,1  # (batch, sequence_length, features)
```

**검증**:

```bash
# ONNX 모델 검증
python scripts/verify_onnx_model.py \
    --onnx-model ocad/models/onnx/udp_echo_v2.0.0.onnx \
    --pytorch-model ocad/models/tcn/udp_echo_v2.0.0.pth \
    --test-data data/processed/timeseries_test.parquet
```

**출력**:
- `ocad/models/onnx/udp_echo_v2.0.0.onnx`
- `ocad/models/onnx/ecpri_v2.0.0.onnx`
- `ocad/models/onnx/lbm_v2.0.0.onnx`

---

### 7. **ONNX 추론 성능 테스트** (30분)

**목표**: ONNX vs PyTorch 추론 속도 비교

```bash
python scripts/benchmark_onnx_inference.py \
    --pytorch-model ocad/models/tcn/udp_echo_v2.0.0.pth \
    --onnx-model ocad/models/onnx/udp_echo_v2.0.0.onnx \
    --test-data data/processed/timeseries_test.parquet \
    --num-samples 1000
```

**출력**:
- `reports/onnx_benchmark_report.md` (추론 속도, 메모리 사용량 비교)

---

## 🟢 개선 작업 (우선순위 낮음)

### 8. **전체 파이프라인 통합 테스트** (1시간)

**목표**: 데이터 생성 → 학습 → 추론 → 보고서 전체 워크플로우 검증

```bash
# 전체 파이프라인 실행 스크립트
./scripts/run_full_pipeline.sh
```

**테스트 시나리오**:
1. 정상 데이터로 학습
2. 정상+비정상 데이터로 추론
3. 보고서 생성 및 검증
4. 성능 지표 검증 (Precision > 95%, Recall > 70%)

---

### 9. **하이퍼파라미터 튜닝** (2-4시간)

**목표**: 각 모델의 최적 하이퍼파라미터 탐색

```bash
# Optuna를 사용한 하이퍼파라미터 튜닝
python scripts/tune_hyperparameters.py \
    --model-type tcn \
    --metric-type udp_echo \
    --n-trials 50 \
    --output config/best_hyperparams.yaml
```

**탐색 공간**:
- TCN: hidden_size, num_layers, kernel_size, dropout
- Isolation Forest: n_estimators, max_samples, contamination

---

### 10. **모델 앙상블 최적화** (1-2시간)

**목표**: RuleBasedDetector, ChangepointDetector, ResidualDetector, MultivariateDetector의 가중치 최적화

**현재 가중치** (CompositeDetector):
- Rule-based: 0.4
- Changepoint: 0.2
- Residual: 0.3
- Multivariate: 0.1

**최적화 방법**:

```python
# scripts/optimize_ensemble_weights.py
# - 검증 데이터로 가중치 최적화
# - Grid search 또는 Bayesian Optimization
# - F1 Score 최대화
```

---

### 11. **실시간 스트리밍 데이터 소스 구현** (2-3시간)

**목표**: Kafka/WebSocket 스트리밍 데이터 지원

```python
# ocad/core/data_source.py 확장
class StreamingDataSource(DataSource):
    def __init__(self, kafka_broker: str, topic: str):
        self.consumer = KafkaConsumer(topic, ...)

    def __iter__(self):
        for message in self.consumer:
            yield self._parse_message(message)
```

---

## 📊 작업 우선순위 요약

### 🔴 내일 (2025-10-29) 필수 작업

1. **시계열 데이터셋 준비** (1-2시간)
   - `scripts/prepare_timeseries_data.py` 작성
   - 슬라이딩 윈도우로 sequence 생성

2. **TCN 모델 학습** (3-6시간, 시간 제한 없음)
   - UDP Echo 학습
   - eCPRI 학습
   - LBM 학습

3. **Isolation Forest 학습** (1시간)
   - 다변량 데이터 준비
   - 모델 학습

### 🟡 이번 주 (2025-10-30 ~ 11-01) 작업

4. **학습된 모델 통합** (1-2시간)
5. **ONNX 변환** (1시간)
6. **ONNX 성능 테스트** (30분)
7. **CUSUM 파라미터 튜닝** (30분)

### 🟢 다음 주 작업

8. **전체 파이프라인 테스트** (1시간)
9. **하이퍼파라미터 튜닝** (2-4시간)
10. **앙상블 가중치 최적화** (1-2시간)
11. **스트리밍 데이터 소스** (2-3시간)

---

## 📁 예상 디렉토리 구조 (완료 후)

```
OCAD/
├── data/
│   ├── training/
│   │   ├── normal_data.parquet          # 정상 데이터 (28,800개)
│   │   ├── timeseries_train.parquet     # 시계열 학습 (sequence + target)
│   │   ├── timeseries_val.parquet       # 시계열 검증
│   │   ├── timeseries_test.parquet      # 시계열 테스트
│   │   ├── multivariate_train.parquet   # 다변량 학습
│   │   └── multivariate_test.parquet    # 다변량 테스트
│   ├── inference/
│   │   ├── normal_only.csv              # 정상만
│   │   └── anomaly_only.csv             # 비정상만
│   └── results/
│       ├── inference_results_*.csv      # 추론 결과
│
├── ocad/models/
│   ├── tcn/
│   │   ├── udp_echo_v2.0.0.pth          # ✅ 학습 완료
│   │   ├── ecpri_v2.0.0.pth             # ✅ 학습 완료
│   │   └── lbm_v2.0.0.pth               # ✅ 학습 완료
│   ├── isolation_forest/
│   │   └── multivariate_v2.0.0.pkl      # ✅ 학습 완료
│   ├── onnx/
│   │   ├── udp_echo_v2.0.0.onnx         # ✅ ONNX 변환
│   │   ├── ecpri_v2.0.0.onnx
│   │   └── lbm_v2.0.0.onnx
│   └── metadata/
│       └── performance_reports/
│
├── reports/
│   ├── inference_report_*.md            # 타임스탬프 포함
│   ├── training_report_*.md             # 학습 리포트
│   ├── onnx_benchmark_report.md         # ONNX 성능 비교
│   └── hyperparameter_tuning_report.md  # 튜닝 결과
│
└── scripts/
    ├── prepare_timeseries_data.py       # ⚠️ 작성 필요
    ├── prepare_multivariate_data.py     # ⚠️ 작성 필요
    ├── train_tcn_model.py               # ✅ 기존 (수정 필요)
    ├── train_isolation_forest.py        # ⚠️ 확인 필요
    ├── tune_cusum_parameters.py         # ⚠️ 작성 필요
    ├── convert_to_onnx.py               # ⚠️ 작성 필요
    ├── verify_onnx_model.py             # ⚠️ 작성 필요
    ├── benchmark_onnx_inference.py      # ⚠️ 작성 필요
    ├── inference_with_report.py         # ✅ 완료
    └── run_full_pipeline.sh             # ⚠️ 작성 필요
```

---

## 🎯 최종 목표 (이번 주 내)

1. ✅ **완전한 학습 파이프라인**: 시계열 데이터 → TCN 학습 → 모델 저장
2. ✅ **완전한 추론 파이프라인**: 데이터 입력 → 4가지 탐지기 (Rule, Changepoint, TCN, IF) → 보고서
3. ✅ **ONNX 모델**: 배포 가능한 ONNX 형식 변환
4. ✅ **성능 검증**: Precision > 95%, Recall > 70%, False Positive < 5%

---

## 📝 참고사항

### 데이터 형식 이슈

**현재 문제**:
- `TimeSeriesDataset`은 `sequence`와 `target` 컬럼을 요구
- 현재 데이터는 평면 구조 (timestamp, value)

**해결책**:
```python
# 슬라이딩 윈도우 예시
# Input: [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]
# Output:
# - sequence: [t1, t2, ..., t10], target: t11
# - sequence: [t2, t3, ..., t11], target: t12
# - ...
```

### 학습 시간 예상

- **TCN (각 메트릭)**: CPU 30-60분, GPU 5-10분
- **Isolation Forest**: CPU 10-20분
- **전체**: 약 2-4시간 (CPU 기준)

### 체크리스트

- [ ] 시계열 데이터 변환 스크립트 작성
- [ ] TCN UDP Echo 학습 (제한 없음)
- [ ] TCN eCPRI 학습 (제한 없음)
- [ ] TCN LBM 학습 (제한 없음)
- [ ] Isolation Forest 학습
- [ ] CUSUM 파라미터 튜닝
- [ ] 학습된 모델 통합 (ResidualDetector, MultivariateDetector)
- [ ] ONNX 변환 스크립트 작성
- [ ] ONNX 변환 실행
- [ ] ONNX 추론 성능 테스트
- [ ] 전체 파이프라인 통합 테스트

---

**문서 업데이트**: 작업 진행 시 이 문서를 체크리스트로 활용하고, 완료된 항목은 ✅로 표시해주세요.
