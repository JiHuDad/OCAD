# OCAD 스크립트 가이드

**최종 업데이트**: 2025-10-30
**현재 상태**: Phase 1-4 완료 (학습된 모델 즉시 사용 가능)

---

## 📁 스크립트 디렉토리 구조

```
scripts/
├── 📊 데이터 준비
│   ├── prepare_timeseries_data.py       # Phase 1 (UDP Echo 전용)
│   ├── prepare_timeseries_data_v2.py    # Phase 2 (UDP/eCPRI/LBM 범용)
│   └── prepare_multivariate_data.py     # Phase 3 (다변량 피처 생성)
│
├── 🎓 모델 학습
│   ├── train_tcn_model.py               # TCN 모델 학습
│   └── train_isolation_forest.py        # Isolation Forest 학습
│
├── 🚀 추론 실행
│   ├── inference_simple.py              # ⭐ 추천: 간단한 추론
│   └── run_inference.py                 # 기존 추론 스크립트
│
├── ✅ 모델 검증
│   ├── test_integrated_detectors.py     # 통합 탐지기 테스트
│   ├── test_all_tcn_models.py           # TCN 모델 검증
│   ├── test_isolation_forest.py         # Isolation Forest 검증
│   └── validate_all_models.py           # 전체 모델 검증 (4개 데이터셋)
│
└── 📈 성능 테스트
    ├── test_inference_performance.py    # 추론 성능 측정
    └── test_system_integration.py       # 시스템 통합 테스트
```

---

## 🚀 가장 많이 사용하는 스크립트

### 1. 추론 실행 (⭐ 가장 중요!)

**`inference_simple.py`** - 자신의 데이터로 이상 탐지

```bash
python scripts/inference_simple.py \
    --input YOUR_DATA.csv \
    --output results.csv

# 옵션:
#   --input PATH        입력 데이터 (CSV/Excel/Parquet)
#   --output PATH       결과 저장 경로 (기본: 자동 생성)
#   --model-dir PATH    모델 디렉토리 (기본: ocad/models)
#   --no-residual       ResidualDetector 비활성화
#   --no-multivariate   MultivariateDetector 비활성화
```

**입력 데이터 형식**:
- **파일 형식**: CSV, Excel (.xlsx, .xls), Parquet
- **필수 컬럼**: `timestamp, endpoint_id, udp_echo_rtt_ms, ecpri_delay_us, lbm_rtt_ms, ccm_miss_count`

**출력 데이터 형식**:
```csv
timestamp,endpoint_id,residual_score,residual_anomaly,multivariate_score,multivariate_anomaly,final_score,is_anomaly
2025-10-30 00:00:00,endpoint-1,0.0,0,0.0017,0,0.0017,0
```

### 2. 통합 테스트

**`test_integrated_detectors.py`** - 모든 모델 로드 확인

```bash
python scripts/test_integrated_detectors.py

# 출력:
# ✅ udp_echo TCN v2.0.0 (17 epochs)
# ✅ ecpri TCN v2.0.0 (7 epochs)
# ✅ lbm TCN v2.0.0 (6 epochs)
# ✅ Isolation Forest v1.0.0 (20 features)
```

### 3. 전체 모델 검증

**`validate_all_models.py`** - 4개 데이터셋으로 성능 확인

```bash
python scripts/validate_all_models.py

# 출력:
# 정상 데이터: 10.0% 이상 탐지율 (정상)
# 드리프트 이상: 81.0% 이상 탐지율 (우수)
# 스파이크 이상: 26.2% 이상 탐지율 (양호)
# 멀티 엔드포인트: 11.9% 이상 탐지율 (정상)
```

---

## 🎓 모델 학습 스크립트

### 1. TCN 모델 학습

**`train_tcn_model.py`** - 시계열 예측-잔차 탐지 모델

```bash
# Step 1: 데이터 준비
python scripts/prepare_timeseries_data_v2.py \
    --input-csv YOUR_TRAINING_DATA.csv \
    --output-dir data/processed \
    --metric-type udp_echo

# Step 2: 모델 학습
python scripts/train_tcn_model.py \
    --train-data data/processed/timeseries_train.parquet \
    --val-data data/processed/timeseries_val.parquet \
    --test-data data/processed/timeseries_test.parquet \
    --metric-type udp_echo \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001

# 옵션:
#   --metric-type       udp_echo / ecpri / lbm
#   --hidden-size       TCN 히든 레이어 크기 (기본: 32)
#   --sequence-length   입력 시퀀스 길이 (기본: 10)
#   --early-stopping    조기 종료 활성화
```

### 2. Isolation Forest 학습

**`train_isolation_forest.py`** - 다변량 이상 탐지 모델

```bash
# Step 1: 다변량 피처 생성
python scripts/prepare_multivariate_data.py \
    --input YOUR_TRAINING_DATA.csv \
    --output-dir data/processed

# Step 2: 모델 학습
python scripts/train_isolation_forest.py \
    --train-data data/processed/multivariate_train.parquet \
    --val-data data/processed/multivariate_val.parquet \
    --test-data data/processed/multivariate_test.parquet \
    --output ocad/models/isolation_forest/my_model_v1.0.0.pkl \
    --contamination 0.1

# 옵션:
#   --n-estimators      트리 개수 (기본: 100)
#   --contamination     이상 비율 (기본: 0.1)
```

---

## 📊 데이터 준비 스크립트

### 1. 시계열 데이터 준비 (TCN용)

**`prepare_timeseries_data_v2.py`** - 시퀀스 데이터 생성

```bash
python scripts/prepare_timeseries_data_v2.py \
    --input-csv YOUR_DATA.csv \
    --output-dir data/processed \
    --metric-type udp_echo \
    --sequence-length 10 \
    --test-split 0.1 \
    --val-split 0.1

# 생성 파일:
#   - timeseries_train.parquet (학습 데이터 80%)
#   - timeseries_val.parquet (검증 데이터 10%)
#   - timeseries_test.parquet (테스트 데이터 10%)
```

### 2. 다변량 피처 데이터 준비 (Isolation Forest용)

**`prepare_multivariate_data.py`** - 다변량 통계 피처 생성

```bash
python scripts/prepare_multivariate_data.py \
    --input YOUR_DATA.csv \
    --output-dir data/processed \
    --window-size 10

# 생성 파일:
#   - multivariate_train.parquet (20개 피처)
#   - multivariate_val.parquet
#   - multivariate_test.parquet
```

---

## ✅ 검증 스크립트

### 1. 통합 탐지기 테스트

**`test_integrated_detectors.py`**

```bash
python scripts/test_integrated_detectors.py
```

모든 탐지기(ResidualDetector + MultivariateDetector)가 사전 학습된 모델을 정상적으로 로드하는지 확인합니다.

### 2. TCN 모델 검증

**`test_all_tcn_models.py`**

```bash
PYTHONPATH=/home/finux/dev/OCAD:$PYTHONPATH python scripts/test_all_tcn_models.py
```

3개 TCN 모델(UDP Echo, eCPRI, LBM)의 로드 및 추론을 테스트합니다.

### 3. Isolation Forest 검증

**`test_isolation_forest.py`**

```bash
python scripts/test_isolation_forest.py
```

Isolation Forest 모델의 로드 및 이상 탐지를 테스트합니다.

### 4. 전체 모델 검증 (4개 데이터셋)

**`validate_all_models.py`**

```bash
python scripts/validate_all_models.py
```

4개 샘플 데이터셋(정상, 드리프트, 스파이크, 멀티)으로 모든 모델의 성능을 검증합니다.

---

## 📈 성능 테스트 스크립트

### 1. 추론 성능 측정

**`test_inference_performance.py`**

```bash
python scripts/test_inference_performance.py
```

추론 속도, 메모리 사용량, 처리량 등을 측정합니다.

### 2. 시스템 통합 테스트

**`test_system_integration.py`**

```bash
python scripts/test_system_integration.py
```

전체 OCAD 시스템의 end-to-end 통합을 테스트합니다.

---

## 🔍 존재하지 않는 스크립트

다음 스크립트들은 **과거 문서에 언급되었지만 현재는 존재하지 않습니다**:

- ❌ `test_udp_echo_model.py` → ✅ `test_all_tcn_models.py` 사용
- ❌ `generate_training_data.py` → ✅ `prepare_timeseries_data_v2.py` 사용

---

## 💡 Quick Start 예시

### 시나리오 1: 즉시 추론 실행 (학습된 모델 사용)

```bash
source .venv/bin/activate
python scripts/inference_simple.py \
    --input data/samples/01_normal_operation_24h.csv \
    --output data/results/my_result.csv
```

### 시나리오 2: 새 데이터로 모델 재학습

```bash
# TCN 학습
python scripts/prepare_timeseries_data_v2.py \
    --input-csv my_data.csv \
    --output-dir data/processed \
    --metric-type udp_echo

python scripts/train_tcn_model.py \
    --train-data data/processed/timeseries_train.parquet \
    --metric-type udp_echo \
    --epochs 50

# Isolation Forest 학습
python scripts/prepare_multivariate_data.py \
    --input my_data.csv \
    --output-dir data/processed

python scripts/train_isolation_forest.py \
    --train-data data/processed/multivariate_train.parquet \
    --output ocad/models/isolation_forest/new_model_v1.0.0.pkl
```

### 시나리오 3: 모델 검증

```bash
# 통합 테스트
python scripts/test_integrated_detectors.py

# 전체 검증
python scripts/validate_all_models.py
```

---

## 📝 참고 문서

- [README.md](../README.md) - 프로젝트 개요 및 빠른 시작
- [QUICK-STATUS.md](./QUICK-STATUS.md) - 현재 진행 상황
- [PHASE4-COMPLETION-REPORT.md](./PHASE4-COMPLETION-REPORT.md) - Phase 4 완료 리포트
