# TCN 모델 학습 가이드

## 개요

이 가이드는 OCAD 시스템에서 TCN (Temporal Convolutional Network) 모델을 학습하고 배포하는 방법을 설명합니다.

## 빠른 시작

### 전체 프로세스 자동 실행

가장 간단한 방법은 통합 스크립트를 사용하는 것입니다:

```bash
# 데이터 생성 + 모든 모델 학습 (약 10-15분 소요)
./scripts/train_all_models.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
1. 훈련 데이터 생성 (64,500개 시퀀스)
2. UDP Echo 모델 학습
3. eCPRI 모델 학습
4. LBM 모델 학습

### 단계별 실행

#### 옵션 1: 데이터만 생성

```bash
./scripts/train_all_models.sh --data-only
```

#### 옵션 2: 학습만 수행 (데이터가 이미 있는 경우)

```bash
./scripts/train_all_models.sh --train-only
```

## 데이터 확인

### 데이터 요약 보기

```bash
# Train 데이터 확인 (샘플 3개 출력)
python scripts/view_training_data.py --split train --samples 3

# Validation 데이터 확인
python scripts/view_training_data.py --split val --samples 3

# Test 데이터 확인
python scripts/view_training_data.py --split test --samples 3

# 전체 분할 비교
python scripts/view_training_data.py --split all
```

**출력 예시:**
```
================================================================================
데이터 파일: ocad/data/training/timeseries_train.parquet
================================================================================

총 시퀀스 수: 45,150
컬럼 수: 7
파일 크기: 2.15 MB

메트릭 타입 분포:
  lbm            : 15,060개 (33.36%)
  ecpri          : 15,058개 (33.35%)
  udp_echo       : 15,032개 (33.29%)

이상 레이블 분포:
  정상:   38,411개 (85.07%)
  이상:    6,739개 (14.93%)

이상 타입 분포:
  spike     :  2,321개
  loss      :  2,211개
  drift     :  2,207개
```

### CSV로 내보내기

데이터를 Excel이나 다른 도구로 확인하려면 CSV로 내보낼 수 있습니다:

```bash
# 처음 50개 샘플을 CSV로 내보내기
python scripts/view_training_data.py --split train --samples 0 --export-csv --csv-samples 50

# 출력: ocad/data/training/timeseries_train_sample.csv
```

CSV 파일은 다음과 같은 형식입니다:
```csv
endpoint_id,metric_type,timestamp_ms,target,is_anomaly,anomaly_type,seq_0,seq_1,seq_2,...,seq_9
o-ru-004,lbm,1761129123404,1.9196,False,,1.68,1.95,2.79,2.81,...
o-ru-006,lbm,1761113323535,24.33,True,spike,5.79,7.31,7.35,5.22,...
```

### 데이터 위치

생성된 데이터는 다음 위치에 저장됩니다:

```
ocad/data/training/
├── timeseries_train.parquet     # 훈련 데이터 (70%, 45,150개)
├── timeseries_val.parquet       # 검증 데이터 (15%, 9,675개)
├── timeseries_test.parquet      # 테스트 데이터 (15%, 9,675개)
└── timeseries_train_sample.csv  # CSV 샘플 (선택적)
```

## 개별 모델 학습

### 1. 훈련 데이터 생성

```bash
python scripts/generate_training_data.py \
    --dataset-type timeseries \
    --endpoints 10 \
    --duration-hours 6 \
    --anomaly-rate 0.15 \
    --output-dir ocad/data/training
```

**파라미터 설명:**
- `--endpoints`: 시뮬레이션할 엔드포인트 수 (기본: 10)
- `--duration-hours`: 데이터 수집 기간 (시간) (기본: 24)
- `--anomaly-rate`: 이상 비율 0.0-1.0 (기본: 0.1)
- `--output-dir`: 출력 디렉토리

### 2. 모델 학습

#### UDP Echo 모델

```bash
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --epochs 50 \
    --batch-size 32 \
    --hidden-size 32 \
    --learning-rate 0.001 \
    --output-dir ocad/models/tcn
```

#### eCPRI 모델

```bash
python scripts/train_tcn_model.py \
    --metric-type ecpri \
    --epochs 50 \
    --batch-size 32 \
    --hidden-size 32 \
    --learning-rate 0.001 \
    --output-dir ocad/models/tcn
```

#### LBM 모델

```bash
python scripts/train_tcn_model.py \
    --metric-type lbm \
    --epochs 50 \
    --batch-size 32 \
    --hidden-size 32 \
    --learning-rate 0.001 \
    --output-dir ocad/models/tcn
```

**파라미터 설명:**
- `--metric-type`: 학습할 메트릭 (udp_echo, ecpri, lbm)
- `--epochs`: 최대 에포크 수 (조기 종료 적용)
- `--batch-size`: 배치 크기
- `--hidden-size`: TCN 히든 레이어 크기
- `--learning-rate`: 학습률
- `--output-dir`: 모델 저장 디렉토리

## 학습 결과 확인

### 모델 파일

학습이 완료되면 다음 파일들이 생성됩니다:

```
ocad/models/tcn/
├── udp_echo_v1.0.0.pth          # UDP Echo 모델 (7.2 KB)
├── udp_echo_v1.0.0.json         # 메타데이터
├── ecpri_v1.0.0.pth             # eCPRI 모델 (17 KB)
├── ecpri_v1.0.0.json            # 메타데이터
├── lbm_v1.0.0.pth               # LBM 모델 (17 KB)
└── lbm_v1.0.0.json              # 메타데이터

ocad/models/metadata/performance_reports/
├── udp_echo_v1.0.0_report.json  # 성능 리포트
├── ecpri_v1.0.0_report.json     # 성능 리포트
└── lbm_v1.0.0_report.json       # 성능 리포트
```

### 모델 메타데이터 확인

```bash
# 메타데이터 확인
cat ocad/models/tcn/udp_echo_v1.0.0.json | python -m json.tool

# 성능 리포트 확인
cat ocad/models/metadata/performance_reports/udp_echo_v1.0.0_report.json | python -m json.tool
```

**메타데이터 예시:**
```json
{
  "metric_type": "udp_echo",
  "version": "1.0.0",
  "model_architecture": "SimpleTCN",
  "hidden_size": 32,
  "sequence_length": 10,
  "trained_at": "2025-10-22 05:32:15"
}
```

**성능 리포트 예시:**
```json
{
  "test_mse": 1.2345,
  "test_mae": 0.6585,
  "test_rmse": 1.1111,
  "test_r2": 0.1341,
  "residual_mean": 0.0153,
  "residual_std": 1.1040
}
```

## 추론 성능 테스트

학습된 모델의 추론 성능을 측정할 수 있습니다:

```bash
python scripts/test_inference_performance.py \
    --model-dir ocad/models/tcn \
    --num-samples 100 \
    --device cpu
```

**출력 예시:**
```
추론 지연 시간 통계
평균:     1.92 ms
중앙값:   1.66 ms
P95:      3.64 ms
P99:      8.33 ms

메모리 사용량
시작:         325.20 MB
모델 로드 후: 326.96 MB (+1.77 MB)
추론 후:      328.96 MB (+2.00 MB)

처리량
총 샘플:      100
총 시간:      0.192 초
처리량:       519.56 샘플/초

목표 달성 여부
P95 < 100.0 ms: ✅ 달성 (실제: 3.64 ms)
P99 < 200.0 ms: ✅ 달성 (실제: 8.33 ms)
```

## 시스템 통합 테스트

모델이 전체 시스템에서 정상 작동하는지 테스트:

```bash
python scripts/test_system_integration.py
```

**출력 예시:**
```
SystemOrchestrator 초기화 테스트
✅ Orchestrator 초기화 성공

등록된 탐지기 수: 4
  1. RuleBasedDetector
  2. ChangePointDetector
  3. ResidualDetectorV2
     로드된 모델:
       - udp_echo: ✅
       - ecpri: ✅
       - lbm: ✅
  4. MultivariateDetector

피처 기반 이상 탐지 테스트
✅ 정상 피처 벡터: 정상으로 판정됨
✅ 이상 피처 벡터: 추론 정상 작동
```

## 모델 성능 개선

### LBM 모델 성능 향상

현재 LBM 모델의 R² 점수가 낮은 경우 (-0.072), 다음 방법으로 개선할 수 있습니다:

#### 1. 더 많은 데이터 생성

```bash
python scripts/generate_training_data.py \
    --endpoints 20 \
    --duration-hours 24 \
    --anomaly-rate 0.15 \
    --output-dir ocad/data/training_large
```

#### 2. 하이퍼파라미터 튜닝

```bash
# 히든 크기 증가
python scripts/train_tcn_model.py \
    --metric-type lbm \
    --hidden-size 64 \
    --epochs 100 \
    --learning-rate 0.0005

# 더 깊은 네트워크
python scripts/train_tcn_model.py \
    --metric-type lbm \
    --hidden-size 64 \
    --num-layers 4 \
    --epochs 100
```

#### 3. 시퀀스 길이 증가

더 긴 시퀀스를 사용하려면 데이터 생성 스크립트를 수정해야 합니다:

```python
# scripts/generate_training_data.py 수정
SEQUENCE_LENGTH = 20  # 기본 10에서 20으로 변경
```

## 모델 배포

### 1. 설정 파일 업데이트

`config/local.yaml`:
```yaml
detection:
  use_pretrained_models: true
  pretrained_model_dir: "ocad/models/tcn"
  inference_device: "cpu"  # 또는 "cuda" / "mps"
```

### 2. 모델 버전 관리

새 버전의 모델을 배포할 때:

```bash
# 기존 모델 백업
cp ocad/models/tcn/udp_echo_v1.0.0.pth ocad/models/tcn/backup/

# 새 모델 학습
python scripts/train_tcn_model.py --metric-type udp_echo

# 새 버전으로 저장 (스크립트가 자동으로 v1.0.0 → v1.1.0으로 증가)
```

### 3. 시스템 재시작

```bash
# API 서버 재시작
python -m ocad.api.main
```

새로운 모델은 자동으로 로드됩니다.

## 트러블슈팅

### 문제: 데이터 파일을 찾을 수 없음

```
FileNotFoundError: [Errno 2] No such file or directory: 'ocad/data/training/timeseries_train.parquet'
```

**해결책:**
```bash
# 데이터 생성
python scripts/generate_training_data.py --output-dir ocad/data/training
```

### 문제: 모델 학습 중 메모리 부족

```
RuntimeError: CUDA out of memory
```

**해결책:**
```bash
# 배치 크기 감소
python scripts/train_tcn_model.py --metric-type udp_echo --batch-size 16

# 또는 CPU 사용
python scripts/train_tcn_model.py --metric-type udp_echo --device cpu
```

### 문제: 모델 성능이 낮음 (R² < 0)

**해결책:**
1. 더 많은 데이터 생성
2. 하이퍼파라미터 튜닝
3. 데이터 품질 확인 (이상 비율, 분포 등)

```bash
# 데이터 확인
python scripts/view_training_data.py --split all

# 더 많은 데이터로 재학습
python scripts/generate_training_data.py --endpoints 20 --duration-hours 24
```

### 문제: 추론 속도가 느림

**해결책:**
```bash
# GPU 사용 (CUDA 지원 시스템)
python scripts/train_tcn_model.py --device cuda

# 설정 파일 업데이트
# config/local.yaml:
#   inference_device: "cuda"
```

## 참고 자료

- **전체 설계 문서**: [docs/Training-Inference-Separation-Design.md](Training-Inference-Separation-Design.md)
- **Phase 3 결과**: [docs/Phase3-Implementation-Summary.md](Phase3-Implementation-Summary.md)
- **Phase 4 결과**: [docs/Phase4-Implementation-Summary.md](Phase4-Implementation-Summary.md)

## 요약: 처음부터 끝까지 전체 프로세스

```bash
# 1. 데이터 생성 및 모든 모델 학습
./scripts/train_all_models.sh

# 2. 데이터 확인
python scripts/view_training_data.py --split all

# 3. 추론 성능 테스트
python scripts/test_inference_performance.py --model-dir ocad/models/tcn

# 4. 시스템 통합 테스트
python scripts/test_system_integration.py

# 5. OCAD 시스템 시작
python -m ocad.api.main
```

**예상 소요 시간:**
- 데이터 생성: 1-2분
- 전체 모델 학습: 5-10분
- 테스트: 1-2분
- **총 소요 시간: 약 10-15분**

---

**작성 일시**: 2025-10-22
**작성자**: Claude Code
**버전**: 1.0.0
