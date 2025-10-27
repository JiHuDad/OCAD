# 추론 가이드

**목적**: 학습된 모델로 이상 탐지를 실행하고 결과를 분석하는 완전한 가이드

**대상**: 모델 학습을 완료하고 실제 데이터에서 이상을 탐지하려는 사용자

---

## 📋 사전 요구사항

### 1. 환경 설정
```bash
# Python venv 활성화
source .venv/bin/activate

# 의존성 설치 확인
pip list | grep -E "pandas|numpy|scikit-learn"
```

### 2. 학습된 모델 (향후)
```bash
# 모델 파일 확인
ls -lh ocad/models/tcn/
ls -lh ocad/models/isolation_forest/
```

**현재**: 룰 기반 탐지만 사용 (모델 로드 불필요)
**향후**: TCN, Isolation Forest 모델 통합

### 3. 추론 데이터
```bash
# 테스트 데이터 생성
python scripts/generate_training_inference_data.py --mode inference
```

---

## 🚀 빠른 시작

### Step 1: 추론 테스트 데이터 생성

```bash
python scripts/generate_training_inference_data.py --mode inference
```

**생성 파일**: `data/inference_test_scenarios.csv`
- 총 780개 레코드
- 정상: 478개 (61.3%)
- 이상: 302개 (38.7%)
- 6가지 시나리오 포함

### Step 2: 추론 실행

```bash
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv \
    --output data/inference_results.csv
```

**예상 출력**:
```
======================================================================
OCAD 추론 실행 (데이터 소스 추상화)
======================================================================
시작 시간: 2025-10-27 07:57:48

설정:
  데이터 소스: data/inference_test_scenarios.csv
  모델 경로: ocad/models/tcn
  임계값: 0.5
  배치 크기: 100
======================================================================

✅ 데이터 소스 생성 완료

데이터 소스 정보:
  source_type: file
  total_records: 780
  batch_size: 100
  start_time: 2025-10-15 12:00:00
  end_time: 2025-10-15 14:09:50
  label_distribution: {'normal': 478, 'anomaly': 302}

추론 실행 중...
  배치 5: 500개 처리됨

✅ 총 780개 레코드 처리 완료

======================================================================
결과 분석
======================================================================

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

✅ 결과 저장: data/inference_results.csv
======================================================================
```

### Step 3: 결과 확인

```bash
# 처음 20개 레코드 확인
head -20 data/inference_results.csv

# 이상 탐지된 케이스만 확인
grep ",anomaly$" data/inference_results.csv | head -10
```

---

## 📊 결과 분석

### 결과 파일 구조

**data/inference_results.csv**:
```csv
timestamp,endpoint_id,udp_echo_rtt,ecpri_delay,lbm_rtt,label,rule_based_score,ecpri_score,lbm_score,composite_score,predicted_label
1760529600000,o-ru-test-001,4.76,93.93,7.33,normal,0.0,0.0,0.0,0.0,normal
1760532080000,o-ru-test-001,12.50,180.45,18.23,anomaly,1.0,0.0,1.0,0.666667,anomaly
```

**컬럼 설명**:
- `timestamp`: Unix timestamp (ms)
- `endpoint_id`: 엔드포인트 ID
- `udp_echo_rtt`: UDP Echo RTT (ms)
- `ecpri_delay`: eCPRI delay (µs)
- `lbm_rtt`: LBM RTT (ms)
- `label`: 실제 라벨 (normal/anomaly)
- `rule_based_score`: 룰 기반 점수 (0-1)
- `ecpri_score`: eCPRI 점수 (0-1)
- `lbm_score`: LBM 점수 (0-1)
- `composite_score`: 종합 점수 (0-1)
- `predicted_label`: 예측 라벨 (normal/anomaly)

### Python으로 분석

```python
import pandas as pd
import matplotlib.pyplot as plt

# 결과 로드
df = pd.read_csv('data/inference_results.csv')

# 1. 예측 분포
print(df['predicted_label'].value_counts())
# normal    565
# anomaly   215

# 2. 정확도
accuracy = (df['label'] == df['predicted_label']).mean() * 100
print(f'정확도: {accuracy:.2f}%')  # 88.85%

# 3. Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df['label'], df['predicted_label'])
print(cm)
# [[478   0]    # 정상을 정상으로: 478, 정상을 이상으로: 0
#  [ 87 215]]   # 이상을 정상으로: 87,  이상을 이상으로: 215

# 4. 성능 지표
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(
    df['label'], df['predicted_label'], average='binary', pos_label='anomaly'
)
print(f'Precision: {precision:.2%}')  # 100.00% (False Positive 0개)
print(f'Recall: {recall:.2%}')        # 71.19% (215/302)
print(f'F1 Score: {f1:.2%}')          # 83.14%

# 5. False Negative 분석
false_negatives = df[(df['label'] == 'anomaly') & (df['predicted_label'] == 'normal')]
print(f'False Negative: {len(false_negatives)}개')
print(false_negatives[['timestamp', 'udp_echo_rtt', 'ecpri_delay', 'lbm_rtt', 'composite_score']].head(10))

# 6. 시계열 시각화
df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
plt.figure(figsize=(15, 5))
plt.plot(df['timestamp_dt'], df['udp_echo_rtt'], label='UDP Echo RTT', alpha=0.7)
plt.scatter(df[df['predicted_label'] == 'anomaly']['timestamp_dt'],
            df[df['predicted_label'] == 'anomaly']['udp_echo_rtt'],
            color='red', label='Anomaly', s=50)
plt.axhline(y=10, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.xlabel('Time')
plt.ylabel('UDP Echo RTT (ms)')
plt.title('Anomaly Detection Results')
plt.show()
```

---

## 🔧 고급 사용법

### 1. 임계값 조정

```bash
# 기본 임계값 (보수적)
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --threshold 0.5

# 민감하게 (더 많이 탐지)
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --threshold 0.3 \
    --rule-threshold 8.0

# 보수적으로 (오탐 최소화)
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --threshold 0.7 \
    --rule-threshold 15.0
```

**파라미터 설명**:
- `--threshold`: 종합 점수 임계값 (0.0-1.0)
- `--rule-threshold`: 룰 기반 임계값 (ms)

### 2. 배치 크기 조정

```bash
# 작은 배치 (메모리 절약)
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --batch-size 50

# 큰 배치 (처리 속도 향상)
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --batch-size 500
```

### 3. 다양한 데이터 형식

```bash
# CSV 파일
python scripts/run_inference.py \
    --data-source data/metrics.csv

# Excel 파일
python scripts/run_inference.py \
    --data-source data/metrics.xlsx

# Parquet 파일 (대용량)
python scripts/run_inference.py \
    --data-source data/metrics.parquet
```

### 4. 실시간 스트리밍 (향후)

```bash
# Kafka
python scripts/run_inference.py \
    --streaming \
    --kafka-broker localhost:9092 \
    --kafka-topic oran-metrics \
    --output results_stream.csv

# WebSocket
python scripts/run_inference.py \
    --streaming \
    --websocket-url ws://oran-server:8080/metrics
```

---

## 🎯 성능 튜닝

### False Negative 줄이기

**문제**: 현재 87개의 False Negative (이상을 정상으로 오판)

**원인 분석**:
```python
# False Negative 레코드 확인
false_negatives = df[(df['label'] == 'anomaly') & (df['predicted_label'] == 'normal')]
print(false_negatives[['udp_echo_rtt', 'composite_score']].describe())

# 대부분 5-10ms 범위 (임계값 10ms 미만)
```

**해결책**:

#### 1. 임계값 낮추기
```bash
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv \
    --rule-threshold 8.0 \
    --threshold 0.3
```

#### 2. 변화점 탐지기 추가
- CUSUM 알고리즘으로 점진적 변화 탐지
- Drift 초기 단계 포착

#### 3. 다변량 탐지 활용
- 여러 메트릭 동시 고려
- UDP RTT는 낮지만 eCPRI delay가 높은 경우 탐지

### False Positive 최소화

**현재**: False Positive 0개 (매우 보수적)

**유지 방법**:
- 임계값을 너무 낮추지 않기
- 여러 탐지기의 합의 요구
- Hold-down timer 적용 (일시적 spike 무시)

---

## 📈 시나리오별 탐지 성능

### 1. Normal (정상 운영)
```
데이터: 180개
탐지 결과: 모두 정상 (100%)
```

### 2. Drift (점진적 증가)
```
데이터: 180개
탐지 결과:
  - 초기 (5→10ms): 미탐지 (임계값 미만)
  - 중기 (10→15ms): 탐지 시작
  - 후기 (15→20ms): 모두 탐지
평균 탐지율: ~70%
```

**개선 방법**: 변화점 탐지기 (CUSUM) 추가

### 3. Spike (급격한 일시적 증가)
```
데이터: 120개
탐지 결과: Spike 발생 시 모두 탐지 (100%)
```

### 4. Jitter (불규칙 변동)
```
데이터: 120개
탐지 결과:
  - 큰 변동: 탐지
  - 작은 변동: 미탐지
평균 탐지율: ~60%
```

### 5. Multi-metric Failure (다중 메트릭 장애)
```
데이터: 90개
탐지 결과: 모두 탐지 (100%)
  - UDP + eCPRI + LBM 동시 이상
```

### 6. Recovery (복구)
```
데이터: 90개
탐지 결과:
  - 이상 구간: 탐지
  - 복구 구간: 정상 판정
```

---

## 🔍 디버깅 가이드

### 1. 추론이 너무 느림

**원인**: 배치 크기가 너무 작음

**해결**:
```bash
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --batch-size 500  # 기본값 100에서 증가
```

### 2. 메모리 부족

**원인**: 대용량 파일 전체 로드

**해결**:
```bash
# 파일 분할
split -l 10000 data/large_metrics.csv data/chunk_

# 청크별 추론
for file in data/chunk_*; do
    python scripts/run_inference.py --data-source $file --output "results_${file}.csv"
done
```

### 3. 정확도가 너무 낮음

**원인**:
- 임계값 부적절
- 학습 데이터와 추론 데이터 분포 다름
- 모델 미통합 (현재 룰 기반만 사용)

**해결**:
```bash
# 1. 임계값 조정
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --threshold 0.3

# 2. 재학습
python scripts/train_model.py \
    --data-source data/new_training_data.csv

# 3. 여러 탐지기 결합 (향후)
```

### 4. False Positive가 많음

**원인**: 임계값이 너무 낮음

**해결**:
```bash
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --threshold 0.7 \
    --rule-threshold 15.0
```

---

## 📚 관련 문서

- [Overview.md](Overview.md) - 학습-추론 개요
- [Training-Guide.md](Training-Guide.md) - 학습 가이드
- [Model-Architecture.md](Model-Architecture.md) - 모델 아키텍처
- [Data-Source-Guide.md](../03-data-management/Data-Source-Guide.md) - 데이터 소스 가이드
- [Training-Inference-Workflow.md](../02-user-guides/Training-Inference-Workflow.md) - 전체 워크플로우

---

## ❓ FAQ

### Q1: 추론 시간은 얼마나 걸리나요?

**A**:
- 780개 레코드: < 1초
- 10,000개 레코드: ~2-3초
- 100,000개 레코드: ~20-30초

### Q2: 실시간 추론이 가능한가요?

**A**: 네, 향후 Kafka/WebSocket 통합 시 실시간 추론 가능합니다.

### Q3: 모델 재학습은 언제 하나요?

**A**:
- ORAN 장비 변경 시
- 정상 패턴 변화 시
- 정확도 저하 시

### Q4: 여러 파일을 한번에 추론할 수 있나요?

**A**: 현재는 개별 실행 필요. 향후 배치 추론 기능 추가 예정.

```bash
# 임시 방법: 스크립트로 반복
for file in data/*.csv; do
    python scripts/run_inference.py --data-source "$file" --output "results_$(basename $file)"
done
```

### Q5: GPU를 사용하나요?

**A**: 현재는 CPU만 사용. 향후 TCN/LSTM 통합 시 GPU 지원 예정.

---

**작성자**: Claude Code
**최종 업데이트**: 2025-10-27
**버전**: 1.0.0
