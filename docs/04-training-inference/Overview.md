# 학습 및 추론 개요

**목적**: OCAD의 학습-추론 분리 아키텍처와 전체 워크플로우를 이해하기

**읽는 시간**: 5-10분

---

## 🎯 핵심 개념

### 학습-추론 분리 아키텍처

OCAD는 **온라인 학습**에서 **오프라인 학습 + 온라인 추론**으로 전환되었습니다.

```
┌─────────────────────────────────────────────┐
│          전통적 온라인 학습 (AS-IS)           │
├─────────────────────────────────────────────┤
│  데이터 수집 → [학습 + 추론] → 알람          │
│  - 장점: 자동화                              │
│  - 단점: 예측 불가능한 지연, 재현 불가       │
└─────────────────────────────────────────────┘

                    ↓ 변경

┌─────────────────────────────────────────────┐
│        학습-추론 분리 아키텍처 (TO-BE)        │
├─────────────────────────────────────────────┤
│  📚 학습 (Offline):                         │
│     정상 데이터 → 모델 학습 → 모델 저장      │
│                                             │
│  🔍 추론 (Online):                          │
│     데이터 수집 → [모델 로드] → 추론 → 알람  │
│  - 장점: 일관된 지연, 재현 가능, 버전 관리   │
└─────────────────────────────────────────────┘
```

---

## 📚 학습 (Training)

### 왜 정상 데이터만 사용하나요?

OCAD는 **이상 탐지(Anomaly Detection)** 시스템입니다:

- **정상 상태**를 학습하여 패턴 이해
- **정상에서 벗어난 것**을 이상으로 탐지
- 이상 데이터는 종류가 무한하므로 정상만 학습

```python
# 이상 탐지의 원리
학습 데이터: [정상, 정상, 정상, ...] → 모델
추론 데이터: [정상, 정상, 이상!, 정상, 이상!]
             ↓
결과:       [OK,   OK,   ⚠️,   OK,   ⚠️ ]
```

### 학습 데이터 준비

```bash
# 1. 학습 데이터 생성 (정상 데이터 28,800개)
python scripts/generate_training_inference_data.py --mode training

# 생성 파일: data/training_normal_only.csv
# - 8개 엔드포인트
# - 1시간 × 60초 간격 = 3,600개/엔드포인트
# - 총 28,800개 레코드
```

**학습 데이터 특징**:
- UDP Echo RTT: 5.0 ± 0.3 ms
- eCPRI delay: 100.0 ± 5.0 µs
- LBM RTT: 7.0 ± 0.4 ms
- 라벨: 모두 "normal"

### 모델 학습

```bash
# 2. 모델 학습
python scripts/train_model.py \
    --data-source data/training_normal_only.csv \
    --epochs 50 \
    --batch-size 32
```

**학습되는 모델**:
1. **TCN (Temporal Convolutional Network)** - 시계열 예측
2. **Isolation Forest** - 다변량 이상 탐지
3. **CUSUM** - 변화점 탐지 (통계 기반, 학습 불필요)

**출력**:
- `ocad/models/tcn/udp_echo_v1.0.0.pth`
- `ocad/models/tcn/ecpri_delay_v1.0.0.pth`
- `ocad/models/isolation_forest/multivariate_v1.0.0.pkl`

**상세 가이드**: [Training-Guide.md](Training-Guide.md)

---

## 🔍 추론 (Inference)

### 추론 데이터 준비

```bash
# 1. 추론 테스트 데이터 생성 (6가지 시나리오)
python scripts/generate_training_inference_data.py --mode inference

# 생성 파일: data/inference_test_scenarios.csv
# - 정상: 478개 (61.3%)
# - 이상: 302개 (38.7%)
```

**6가지 시나리오**:
1. **Normal** - 정상 운영
2. **Drift** - 점진적 증가 (30분)
3. **Spike** - 급격한 일시적 증가
4. **Jitter** - 불규칙 변동
5. **Multi-metric Failure** - 다중 메트릭 장애
6. **Recovery** - 이상 → 정상 복구

### 추론 실행

```bash
# 2. 추론 실행
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv \
    --output data/inference_results.csv

# 결과:
# - 정확도: 88.85%
# - True Positive: 215개 (이상 정확히 탐지)
# - False Negative: 87개 (이상 미탐지)
# - False Positive: 0개 (정상을 이상으로 오판 없음)
```

### 결과 분석

```bash
# 3. 결과 확인
head -20 data/inference_results.csv

# Python으로 분석
python3 -c "
import pandas as pd
df = pd.read_csv('data/inference_results.csv')
print(df['predicted_label'].value_counts())
print(f'Accuracy: {(df[\"label\"] == df[\"predicted_label\"]).mean() * 100:.2f}%')
"
```

**상세 가이드**: [Inference-Guide.md](Inference-Guide.md)

---

## 🔄 전체 워크플로우

### 최초 1회: 모델 학습

```bash
# Step 1: 정상 데이터 수집 (실제 환경)
# - 프로토콜 담당자가 네트워크 장비에서 수집
# - 또는 scripts/generate_training_inference_data.py로 생성

# Step 2: 모델 학습
python scripts/train_model.py \
    --data-source data/training_normal_only.csv \
    --epochs 50

# Step 3: 모델 검증
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv
```

### 운영: 실시간 추론

```bash
# Option 1: 파일 기반
python scripts/run_inference.py \
    --data-source data/live_metrics.csv

# Option 2: 스트리밍 (향후)
python scripts/run_inference.py \
    --streaming \
    --kafka-broker localhost:9092 \
    --kafka-topic oran-metrics
```

---

## 📊 데이터 소스 선택

### 현재 지원: 파일 기반

```bash
# CSV 파일
python scripts/run_inference.py --data-source data/metrics.csv

# Excel 파일
python scripts/run_inference.py --data-source data/metrics.xlsx

# Parquet 파일 (대용량)
python scripts/run_inference.py --data-source data/metrics.parquet
```

### 향후 지원: 스트리밍

```bash
# Kafka
python scripts/run_inference.py \
    --streaming \
    --kafka-broker localhost:9092 \
    --kafka-topic oran-metrics

# WebSocket
python scripts/run_inference.py \
    --streaming \
    --websocket-url ws://oran-server:8080/metrics
```

**상세 가이드**: [Data-Source-Guide.md](../03-data-management/Data-Source-Guide.md)

---

## 🎨 탐지 알고리즘

OCAD는 4가지 탐지 알고리즘을 결합합니다:

### 1. 룰 기반 탐지 (Rule-Based)
```python
if udp_echo_rtt > 10ms:
    score = 1.0  # 이상
```
- 간단하고 빠름
- 임계값 기반
- 현재 추론 스크립트에서 사용 중

### 2. 변화점 탐지 (Changepoint)
```python
CUSUM 알고리즘으로 점진적 변화 탐지
```
- Drift(점진적 증가) 탐지
- 통계 기반

### 3. 잔차 탐지 (Residual)
```python
예측값: 5.0ms
실제값: 15.0ms
잔차: 10.0ms → 이상!
```
- TCN/LSTM 모델 사용
- 시계열 패턴 학습
- 향후 통합 예정

### 4. 다변량 탐지 (Multivariate)
```python
[UDP, eCPRI, LBM] 동시 고려
```
- Isolation Forest 사용
- 여러 메트릭 간 상관관계
- 향후 통합 예정

**종합 점수**:
```python
composite_score = (
    0.3 × rule_score +
    0.4 × changepoint_score +
    0.2 × residual_score +
    0.1 × multivariate_score
)
```

**상세 설명**: [Model-Architecture.md](Model-Architecture.md)

---

## 🔧 성능 튜닝

### 임계값 조정

```bash
# 기본 임계값 (보수적)
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --threshold 0.5  # 종합 점수 임계값

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

### False Negative 줄이기

현재 정확도 88.85%에서 False Negative 87개 발생:

**원인**:
- Drift 초기 단계 미탐지
- 임계값보다 낮은 이상값

**해결책**:
1. 임계값 낮추기: `--threshold 0.3`
2. 변화점 탐지기 추가 (CUSUM)
3. 더 많은 학습 데이터

---

## 📚 다음 단계

### 신규 사용자
1. [학습 가이드](Training-Guide.md) - 실제로 모델 학습해보기
2. [추론 가이드](Inference-Guide.md) - 추론 실행 및 결과 분석
3. [전체 워크플로우](../02-user-guides/Training-Inference-Workflow.md) - 상세 단계별 설명

### 운영자
1. [Operations-Guide.md](../02-user-guides/Operations-Guide.md) - 시스템 운영
2. [Logging-Guide.md](../02-user-guides/Logging-Guide.md) - 로그 분석

### 개발자
1. [Training-Inference-Separation-Design.md](../05-architecture/Training-Inference-Separation-Design.md) - 아키텍처 상세 설계
2. [Data-Source-Abstraction-Design.md](../05-architecture/Data-Source-Abstraction-Design.md) - 데이터 소스 추상화

---

## ❓ FAQ

### Q1: 학습은 몇 번 해야 하나요?

**A**: 일반적으로 최초 1회만 하면 됩니다. 다만 다음 경우 재학습:
- 네트워크 장비 추가/변경
- 정상 패턴 변화 (예: 네트워크 업그레이드)
- 성능 개선 필요

### Q2: 추론은 얼마나 자주 실행하나요?

**A**:
- **파일 기반**: 필요 시 수동 실행
- **실시간**: 24/7 연속 실행 (향후)

### Q3: 모델 학습 시간은?

**A**:
- TCN: 약 5-10분 (28,800개 레코드, 50 epochs)
- Isolation Forest: 약 1-2분
- 총 10-15분

### Q4: 정확도를 높이려면?

**A**:
1. 더 많은 정상 데이터로 학습 (현재 28,800개 → 수십만 개)
2. 하이퍼파라미터 튜닝 (epochs, batch size)
3. 여러 탐지기 결합 (현재는 룰 기반만 사용)

### Q5: 실시간 스트리밍은 언제 지원되나요?

**A**: CFM 담당자와 협의 후 Kafka/WebSocket 구현 예정입니다.

---

**작성자**: Claude Code
**최종 업데이트**: 2025-10-27
**버전**: 1.0.0
