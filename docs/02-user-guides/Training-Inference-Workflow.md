# 학습-추론 워크플로우 가이드

**날짜**: 2025-10-23
**대상**: OCAD 이상 탐지 모델 학습 및 추론

## 목차

1. [이상 탐지 개념](#이상-탐지-개념)
2. [데이터 생성](#데이터-생성)
3. [모델 학습](#모델-학습)
4. [추론 테스트](#추론-테스트)
5. [결과 분석](#결과-분석)

---

## 이상 탐지 개념

### 핵심 원리

**이상 탐지는 정상 데이터로만 학습합니다!**

```
학습 데이터:  정상 패턴만 ✅
           (이상 데이터 ❌)

추론 데이터:  정상 + 이상 모두 테스트 ✓
```

### 왜 정상 데이터로만 학습하나요?

1. **정상 패턴 학습**: 모델이 "정상"이 무엇인지 학습
2. **이상은 예측 불가**: 어떤 종류의 이상이 발생할지 모름
3. **일반화**: 학습 시 보지 못한 새로운 이상도 탐지 가능

### 탐지 방식

```
입력 데이터 → 모델 → 예측값
                ↓
        예측값 vs 실제값 비교
                ↓
          잔차(residual) 계산
                ↓
    잔차가 크면 → 이상 ⚠️
    잔차가 작으면 → 정상 ✅
```

---

## 데이터 생성

### 1. 데이터 생성 스크립트 실행

```bash
python scripts/generate_training_inference_data.py
```

### 2. 생성되는 파일

#### 학습 데이터 (정상만)

**파일**: `data/training_normal_only.csv`
- **용도**: 모델 학습
- **내용**: 정상 데이터만 (28,800개)
- **특징**:
  - 5개 엔드포인트
  - 48시간 분량
  - 30초 간격
  - 일일 주기 패턴 포함 (야간/주간 변동)

**샘플**:
```csv
timestamp,endpoint_id,site_name,zone,udp_echo_rtt_ms,ecpri_delay_us,lbm_rtt_ms,lbm_success,ccm_interval_ms,ccm_miss_count,label
2025-10-01 00:00:00,o-ru-001,Tower-A,Urban,4.98,99.23,6.95,True,1000,0,normal
2025-10-01 00:00:30,o-ru-001,Tower-A,Urban,5.12,101.45,7.08,True,1000,0,normal
...
```

#### 추론 테스트 데이터 (정상 + 비정상)

**파일**: `data/inference_test_scenarios.csv`
- **용도**: 추론 테스트
- **내용**: 정상 + 비정상 (780개)
- **시나리오**:
  1. 정상 운영 (180개) - 정상 ✅
  2. Drift (180개) - 점진적 증가 ⚠️
  3. Spike (120개) - 일시적 급증 ⚠️
  4. Jitter (120개) - 불규칙 변동 ⚠️
  5. 복합 장애 (90개) - 여러 메트릭 동시 이상 🚨
  6. 정상 복구 (90개) - 복구 후 정상 ✅

**라벨 분포**:
- `normal`: 478개 (61.3%)
- `anomaly`: 302개 (38.7%)

---

## 모델 학습

### 사용 가능한 방법

#### 방법 1: 기존 학습 스크립트 (TCN)

```bash
# TCN 모델 학습
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --data-path data/training/normal_data.parquet \
    --epochs 50 \
    --batch-size 32
```

**출력**:
- 학습된 모델: `ocad/models/tcn/udp_echo_v1.0.0.pth`
- 메타데이터: `ocad/models/metadata/udp_echo_v1.0.0.json`

#### 방법 2: 전체 학습 스크립트

```bash
# 모든 메트릭에 대해 학습
./scripts/train_all_models.sh
```

**생성되는 모델**:
- `udp_echo_v1.0.0.pth`
- `ecpri_v1.0.0.pth`
- `lbm_v1.0.0.pth`

### 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--epochs` | 50 | 학습 에폭 수 |
| `--batch-size` | 32 | 배치 크기 |
| `--learning-rate` | 0.001 | 학습률 |
| `--sequence-length` | 60 | 시퀀스 길이 |
| `--hidden-dim` | 64 | 은닉층 차원 |

### 학습 시간

| 데이터 크기 | 예상 시간 |
|------------|----------|
| 10,000개 | ~2분 |
| 30,000개 | ~5분 |
| 100,000개 | ~15분 |

---

## 추론 테스트

### 1. 추론 테스트 실행

```bash
python scripts/test_inference.py
```

### 2. 사용되는 탐지기

OCAD는 **하이브리드 탐지** 방식을 사용합니다:

| 탐지기 | 방식 | 설명 |
|-------|------|------|
| **Rule-based** | 임계값 | 설정된 임계값 초과 여부 확인 |
| **Changepoint** | CUSUM | 급격한 변화점 탐지 |
| **Residual (TCN)** | 예측-잔차 | 정상 패턴 예측 후 잔차 계산 |

**종합 점수** = 각 탐지기 점수의 평균

### 3. 출력 예시

```
======================================================================
추론 결과 평가
======================================================================

전체 정확도: 85.64%

Confusion Matrix:
예측         anomaly  normal
실제
anomaly         245      57
normal           55     423

클래스별 정확도:
  normal  : 88.49% (423/478)
  anomaly : 81.13% (245/302)

시나리오별 정확도:
  Drift (점진적 증가)          :  95.6% (180개)
  Jitter (불안정)             :  73.3% (120개)
  Spike (일시적 급증)          :  80.0% (120개)
  복합 장애 (Multi-metric)    : 100.0% (90개)
  정상 복구                   :  88.9% (90개)
  정상 운영                   :  90.0% (180개)

탐지기별 평균 점수:
  rule_based     : 정상 0.125, 이상 0.782
  changepoint    : 정상 0.089, 이상 0.654
  residual       : 정상 0.102, 이상 0.723

⚠️  오탐 (False Positive): 55개
   오탐률: 11.51%

❌ 미탐 (False Negative): 57개
   미탐률: 18.87%
```

### 4. 결과 파일

**파일**: `data/inference_results.csv`

**내용**:
- 원본 데이터
- 각 탐지기 점수
- 종합 점수
- 예측 레이블
- 정답 여부

**샘플**:
```csv
timestamp,endpoint_id,udp_echo_rtt_ms,ecpri_delay_us,label,rule_based_score,changepoint_score,residual_score,composite_score,predicted_label,correct
2025-10-15 12:00:00,o-ru-test-001,5.12,101.45,normal,0.05,0.03,0.08,0.053,normal,True
2025-10-15 12:30:00,o-ru-test-001,15.32,255.87,anomaly,0.65,0.72,0.78,0.717,anomaly,True
...
```

---

## 결과 분석

### 1. CSV 파일로 분석

```bash
# 처음 20개 결과 확인
head -20 data/inference_results.csv

# 이상으로 예측된 케이스만
grep "anomaly" data/inference_results.csv | head -10
```

### 2. Excel로 분석

```bash
# Excel에서 열기
open data/inference_results.csv  # macOS
xdg-open data/inference_results.csv  # Linux
```

### 3. Python으로 분석

```python
import pandas as pd

# 결과 로드
df = pd.read_csv('data/inference_results.csv')

# 오탐 분석 (False Positive)
false_positives = df[(df['label'] == 'normal') & (df['predicted_label'] == 'anomaly')]
print(f"오탐: {len(false_positives)}개")
print(false_positives[['timestamp', 'udp_echo_rtt_ms', 'composite_score']])

# 미탐 분석 (False Negative)
false_negatives = df[(df['label'] == 'anomaly') & (df['predicted_label'] == 'normal')]
print(f"미탐: {len(false_negatives)}개")
print(false_negatives[['timestamp', 'udp_echo_rtt_ms', 'composite_score']])

# 점수 분포
import matplotlib.pyplot as plt
df.boxplot(column='composite_score', by='label')
plt.show()
```

### 4. 성능 지표

| 지표 | 계산 방식 | 목표 |
|------|----------|------|
| **정확도 (Accuracy)** | (TP + TN) / Total | > 85% |
| **정밀도 (Precision)** | TP / (TP + FP) | > 80% |
| **재현율 (Recall)** | TP / (TP + FN) | > 80% |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | > 80% |
| **오탐률 (FPR)** | FP / (FP + TN) | < 15% |

- TP: True Positive (정확히 이상 탐지)
- TN: True Negative (정확히 정상 판단)
- FP: False Positive (정상인데 이상으로 오탐)
- FN: False Negative (이상인데 정상으로 미탐)

---

## 전체 워크플로우 요약

### 📋 체크리스트

```bash
# 1. 데이터 생성
python scripts/generate_training_inference_data.py
# ✅ data/training_normal_only.csv (학습용)
# ✅ data/inference_test_scenarios.csv (추론용)

# 2. 모델 학습
./scripts/train_all_models.sh
# ✅ ocad/models/tcn/udp_echo_v1.0.0.pth
# ✅ ocad/models/tcn/ecpri_v1.0.0.pth
# ✅ ocad/models/tcn/lbm_v1.0.0.pth

# 3. 추론 테스트
python scripts/test_inference.py
# ✅ data/inference_results.csv

# 4. 결과 분석
head -20 data/inference_results.csv
```

### 🎯 핵심 포인트

1. **학습**: 정상 데이터만 사용 ✅
2. **추론**: 정상 + 비정상 모두 테스트 ✓
3. **평가**: 오탐률/미탐률 확인 📊
4. **개선**: 임계값 조정, 모델 재학습 🔄

---

## 자주 묻는 질문 (FAQ)

### Q1. 왜 정상 데이터로만 학습하나요?

**A**: 이상 탐지의 특성상:
- 정상 패턴은 예측 가능
- 이상 패턴은 다양하고 예측 불가
- 학습 시 본 적 없는 새로운 이상도 탐지 가능

### Q2. 비정상 데이터가 필요 없나요?

**A**: 학습에는 불필요, 평가에는 필요:
- **학습**: 정상만 ✅
- **평가**: 정상 + 비정상 ✓

### Q3. 모델 성능이 낮으면?

**A**: 다음을 시도:
1. 학습 데이터 양 증가
2. 학습 에폭 수 증가
3. 임계값 조정
4. 다른 모델 아키텍처 시도

### Q4. 오탐이 많으면?

**A**: 임계값을 높이기:
```python
# test_inference.py에서
predicted_label = "anomaly" if composite_score > 0.7 else "normal"
# 기본값 0.5 → 0.7로 변경
```

### Q5. 미탐이 많으면?

**A**: 임계값을 낮추기:
```python
predicted_label = "anomaly" if composite_score > 0.3 else "normal"
# 기본값 0.5 → 0.3으로 변경
```

---

## 참고 문서

- [Model-Training-Guide.md](Model-Training-Guide.md) - 모델 학습 상세 가이드
- [Training-Inference-Separation-Design.md](Training-Inference-Separation-Design.md) - 학습-추론 분리 설계
- [Quick-Start-Guide.md](Quick-Start-Guide.md) - 빠른 시작 가이드

---

**작성자**: Claude Code
**날짜**: 2025-10-23
**버전**: 1.0.0
