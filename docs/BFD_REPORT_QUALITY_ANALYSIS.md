# BFD 리포트 품질 분석 및 개선 방안

**분석 일시**: 2025-11-07
**분석 대상**: BFD 프로토콜 이상 탐지 성능 리포트 (HMM, LSTM)

---

## 문제 제기

사용자 리포트:
> "생성된 bfd 리포트를 하나 봤는데.. 권장사항에 모델 아키텍처를 재검토하라는데? 전반적으로 리포트 품질이 떨어지는것 같아. 이해도 어렵고."

### 기존 리포트의 문제점

1. **낮은 성능 (정확도 54.9%, F1-score 13.3%)**
   - 거의 랜덤 수준 (50%)
   - 재현율 7.9%: 이상의 92%를 놓침

2. **부적절한 권장사항**
   ```
   필수 개선 작업:
   1. 모델 아키텍처 재검토 (LSTM → Transformer, Ensemble 등)
   2. 학습 데이터 품질 점검 및 정제
   3. 피처 엔지니어링 강화
   4. 전문가와 함께 이상 패턴 재정의
   ```
   - 일반적인 조언만 나열
   - **근본 원인 분석 없음**
   - 실행 가능한 구체적 액션 없음

3. **이해하기 어려운 내용**
   - BFD 프로토콜 특성을 고려하지 않은 일반적인 지표 나열
   - 왜 성능이 낮은지 설명 없음
   - 다음 단계가 불명확

---

## 근본 원인 분석

### 단계 1: 예측 결과 분석

**발견사항**: 모든 예측의 `anomaly_score = 0.0`

```csv
timestamp,value,true_label,predicted_label,anomaly_score,confidence
2025-11-05 11:29:20,1,True,False,0.0,0.0  # ❌ 이상인데 점수 0
2025-11-05 11:29:25,1,True,False,0.0,0.0  # ❌ 이상인데 점수 0
2025-11-05 11:29:30,3,False,False,0.0,0.5  # ✅ 정상
```

**Evidence 분석**:
```json
{
  "log_likelihood": -0.5,
  "anomaly_threshold": -0.72,
  "sequence_length": 10
}
```

- **log_likelihood (-0.5) > threshold (-0.72)** → 정상 판단
- HMM은 `likelihood < threshold`일 때 이상으로 판단하는데, 모든 샘플이 threshold보다 높음
- 즉, **임계값이 너무 낮게 설정**됨

### 단계 2: 학습 데이터 분석

**학습 데이터 (data/bfd/train/)**: is_anomaly=False만 사용해야 함

```csv
timestamp,source_id,local_state,is_anomaly,flap_count
2025-11-05 11:29:20,bfd-session-1,3 (UP),False,0     # ✅ 정상
2025-11-05 11:29:20,bfd-session-2,3 (UP),False,0     # ✅ 정상
2025-11-05 11:29:20,bfd-session-3,3 (UP),False,0     # ✅ 정상
2025-11-05 11:29:25,bfd-session-1,3 (UP),False,0     # ✅ 정상
2025-11-05 11:29:25,bfd-session-2,3 (UP),False,0     # ✅ 정상
2025-11-05 11:29:25,bfd-session-3,3 (UP),False,0     # ✅ 정상
...
2025-11-05 11:29:30,bfd-session-3,1 (DOWN),False,1   # ❌ 문제!!!
2025-11-05 11:29:35,bfd-session-3,1 (DOWN),False,1   # ❌ DOWN 상태 계속
2025-11-05 11:29:40,bfd-session-3,1 (DOWN),False,1   # ❌ DOWN 상태 계속
...
```

**문제 발견**:
- bfd-session-3이 DOWN 상태로 계속 머무름
- **is_anomaly=False**로 마킹되어 있음
- HMM이 이걸 "정상"으로 학습함!

### 단계 3: 검증 데이터 분석

**검증 정상 데이터 (val_normal/)**:
- 모두 `local_state=3` (UP)
- 정상 패턴

**검증 이상 데이터 (val_anomaly/)**:
```csv
timestamp,source_id,local_state,is_anomaly,flap_count
2025-11-05 11:29:20,bfd-session-1,1 (DOWN),True,1
2025-11-05 11:29:25,bfd-session-1,1 (DOWN),True,1
2025-11-05 11:29:30,bfd-session-1,1 (DOWN),True,1
```
- 많은 샘플이 `local_state=1` (DOWN)
- 플래핑 패턴

**결과**:
- HMM이 DOWN 상태를 "정상"으로 학습했기 때문에
- 검증 이상 데이터를 정상으로 판단
- 재현율 7.9%, 정확도 54.9%

### 단계 4: 데이터 생성 로직 분석

**`scripts/generate_bfd_data.py` line 154-164**:

```python
else:
    # Normal behavior: very rare state changes
    if random.random() < 0.01:  # 1% chance
        state_changed = True
        if self.local_state == BFDState.UP:
            self.local_state = BFDState.DOWN  # ❌ 정상 기간에 DOWN 발생!
            diagnostic = BFDDiagnostic.CONTROL_DETECTION_TIME_EXPIRED
        else:
            self.local_state = BFDState.UP
            diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
        self.flap_count += 1
```

**버그 발견**:

1. **정상 기간에도 1% 확률로 DOWN 상태 발생**
2. 한번 DOWN되면 다시 UP으로 돌아오려면 또 1% 확률 필요
3. 결과: 일부 세션이 DOWN 상태에 계속 머무름
4. **is_anomaly 필드는 is_anomaly_period만 체크** (line 196)
   - 실제 상태(UP/DOWN)는 체크하지 않음
   - DOWN 상태인데 `is_anomaly=False`로 마킹됨

**근본 원인**:
- 데이터 생성 로직의 버그로 인해
- 학습 데이터에 DOWN 상태가 "정상"으로 포함됨
- HMM이 잘못된 패턴을 학습
- 모델 아키텍처 문제가 **아님**!

---

## 문제 요약

| 레벨 | 문제 | 원인 |
|------|------|------|
| **L1: 증상** | 정확도 54.9%, F1-score 13.3% | 모든 이상을 정상으로 판단 |
| **L2: 모델** | 모든 anomaly_score = 0.0 | 임계값(-0.72)이 너무 낮음 |
| **L3: 학습** | 임계값이 낮게 설정됨 | 학습 데이터에 DOWN 패턴 포함 |
| **L4: 데이터** | DOWN 상태가 정상으로 마킹 | 데이터 생성 로직 버그 |
| **L5: 근본** | 정상 기간 상태 변경 로직 | `if random.random() < 0.01` |

---

## 해결 방안

### 우선순위 1: 데이터 생성 로직 수정 (긴급)

**파일**: `scripts/generate_bfd_data.py`

#### 수정 1: 정상 기간 상태 변경 제거

**현재** (line 154-164):
```python
else:
    # Normal behavior: very rare state changes
    if random.random() < 0.01:  # ❌ 정상인데 상태 변경
        state_changed = True
        if self.local_state == BFDState.UP:
            self.local_state = BFDState.DOWN
```

**수정 후**:
```python
else:
    # Normal behavior: stable UP state
    # Ensure state is UP during normal periods
    if self.local_state != BFDState.UP:
        self.local_state = BFDState.UP  # 정상 기간에는 항상 UP
        state_changed = True
        diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
    else:
        state_changed = False
```

#### 수정 2: is_anomaly 필드 로직 개선

**현재** (line 196):
```python
"is_anomaly": is_anomaly_period,  # ❌ 기간만 체크
```

**수정 후**:
```python
# 실제 이상 상태 확인
is_actual_anomaly = (
    is_anomaly_period  # Anomaly period
    or self.local_state != BFDState.UP  # DOWN 상태
    or self.flap_count > (previous_flap_count + 1)  # 빈번한 플래핑
)
"is_anomaly": is_actual_anomaly,
```

### 우선순위 2: 리포트 생성 로직 개선

**파일**: `scripts/report_bfd.py`

#### 개선 1: 근본 원인 분석 추가

```python
def analyze_root_cause(df: pd.DataFrame, metrics: dict) -> dict:
    """Analyze root cause of poor performance.

    Returns:
        Dictionary with analysis results:
        - issue_type: str (e.g., "data_quality", "model_architecture", "threshold")
        - evidence: list of findings
        - recommended_actions: list of specific actions
    """
    findings = []

    # Check 1: All predictions have zero score?
    if df["anomaly_score"].max() == 0.0:
        findings.append({
            "issue": "model_not_detecting",
            "evidence": "All anomaly scores are 0.0",
            "likely_cause": "Threshold too low or model not trained properly",
        })

    # Check 2: Training data quality
    if "value" in df.columns:
        # Analyze value distribution
        anomaly_values = df[df["true_label"] == True]["value"].values
        normal_values = df[df["true_label"] == False]["value"].values

        # Check if distributions overlap significantly
        anomaly_mean = anomaly_values.mean() if len(anomaly_values) > 0 else 0
        normal_mean = normal_values.mean() if len(normal_values) > 0 else 0

        if abs(anomaly_mean - normal_mean) < 0.5:
            findings.append({
                "issue": "data_quality",
                "evidence": f"Anomaly mean ({anomaly_mean:.2f}) vs Normal mean ({normal_mean:.2f})",
                "likely_cause": "Training data may contain anomalies labeled as normal",
            })

    # Check 3: BFD-specific patterns
    if "metric_name" in df.columns and "local_state" in df.columns:
        # Check if DOWN states are predicted as normal
        down_predictions = df[df["value"] == 1]  # local_state=1 (DOWN)
        if len(down_predictions) > 0:
            down_pred_normal = (down_predictions["predicted_label"] == False).sum()
            if down_pred_normal / len(down_predictions) > 0.8:
                findings.append({
                    "issue": "model_learned_wrong_pattern",
                    "evidence": f"{down_pred_normal}/{len(down_predictions)} DOWN states predicted as normal",
                    "likely_cause": "Model learned DOWN state as normal during training",
                })

    return {"findings": findings}
```

#### 개선 2: BFD-특화 권장사항

```python
def generate_bfd_specific_recommendations(metrics: dict, root_cause: dict) -> list:
    """Generate BFD protocol-specific recommendations."""
    recommendations = []

    findings = root_cause.get("findings", [])

    for finding in findings:
        if finding["issue"] == "data_quality":
            recommendations.append({
                "priority": "HIGH",
                "action": "재학습 데이터 생성",
                "details": [
                    "정상 데이터(train/)는 local_state=3 (UP)만 포함되어야 함",
                    "DOWN 상태는 이상 데이터에만 포함",
                    "is_anomaly 필드 재검증 필요",
                ],
                "command": "python scripts/generate_bfd_ml_data.py --sessions 10 --train-hours 2.0",
            })

        elif finding["issue"] == "model_learned_wrong_pattern":
            recommendations.append({
                "priority": "CRITICAL",
                "action": "학습 데이터 정제 및 재학습",
                "details": [
                    "현재 모델이 DOWN 상태를 정상으로 학습함",
                    "학습 데이터에서 local_state=1 (DOWN) 제거 필요",
                    "데이터 생성 로직 버그 수정 필요",
                ],
                "command": "python scripts/fix_bfd_training_data.py && python scripts/train_bfd_hmm.py",
            })

    return recommendations
```

#### 개선 3: 이해하기 쉬운 설명 추가

```markdown
## BFD 프로토콜 특성 기반 분석

### BFD 상태 (local_state)

- **UP (3)**: 정상 - BFD 세션이 정상 동작 중
- **DOWN (1)**: 이상 - 경로 장애, 플래핑 등
- **INIT (2)**: 초기화 중
- **ADMIN_DOWN (0)**: 관리자에 의해 비활성화

### 이상 패턴

1. **플래핑 (Flapping)**: UP ↔ DOWN 빠른 반복
   - 네트워크 불안정
   - flap_count 증가

2. **지속 장애**: DOWN 상태 장시간 유지
   - 경로 단절
   - detection_time_ms 증가

### 현재 모델의 문제

**발견된 패턴**:
- DOWN 상태 샘플 63개 중 58개를 정상으로 잘못 판별 (92%)
- 모델이 DOWN 상태를 "정상"으로 학습한 것으로 보임

**원인 분석**:
- 학습 데이터에 DOWN 상태가 is_anomaly=False로 포함되었을 가능성
- 데이터 생성 스크립트의 is_anomaly 필드 로직 재검토 필요

**즉시 조치 사항**:
1. 학습 데이터 확인: `cat data/bfd/train/*.csv | grep "local_state,1"`
2. DOWN 상태 샘플 제거 또는 재라벨링
3. 모델 재학습
```

### 우선순위 3: 데이터 검증 스크립트 추가

**새 파일**: `scripts/validate_bfd_training_data.py`

```python
#!/usr/bin/env python3
"""Validate BFD training data quality."""

import pandas as pd
from pathlib import Path

def validate_training_data(data_path: Path) -> dict:
    """Validate BFD training data.

    Returns:
        Validation report with issues found
    """
    df = pd.read_parquet(data_path)

    issues = []

    # Rule 1: Training data (is_anomaly=False) should only have UP states
    normal_data = df[df["is_anomaly"] == False]
    down_in_normal = normal_data[normal_data["local_state"] != 3]

    if len(down_in_normal) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "Normal data should only have UP states",
            "found": f"{len(down_in_normal)} records with local_state != 3 (UP)",
            "samples": down_in_normal.head(5).to_dict("records"),
        })

    # Rule 2: Anomaly data should have DOWN states or flapping
    anomaly_data = df[df["is_anomaly"] == True]
    up_in_anomaly = anomaly_data[
        (anomaly_data["local_state"] == 3) & (anomaly_data["flap_count"] == 0)
    ]

    if len(up_in_anomaly) > 0:
        issues.append({
            "severity": "WARNING",
            "rule": "Anomaly data should have DOWN states or flapping",
            "found": f"{len(up_in_anomaly)} records with UP state and no flapping",
        })

    # Rule 3: Check flap_count consistency
    flap_changes = df.groupby("source_id")["flap_count"].diff().fillna(0)
    invalid_flaps = (flap_changes < 0) | (flap_changes > 1)

    if invalid_flaps.sum() > 0:
        issues.append({
            "severity": "WARNING",
            "rule": "flap_count should only increase by 0 or 1",
            "found": f"{invalid_flaps.sum()} invalid transitions",
        })

    return {
        "total_records": len(df),
        "normal_records": len(normal_data),
        "anomaly_records": len(anomaly_data),
        "issues": issues,
        "status": "PASS" if len(issues) == 0 else "FAIL",
    }
```

---

## 실행 계획

### Phase 1: 긴급 수정 (1-2일)

1. ✅ 근본 원인 분석 완료
2. **데이터 생성 로직 수정** (scripts/generate_bfd_data.py)
   - 정상 기간 상태 변경 제거
   - is_anomaly 필드 로직 개선
3. **데이터 검증 스크립트 작성** (scripts/validate_bfd_training_data.py)
4. **새 데이터 생성 및 검증**
   ```bash
   python scripts/generate_bfd_ml_data.py --sessions 10 --train-hours 2.0
   python scripts/validate_bfd_training_data.py --data data/bfd/train
   ```

### Phase 2: 재학습 및 검증 (1일)

5. **모델 재학습**
   ```bash
   python scripts/train_bfd_hmm.py --data data/bfd/train
   python scripts/train_bfd_lstm.py --data data/bfd/train
   ```
6. **재평가**
   ```bash
   python scripts/infer_bfd.py --model models/bfd/hmm_v1.0.0.pkl
   ```
7. **성능 비교**
   - 목표: Accuracy > 90%, Recall > 85%

### Phase 3: 리포트 개선 (2-3일)

8. **리포트 생성 로직 개선** (scripts/report_bfd.py)
   - 근본 원인 분석 추가
   - BFD-특화 권장사항 추가
   - 이해하기 쉬운 설명 추가
9. **리포트 재생성 및 검토**
10. **문서 업데이트**

---

## 예상 결과

### 수정 전 (현재)

- Accuracy: 54.9%
- Precision: 41.7%
- Recall: 7.9%
- F1-score: 13.3%
- **문제**: HMM이 DOWN 상태를 정상으로 학습

### 수정 후 (예상)

- Accuracy: > 90%
- Precision: > 85%
- Recall: > 85%
- F1-score: > 85%
- **개선**: 정상 데이터에 UP 상태만 포함, 명확한 패턴 학습

---

## 교훈 및 개선 사항

### 1. 데이터 품질이 가장 중요

- 모델 아키텍처보다 **데이터 품질**이 더 중요함
- "Garbage In, Garbage Out"
- 학습 전 데이터 검증 필수

### 2. 도메인 지식 활용

- BFD 프로토콜 특성을 고려한 분석 필요
- UP/DOWN 상태의 의미 이해
- 프로토콜별 특화 검증 로직 필요

### 3. 근본 원인 분석

- 단순히 "모델 아키텍처 재검토" 권장하지 말 것
- 레이어별 분석 (증상 → 모델 → 데이터 → 로직)
- 구체적이고 실행 가능한 액션 제시

### 4. 자동화된 데이터 검증

- 데이터 생성 후 자동 검증
- CI/CD 파이프라인에 통합
- 이상 패턴 조기 발견

---

## 다른 프로토콜 적용

동일한 분석 프레임워크를 BGP, PTP, CFM에도 적용:

1. **데이터 생성 로직 검증**
   - 정상/이상 패턴 명확히 구분
   - is_anomaly 필드 로직 정확성

2. **학습 데이터 검증**
   - 정상 데이터에 이상 패턴 포함 여부
   - 프로토콜별 특성 반영

3. **리포트 품질 개선**
   - 프로토콜별 특화 분석
   - 근본 원인 자동 분석
   - 실행 가능한 권장사항

---

## 참고 자료

- [BFD Protocol RFC 5880](https://tools.ietf.org/html/rfc5880)
- [PROTOCOL-ANOMALY-DETECTION-PLAN.md](./PROTOCOL-ANOMALY-DETECTION-PLAN.md)
- [ML_PIPELINE_VALIDATION_REPORT.md](./ML_PIPELINE_VALIDATION_REPORT.md)
- [DATA_GENERATION_ARGUMENT_VERIFICATION.md](./DATA_GENERATION_ARGUMENT_VERIFICATION.md)

---

**문서 정보**

- 작성: 2025-11-07
- 버전: 1.0.0
- 상태: ✅ 근본 원인 분석 완료, 수정 작업 대기 중
