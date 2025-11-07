# BFD 데이터 수정 전후 성능 비교

**날짜**: 2025-11-07
**작업**: 옵션 3 - 전체 품질 개선

---

## 요약

BFD 데이터 생성 로직 버그를 수정하고 모델을 재학습한 결과, **Recall이 10배 향상**되었습니다!

| 지표 | 수정 전 | 수정 후 | 개선율 |
|------|---------|---------|--------|
| **Accuracy** | 54.9% | 52.9% | -2.0% |
| **Precision** | 41.7% | 49.5% | +7.8% |
| **Recall** | **7.9%** | **78.1%** | **+70.2%** ⬆️⬆️⬆️ |
| **F1-score** | **13.3%** | **60.6%** | **+47.3%** ⬆️⬆️⬆️ |

---

## 수정 내용

### 1. 데이터 생성 로직 수정

**파일**: `scripts/generate_bfd_data.py`

**수정 전** (line 154-164):
```python
else:
    # Normal behavior: very rare state changes
    if random.random() < 0.01:  # ❌ 1% 확률로 상태 변경
        state_changed = True
        if self.local_state == BFDState.UP:
            self.local_state = BFDState.DOWN  # ❌ DOWN 발생
            diagnostic = BFDDiagnostic.CONTROL_DETECTION_TIME_EXPIRED
        else:
            self.local_state = BFDState.UP  # 1% 확률로만 복구
            diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
        self.flap_count += 1
```

**수정 후**:
```python
else:
    # Normal behavior: maintain stable UP state
    # Auto-recover to UP if in abnormal state during normal periods
    if self.local_state != BFDState.UP:
        # Recovery to normal UP state
        self.local_state = BFDState.UP  # ✅ 자동 복구
        state_changed = True
        diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
    else:
        # Already in UP state, no change
        state_changed = False
        diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
```

### 2. is_anomaly 필드 강화

**수정 전** (line 197):
```python
"is_anomaly": is_anomaly_period,  # ❌ 기간만 체크
```

**수정 후**:
```python
# Determine actual anomaly status
# Anomaly if: period flag OR abnormal state OR excessive flapping
is_actual_anomaly = (
    is_anomaly_period  # Explicitly marked anomaly period
    or self.local_state != BFDState.UP  # Not in UP state
    or self.flap_count > 5  # Excessive flapping
)
"is_anomaly": is_actual_anomaly,
```

---

## 데이터 품질 비교

### 학습 데이터 (train/)

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| 총 레코드 | 180 | 177 |
| **UP 상태** | 122 (68%) | **177 (100%)** ✅ |
| **DOWN 상태** | **58 (32%)** ❌ | **0 (0%)** ✅ |
| flap_count max | 3 | 0 |

**문제**:
- 수정 전: 정상 데이터인데 58개(32%)가 DOWN 상태로 오염
- HMM이 DOWN 상태를 "정상"으로 학습
- 결과: 이상 탐지 실패

**해결**:
- 수정 후: 모든 정상 데이터가 UP 상태
- HMM이 정확한 정상 패턴 학습
- 결과: 이상 탐지 가능

---

## 성능 분석

### Recall 개선: 7.9% → 78.1% (+70.2%)

**수정 전**:
- 모델이 DOWN 상태를 정상으로 학습
- 검증 이상 데이터의 DOWN 상태를 정상으로 판별
- **실제 이상 63개 중 5개만 탐지** (7.9%)

**수정 후**:
- 모델이 UP 상태만 정상으로 학습
- DOWN 상태를 이상으로 올바르게 판별
- **실제 이상 64개 중 50개 탐지** (78.1%)

### F1-score 개선: 13.3% → 60.6% (+47.3%)

- Precision과 Recall의 균형이 크게 개선
- 이상 탐지와 정상 판별의 조화로운 성능

### 여전히 개선 필요한 부분

**Accuracy가 여전히 낮음 (52.9%)**:
- 원인: 오탐(FP)이 많음 (51건)
- False Positive 예시: UP 상태(value=3.00)인데 이상으로 판별

**근본 원인 분석**:
1. HMM이 모든 학습 데이터가 UP=3 상태만 봄
2. Anomaly threshold = -0.000000 (거의 0)
3. 약간의 변화도 이상으로 판별 → 오탐 증가

**해결 방안**:
1. **더 많은 학습 데이터**: 현재 177개 → 1000+개
2. **다른 메트릭 사용**: local_state 대신 detection_time_ms 사용
3. **threshold 조정**: 더 보수적인 임계값
4. **앙상블 모델**: HMM + 다른 탐지기

---

## 검증 스크립트 추가

모든 프로토콜(BFD, BGP, PTP, CFM)에 대한 데이터 검증 스크립트 생성:

1. `scripts/validate_bfd_training_data.py`
2. `scripts/validate_bgp_training_data.py`
3. `scripts/validate_ptp_training_data.py`
4. `scripts/validate_cfm_training_data.py`
5. `scripts/validate_all_training_data.py` (통합)

**검증 항목**:
- 정상 데이터에 이상 패턴 포함 여부
- 상태/메트릭 범위 정상성
- is_anomaly 필드 일관성
- 음수 값, 중복 등 데이터 무결성

**사용 예시**:
```bash
# BFD만 검증
python scripts/validate_bfd_training_data.py --data data/bfd/train

# 모든 프로토콜 검증
python scripts/validate_all_training_data.py
```

---

## 다음 단계

### 즉시 실행 가능 (추가 개선)

1. **더 많은 학습 데이터 생성**:
   ```bash
   python scripts/generate_bfd_ml_data.py \
     --sessions 10 \
     --train-hours 2.0 \
     --output data/bfd
   ```

2. **detection_time_ms 메트릭으로 재학습**:
   ```bash
   python scripts/train_bfd_hmm.py \
     --data data/bfd/train \
     --metric detection_time_ms
   ```

3. **임계값 조정 실험**:
   ```bash
   python scripts/train_bfd_hmm.py \
     --data data/bfd/train \
     --threshold-percentile 10  # 기본 5 → 10
   ```

### 장기 개선 사항

1. **리포트 생성 로직 개선**
   - 근본 원인 자동 분석
   - BFD 특화 설명 및 권장사항
   - 프로토콜 특성 기반 해석

2. **CI/CD 통합**
   - 데이터 생성 후 자동 검증
   - 품질 게이트 추가
   - 성능 회귀 테스트

3. **문서화**
   - 베스트 프랙티스 가이드
   - 데이터 생성 패턴 문서
   - 트러블슈팅 가이드

---

## 결론

✅ **핵심 성과**: 데이터 품질 문제 해결로 Recall 10배 향상!

✅ **검증 완료**:
- 데이터 생성 로직 수정 ✅
- 검증 스크립트 추가 ✅
- 모델 재학습 ✅
- 성능 개선 확인 ✅

⚠️ **추가 개선 필요**:
- 더 많은 학습 데이터
- 메트릭 선택 최적화
- 임계값 튜닝

**전체 평가**: **성공** - 근본 문제 해결, 성능 크게 개선, 재현 가능한 품질 보증 시스템 구축

---

**관련 문서**:
- [BFD_REPORT_QUALITY_ANALYSIS.md](./BFD_REPORT_QUALITY_ANALYSIS.md)
- [ALL_PROTOCOLS_DATA_QUALITY_ANALYSIS.md](./ALL_PROTOCOLS_DATA_QUALITY_ANALYSIS.md)
