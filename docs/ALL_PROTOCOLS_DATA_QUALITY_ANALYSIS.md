# 전체 프로토콜 데이터 생성 로직 검증 결과

**분석 일시**: 2025-11-07
**분석 대상**: BFD, BGP, PTP, CFM 데이터 생성 로직 및 학습 데이터 품질

---

## Executive Summary

4개 프로토콜의 데이터 생성 로직과 실제 학습 데이터를 검증한 결과:

| 프로토콜 | 상태 | 문제 | 우선순위 |
|---------|------|------|----------|
| **BFD** | ❌ **FAIL** | 정상 기간 상태 변경 버그 | **긴급 (P0)** |
| BGP | ✅ PASS | 자동 복구 로직 정상 작동 | - |
| PTP | ✅ PASS | 강제 상태 복원 정상 작동 | - |
| CFM | ✅ PASS | 메트릭 범위 정상 유지 | - |

**결론**: **BFD만 수정 필요**, 다른 프로토콜은 정상

---

## 프로토콜별 상세 분석

### 1. BFD (Bidirectional Forwarding Detection) - ❌ FAIL

#### 문제 요약
- **정상 기간에도 1% 확률로 DOWN 상태 발생**
- **한번 DOWN되면 복구되지 않음** (다시 1% 확률 필요)
- **결과**: 학습 데이터에 DOWN 상태가 `is_anomaly=False`로 마킹됨

#### 버그 코드 위치

**파일**: `scripts/generate_bfd_data.py` line 154-164

```python
else:
    # Normal behavior: very rare state changes
    if random.random() < 0.01:  # ❌ 1% 확률로 상태 변경
        state_changed = True
        if self.local_state == BFDState.UP:
            self.local_state = BFDState.DOWN  # ❌ DOWN으로 변경
            diagnostic = BFDDiagnostic.CONTROL_DETECTION_TIME_EXPIRED
        else:
            self.local_state = BFDState.UP  # 1% 확률로만 복구
            diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
        self.flap_count += 1
```

#### 실제 데이터 검증

```csv
# data/bfd/train/bfd_data_20251105_112920.csv
timestamp,source_id,local_state,local_state_name,is_anomaly
2025-11-05 11:29:20,bfd-session-1,3,UP,False        # ✅ 정상
2025-11-05 11:29:20,bfd-session-2,3,UP,False        # ✅ 정상
2025-11-05 11:29:20,bfd-session-3,3,UP,False        # ✅ 정상
...
2025-11-05 11:29:30,bfd-session-3,1,DOWN,False      # ❌ DOWN인데 정상!
2025-11-05 11:29:35,bfd-session-3,1,DOWN,False      # ❌ 계속 DOWN
2025-11-05 11:29:40,bfd-session-3,1,DOWN,False      # ❌ 계속 DOWN
...
```

**통계**:
- 학습 데이터 총 177개 레코드
- 정상(is_anomaly=False) 전체 177개
- **하지만 일부가 local_state=1 (DOWN)** ← 문제!

#### 영향

1. **HMM 학습**: DOWN 상태를 "정상"으로 학습
2. **추론 결과**:
   - Accuracy: 54.9% (거의 랜덤)
   - Recall: 7.9% (이상의 92% 놓침)
   - 모든 anomaly_score = 0.0
3. **리포트 품질**: "모델 아키텍처 재검토" 등 부적절한 권장사항

---

### 2. BGP (Border Gateway Protocol) - ✅ PASS

#### 정상 작동 확인

**파일**: `scripts/generate_bgp_ml_data.py` line 131-136

```python
else:
    # Normal behavior
    if self.state != BGPState.ESTABLISHED:
        self.state = BGPState.ESTABLISHED  # ✅ 자동 복구!
        self.prefix_count = random.randint(50, 200)
        update_delta = self.prefix_count
```

**핵심 패턴**: 정상 기간에 ESTABLISHED가 아니면 **강제로 ESTABLISHED로 복구**

#### 실제 데이터 검증

```
=== BGP TRAINING DATA ===
Total records: 900

=== STATE DISTRIBUTION (is_anomaly=False) ===
ESTABLISHED    900  ← 모두 정상 상태!

✅ All normal records are ESTABLISHED
```

#### 왜 BGP는 괜찮은가?

- 이상 시나리오(session_reset)가 IDLE로 변경 (line 123-128)
- **하지만 다음 정상 기간에 자동으로 ESTABLISHED로 복구**
- 결과: 정상 데이터에는 ESTABLISHED만 포함

---

### 3. PTP (Precision Time Protocol) - ✅ PASS

#### 정상 작동 확인

**파일**: `scripts/generate_ptp_ml_data.py` line 127-141

```python
else:
    # Normal operation: small random walk
    self.port_state = PTPPortState.SLAVE  # ✅ 강제로 SLAVE 설정!

    # Offset random walk (stays within ±100ns)
    self.offset_from_master_ns += random.uniform(-20, 20)
    self.offset_from_master_ns = max(-100, min(100, self.offset_from_master_ns))

    # Path delay random walk (stays within 1-10μs)
    self.mean_path_delay_ns += random.uniform(-100, 100)
    self.mean_path_delay_ns = max(1000, min(10000, self.mean_path_delay_ns))
```

**핵심 패턴**:
1. 정상 기간에 **port_state를 무조건 SLAVE로 설정**
2. 모든 메트릭을 **정상 범위 내로 제한** (max/min 사용)

#### 실제 데이터 검증

```
=== PTP TRAINING DATA ===
Total records: 7200

=== STATE DISTRIBUTION (is_anomaly=False) ===
SLAVE    7200  ← 모두 SLAVE 상태!

=== METRIC RANGES (normal data) ===
offset_from_master_ns: [-100.00, 100.00]  ← 정상 범위
mean_path_delay_ns: [1000.00, 7416.72]    ← 정상 범위
clock_drift_ppb: [-20.00, 20.00]          ← 정상 범위

✅ All normal records are SLAVE state
```

---

### 4. CFM (Connectivity Fault Management) - ✅ PASS

#### 정상 작동 확인

**파일**: `scripts/generate_cfm_ml_data.py` line 139-145

```python
else:
    # Normal behavior
    # Small random variations around baseline
    udp_rtt = self.base_udp_rtt * random.uniform(0.9, 1.1)  # ±10%
    ecpri_delay = self.base_ecpri_delay * random.uniform(0.9, 1.1)
    lbm_rtt = self.base_lbm_rtt * random.uniform(0.9, 1.1)
    ccm_miss = 1 if random.random() < 0.01 else 0  # 1% rare miss
```

**핵심 패턴**:
- CFM은 상태 기반이 아니라 **메트릭 기반**
- 정상 기간: baseline의 **90-110% 범위**
- 이상 기간: **3-10배 증가**
- 명확한 분리

#### 실제 데이터 검증

```
=== CFM TRAINING DATA ===
Total records: 180

=== ANOMALY TYPE DISTRIBUTION ===
normal    180  ← 모두 정상!

=== METRIC RANGES (is_anomaly=False) ===
udp_echo_rtt_ms: [5.41, 7.80]      ← baseline 6.0 ± 10%
ecpri_delay_us: [90.28, 205.48]    ← baseline varies
lbm_rtt_ms: [7.50, 12.19]          ← baseline 10.0 ± 10%

✅ All normal records have anomaly_type="normal"
```

---

## 패턴 비교: 정상 기간 처리

| 프로토콜 | 정상 기간 처리 방식 | 장점 |
|---------|-------------------|------|
| **BFD** | ❌ 1% 확률로 상태 변경 | **버그** - 복구 안 됨 |
| **BGP** | ✅ ESTABLISHED로 강제 복구 | 자동 복구, 상태 안정성 |
| **PTP** | ✅ SLAVE로 강제 설정 + 범위 제한 | 상태 + 메트릭 모두 안정 |
| **CFM** | ✅ 메트릭 범위 제한 (±10%) | 명확한 정상/이상 분리 |

### 베스트 프랙티스 (PTP 패턴)

```python
else:
    # Normal operation
    # 1. 상태를 정상으로 강제 설정
    self.state = NORMAL_STATE

    # 2. 메트릭을 정상 범위 내로 제한
    self.metric += random.uniform(-delta, +delta)
    self.metric = max(min_value, min(max_value, self.metric))
```

---

## BFD 수정 방안

### 우선순위 1: 정상 기간 상태 복원 (긴급)

**현재 코드** (`scripts/generate_bfd_data.py` line 154-164):
```python
else:
    # Normal behavior: very rare state changes
    if random.random() < 0.01:  # ❌ 문제!
        state_changed = True
        if self.local_state == BFDState.UP:
            self.local_state = BFDState.DOWN
        else:
            self.local_state = BFDState.UP
        self.flap_count += 1
```

**수정 방안 A** (BGP 패턴 참고):
```python
else:
    # Normal behavior: maintain UP state
    if self.local_state != BFDState.UP:
        # Auto-recover to UP during normal periods
        self.local_state = BFDState.UP
        state_changed = True
        diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
    else:
        state_changed = False
        diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
```

**수정 방안 B** (더 보수적):
```python
else:
    # Normal behavior: stable UP state
    # No state changes during normal periods
    self.local_state = BFDState.UP
    state_changed = False
    diagnostic = BFDDiagnostic.NO_DIAGNOSTIC
    # Detection time remains normal (15-50ms)
```

**권장**: **수정 방안 B** (단순하고 명확)

### 우선순위 2: is_anomaly 필드 검증 강화

**현재 코드** (line 196):
```python
"is_anomaly": is_anomaly_period,  # ❌ 기간만 체크
```

**수정 후**:
```python
# Actual anomaly: period flag OR abnormal state
is_actual_anomaly = (
    is_anomaly_period
    or self.local_state != BFDState.UP  # DOWN/INIT/ADMIN_DOWN
    or self.flap_count > 5  # Excessive flapping
)
"is_anomaly": is_actual_anomaly,
```

---

## 실행 계획

### Phase 1: BFD 긴급 수정 (1일)

**작업 순서**:
1. `scripts/generate_bfd_data.py` 수정
   - line 154-164: 정상 기간 상태 복원 로직
   - line 196: is_anomaly 필드 검증 강화
2. 데이터 검증 스크립트 작성
   ```python
   # scripts/validate_bfd_training_data.py
   def validate(df):
       normal = df[df['is_anomaly'] == False]
       bad = normal[normal['local_state'] != BFDState.UP]
       assert len(bad) == 0, f"Found {len(bad)} non-UP in normal data"
   ```
3. 새 데이터 생성 및 검증
   ```bash
   python scripts/generate_bfd_ml_data.py --sessions 10 --train-hours 2.0
   python scripts/validate_bfd_training_data.py --data data/bfd/train
   ```

### Phase 2: 재학습 및 검증 (1일)

4. 모델 재학습
   ```bash
   python scripts/train_bfd_hmm.py --data data/bfd/train
   python scripts/train_bfd_lstm.py --data data/bfd/train
   ```
5. 재평가
   ```bash
   python scripts/infer_bfd.py --model models/bfd/hmm_v1.0.0.pkl \
       --detector hmm --metric local_state \
       --data data/bfd/val_normal data/bfd/val_anomaly
   ```
6. 성능 비교
   - **현재**: Accuracy 54.9%, Recall 7.9%
   - **목표**: Accuracy >90%, Recall >85%

### Phase 3: 리포트 개선 (2일)

7. 리포트 생성 로직 개선 (`scripts/report_bfd.py`)
   - 근본 원인 자동 분석 추가
   - BFD-특화 설명 및 권장사항
   - 프로토콜 특성 기반 해석
8. 다른 프로토콜 리포트도 동일한 패턴 적용
9. 문서 업데이트

---

## 다른 프로토콜 리포트 확인 필요성

BGP, PTP, CFM 데이터는 괜찮지만, **리포트 품질**도 확인 필요:

```bash
# 다른 프로토콜 리포트 확인
ls -la results/bgp/report*.md
ls -la results/ptp/report*.md
ls -la results/cfm/report*.md

# 성능 메트릭 확인
cat results/bgp/predictions.metrics.txt
cat results/ptp/predictions.metrics.txt
cat results/cfm/predictions.metrics.txt
```

**예상**:
- BGP, PTP, CFM은 **데이터가 정상**이므로 모델 성능도 좋을 것
- 하지만 리포트 생성 로직은 **동일한 개선** 필요
  - 근본 원인 분석
  - 프로토콜별 특화 설명
  - 실행 가능한 권장사항

---

## 교훈 및 권장사항

### 1. 데이터 생성 패턴

**✅ 좋은 패턴** (BGP, PTP, CFM):
- 정상 기간에 **상태를 강제로 복원**
- 메트릭을 **정상 범위 내로 제한**
- 명확한 **정상/이상 분리**

**❌ 나쁜 패턴** (BFD):
- 정상 기간에 **확률적 상태 변경**
- 복구 로직 없음
- 애매한 경계

### 2. is_anomaly 필드 검증

**현재 대부분의 스크립트**:
```python
"is_anomaly": is_anomaly_period  # ❌ 기간만 체크
```

**개선안**:
```python
# 실제 이상 상태 확인
is_actual_anomaly = (
    is_anomaly_period
    or abnormal_state_detected()  # 상태 체크
    or abnormal_metric_detected()  # 메트릭 체크
)
"is_anomaly": is_actual_anomaly
```

### 3. 자동화된 데이터 검증

모든 프로토콜에 대해:
```bash
# 데이터 생성 후 자동 검증
python scripts/generate_XXX_ml_data.py ...
python scripts/validate_XXX_training_data.py ...  # ← 필수!
```

**검증 항목**:
- 정상 데이터에 이상 패턴 포함 여부
- 이상 데이터에 정상 패턴만 있는지
- 메트릭 범위 정상성
- is_anomaly 필드 일관성

### 4. CI/CD 통합

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Check
on: [push]
jobs:
  validate:
    steps:
      - run: python scripts/generate_all_ml_data.sh --quick
      - run: python scripts/validate_all_training_data.py
      - run: python scripts/test_all_models.py
```

---

## 다음 단계 권장사항

### 즉시 실행 (오늘)

1. ✅ **BFD 데이터 생성 로직 수정** (1-2시간)
2. ✅ **데이터 검증 스크립트 작성** (1시간)
3. ✅ **새 데이터 생성 및 검증** (30분)

### 내일 실행

4. ✅ **BFD 모델 재학습** (1시간)
5. ✅ **BFD 재평가 및 성능 확인** (1시간)
6. ✅ **다른 프로토콜 리포트 품질 확인** (2시간)

### 이번 주 내

7. ✅ **리포트 생성 로직 개선** (1일)
8. ✅ **모든 프로토콜에 검증 스크립트 추가** (1일)
9. ✅ **문서 업데이트 및 정리** (반나절)

---

## 참고 자료

### BFD 관련
- [BFD_REPORT_QUALITY_ANALYSIS.md](./BFD_REPORT_QUALITY_ANALYSIS.md) - BFD 상세 분석
- [RFC 5880 - BFD Protocol](https://tools.ietf.org/html/rfc5880)

### 프로토콜 확장
- [PROTOCOL-ANOMALY-DETECTION-PLAN.md](./PROTOCOL-ANOMALY-DETECTION-PLAN.md)
- [Plugin-Development-Guide.md](./07-development/Plugin-Development-Guide.md)

### ML 파이프라인
- [ML_PIPELINE_VALIDATION_REPORT.md](./ML_PIPELINE_VALIDATION_REPORT.md)
- [ML_PIPELINE_TRAIN_INFERENCE_CONSISTENCY_SUMMARY.md](./ML_PIPELINE_TRAIN_INFERENCE_CONSISTENCY_SUMMARY.md)

---

**문서 정보**

- 작성: 2025-11-07
- 버전: 1.0.0
- 상태: ✅ 전체 프로토콜 검증 완료
- 결론: **BFD만 수정 필요, 다른 프로토콜은 정상**
