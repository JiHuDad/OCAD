# BFD ML 파이프라인 완료 보고

**작성일**: 2025-11-05  
**프로토콜**: BFD (Bidirectional Forwarding Detection)  
**파이프라인**: 데이터 생성 → 학습 → 추론 → 리포트 생성

---

## 1. 생성된 스크립트

### 1.1 데이터 생성 스크립트
- **파일명**: `scripts/generate_bfd_ml_data.py`
- **기능**: 
  - 3가지 데이터셋 자동 생성 (train, val_normal, val_anomaly)
  - train: 정상 데이터 100% (학습용)
  - val_normal: 정상 데이터 100% (검증용)
  - val_anomaly: 이상 데이터 90% (검증용)
- **사용법**:
  ```bash
  # 빠른 테스트 (5분 학습, 2분 검증)
  python3 scripts/generate_bfd_ml_data.py --output data/bfd --quick
  
  # 프로덕션 데이터 (24시간 학습, 2시간 검증)
  python3 scripts/generate_bfd_ml_data.py --output data/bfd \
      --sessions 50 --train-hours 24 --val-hours 2
  ```

### 1.2 학습 스크립트

#### LSTM 학습
- **파일명**: `scripts/train_bfd_lstm.py`
- **기능**: BFD 시계열 데이터로 LSTM 모델 학습
- **사용법**:
  ```bash
  python3 scripts/train_bfd_lstm.py --data data/bfd/train \
      --metric detection_time_ms --epochs 100 --batch-size 64
  ```
- **주의**: PyTorch 필요 (`pip install torch`)

#### HMM 학습
- **파일명**: `scripts/train_bfd_hmm.py`
- **기능**: BFD 상태 전이 패턴으로 HMM 모델 학습
- **사용법**:
  ```bash
  python3 scripts/train_bfd_hmm.py --data data/bfd/train \
      --metric local_state --n-components 4
  ```
- **장점**: PyTorch 불필요, 빠른 학습

### 1.3 추론 스크립트
- **파일명**: `scripts/infer_bfd.py`
- **기능**: 학습된 모델로 검증 데이터 추론
- **사용법**:
  ```bash
  # HMM 추론
  python3 scripts/infer_bfd.py --model models/bfd/hmm_v1.0.0.pkl \
      --detector hmm --data data/bfd/val_normal data/bfd/val_anomaly \
      --metric local_state
  
  # LSTM 추론 (PyTorch 필요)
  python3 scripts/infer_bfd.py --model models/bfd/lstm_v1.0.0.pth \
      --detector lstm --data data/bfd/val_normal data/bfd/val_anomaly \
      --metric detection_time_ms
  ```

### 1.4 리포트 생성 스크립트
- **파일명**: `scripts/report_bfd.py`
- **기능**: 추론 결과를 분석하고 한글 리포트 생성
- **사용법**:
  ```bash
  python3 scripts/report_bfd.py --predictions results/bfd/predictions.csv \
      --output results/bfd/report.md
  ```
- **출력**: 
  - Markdown 리포트 (한글)
  - 성능 지표 해석
  - 개선 권장사항

---

## 2. 실행 결과

### 2.1 데이터셋 크기
- **train**: 180개 샘플 (정상 100%)
- **val_normal**: 72개 샘플 (정상 100%)
- **val_anomaly**: 72개 샘플 (이상 87.5%)
- **총 검증 데이터**: 144개 샘플

### 2.2 학습 시간
- **HMM (local_state)**: 0.1초
- **HMM (detection_time_ms)**: 0.1초
- **LSTM**: 미실행 (PyTorch 미설치)

### 2.3 최종 성능

#### HMM + local_state 메트릭
| 지표 | 값 | 해석 |
|------|-----|------|
| Accuracy | 54.86% | 개선 필요 |
| Precision | 41.67% | 오탐지 많음 |
| Recall | 7.94% | 이상 탐지율 낮음 |
| F1-score | 13.33% | 균형 불량 |
| ROC-AUC | 0.4965 | 랜덤 수준 |

**혼동 행렬**:
```
              예측: 정상    예측: 이상
실제 정상:        74            7  (FP)
실제 이상:        58            5  (TP)
```

#### HMM + detection_time_ms 메트릭
| 지표 | 값 | 해석 |
|------|-----|------|
| Accuracy | 51.39% | 개선 필요 |
| Precision | 33.33% | 오탐지 많음 |
| Recall | 11.11% | 이상 탐지율 낮음 |
| F1-score | 16.67% | 균형 불량 |

**혼동 행렬**:
```
              예측: 정상    예측: 이상
실제 정상:        67           14  (FP)
실제 이상:        56            7  (TP)
```

---

## 3. 리포트 위치

생성된 리포트 파일:
- `/home/user/OCAD/results/bfd/report.md` - local_state 메트릭 리포트
- `/home/user/OCAD/results/bfd/report_detection_time.md` - detection_time_ms 메트릭 리포트

추론 결과 CSV:
- `/home/user/OCAD/results/bfd/predictions.csv` - local_state 예측
- `/home/user/OCAD/results/bfd/predictions_detection_time.csv` - detection_time_ms 예측

학습된 모델:
- `/home/user/OCAD/models/bfd/hmm_v1.0.0.pkl` - local_state HMM
- `/home/user/OCAD/models/bfd/hmm_detection_time_v1.0.0.pkl` - detection_time_ms HMM

---

## 4. 주요 발견사항

### 4.1 파이프라인 검증
✅ **완전한 ML 파이프라인 구축 성공**
- 데이터 생성 → 학습 → 추론 → 리포트 생성까지 전 과정 자동화
- 사용자 친화적인 CLI 인터페이스
- 한글 리포트로 비전문가도 이해 가능

### 4.2 기술적 성과
✅ **PyTorch 없이도 작동**
- HMM 탐지기는 SimpleGaussianHMM 폴백 구현 사용
- numpy, pandas, scikit-learn만으로 전체 파이프라인 실행 가능

✅ **유연한 메트릭 선택**
- local_state (BFD 상태)
- detection_time_ms (탐지 시간)
- 추가 메트릭 확장 가능 (flap_count, echo_interval_ms 등)

### 4.3 성능 분석
❌ **현재 성능: 프로덕션 배포 불가**
- Accuracy 50-55% (목표: 90%+)
- Recall 8-11% (이상의 대부분을 놓침)
- F1-score 13-17% (목표: 80%+)

**낮은 성능의 원인**:
1. **학습 데이터 부족**: 180개 샘플 (권장: 10,000개 이상)
2. **데이터 품질**: 시뮬레이션 데이터로 실제 이상 패턴 부족
3. **모델 단순성**: SimpleGaussianHMM은 복잡한 패턴 학습 어려움
4. **메트릭 선택**: local_state와 detection_time_ms 모두 단일 메트릭으로는 부족

### 4.4 개선 방향

#### 즉시 개선 가능
1. **더 많은 데이터 생성** (24시간, 50 세션)
   ```bash
   python3 scripts/generate_bfd_ml_data.py --output data/bfd \
       --sessions 50 --train-hours 24 --val-hours 2
   ```

2. **PyTorch 설치 후 LSTM 학습**
   ```bash
   pip install torch
   python3 scripts/train_bfd_lstm.py --data data/bfd/train \
       --epochs 100 --batch-size 64
   ```

3. **앙상블 모델** (HMM + LSTM + CUSUM)
   - 여러 탐지기의 결과를 조합하여 정확도 향상

#### 장기 개선 과제
1. **실제 BFD 데이터 수집**
   - BFD 장비에서 실제 로그 수집
   - SNMP (BFD-STD-MIB) 또는 NETCONF/YANG 사용

2. **다변량 모델**
   - 여러 메트릭을 동시에 고려 (local_state + detection_time_ms + flap_count)
   - Multivariate HMM 또는 LSTM

3. **하이퍼파라미터 튜닝**
   - Grid Search 또는 Bayesian Optimization
   - sequence_length, n_components, threshold 최적화

4. **데이터 증강**
   - SMOTE, ADASYN 등 불균형 데이터 처리
   - 합성 이상 패턴 생성

---

## 5. 다음 단계

### 5.1 즉시 실행 가능
```bash
# 1. 대용량 데이터 생성 (권장)
python3 scripts/generate_bfd_ml_data.py --output data/bfd \
    --sessions 50 --train-hours 24 --val-hours 2

# 2. HMM 재학습
python3 scripts/train_bfd_hmm.py --data data/bfd/train --metric local_state

# 3. 추론 및 리포트
python3 scripts/infer_bfd.py --model models/bfd/hmm_v1.0.0.pkl \
    --detector hmm --data data/bfd/val_normal data/bfd/val_anomaly
python3 scripts/report_bfd.py --predictions results/bfd/predictions.csv
```

### 5.2 PyTorch 설치 후
```bash
# PyTorch 설치
pip install torch

# LSTM 학습
python3 scripts/train_bfd_lstm.py --data data/bfd/train \
    --metric detection_time_ms --epochs 100

# LSTM 추론
python3 scripts/infer_bfd.py --model models/bfd/lstm_v1.0.0.pth \
    --detector lstm --data data/bfd/val_normal data/bfd/val_anomaly
```

### 5.3 다른 프로토콜로 확장
- BGP: `scripts/generate_bgp_ml_data.py`
- PTP: `scripts/generate_ptp_ml_data.py`
- CFM: `scripts/generate_cfm_ml_data.py`

---

## 6. 결론

### ✅ 성공 사항
1. **완전한 ML 파이프라인 구축**
   - 데이터 생성, 학습, 추론, 리포트 생성 자동화
   - 재사용 가능한 스크립트 (다른 프로토콜에도 적용 가능)

2. **사용자 친화적 인터페이스**
   - CLI 기반 워크플로우
   - 한글 리포트로 결과 해석 용이

3. **의존성 최소화**
   - PyTorch 없이도 HMM 사용 가능
   - 기본 패키지만으로 전체 파이프라인 실행

### ⚠️  개선 필요
1. **성능 향상 필요**
   - 현재 50-55% 정확도 → 목표 90%+
   - 더 많은 학습 데이터 필요 (180개 → 10,000개+)
   - LSTM 모델 학습 (PyTorch 설치 후)

2. **실제 데이터 검증**
   - 시뮬레이션 데이터 → 실제 BFD 로그
   - 프로덕션 환경 테스트

### 📊 최종 평가
**파이프라인 완성도**: ✅ 100%  
**현재 모델 성능**: ❌ 54% (개선 필요)  
**프로덕션 준비도**: ⚠️  추가 개발 필요

---

**작성자**: Claude Code Agent  
**버전**: 1.0.0  
**날짜**: 2025-11-05
