# OCAD 모델 학습 Phase별 로드맵

**전체 목표**: 완전한 학습-추론 파이프라인 구축
**예상 기간**: 3-5일

---

## 📊 Phase 개요

| Phase | 목표 | 난이도 | 소요 시간 | 상태 |
|-------|------|--------|----------|------|
| **Phase 1** | UDP Echo TCN 학습 | ⭐⭐☆☆☆ | 2-3시간 | ⏳ 준비됨 |
| **Phase 2** | eCPRI, LBM TCN 학습 | ⭐⭐☆☆☆ | 2-4시간 | 🔜 대기 중 |
| **Phase 3** | Isolation Forest 학습 | ⭐☆☆☆☆ | 1-2시간 | 🔜 대기 중 |
| **Phase 4** | ONNX 변환 및 배포 | ⭐⭐⭐☆☆ | 2-3시간 | 🔜 대기 중 |
| **Phase 5** | 통합 및 최적화 | ⭐⭐⭐⭐☆ | 3-4시간 | 🔜 대기 중 |

---

## Phase 1: UDP Echo TCN 학습 ⏳

**목표**: 가장 중요한 UDP Echo 메트릭 예측 모델 완성

### 작업 내역
1. ✅ 시계열 데이터 준비 스크립트 작성
2. ⚠️ 슬라이딩 윈도우로 시퀀스 생성 (10 timesteps → 1 target)
3. ⚠️ TCN 모델 학습 (30-50 epochs)
4. ⚠️ 학습 결과 검증

### 출력물
- `data/processed/timeseries_train.parquet` (~23,000 시퀀스)
- `ocad/models/tcn/udp_echo_v2.0.0.pth` (학습된 모델)
- `ocad/models/tcn/udp_echo_v2.0.0.json` (메타데이터)

### 성공 기준
- ✅ Validation Loss < 0.1
- ✅ 모델 로드 테스트 통과
- ✅ 더미 데이터로 추론 가능

### 빠른 시작
```bash
# 상세 가이드 확인
cat docs/PHASE1-QUICKSTART.md

# 실행
python scripts/prepare_timeseries_data.py
python scripts/train_tcn_model.py --metric-type udp_echo --epochs 30
```

**예상 소요 시간**: 2-3시간 (CPU 기준)

---

## Phase 2: eCPRI, LBM TCN 학습 🔜

**목표**: 나머지 2개 메트릭 예측 모델 완성

### 작업 내역
1. eCPRI Delay 시퀀스 데이터 생성 (또는 Phase 1에서 함께 생성됨)
2. LBM RTT 시퀀스 데이터 생성
3. eCPRI TCN 모델 학습
4. LBM TCN 모델 학습

### 출력물
- `ocad/models/tcn/ecpri_v2.0.0.pth`
- `ocad/models/tcn/lbm_v2.0.0.pth`

### 실행 (Phase 1 완료 후)
```bash
# eCPRI 학습 (터미널 1)
python scripts/train_tcn_model.py \
    --metric-type ecpri \
    --epochs 30 \
    --version v2.0.0

# LBM 학습 (터미널 2, 병렬 실행)
python scripts/train_tcn_model.py \
    --metric-type lbm \
    --epochs 30 \
    --version v2.0.0
```

### 최적화 팁
- **병렬 학습**: 2개 터미널에서 동시 실행
- **시퀀스 재사용**: Phase 1 데이터에 이미 모든 메트릭 포함됨
- **Epoch 조정**: eCPRI는 변동이 크므로 epochs 증가 고려

**예상 소요 시간**: 2-4시간 (병렬 실행 시 2시간)

---

## Phase 3: Isolation Forest 학습 🔜

**목표**: 다변량 이상 탐지 모델 완성

### 작업 내역
1. 다변량 데이터 준비 (UDP + eCPRI + LBM + CCM)
2. Isolation Forest 학습
3. Contamination 파라미터 튜닝

### 출력물
- `data/processed/multivariate_train.parquet`
- `ocad/models/isolation_forest/multivariate_v2.0.0.pkl`

### 실행
```bash
# 1. 데이터 준비
python scripts/prepare_multivariate_data.py

# 2. 학습
python scripts/train_isolation_forest.py \
    --train-data data/processed/multivariate_train.parquet \
    --test-data data/processed/multivariate_test.parquet \
    --n-estimators 100 \
    --contamination 0.1 \
    --version v2.0.0

# 3. 검증
python scripts/test_isolation_forest.py
```

### 하이퍼파라미터 권장값
- `n_estimators`: 100-200
- `contamination`: 0.05-0.15 (데이터에 따라 조정)
- `max_samples`: 256 (기본값)

**예상 소요 시간**: 1-2시간

---

## Phase 4: ONNX 변환 및 배포 🔜

**목표**: 프로덕션 배포 가능한 ONNX 모델 생성

### 작업 내역
1. PyTorch → ONNX 변환 스크립트 작성
2. UDP Echo 모델 변환
3. eCPRI, LBM 모델 변환
4. ONNX 추론 성능 테스트

### 출력물
- `ocad/models/onnx/udp_echo_v2.0.0.onnx`
- `ocad/models/onnx/ecpri_v2.0.0.onnx`
- `ocad/models/onnx/lbm_v2.0.0.onnx`
- `reports/onnx_benchmark_report.md`

### 실행
```bash
# 1. ONNX 변환
python scripts/convert_to_onnx.py \
    --model-path ocad/models/tcn/udp_echo_v2.0.0.pth \
    --output ocad/models/onnx/udp_echo_v2.0.0.onnx \
    --input-shape 1,10,1

# 2. 검증
python scripts/verify_onnx_model.py \
    --onnx-model ocad/models/onnx/udp_echo_v2.0.0.onnx \
    --pytorch-model ocad/models/tcn/udp_echo_v2.0.0.pth

# 3. 성능 벤치마크
python scripts/benchmark_onnx_inference.py \
    --pytorch-model ocad/models/tcn/udp_echo_v2.0.0.pth \
    --onnx-model ocad/models/onnx/udp_echo_v2.0.0.onnx
```

### 기대 성능 향상
- **추론 속도**: 2-5배 빠름
- **메모리 사용량**: 30-50% 감소
- **배포 용이성**: 플랫폼 독립적

**예상 소요 시간**: 2-3시간

---

## Phase 5: 통합 및 최적화 🔜

**목표**: 학습된 모델을 추론 파이프라인에 통합

### 작업 내역
1. ResidualDetector 업데이트 (TCN 모델 로드)
2. MultivariateDetector 업데이트 (Isolation Forest 로드)
3. CompositeDetector 가중치 최적화
4. 전체 파이프라인 테스트
5. 성능 비교 (룰 기반 vs 학습 모델)

### 수정 파일
- `ocad/detectors/residual.py`
- `ocad/detectors/multivariate.py`
- `scripts/inference_with_report.py`
- `config/detection_config.yaml`

### 실행
```bash
# 1. 통합 추론 실행 (학습된 모델 사용)
python scripts/inference_with_report.py \
    --data-source data/inference_anomaly_only.csv \
    --model-path ocad/models/tcn \
    --use-trained-models

# 2. 성능 비교
python scripts/compare_detection_performance.py \
    --baseline rule_based \
    --new trained_models

# 3. 앙상블 가중치 최적화
python scripts/optimize_ensemble_weights.py \
    --validation-data data/inference_test_scenarios.csv
```

### 성능 목표
- **Precision**: > 95% (현재 100%)
- **Recall**: > 85% (현재 82.45% → 향상)
- **F1 Score**: > 90%
- **False Positive Rate**: < 5%

**예상 소요 시간**: 3-4시간

---

## 📅 권장 일정

### 🗓️ Day 1 (오늘/내일)
- ✅ **Phase 1**: UDP Echo TCN 학습 (2-3시간)
- 📝 학습 결과 검증 및 문서화

### 🗓️ Day 2
- ✅ **Phase 2**: eCPRI, LBM TCN 학습 (2-4시간, 병렬)
- ✅ **Phase 3**: Isolation Forest 학습 (1-2시간)

### 🗓️ Day 3
- ✅ **Phase 4**: ONNX 변환 (2-3시간)
- 🧪 ONNX 성능 벤치마크

### 🗓️ Day 4
- ✅ **Phase 5**: 모델 통합 (3-4시간)
- 🧪 전체 파이프라인 테스트

### 🗓️ Day 5 (선택)
- 🔧 하이퍼파라미터 튜닝
- 🔧 앙상블 가중치 최적화
- 📊 성능 분석 및 보고서 작성

---

## 🎯 최종 목표 (Phase 5 완료 후)

### 시스템 구성
```
데이터 입력
    ↓
4가지 탐지기 (앙상블)
├── RuleBasedDetector      (임계값 기반)
├── ChangepointDetector    (CUSUM/PELT)
├── ResidualDetector       (TCN 예측-잔차) ← NEW!
└── MultivariateDetector   (Isolation Forest) ← NEW!
    ↓
CompositeDetector (가중치 앙상블)
    ↓
AlertManager
    ↓
보고서 생성
```

### 배포 옵션
1. **PyTorch 모델** (개발/테스트)
2. **ONNX 모델** (프로덕션)
3. **실시간 스트리밍** (Kafka 연동)

### 성능 KPI
- ✅ Precision > 95%
- ✅ Recall > 85%
- ✅ F1 Score > 90%
- ✅ 추론 지연시간 < 100ms
- ✅ False Positive Rate < 5%

---

## 📚 참고 문서

| 문서 | 설명 | 링크 |
|------|------|------|
| **Phase 1 Quick Start** | UDP Echo 학습 상세 가이드 | [PHASE1-QUICKSTART.md](PHASE1-QUICKSTART.md) |
| **Tomorrow Quick Start** | 내일 전체 작업 가이드 | [TOMORROW-QUICKSTART.md](TOMORROW-QUICKSTART.md) |
| **전체 TODO** | 모든 작업 상세 목록 | [TODO.md](TODO.md) |
| **학습-추론 개요** | 아키텍처 설명 | [04-training-inference/Overview.md](04-training-inference/Overview.md) |

---

## 💡 성공을 위한 팁

### 1. 단계별 검증
각 Phase 완료 후 반드시 검증:
- 모델 파일 생성 확인
- 모델 로드 테스트
- 메타데이터 확인

### 2. 병렬 작업 활용
- Phase 2: eCPRI + LBM 동시 학습
- Phase 4: ONNX 변환 병렬 실행

### 3. 백그라운드 실행
장시간 학습은 nohup 활용:
```bash
nohup python scripts/train_tcn_model.py ... > logs/train.log 2>&1 &
```

### 4. 체크포인트 저장
중간 결과 저장으로 재시작 가능하게:
- Epoch 10, 20, 30마다 체크포인트
- Early stopping 활용

### 5. 모니터링
실시간 로그 확인:
```bash
tail -f logs/training_*.log
watch -n 5 'ls -lh ocad/models/tcn/'
```

---

## ⚠️ 주의사항

1. **데이터 형식**: Phase 1 데이터 준비가 가장 중요 (sequence + target)
2. **시간 관리**: Epoch 수를 조절하여 시간 단축 가능 (30 vs 50)
3. **메모리**: 배치 크기 조절로 메모리 사용량 제어
4. **GPU**: 있으면 5-10배 빠름, 없어도 CPU로 충분히 가능
5. **Early Stopping**: 자동 종료로 불필요한 학습 방지

---

## 🚀 시작하기

### Phase 1부터 시작
```bash
cd /home/finux/dev/OCAD
source .venv/bin/activate

# Phase 1 가이드 확인
cat docs/PHASE1-QUICKSTART.md

# 시작!
python scripts/prepare_timeseries_data.py
```

### 전체 진행 상황 추적
```bash
# 체크리스트
echo "Phase 1: [ ]"
echo "Phase 2: [ ]"
echo "Phase 3: [ ]"
echo "Phase 4: [ ]"
echo "Phase 5: [ ]"
```

---

**화이팅! 🎉**

Phase 1부터 차근차근 진행하시면 됩니다. 각 Phase는 독립적으로 완료 가능하며, 중간에 멈춰도 다시 이어서 할 수 있습니다.
