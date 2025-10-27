# README.md 업데이트 계획

**날짜**: 2025-10-27
**목적**: README.md를 최신 아키텍처(학습-추론 분리, 데이터 소스 추상화)에 맞게 업데이트

---

## 📊 현재 README.md 분석

### 현재 구조
1. **주요 특징** - OK (변경 불필요)
2. **시스템 구조** - OK (변경 불필요)
3. **🆕 파일 기반 데이터 입력** - ⚠️ 업데이트 필요
   - 샘플 데이터 생성 스크립트 변경됨
   - 학습/추론 데이터 분리 반영 필요
4. **설치 및 실행** - OK (변경 불필요)
5. **🧪 시스템 테스트** - OK (변경 불필요)
6. **주요 컴포넌트** - ⚠️ 업데이트 필요
   - 데이터 소스 추상화 추가
   - 학습/추론 분리 명시

---

## ❌ 제거할 내용

### 1. 오래된 파일 기반 입력 섹션
현재 README.md의 18-103줄:

```markdown
## 🆕 파일 기반 데이터 입력 (NEW!)

### 빠른 시작
python scripts/generate_sample_data.py  # ← 이 스크립트는 샘플 생성용
python scripts/test_file_loaders.py     # ← 로더 테스트용

### 샘플 데이터
- 01_normal_operation_24h.csv
- 02_drift_anomaly.csv
...
```

**문제점**:
- `generate_sample_data.py`는 **데모용 샘플** 생성
- **학습/추론 데이터**는 `generate_training_inference_data.py`로 생성
- 사용자가 혼란스러움

---

## ✅ 추가할 내용

### 1. 학습-추론 워크플로우 섹션 (NEW)

```markdown
## 🤖 학습 및 추론 워크플로우

OCAD는 **학습-추론 분리 아키텍처**를 사용합니다:

### 학습 (Training)
정상 데이터만 사용하여 모델 학습:

```bash
# 1. 학습 데이터 생성 (정상 데이터 28,800개)
python scripts/generate_training_inference_data.py --mode training

# 2. 모델 학습
python scripts/train_model.py \
    --data-source data/training_normal_only.csv \
    --epochs 50 \
    --batch-size 32
```

### 추론 (Inference)
학습된 모델로 이상 탐지:

```bash
# 1. 추론 테스트 데이터 생성 (정상 + 6가지 이상 시나리오)
python scripts/generate_training_inference_data.py --mode inference

# 2. 추론 실행
python scripts/run_inference.py \
    --data-source data/inference_test_scenarios.csv \
    --output data/inference_results.csv

# 3. 결과 확인
head -20 data/inference_results.csv
```

### 데이터 소스 선택

파일 기반 (현재):
```bash
python scripts/run_inference.py --data-source data/metrics.csv
```

스트리밍 (향후):
```bash
python scripts/run_inference.py \
    --streaming \
    --kafka-broker localhost:9092 \
    --kafka-topic oran-metrics
```

**상세 가이드**: [Training-Inference-Workflow.md](docs/Training-Inference-Workflow.md)
```

---

## 🔄 업데이트할 내용

### 1. 파일 기반 입력 섹션 간소화

**현재 (85줄)**:
```markdown
## 🆕 파일 기반 데이터 입력 (NEW!)
[긴 설명...]
```

**변경 후 (20줄)**:
```markdown
## 💾 데이터 입력 방식

OCAD는 두 가지 데이터 입력 방식을 지원합니다:

### 1. 파일 기반 (현재 지원)
- CSV, Excel, Parquet 형식
- 사람이 읽기 쉬운 형식
- 학습/추론 데이터 분리

```bash
# 데모용 샘플 생성 (다양한 시나리오)
python scripts/generate_sample_data.py

# 학습/추론 데이터 생성
python scripts/generate_training_inference_data.py
```

### 2. 실시간 수집 (기본)
- NETCONF/YANG을 통한 실시간 수집
- UDP Echo, eCPRI, LBM 메트릭

**상세 가이드**: [Data-Source-Guide.md](docs/03-data-management/Data-Source-Guide.md)
```

### 2. 주요 컴포넌트 섹션 업데이트

**추가할 내용**:
```markdown
### 6. Data Source Abstraction (NEW)
- 파일 기반 입력 (CSV/Excel/Parquet)
- 스트리밍 입력 (Kafka/WebSocket - 향후)
- 통일된 DataSource 인터페이스
- 학습/추론 모두 동일한 방식으로 처리
```

---

## 📝 새로운 README.md 구조

```markdown
# ORAN CFM-Lite AI Anomaly Detection System

## 주요 특징
[기존 유지]

## 시스템 구조
[기존 유지]

## 🤖 학습 및 추론 워크플로우 (NEW)
- 학습 (정상 데이터만 사용)
- 추론 (학습된 모델로 탐지)
- 데이터 소스 선택 (파일 / 스트리밍)

## 💾 데이터 입력 방식 (간소화)
- 파일 기반 (CSV/Excel/Parquet)
- 실시간 수집 (NETCONF/YANG)

## 설치 및 실행
[기존 유지]

## 🧪 시스템 테스트
[기존 유지]

## 주요 컴포넌트 (업데이트)
1-5. [기존 유지]
6. Data Source Abstraction (NEW)

## 성능 목표
[기존 유지]

## 📚 문서 (NEW)
- [Quick Start Guide](docs/01-getting-started/Quick-Start-Guide.md)
- [Training & Inference Guide](docs/04-training-inference/Overview.md)
- [Data Source Guide](docs/03-data-management/Data-Source-Guide.md)
- [Operations Guide](docs/02-user-guides/Operations-Guide.md)

## 개발 스프린트
[기존 유지]

## 라이선스
[기존 유지]
```

---

## 🎯 핵심 변경사항 요약

### Before (현재)
- 파일 기반 입력이 "NEW" 기능으로 강조 (85줄)
- 학습/추론 분리 언급 없음
- 샘플 데이터에 집중 (6가지 샘플 나열)
- 데이터 소스 추상화 언급 없음

### After (개선)
- 학습/추론 워크플로우가 핵심 (명확한 예제)
- 파일 입력은 데이터 입력 방식 중 하나로 간략히 설명
- 학습용/추론용 데이터 생성 스크립트 명시
- 데이터 소스 추상화 설명 추가

---

## ✅ 실행 순서

### 1. 백업
```bash
cp README.md README.md.backup
```

### 2. 섹션별 업데이트
1. 학습/추론 워크플로우 섹션 추가 (18줄 다음)
2. 파일 기반 입력 섹션 간소화 (18-103줄 → 20줄)
3. 주요 컴포넌트 섹션 업데이트 (280-308줄)
4. 문서 섹션 추가 (324줄 앞)

### 3. 링크 검증
- 모든 문서 링크 동작 확인
- 존재하지 않는 문서는 "향후 작성" 표시

### 4. 검토
- 전체 흐름 확인
- 중복 내용 제거
- 일관성 검증

---

## 📋 체크리스트

- [ ] 학습/추론 워크플로우 섹션 추가
- [ ] 파일 기반 입력 섹션 간소화
- [ ] 데이터 소스 추상화 설명 추가
- [ ] 주요 컴포넌트 업데이트
- [ ] 문서 인덱스 추가
- [ ] 모든 링크 검증
- [ ] 중복 내용 제거
- [ ] CLAUDE.md에도 반영

---

**작성자**: Claude Code
**버전**: 1.0.0
**상태**: 실행 준비 완료
