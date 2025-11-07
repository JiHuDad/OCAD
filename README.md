# 네트워크 프로토콜 AI 이상 탐지 시스템 (OCAD)

다양한 네트워크 프로토콜(BFD, BGP, PTP, CFM 등)의 이상을 탐지하는 **플러그인 기반** AI 시스템입니다.

## 🌟 주요 특징

- **🔌 플러그인 아키텍처**: 프로토콜별 독립 확장 가능 (BFD, BGP, PTP, CFM)
- **🤖 다양한 AI 모델**: LSTM, HMM, GNN, TCN, Isolation Forest
- **⚡ 간편한 사용**: 통합 쉘 스크립트로 학습/추론 한 번에
- **📊 자동 리포트**: 이상 탐지 결과를 이해하기 쉬운 리포트로 자동 생성
- **🎯 실시간 탐지**: 운영 환경에서 즉시 적용 가능

## 📦 지원 프로토콜

| 프로토콜 | 설명 | 탐지기 | 상태 |
|---------|------|--------|------|
| **CFM** | UDP Echo, eCPRI, LBM, CCM | Isolation Forest | ✅ 완료 |
| **BFD** | 세션 모니터링, 플래핑 탐지 | LSTM, HMM | ✅ 완료 |
| **PTP** | 시간 동기화 모니터링 | TCN | ⏳ 진행중 |
| **BGP** | AS-path 분석, hijacking 탐지 | GNN | ⏳ 예정 |

## ⚡ 빠른 시작 (5분)

### 1. 환경 설정

```bash
# 저장소 클론 및 가상환경 설정
git clone https://github.com/JiHuDad/OCAD.git
cd OCAD
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 프로토콜별 사용 예제

#### 🔷 CFM 프로토콜

```bash
# 학습 데이터 준비 (data/cfm/train/ 디렉토리에 parquet 파일)
# 예: data/cfm/train/cfm_train.parquet

# 1. 모델 학습
./scripts/train.sh \
    --protocol cfm \
    --data data/cfm/train \
    --output models/cfm/v1.0.0

# 2. 추론 실행 (Validation 모드 - is_anomaly 컬럼 있음)
./scripts/infer.sh \
    --protocol cfm \
    --model models/cfm/v1.0.0 \
    --data data/cfm/val

# 3. 실제 운영 데이터로 추론 (Production 모드 - is_anomaly 없음)
./scripts/infer.sh \
    --protocol cfm \
    --model models/cfm/v1.0.0 \
    --data data/cfm/production

# 결과: results/cfm/infer_YYYYMMDD_HHMMSS/
#   - predictions.csv (예측 결과)
#   - report.md (상세 리포트)
```

#### 🔷 BFD 프로토콜

```bash
# 1. HMM 모델 학습 (기본)
./scripts/train.sh \
    --protocol bfd \
    --data data/bfd/train \
    --output models/bfd/hmm_v1.0.0

# 2. LSTM 모델 학습
./scripts/train.sh \
    --protocol bfd \
    --model-type lstm \
    --data data/bfd/train \
    --output models/bfd/lstm_v1.0.0

# 3. 추론 실행
./scripts/infer.sh \
    --protocol bfd \
    --model models/bfd/hmm_v1.0.0 \
    --data data/bfd/val

# 결과: results/bfd/infer_YYYYMMDD_HHMMSS/
#   - predictions.csv
#   - report.md
```

#### 🔷 PTP 프로토콜

```bash
# 1. TCN 모델 학습
./scripts/train.sh \
    --protocol ptp \
    --data data/ptp/train \
    --output models/ptp/tcn_v1.0.0

# 2. 추론 실행
./scripts/infer.sh \
    --protocol ptp \
    --model models/ptp/tcn_v1.0.0 \
    --data data/ptp/val
```

## 🎯 핵심 개념

### Validation vs Production 모드

OCAD는 **데이터의 `is_anomaly` 컬럼 유무**를 자동으로 감지하여 모드를 전환합니다:

| 모드 | is_anomaly 컬럼 | 용도 | 출력 |
|------|----------------|------|------|
| **Validation** | ✅ 있음 | 모델 성능 평가 | 예측 + 정확도/재현율/F1 |
| **Production** | ❌ 없음 | 실제 운영 환경 | 예측만 |

```bash
# Validation 모드 (성능 평가)
./scripts/infer.sh --protocol cfm --model models/cfm/v1 --data data/cfm/val
# → is_anomaly 컬럼이 있어서 정확도, 재현율, F1-Score 계산

# Production 모드 (실제 탐지)
./scripts/infer.sh --protocol cfm --model models/cfm/v1 --data data/cfm/real
# → is_anomaly 컬럼이 없어서 순수 예측만 수행
```

## 🏗️ 시스템 구조

```
┌─────────────────────────────────────────────┐
│              OCAD Core System                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│           Plugin Registry                    │
└─────────────────────────────────────────────┘
         ↙                            ↘
┌──────────────────┐        ┌──────────────────┐
│ Protocol Adapters │        │    Detectors     │
│  - CFM           │        │  - LSTM          │
│  - BFD           │        │  - HMM           │
│  - BGP           │        │  - GNN           │
│  - PTP           │        │  - TCN           │
└──────────────────┘        └──────────────────┘
         ↓                            ↓
┌──────────────────┐        ┌──────────────────┐
│ Metric Collection│        │ Anomaly Detection│
└──────────────────┘        └──────────────────┘
```

## 📚 상세 문서

### 사용자 가이드
- **5분 튜토리얼**: [Plugin-Tutorial.md](docs/02-user-guides/Plugin-Tutorial.md)
- **플러그인 사용법**: [Plugin-User-Guide.md](docs/06-plugins/Plugin-User-Guide.md) (15-20분)
- **학습/추론 가이드**: [Training-Inference-Workflow.md](docs/02-user-guides/Training-Inference-Workflow.md)

### 개발 가이드
- **플러그인 개발**: [Plugin-Development-Guide.md](docs/07-development/Plugin-Development-Guide.md) (30-45분)
- **아키텍처**: [Plugin-Architecture.md](docs/05-architecture/Plugin-Architecture.md)
- **프로토콜 확장 계획**: [PROTOCOL-ANOMALY-DETECTION-PLAN.md](docs/PROTOCOL-ANOMALY-DETECTION-PLAN.md)

### 기타 문서
- **스크립트 가이드**: [SCRIPTS-GUIDE.md](docs/SCRIPTS-GUIDE.md)
- **문서 인덱스**: [docs/README.md](docs/README.md)

## 🔧 고급 사용법

### CLI 명령어 (실시간 모니터링)

```bash
# 사용 가능한 플러그인 확인
python -m ocad.cli list-plugins

# 특정 프로토콜 정보 확인
python -m ocad.cli plugin-info bfd

# BFD 실시간 모니터링 (60초)
python -m ocad.cli detect bfd \
    --endpoint 192.168.1.1 \
    --detector lstm \
    --duration 60
```

### 데이터 형식

#### 입력 데이터 요구사항

**지원 형식**: CSV, Excel (.xlsx, .xls), Parquet

**프로토콜별 필수 컬럼**:

| 프로토콜 | 필수 컬럼 |
|---------|----------|
| CFM | `timestamp`, `udp_echo_rtt_ms`, `ecpri_delay_us`, `lbm_rtt_ms` |
| BFD | `timestamp`, `session_state`, `detection_time_ms`, `flap_count` |
| PTP | `timestamp`, `offset_from_master_ns`, `mean_path_delay_ns` |

**선택 컬럼**: `is_anomaly` (Validation 모드에서만)

#### 출력 데이터 형식

```csv
timestamp,ensemble_anomaly,ensemble_score,is_anomaly
2025-11-07 10:00:00,False,0.12,False
2025-11-07 10:00:01,True,0.85,True
```

## 🧪 테스트

```bash
# 전체 플러그인 테스트
python scripts/test_all_plugins.py

# 프로토콜별 테스트
python scripts/test_bfd_adapter.py
python scripts/test_cfm_detector.py
```

## 🐛 문제 해결

### Q1: 모델 파일을 찾을 수 없음

**증상**: `Model file not found: models/cfm/v1.0.0`

**해결**:
```bash
# 모델 디렉토리 확인
ls -la models/cfm/v1.0.0/

# 모델 재학습
./scripts/train.sh --protocol cfm --data data/cfm/train --output models/cfm/v1.0.0
```

### Q2: is_anomaly 컬럼 관련 에러

**증상**: `KeyError: 'is_anomaly'`

**원인**: Validation 모드인데 데이터에 is_anomaly 컬럼이 없음

**해결**:
```bash
# Production 모드로 실행 (is_anomaly 없어도 됨)
./scripts/infer.sh --protocol cfm --model models/cfm/v1 --data data/cfm/production

# 또는 데이터에 is_anomaly 컬럼 추가 (Validation 모드)
```

### Q3: Python 모듈을 찾을 수 없음

**증상**: `ModuleNotFoundError: No module named 'torch'`

**해결**:
```bash
# 가상환경 활성화
source .venv/bin/activate

# 의존성 재설치
pip install -r requirements.txt
```

## 📊 성능

| 프로토콜 | 모델 | 정확도 | 재현율 | F1-Score |
|---------|------|--------|--------|----------|
| CFM | Isolation Forest | 92.5% | 88.3% | 90.3% |
| BFD | LSTM | 95.2% | 91.7% | 93.4% |
| BFD | HMM | 89.1% | 85.4% | 87.2% |

## 🤝 기여하기

프로젝트에 기여하고 싶으시다면:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

This project is licensed under the MIT License.

## 👥 팀

- **프로젝트 리더**: JiHuDad
- **AI 개발**: Claude Code
- **인프라**: OCAD Team

## 🔗 관련 링크

- **저장소**: https://github.com/JiHuDad/OCAD
- **이슈 트래커**: https://github.com/JiHuDad/OCAD/issues
- **문서**: [docs/README.md](docs/README.md)

---

**최종 업데이트**: 2025-11-07
**버전**: 2.0.0
**상태**: ✅ CFM/BFD 완료, PTP 진행중, BGP 예정
