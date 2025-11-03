# ORAN CFM-Lite AI Anomaly Detection System

ORAN 환경에서 축소된 CFM 기능을 활용한 하이브리드 이상탐지 시스템입니다.

## 주요 특징

- **Capability-Driven**: 장비가 지원하는 기능을 자동 인식하여 파이프라인 구성
- **하이브리드 탐지**: 룰 + 변화점(CUSUM/PELT) + 예측-잔차(TCN/LSTM) + 다변량(옵션)
- **조기 경보**: 끊기기 전 징조를 4분 이상 앞서 탐지 (목표)
- **최소 신호**: UDP-echo, eCPRI delay, LBM만으로도 효과적 탐지

## 시스템 구조

```
O-RU/O-DU → Capability Detector → Collectors → Feature Engine → Detectors → Alerts
```

## 🤖 학습 및 추론 워크플로우

OCAD는 **학습-추론 분리 아키텍처**를 사용하며, **Phase 1-4 완료**로 즉시 사용 가능합니다.

### ✅ 현재 상태 (Phase 1-4 완료)

학습된 4개 모델이 준비되어 있어 **즉시 추론 가능**합니다:

- **TCN 모델 3개**: UDP Echo, eCPRI, LBM (예측-잔차 기반 탐지)
- **Isolation Forest 1개**: 다변량 이상 탐지

### ⚡ 원스톱 Shell Scripts (추천!)

복잡한 파이프라인을 간단한 명령어로 실행할 수 있습니다:

```bash
# 1️⃣ 데이터셋 생성 (CSV + Parquet 변환)
./scripts/make_dataset.sh

# 생성되는 파일:
#   - 01_training_normal.csv (학습용 정상 데이터)
#   - 02_validation_normal.csv (검증용 정상 데이터)
#   - 03_validation_drift_anomaly.csv (Drift 이상)
#   - 04_validation_spike_anomaly.csv (Spike 이상)
#   - 05_validation_packet_loss_anomaly.csv (패킷 손실)
#   + Parquet 파일들 (data/processed/)

# 2️⃣ 모델 학습 (TCN 3개 + Isolation Forest)
./scripts/train.sh --train-data data/datasets/01_training_normal.csv

# 학습되는 모델:
#   - udp_echo_v2.0.2.pth + scaler
#   - ecpri_v2.0.2.pth + scaler
#   - lbm_v2.0.2.pth + scaler
#   - isolation_forest_2.0.2.pkl + scaler

# 3️⃣ 추론 + 리포트 생성
./scripts/infer.sh --input data/datasets/03_validation_drift_anomaly.csv

# 결과:
#   - data/results/03_validation_drift_anomaly_result.csv (추론 결과)
#   - reports/03_validation_drift_anomaly_report.md (상세 리포트)
```

**상세 옵션**:
```bash
# 데이터셋 생성 옵션
./scripts/make_dataset.sh --help

# 학습 옵션 (에포크, 배치 크기, 버전 등)
./scripts/train.sh --help

# 추론 옵션 (출력 파일, 리포트 옵션 등)
./scripts/infer.sh --help
```

### 🚀 빠른 시작: 추론 실행 (Python 직접 사용)

#### 1️⃣ 자신의 데이터로 추론 (권장!)

```bash
# 가상환경 활성화
source .venv/bin/activate

# 데이터 파일을 선택하여 추론 실행
python scripts/inference_simple.py \
    --input data/samples/01_normal_operation_24h.csv \
    --output data/results/my_inference.csv

# 지원 형식: CSV, Excel (.xlsx, .xls), Parquet
# 필수 컬럼: timestamp, endpoint_id, udp_echo_rtt_ms, ecpri_delay_us, lbm_rtt_ms, ccm_miss_count
# 출력: residual_score, multivariate_score, is_anomaly 컬럼 포함 CSV

# 출력 예시:
# 📂 입력 데이터: data/samples/01_normal_operation_24h.csv
# ✅ 1440개 레코드 로드 완료
# 🔧 ResidualDetector 초기화 중...
# 🔧 MultivariateDetector 초기화 중...
# 🚀 1440개 샘플에 대한 추론 실행 중...
# 📊 추론 결과 요약
# 총 샘플 수: 1440
# 이상 탐지 수: 0
# 이상 탐지율: 0.00%
# 💾 결과 저장 완료: data/results/my_inference.csv
```

#### 2️⃣ 모델 검증 (시스템 확인용)

```bash
# 통합 테스트 (모든 모델 로드 확인)
python scripts/test_integrated_detectors.py

# 검증 데이터셋으로 모델 성능 확인
python scripts/validate_all_models.py
# 출력: 정상 10.0% / 드리프트 81.0% / 스파이크 26.2%
```

### 📊 결과 확인 방법

추론 결과는 4가지 방법으로 확인할 수 있습니다:

#### 1. 상세 리포트 생성 (⭐ 추천!)

```bash
# 추론 결과 리포트 생성 (왜 이상인지 쉽게 설명)
python scripts/generate_inference_report.py \
    --inference-result data/results/my_inference.csv \
    --original-data data/datasets/03_validation_drift_anomaly.csv \
    --output reports/my_report.md

# 리포트 확인
cat reports/my_report.md
```

**리포트 내용**:
- 📊 전체 요약 (이상 탐지율, 평균 점수)
- ⚠️ 이상 구간 분석 (시작/종료 시간, 지속 시간)
- 🔍 이상 원인 분석 (메트릭별 정상/이상 비교, 변화율, 표준편차)
- 💡 권장 사항 (이상 탐지율에 따른 조치 방법)
- 📋 이상 데이터 샘플 (상위 10개)
  - 각 샘플마다 메트릭별 상세 분석 (상태 표시기: 🔴🟡🟢)
  - 정상 평균 대비 변화율 (%) 및 표준편차 배수 (σ)
  - 종합 판단 (어떤 메트릭에 문제가 있는지 명확히 설명)

**샘플 예시**:

```markdown
### 🔴 이상 샘플 #1
**시간**: 2025-10-02 01:40:00
**최종 이상 점수**: 0.6585

**메트릭 분석**:
- 🔴 **UDP Echo RTT**: 19.70 ms
  - 정상 평균: 5.00 ms
  - 차이: +294.4% (+26.95σ)
- 🔴 **eCPRI Delay**: 226.12 μs
  - 정상 평균: 99.47 μs
  - 차이: +127.3% (+13.32σ)

**💡 종합 판단**:
- UDP Echo RTT가 정상 대비 294% 증가
- eCPRI 지연이 정상 대비 127% 증가
```

#### 2. CSV 파일

```bash
# 추론 결과 CSV 파일 확인
cat data/results/my_inference.csv

# 출력 예시:
# timestamp,endpoint_id,residual_score,residual_anomaly,multivariate_score,multivariate_anomaly,final_score,is_anomaly
# 2025-10-30 00:00:00,endpoint-1,0.0,0,0.0017,0,0.0017,0
# 2025-10-30 00:01:00,endpoint-1,0.0,0,0.0023,0,0.0023,0
# ...

# Excel로 열거나 pandas로 분석 가능
```

#### 3. 콘솔 출력

```bash
python scripts/inference_simple.py --input YOUR_DATA.csv

# 터미널에 실시간으로 출력됩니다:
# - 로드된 레코드 수
# - 탐지기 초기화 상태
# - 진행률 (100개 단위)
# - 최종 요약 (이상 탐지율, 평균 점수)
```

#### 4. 개별 모델 테스트

```bash
# TCN 모델만 테스트
python scripts/test_all_tcn_models.py

# Isolation Forest만 테스트
python scripts/test_isolation_forest.py
```

### 🔧 모델 학습 (처음부터 학습하기)

자신의 데이터가 없다면 먼저 데이터셋을 생성하고, 모델을 학습할 수 있습니다:

#### 0️⃣ 데이터셋 생성 (자신의 데이터가 없는 경우)

```bash
# 학습용 정상 데이터 + 검증용 정상/비정상 데이터 자동 생성
python scripts/generate_datasets.py \
    --training-hours 24 \
    --validation-hours 12 \
    --anomaly-hours 6 \
    --formats csv parquet

# 생성되는 파일:
# 1. data/datasets/01_training_normal.csv (학습용 정상 데이터, 24시간)
# 2. data/datasets/02_validation_normal.csv (검증용 정상 데이터, 12시간)
# 3. data/datasets/03_validation_drift_anomaly.csv (점진적 증가 이상)
# 4. data/datasets/04_validation_spike_anomaly.csv (급격한 증가 이상)
# 5. data/datasets/05_validation_packet_loss_anomaly.csv (패킷 손실 이상)
```

**데이터셋 구성**:

- **학습 데이터**: 정상 운영 데이터만 포함 (24시간, 1440개 샘플)
- **검증 정상 데이터**: 모델 성능 확인용 정상 데이터 (12시간, 720개 샘플)
- **검증 비정상 데이터**: 3가지 이상 패턴 (각 6시간, 360개 샘플)
  - Drift: 점진적 성능 저하
  - Spike: 급격한 지연 증가
  - Packet Loss: 패킷 손실 발생

#### 1️⃣ TCN 모델 학습 (시계열 예측-잔차 탐지)

```bash
# Step 1: 학습 데이터 준비 (Parquet 포맷으로 변환)
python scripts/prepare_timeseries_data_v2.py \
    --input data/datasets/01_training_normal.csv \
    --output-dir data/processed \
    --metric-type udp_echo

# Step 2: TCN 모델 학습
python scripts/train_tcn_model.py \
    --train-data data/processed/timeseries_train.parquet \
    --val-data data/processed/timeseries_val.parquet \
    --test-data data/processed/timeseries_test.parquet \
    --metric-type udp_echo \
    --epochs 50 \
    --batch-size 32

# 저장 위치:
#   - 모델: ocad/models/tcn/udp_echo_v1.0.0.pth
#   - 메타데이터: ocad/models/tcn/udp_echo_v1.0.0.json
#   - 성능 리포트: ocad/models/metadata/performance_reports/udp_echo_v1.0.0_report.json

# 다른 메트릭 학습: --metric-type ecpri 또는 lbm
```

#### 2️⃣ Isolation Forest 학습 (다변량 이상 탐지)

```bash
# Step 1: 다변량 피처 데이터 준비
python scripts/prepare_multivariate_data.py \
    --input data/datasets/01_training_normal.csv \
    --output-dir data/processed

# Step 2: Isolation Forest 학습
python scripts/train_isolation_forest.py \
    --train-data data/processed/multivariate_train.parquet \
    --val-data data/processed/multivariate_val.parquet \
    --test-data data/processed/multivariate_test.parquet

# 저장 위치 (기본값):
#   - 모델: ocad/models/isolation_forest/isolation_forest_v1.0.0.pkl
#   - 스케일러: ocad/models/isolation_forest/isolation_forest_v1.0.0_scaler.pkl
#   - 메타데이터: ocad/models/isolation_forest/isolation_forest_v1.0.0.json
```

#### 3️⃣ 학습된 모델 검증

```bash
# 정상 데이터로 검증 (낮은 이상 탐지율 기대)
python scripts/inference_simple.py \
    --input data/datasets/02_validation_normal.csv \
    --output results_normal.csv

# Drift 이상 데이터로 검증 (높은 이상 탐지율 기대)
python scripts/inference_simple.py \
    --input data/datasets/03_validation_drift_anomaly.csv \
    --output results_drift.csv

# Spike 이상 데이터로 검증
python scripts/inference_simple.py \
    --input data/datasets/04_validation_spike_anomaly.csv \
    --output results_spike.csv
```

**입력 데이터 형식**:

- **파일 형식**: CSV, Excel (.xlsx, .xls), Parquet
- **필수 컬럼**: `timestamp, endpoint_id, udp_echo_rtt_ms, ecpri_delay_us, lbm_rtt_ms, ccm_miss_count`
- **데이터 요구사항**:
  - 학습: 정상 데이터만 사용 (이상 데이터 제외)
  - 검증: 정상/비정상 데이터 모두 사용 가능

### 📈 학습 결과

**Phase 1-4 완료 상태** (2025-10-30):

| 모델 | 데이터 | 성능 | 크기 |
|------|--------|------|------|
| UDP Echo TCN v2.0.0 | 28,750 시퀀스 (17 epochs) | R² = 0.19 | 17KB |
| eCPRI TCN v2.0.0 | 1,430 시퀀스 (7 epochs) | R² = -0.003 | 17KB |
| LBM TCN v2.0.0 | 1,430 시퀀스 (6 epochs) | R² = -0.008 | 17KB |
| Isolation Forest v1.0.0 | 1,431 샘플 (20 피처) | Drift 81% / Spike 26% | 1.14MB |

**상세 리포트**:

- [Phase 4 완료 리포트](docs/PHASE4-COMPLETION-REPORT.md) - 모델 통합
- [Phase 3 완료 리포트](docs/PHASE3-COMPLETION-REPORT.md) - Isolation Forest
- [Quick Status](docs/QUICK-STATUS.md) - 전체 진행 상황

## 💾 데이터 입력 방식

OCAD는 두 가지 데이터 입력 방식을 지원합니다:

### 1. 파일 기반 (현재 지원)
- **CSV, Excel, Parquet** 형식
- 사람이 읽기 쉬운 형식
- 학습/추론 데이터 분리

```bash
# 데모용 샘플 생성 (다양한 시나리오)
python scripts/generate_sample_data.py

# 학습/추론 데이터 생성
python scripts/generate_training_inference_data.py
```

### 2. 실시간 수집 (기본)
- **NETCONF/YANG**을 통한 실시간 수집
- UDP Echo, eCPRI, LBM 메트릭
- Capability 자동 감지

**상세 가이드**: [Data-Source-Guide.md](docs/03-data-management/Data-Source-Guide.md)

---

## 설치 및 실행

### 빠른 시작

```bash
# 자동 설치 및 실행
./scripts/start.sh
```

### 수동 설치

```bash
# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 설정
cp config/example.yaml config/local.yaml
# config/local.yaml 편집

# 서비스 실행 (API 서버)
python -m ocad.api.main

# 또는 백그라운드 시스템만 실행
python -m ocad.main
```

### CLI 사용

```bash
# 엔드포인트 추가
python -m ocad.cli add-endpoint 192.168.1.100 --role o-ru

# 시스템 상태 확인
python -m ocad.cli status

# 시뮬레이션 실행
python -m ocad.cli simulate --count 10 --duration 300

# 또는 스크립트 사용
./scripts/simulate.py
```

### 환경 설정

환경변수를 통해 시스템 설정을 변경할 수 있습니다:

```bash
# .env 파일 생성 (선택사항)
cat > .env << EOF
ENVIRONMENT=development
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8080
DATABASE__URL=postgresql+asyncpg://ocad:ocad@localhost:5432/ocad
REDIS__URL=redis://localhost:6379/0
MONITORING__LOG_LEVEL=INFO

# 로그 디렉토리 커스터마이징 (선택사항)
OCAD_LOG_DIR=/var/log/ocad  # 기본값: <프로젝트루트>/logs
EOF
```

### 로그 디렉토리 설정
```bash
# 기본값: 프로젝트 루트의 logs 디렉토리
python3 scripts/quick_test.py

# 커스텀 로그 디렉토리 사용
export OCAD_LOG_DIR=/tmp/ocad_logs
python3 scripts/quick_test.py

# 시스템 로그 디렉토리 사용
export OCAD_LOG_DIR=/var/log/ocad
sudo mkdir -p /var/log/ocad
sudo chown $USER:$USER /var/log/ocad
python3 scripts/quick_test.py
```

### 📁 로그 구조
테스트 실행 시 다음과 같은 구조로 로그가 생성됩니다:
```
logs/test_YYYYMMDD_HHMMSS/
├── debug/detailed.log              # 🔍 모든 디버그 정보 (CUSUM 계산, 피처 추출 등)
├── summary/summary.log             # 📋 중요 이벤트만 (시스템 시작/종료, 알람 등)
└── alerts/
    ├── alert_details.log           # 🚨 기술적 알람 정보
    └── human_readable_analysis.txt # 📖 사람 친화적 분석 보고서
```

## 🧪 시스템 테스트 (실제 CFM 장비 없이)

OCAD 시스템은 실제 ORAN 장비 없이도 완전한 검증이 가능합니다:

### 0. 플랫폼 호환성 체크 (30초)
```bash
# 다른 플랫폼에서 먼저 호환성 확인
python3 scripts/compatibility_check.py
```
- ✅ 기본 Import 및 모델 호환성 확인
- ✅ Alert 모델 속성 접근 방식 검증
- ✅ 로깅 시스템 및 경로 처리 검증
- 🎯 **다른 환경에서 실행 전 필수 체크**

### 1. 빠른 검증 테스트 (2분)
```bash
# 시스템 기본 동작 확인 (구조화된 로그 포함)
./scripts/quick_test.py
```
- ✅ 데이터 수집/피처 추출/이상 탐지 파이프라인 검증
- ✅ 가상 O-RU/O-DU/Transport 엔드포인트 시뮬레이션
- ✅ 지연 스파이크 주입으로 알람 생성 확인
- 📝 **구조화된 로그 시스템**: debug/summary/alerts 폴더별 분리
- 📖 **사람 친화적 분석 보고서**: 운영자가 이해하기 쉬운 알람 분석

### 2. 상세 시나리오 테스트 (10분)
```bash
# 다양한 장애 상황 시뮬레이션
./scripts/scenario_test.py
```
- 🔥 급격한 지연 증가 (네트워크 혼잡)
- 📈 점진적 성능 저하 (하드웨어 열화)
- 📉 패킷 손실 (전송 품질 저하)
- 💥 동시 다발적 문제 (복합 장애)

### 3. 사람 친화적 보고서 샘플 생성
```bash
# 알람 분석 보고서 샘플 생성 및 확인
python3 scripts/generate_sample_report.py
```
- 📖 **WARNING/CRITICAL 레벨 샘플 보고서** 생성
- 🔍 **문제 요약**: 기술 용어 없이 이해하기 쉬운 설명
- 💡 **구체적 조치사항**: 실행 가능한 권장사항 제공
- 👀 **모니터링 포인트**: 지속 관찰이 필요한 지표 안내

### 4. 실시간 대시보드 테스트 (지속적)
```bash
# 웹 대시보드와 실시간 모니터링
./scripts/dashboard_test.py
```
- 📊 실시간 통계 및 KPI 모니터링
- 🚨 알람 생성/관리 확인
- 🌐 REST API 엔드포인트 테스트
- 브라우저에서 http://localhost:8080 접속

### 타당성 검증 포인트

**✅ Capability-Driven 아키텍처**
- 장비별 지원 기능 자동 인식
- UDP Echo, eCPRI, LBM 등 CFM-Lite 기능 시뮬레이션

**✅ 하이브리드 이상 탐지**
- 룰 기반: 즉시 임계값 위반 탐지
- 변화점: CUSUM으로 급격한 변화 포착
- 잔차: TCN으로 드리프트 조기 탐지
- 다변량: 동시성 패턴 감지

**✅ 스마트 알람 시스템**
- 근거 3개 원칙으로 오탐 감소
- Hold-down/중복제거로 알람 폭주 방지
- 심각도별 자동 분류 (CRITICAL/WARNING/INFO)
- 📖 **사람 친화적 분석 보고서**: 운영자가 바로 이해할 수 있는 상세 분석
- 💡 **구체적 조치사항**: 실행 가능한 권장사항 자동 생성
- 📁 **구조화된 로그**: debug/summary/alerts 레벨별 분리

**✅ 실시간 성능**
- 처리 지연 < 30초 (목표)
- 수천 엔드포인트 확장성
- 메모리 효율적 스트리밍 처리

## 주요 컴포넌트

### 1. Capability Detection
- NETCONF/YANG을 통한 장비 기능 자동 인식
- 지원 기능에 따른 수집 파이프라인 자동 구성

### 2. Data Collection
- UDP Echo RTT 측정
- eCPRI One-way Delay 수집  
- LBM (Loopback) RTT 및 성공률
- CCM 최소 통계 (지원시)

### 3. Feature Engineering
- 백분위 (p95, p99) 계산
- EWMA, 기울기, CUSUM 누적량
- 예측-잔차 (TCN/LSTM)
- Run-length, 동시성 지표

### 4. Anomaly Detection
- **룰 기반**: 명확한 임계값 (빠른 탐지)
- **변화점**: CUSUM/PELT (급격한 변화)  
- **예측-잔차**: TCN/LSTM (드리프트 탐지)
- **다변량**: MSCRED/Isolation Forest (그룹 이상)

### 5. Alert Management
- 근거 3개 원칙 (드리프트/급등/동시성 중 2-3개)
- Hold-down, 중복 제거, 억제
- 스파크라인 및 capability 스냅샷

### 6. Data Source Abstraction
- 파일 기반 입력 (CSV/Excel/Parquet)
- 스트리밍 입력 (Kafka/WebSocket - 향후)
- 통일된 DataSource 인터페이스
- 학습/추론 모두 동일한 방식으로 처리

## 📚 문서

### 시작하기
- [빠른 시작 가이드](docs/01-getting-started/Quick-Start-Guide.md) - 5분 내 시작
- [학습-추론 워크플로우](docs/02-user-guides/Training-Inference-Workflow.md) - 전체 흐름 이해

### 학습 및 추론
- [학습-추론 개요](docs/04-training-inference/Overview.md) - 핵심 개념 (5-10분)
- [학습 가이드](docs/04-training-inference/Training-Guide.md) - 모델 학습 방법
- [추론 가이드](docs/04-training-inference/Inference-Guide.md) - 이상 탐지 실행

### 데이터 관리
- [데이터 소스 가이드](docs/03-data-management/Data-Source-Guide.md) - 파일/스트리밍 입력
- [CFM 데이터 요구사항](docs/03-data-management/CFM-Data-Requirements.md) - CFM 담당자용

### 운영 및 개발
- [운영 가이드](docs/02-user-guides/Operations-Guide.md) - 시스템 운영
- [로깅 가이드](docs/02-user-guides/Logging-Guide.md) - 로그 분석
- [API 참조](docs/02-user-guides/API.md) - REST API

### 아키텍처
- [학습-추론 분리 설계](docs/05-architecture/Training-Inference-Separation-Design.md) - 온라인/오프라인 분리
- [데이터 소스 추상화](docs/05-architecture/Data-Source-Abstraction-Design.md) - 파일/스트리밍 지원

**전체 문서 인덱스**: [docs/README.md](docs/README.md)

## 성능 목표

- MTTD 20-30% 단축 (룰 대비)
- 사전 경고 리드타임 p50 ≥ 4분
- 오경보율 ≤ 6%
- 운영자 승인율 ≥ 80%
- 전체 지연 ≤ 30초 (95th)

## 개발 스프린트

- Sprint 0: 프로젝트 초기화 및 기본 구조
- Sprint 1: Capability 감지 및 수집기
- Sprint 2: 피처링 및 탐지 엔진 v1
- Sprint 3: 운영 및 대시보드
- Sprint 4: 다변량 탐지 및 하드닝

## 라이선스

MIT License
