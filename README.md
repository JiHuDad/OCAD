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

## 🆕 파일 기반 데이터 입력 (NEW!)

OCAD는 실시간 수집 외에도 **파일 기반 입력**을 지원합니다. CFM 담당자가 수집한 데이터를 파일로 제공받아 분석할 수 있습니다.

### 지원 형식

- **CSV**: 사람이 읽기 쉬운 형식 (Wide/Long Format 자동 감지)
- **Excel**: 여러 Sheet 지원 (.xlsx)
- **Parquet**: 대용량 데이터 고성능 처리

### 빠른 시작

```bash
# 1. 샘플 데이터 생성
python scripts/generate_sample_data.py

# 2. 파일 로더 테스트
python scripts/test_file_loaders.py

# 3. 생성된 샘플 확인
ls -lh data/samples/
```

### 샘플 데이터

생성된 샘플 데이터는 사람이 읽을 수 있도록 설계되었습니다:

```csv
timestamp,endpoint_id,site_name,zone,udp_echo_rtt_ms,ecpri_delay_us,lbm_rtt_ms,notes
2025-10-22 09:00:00,o-ru-001,Tower-A,Urban,5.2,102.3,7.1,정상 운영
2025-10-22 09:00:10,o-ru-001,Tower-A,Urban,8.2,158.3,10.5,⚠️ RTT 증가 시작
2025-10-22 09:00:20,o-ru-001,Tower-A,Urban,25.8,350.1,25.5,🚨 CRITICAL: 높은 지연
```

**생성 가능한 샘플:**
1. `01_normal_operation_24h.csv` - 정상 운영 (24시간, 1,440개 레코드)
2. `02_drift_anomaly.csv/.xlsx` - Drift 이상 패턴 (점진적 증가)
3. `03_spike_anomaly.csv` - Spike 이상 패턴 (일시적 급증)
4. `04_multi_endpoint.csv/.parquet` - 여러 엔드포인트 데이터
5. `05_weekly_data.parquet` - 주간 데이터 (7일, 2,016개 레코드)
6. `06_comprehensive_example.xlsx` - 종합 예제 (정상+Drift+Spike)

### 파일 로더 사용법

```python
from pathlib import Path
from ocad.loaders import CSVLoader, ExcelLoader

# CSV 파일 로드
loader = CSVLoader(strict_mode=False)
result = loader.load(Path("data/samples/01_normal_operation_24h.csv"))

if result.success:
    print(f"✅ {result.valid_records}개 메트릭 로드 완료")
    for metric in result.metrics:
        # 탐지 파이프라인으로 전달
        process_metric(metric)

# Excel 파일 로드
excel_loader = ExcelLoader(sheet_name="메트릭 데이터")
result = excel_loader.load(Path("data/samples/sample_oran_metrics.xlsx"))
```

### 파일 형식 변환

```bash
# CSV → Parquet (대용량 처리 최적화)
python -c "
from ocad.loaders import FormatConverter
FormatConverter.csv_to_parquet('data/input/metrics.csv', 'data/processed/metrics.parquet')
"

# Wide Format → Long Format (분석 용이)
python -c "
from ocad.loaders import FormatConverter
FormatConverter.wide_to_long('data/input/metrics_wide.csv', 'data/processed/metrics_long.csv')
"
```

### CFM 담당자용 문서

CFM 담당자와 협의 시 사용할 문서:
- [CFM-Data-Requirements.md](docs/CFM-Data-Requirements.md) - 데이터 수집 요구사항
- [sample_oran_metrics.xlsx](data/samples/sample_oran_metrics.xlsx) - Excel 샘플 (3 sheets)

**상세 문서**: [File-Based-Input-Implementation-Summary.md](docs/File-Based-Input-Implementation-Summary.md)

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
