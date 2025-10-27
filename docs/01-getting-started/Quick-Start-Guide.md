# OCAD 빠른 시작 가이드

**버전**: 1.0.0
**최종 업데이트**: 2025-10-23

## 목차

1. [개요](#개요)
2. [파일 기반 입력으로 시작하기](#파일-기반-입력으로-시작하기)
3. [샘플 데이터 생성](#샘플-데이터-생성)
4. [데이터 로드 및 분석](#데이터-로드-및-분석)
5. [CFM 담당자 협의](#cfm-담당자-협의)
6. [다음 단계](#다음-단계)

---

## 개요

OCAD (ORAN CFM-Lite AI Anomaly Detection System)는 ORAN 네트워크에서 CFM 메트릭을 기반으로 이상을 탐지하는 시스템입니다.

**주요 특징:**
- ✅ 파일 기반 입력 지원 (CSV, Excel, Parquet)
- ✅ 실시간 REST API 입력 지원 (선택)
- ✅ 하이브리드 이상 탐지 (룰 + 변화점 + 예측-잔차 + 다변량)
- ✅ 사람이 읽을 수 있는 샘플 데이터
- ✅ 자동 검증 및 형식 변환

---

## 파일 기반 입력으로 시작하기

실제 ORAN 장비나 실시간 수집 없이도 OCAD를 사용할 수 있습니다!

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 샘플 데이터 생성

```bash
# 다양한 시나리오의 샘플 데이터 생성
python scripts/generate_sample_data.py
```

**생성되는 파일:**
```
data/samples/
├── 01_normal_operation_24h.csv      # 정상 운영 (24시간, 1,440개)
├── 02_drift_anomaly.csv             # Drift 이상 (점진적 증가)
├── 02_drift_anomaly.xlsx            # Drift 이상 (Excel)
├── 03_spike_anomaly.csv             # Spike 이상 (일시적 급증)
├── 04_multi_endpoint.csv            # 여러 엔드포인트
├── 04_multi_endpoint.parquet        # 여러 엔드포인트 (Parquet)
├── 05_weekly_data.parquet           # 주간 데이터 (7일, 2,016개)
└── 06_comprehensive_example.xlsx    # 종합 예제
```

### 3. 샘플 데이터 확인

**CSV 파일 (텍스트 에디터 또는 Excel):**
```bash
# 명령줄에서 확인
head -20 data/samples/01_normal_operation_24h.csv

# 또는 Excel에서 열기
open data/samples/02_drift_anomaly.xlsx  # macOS
xdg-open data/samples/02_drift_anomaly.xlsx  # Linux
```

**Parquet 파일 (Python):**
```python
import pandas as pd

# Parquet 파일 읽기
df = pd.read_parquet('data/samples/05_weekly_data.parquet')

# 데이터 확인
print(df.head(10))
print(f"\n총 {len(df):,}개 레코드")
print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
```

---

## 샘플 데이터 생성

### 사용자 정의 샘플 생성

```python
from scripts.generate_sample_data import SampleDataGenerator
from pathlib import Path

generator = SampleDataGenerator(Path("data/custom"))

# 1. 정상 운영 데이터 (7일)
df_normal = generator.generate_normal_operation(
    endpoint_id="o-ru-custom-001",
    site_name="My-Site",
    zone="Urban",
    duration_hours=24 * 7,  # 7일
    interval_seconds=300     # 5분 간격
)
generator.save_as_csv(df_normal, "my_normal_data.csv")

# 2. Drift 이상 패턴
df_drift = generator.generate_drift_anomaly(
    endpoint_id="o-ru-custom-002",
    site_name="Problem-Site",
    zone="Rural"
)
generator.save_as_excel(df_drift, "my_drift_data.xlsx")

# 3. Spike 이상 패턴
df_spike = generator.generate_spike_anomaly(
    endpoint_id="o-ru-custom-003",
    site_name="Spike-Site",
    zone="Suburban"
)
generator.save_as_csv(df_spike, "my_spike_data.csv")
```

### 데이터 형식

**Wide Format (사람이 읽기 쉬움):**
```csv
timestamp,endpoint_id,site_name,udp_echo_rtt_ms,ecpri_delay_us,lbm_rtt_ms,...
2025-10-22 09:00:00,o-ru-001,Tower-A,5.2,102.3,7.1,...
```

**Long Format (프로그래밍 처리에 유리):**
```csv
timestamp,endpoint_id,site_name,metric_name,value,unit
2025-10-22 09:00:00,o-ru-001,Tower-A,udp_echo_rtt,5.2,ms
2025-10-22 09:00:00,o-ru-001,Tower-A,ecpri_delay,102.3,us
```

---

## 데이터 로드 및 분석

### 파일 로더 사용

```python
from pathlib import Path
from ocad.loaders import CSVLoader, ExcelLoader, ParquetLoader

# 1. CSV 파일 로드 (Wide/Long 자동 감지)
csv_loader = CSVLoader(strict_mode=False)
result = csv_loader.load(Path("data/samples/01_normal_operation_24h.csv"))

if result.success:
    print(f"✅ {result.valid_records}개 메트릭 로드 완료")
    print(f"   성공률: {result.success_rate * 100:.1f}%")

    # 메트릭 처리
    for metric in result.metrics:
        print(f"  {metric.endpoint_id}: {metric.metric_type} = {metric.value} {metric.unit}")
else:
    print(f"❌ 로드 실패")
    for error in result.errors[:5]:  # 처음 5개 에러만
        print(f"  - {error}")

# 2. Excel 파일 로드
excel_loader = ExcelLoader(sheet_name="메트릭 데이터")
result = excel_loader.load(Path("data/samples/sample_oran_metrics.xlsx"))

# 3. Parquet 파일 로드 (대용량 데이터)
parquet_loader = ParquetLoader()
result = parquet_loader.load(Path("data/samples/05_weekly_data.parquet"))
```

### 파일 형식 변환

```python
from ocad.loaders import FormatConverter
from pathlib import Path

# CSV → Parquet (압축 및 성능 향상)
FormatConverter.csv_to_parquet(
    csv_path=Path("data/input/large_data.csv"),
    parquet_path=Path("data/processed/large_data.parquet"),
    compression="snappy"
)

# Wide → Long (분석 용이)
FormatConverter.wide_to_long(
    input_path=Path("data/input/metrics_wide.csv"),
    output_path=Path("data/processed/metrics_long.csv")
)

# Long → Wide (사람이 읽기 쉬움)
FormatConverter.long_to_wide(
    input_path=Path("data/input/metrics_long.csv"),
    output_path=Path("data/processed/metrics_wide.csv")
)
```

### 로더 테스트

```bash
# 모든 로더 및 변환기 테스트
python scripts/test_file_loaders.py
```

**예상 출력:**
```
============================================================
파일 로더 테스트
============================================================

Test 1: CSV Loader (Wide Format)
✅ 통과

Test 2: CSV Loader (Long Format)
✅ 통과

Test 3: Excel Loader
✅ 통과

Test 4: CSV → Parquet 변환
✅ 통과

Test 5: Wide → Long 변환
✅ 통과

Test 6: Long → Wide 변환
✅ 통과

------------------------------------------------------------
총 6개 테스트
통과: 6개
실패: 0개
```

---

## CFM 담당자 협의

### 준비 자료

1. **Excel 샘플 파일** ([data/samples/sample_oran_metrics.xlsx](../data/samples/sample_oran_metrics.xlsx))
   - Sheet 1: 메트릭 데이터 (실제 샘플)
   - Sheet 2: 필드 설명 (11개 필드, Level 1/2/3 분류)
   - Sheet 3: 예상 이상 케이스 (5가지 시나리오)

2. **요구사항 문서** ([docs/CFM-Data-Requirements.md](CFM-Data-Requirements.md))
   - 필수/권장/선택 필드 목록
   - 수집 가능 여부 체크리스트
   - 예상 이상 시나리오

### 협의 체크리스트

- [ ] 필수 필드 (Level 1) 수집 가능 여부
  - [ ] timestamp (측정 시각)
  - [ ] endpoint_id (장비 식별자)
  - [ ] udp_echo_rtt_ms (UDP Echo RTT)

- [ ] 권장 필드 (Level 2) 수집 가능 여부
  - [ ] ecpri_delay_us (eCPRI 지연)
  - [ ] lbm_rtt_ms (Loopback RTT)
  - [ ] lbm_success (Loopback 성공 여부)
  - [ ] ccm_interval_ms (CCM 간격)
  - [ ] ccm_miss_count (CCM 누락 횟수)

- [ ] 선택 필드 (Level 3)
  - [ ] site_name (사이트 이름)
  - [ ] zone (지역 구분)

- [ ] 파일 형식 선호도
  - [ ] CSV
  - [ ] Excel
  - [ ] Parquet

- [ ] 데이터 전송 주기
  - [ ] 실시간 (REST API/Kafka)
  - [ ] 배치 (1시간, 1일, 1주일)

### 피드백 반영

CFM 담당자 피드백 받은 후:

1. **스키마 조정**
   ```bash
   # ocad/core/schemas.py 수정
   # 수집 불가능한 필드를 Optional로 변경 또는 제거
   ```

2. **로더 업데이트**
   ```bash
   # ocad/loaders/csv_loader.py 등 업데이트
   # 새로운 필드 매핑 추가
   ```

3. **재검증**
   ```bash
   # 수정된 스키마로 테스트
   python scripts/test_file_loaders.py
   ```

---

## 다음 단계

### 단기 (1-2주)

1. **파일 로더 파이프라인 통합**
   ```python
   # SystemOrchestrator에 파일 로더 연동
   from ocad.loaders import CSVLoader
   from ocad.system.orchestrator import SystemOrchestrator

   orchestrator = SystemOrchestrator()
   loader = CSVLoader()

   # 파일에서 데이터 로드 → 파이프라인 처리
   result = loader.load("data/input/daily_metrics.csv")
   for metric in result.metrics:
       orchestrator.process_metric(metric)
   ```

2. **배치 처리 스케줄러**
   ```bash
   # 일일 배치 처리 (cron)
   0 1 * * * /path/to/process_daily_files.sh
   ```

3. **자동 파일 감시**
   ```python
   # watchdog로 파일 자동 감지
   from watchdog.observers import Observer
   from watchdog.events import FileSystemEventHandler

   class MetricFileHandler(FileSystemEventHandler):
       def on_created(self, event):
           if event.src_path.endswith('.csv'):
               process_file(event.src_path)
   ```

### 중기 (1개월)

1. **REST API 활성화** (필요 시)
   ```bash
   # API 서버 실행
   python -m ocad.api.main

   # http://localhost:8080/api/docs
   ```

2. **대시보드 UI 개선**
   - 파일 업로드 UI
   - 실시간 처리 진행 상황
   - 이상 탐지 결과 시각화

3. **성능 최적화**
   - 대용량 파일 청크 처리
   - 병렬 처리
   - 메모리 효율화

### 장기 (3개월+)

1. **실시간 스트리밍** (CFM에서 지원 시)
   - Kafka 통합
   - WebSocket 실시간 푸시

2. **모델 레지스트리**
   - 모델 버전 관리
   - A/B 테스팅

3. **대규모 확장**
   - Kubernetes 배포
   - 분산 처리

---

## 참고 문서

### 필수 문서
- [README.md](../README.md) - 전체 시스템 개요
- [CLAUDE.md](../CLAUDE.md) - 개발 가이드

### 파일 기반 입력
- [File-Based-Input-Implementation-Summary.md](File-Based-Input-Implementation-Summary.md) - 구현 완료 요약
- [File-Based-Data-Input-Plan.md](File-Based-Data-Input-Plan.md) - 구현 계획
- [CFM-Data-Requirements.md](CFM-Data-Requirements.md) - CFM 요구사항 문서

### 리팩토링
- [Refactoring-Plan.md](Refactoring-Plan.md) - 전체 리팩토링 계획
- [Refactoring-Summary.md](Refactoring-Summary.md) - 리팩토링 요약
- [Data-Interface-Specification.md](Data-Interface-Specification.md) - 데이터 인터페이스 명세

### 학습 및 모델
- [Training-Inference-Separation-Design.md](Training-Inference-Separation-Design.md) - 학습-추론 분리 설계
- [Model-Training-Guide.md](Model-Training-Guide.md) - 모델 학습 가이드

---

## 문의 및 지원

문제가 발생하거나 질문이 있으면:
1. [GitHub Issues](https://github.com/your-org/OCAD/issues)
2. 개발팀 이메일
3. Slack 채널

---

**작성자**: Claude Code
**최종 업데이트**: 2025-10-23
**버전**: 1.0.0
