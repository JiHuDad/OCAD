# 파일 기반 데이터 입력 설계

## 배경 및 목적

### 왜 파일 기반으로 시작하는가?

1. **요구사항 명확화**
   - CFM 담당자에게 "이런 데이터가 필요합니다"라고 구체적으로 보여줄 수 있음
   - 실제 샘플 파일을 보면서 수집 가능 여부 논의 가능

2. **반복 개선**
   - 수집 불가능한 필드 → 제거
   - 추가 필요 필드 → 추가
   - 파일만 수정하면 되므로 빠른 피드백 가능

3. **검증 및 디버깅**
   - 사람이 직접 파일을 열어 데이터 확인 가능
   - 이상 탐지 결과를 실제 데이터와 비교 가능
   - 문제 발생 시 파일만 있으면 재현 가능

4. **점진적 발전**
   ```
   Phase 1: 파일 입력 (CSV/Excel)
      ↓
   Phase 2: 파일 변환 (Parquet)
      ↓
   Phase 3: 실시간 API (나중에)
   ```

## 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│                   현재 단계 (Phase 1)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 샘플 파일 작성 (Excel/CSV)                              │
│     └─> CFM 담당자에게 전달                                 │
│                                                             │
│  2. CFM 담당자 검토                                         │
│     ├─> 수집 가능 필드: ✅                                  │
│     ├─> 수집 불가 필드: ❌ (제거)                           │
│     └─> 추가 필요 필드: ➕ (추가)                           │
│                                                             │
│  3. 샘플 파일 수정 (피드백 반영)                            │
│     └─> 2번으로 돌아가서 반복                               │
│                                                             │
│  4. 최종 파일 포맷 확정                                     │
│     └─> 이 포맷으로 실제 데이터 수집                        │
│                                                             │
│  5. OCAD에서 파일 읽기                                      │
│     ├─> CSV/Excel 읽기                                      │
│     ├─> Parquet 변환 (옵션)                                 │
│     ├─> 검증 (스키마)                                       │
│     └─> 파이프라인 처리                                     │
│                                                             │
│  6. 결과 확인 및 분석                                       │
│     ├─> 탐지 결과 CSV 출력                                  │
│     ├─> 알람 목록 Excel 출력                                │
│     └─> 사람이 직접 확인                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   미래 단계 (Phase 2+)                      │
├─────────────────────────────────────────────────────────────┤
│  - 자동 파일 수집 (cron job)                                │
│  - REST API 구축                                            │
│  - Kafka 스트리밍                                           │
│  - 실시간 대시보드                                          │
└─────────────────────────────────────────────────────────────┘
```

## 파일 포맷 설계

### 포맷 1: 사람이 읽기 쉬운 형태 (Excel/CSV)

#### 옵션 A: Wide Format (시계열을 열로 펼침)

**특징**: Excel에서 보기 편함, 사람이 이해하기 쉬움

**파일명**: `oran_metrics_wide_YYYYMMDD_HHMMSS.csv`

```csv
timestamp,endpoint_id,site_name,zone,udp_echo_rtt_ms,ecpri_delay_us,lbm_rtt_ms,lbm_success,ccm_interval_ms,ccm_miss_count,port_crc_errors,queue_drops,notes
2025-10-22 09:00:00,o-ru-001,Tower-A,Urban,5.2,102.3,7.1,TRUE,1000,0,0,0,정상
2025-10-22 09:00:01,o-ru-001,Tower-A,Urban,5.4,105.2,7.3,TRUE,1000,0,0,0,정상
2025-10-22 09:00:02,o-ru-001,Tower-A,Urban,25.8,250.1,22.5,FALSE,1000,1,5,10,⚠️ 높은 RTT
2025-10-22 09:00:00,o-ru-002,Tower-B,Rural,4.8,98.5,6.9,TRUE,1000,0,0,0,정상
```

**장점**:
- ✅ Excel에서 한눈에 들어옴
- ✅ 필터/정렬 쉬움
- ✅ 그래프 그리기 쉬움
- ✅ CFM 담당자가 이해하기 쉬움

**단점**:
- ❌ 컬럼이 많아질 수 있음
- ❌ 새 메트릭 추가 시 구조 변경 필요

#### 옵션 B: Long Format (시계열을 행으로 쌓음)

**특징**: 확장성 좋음, 데이터베이스 친화적

**파일명**: `oran_metrics_long_YYYYMMDD_HHMMSS.csv`

```csv
timestamp,endpoint_id,site_name,metric_name,value,unit,status,notes
2025-10-22 09:00:00,o-ru-001,Tower-A,udp_echo_rtt,5.2,ms,OK,
2025-10-22 09:00:00,o-ru-001,Tower-A,ecpri_delay,102.3,us,OK,
2025-10-22 09:00:00,o-ru-001,Tower-A,lbm_rtt,7.1,ms,OK,
2025-10-22 09:00:01,o-ru-001,Tower-A,udp_echo_rtt,5.4,ms,OK,
2025-10-22 09:00:02,o-ru-001,Tower-A,udp_echo_rtt,25.8,ms,WARN,높은 RTT 발생
```

**장점**:
- ✅ 새 메트릭 추가 쉬움 (행만 추가)
- ✅ 확장성 좋음
- ✅ 데이터베이스 저장 쉬움

**단점**:
- ❌ 사람이 보기 어려움 (같은 시간대가 여러 행에 분산)
- ❌ Excel에서 피벗 테이블 필요

#### 권장: Wide Format → Long Format 변환 지원

**제안**:
1. CFM 담당자에게는 **Wide Format** (Excel) 전달
2. OCAD는 **양쪽 다 읽기** 가능
3. 내부적으로 필요하면 **Long Format으로 변환**

### 포맷 2: 기계가 읽기 좋은 형태 (Parquet)

**특징**: 대용량, 빠른 처리, 컬럼 기반 저장

**파일명**: `oran_metrics_YYYYMMDD_HHMMSS.parquet`

```python
# 스키마
schema = pa.schema([
    ('timestamp', pa.timestamp('ms')),
    ('endpoint_id', pa.string()),
    ('site_name', pa.string()),
    ('zone', pa.string()),

    # 메트릭 값들
    ('udp_echo_rtt_ms', pa.float64()),
    ('ecpri_delay_us', pa.float64()),
    ('lbm_rtt_ms', pa.float64()),
    ('lbm_success', pa.bool_()),
    ('ccm_interval_ms', pa.int64()),
    ('ccm_miss_count', pa.int64()),
    ('port_crc_errors', pa.int64()),
    ('queue_drops', pa.int64()),

    # 메타데이터
    ('notes', pa.string()),
])
```

**사용 시점**:
- 데이터가 10,000행 이상일 때
- 반복적인 분석이 필요할 때
- 성능이 중요할 때

## 필수 필드 정의

### 레벨 1: 최소 필수 필드 (없으면 안 됨)

| 필드명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| `timestamp` | datetime | 측정 시각 | 2025-10-22 09:00:00 |
| `endpoint_id` | string | 엔드포인트 ID | o-ru-001 |
| `udp_echo_rtt_ms` | float | UDP Echo RTT | 5.2 |

**최소 데이터셋 예시** (CFM 담당자에게 먼저 요청):
```csv
timestamp,endpoint_id,udp_echo_rtt_ms
2025-10-22 09:00:00,o-ru-001,5.2
2025-10-22 09:00:01,o-ru-001,5.4
2025-10-22 09:00:02,o-ru-001,25.8
```

### 레벨 2: 권장 필드 (있으면 좋음)

| 필드명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| `site_name` | string | 사이트 이름 | Tower-A |
| `zone` | string | 지역 구분 | Urban/Rural |
| `ecpri_delay_us` | float | eCPRI 지연 | 102.3 |
| `lbm_rtt_ms` | float | LBM RTT | 7.1 |
| `lbm_success` | boolean | LBM 성공 여부 | TRUE/FALSE |

### 레벨 3: 선택 필드 (추가 분석용)

| 필드명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| `ccm_interval_ms` | int | CCM 간격 | 1000 |
| `ccm_miss_count` | int | CCM 누락 수 | 0 |
| `port_crc_errors` | int | CRC 에러 수 | 0 |
| `queue_drops` | int | 큐 드롭 수 | 0 |
| `notes` | string | 비고 | "정상", "높은 RTT" |

## CFM 담당자 요청 문서

### 문서 1: 데이터 요구사항 명세서

**제목**: ORAN 이상 탐지 시스템을 위한 CFM 데이터 요구사항

**수신**: CFM 담당자
**발신**: OCAD 개발팀
**일자**: 2025-10-22

---

#### 1. 목적

ORAN 네트워크의 이상 탐지를 위해 CFM-Lite 메트릭 데이터가 필요합니다.

#### 2. 요청 데이터 형식

**파일 형식**: CSV 또는 Excel (.csv, .xlsx)
**제공 주기**: [일일 / 시간별 / 실시간] ← CFM 담당자와 협의
**파일 크기**: 예상 [X] MB ~ [Y] MB

#### 3. 필수 데이터 필드

**최소 요구사항** (이것만이라도 주세요!):

```csv
timestamp,endpoint_id,udp_echo_rtt_ms
2025-10-22 09:00:00,o-ru-001,5.2
```

| 필드 | 설명 | 단위 | 수집 가능? |
|------|------|------|-----------|
| timestamp | 측정 시각 | YYYY-MM-DD HH:MM:SS | [ ] 가능 [ ] 불가 |
| endpoint_id | 장비 식별자 | 문자열 | [ ] 가능 [ ] 불가 |
| udp_echo_rtt_ms | UDP Echo 왕복 시간 | 밀리초(ms) | [ ] 가능 [ ] 불가 |

#### 4. 권장 데이터 필드

가능하다면 다음 필드도 함께 제공해주세요:

| 필드 | 설명 | 단위 | 수집 가능? |
|------|------|------|-----------|
| ecpri_delay_us | eCPRI 일방향 지연 | 마이크로초(μs) | [ ] 가능 [ ] 불가 |
| lbm_rtt_ms | LBM 왕복 시간 | 밀리초(ms) | [ ] 가능 [ ] 불가 |
| lbm_success | LBM 성공 여부 | TRUE/FALSE | [ ] 가능 [ ] 불가 |
| ccm_interval_ms | CCM 도착 간격 | 밀리초(ms) | [ ] 가능 [ ] 불가 |

#### 5. 선택 데이터 필드

추가 분석을 위해 다음 필드가 있으면 더 좋습니다:

| 필드 | 설명 | 수집 가능? | 대안 |
|------|------|-----------|------|
| site_name | 사이트 이름 | [ ] 가능 [ ] 불가 | endpoint_id로 매핑 가능 |
| zone | 지역 구분 (Urban/Rural) | [ ] 가능 [ ] 불가 | 별도 매핑 파일 제공 |
| port_crc_errors | 포트 CRC 에러 수 | [ ] 가능 [ ] 불가 | 없어도 됨 |
| queue_drops | 큐 드롭 수 | [ ] 가능 [ ] 불가 | 없어도 됨 |

#### 6. 샘플 파일

첨부: `sample_oran_metrics.xlsx`

실제 데이터가 이런 형태로 제공 가능한지 확인 부탁드립니다.

#### 7. 질문 사항

1. **수집 주기**: 메트릭이 몇 초마다 수집되나요?
   - [ ] 1초  [ ] 5초  [ ] 10초  [ ] 기타: _______

2. **데이터 보관**: 과거 데이터는 얼마나 보관하나요?
   - [ ] 1일  [ ] 1주일  [ ] 1개월  [ ] 기타: _______

3. **장비 수**: 모니터링 대상 O-RU/O-DU는 몇 대인가요?
   - 답변: _______

4. **추가 메트릭**: 위 목록 외에 제공 가능한 메트릭이 있나요?
   - 답변: _______

5. **제약 사항**: 데이터 제공 시 제약사항이 있나요? (보안, 개인정보 등)
   - 답변: _______

---

### 문서 2: 샘플 데이터 파일

**파일명**: `sample_oran_metrics.xlsx`

**Sheet 1: 메트릭 데이터**

| timestamp | endpoint_id | site_name | zone | udp_echo_rtt_ms | ecpri_delay_us | lbm_rtt_ms | lbm_success | notes |
|-----------|-------------|-----------|------|-----------------|----------------|------------|-------------|-------|
| 2025-10-22 09:00:00 | o-ru-001 | Tower-A | Urban | 5.2 | 102.3 | 7.1 | TRUE | 정상 |
| 2025-10-22 09:00:01 | o-ru-001 | Tower-A | Urban | 5.4 | 105.2 | 7.3 | TRUE | 정상 |
| 2025-10-22 09:00:02 | o-ru-001 | Tower-A | Urban | 25.8 | 250.1 | 22.5 | FALSE | ⚠️ 높은 RTT |
| 2025-10-22 09:00:03 | o-ru-001 | Tower-A | Urban | 26.2 | 255.3 | 23.1 | FALSE | ⚠️ 높은 RTT |
| 2025-10-22 09:00:04 | o-ru-001 | Tower-A | Urban | 5.1 | 100.8 | 7.0 | TRUE | 정상 복구 |

**Sheet 2: 필드 설명**

| 필드명 | 설명 | 단위 | 필수 여부 |
|--------|------|------|----------|
| timestamp | 측정 시각 | YYYY-MM-DD HH:MM:SS | 필수 |
| endpoint_id | 엔드포인트 식별자 | 문자열 (예: o-ru-001) | 필수 |
| site_name | 사이트 이름 | 문자열 | 권장 |
| zone | 지역 구분 | Urban/Rural/Suburban | 선택 |
| udp_echo_rtt_ms | UDP Echo 왕복 시간 | 밀리초 (ms) | 필수 |
| ecpri_delay_us | eCPRI 일방향 지연 | 마이크로초 (μs) | 권장 |
| lbm_rtt_ms | LBM 왕복 시간 | 밀리초 (ms) | 권장 |
| lbm_success | LBM 성공 여부 | TRUE/FALSE | 권장 |
| notes | 비고 | 자유 텍스트 | 선택 |

**Sheet 3: 예상 이상 케이스**

| 이상 유형 | 특징 | 예시 |
|----------|------|------|
| Spike (급증) | RTT가 갑자기 5배 이상 증가 | 5ms → 25ms |
| Drift (점진 증가) | RTT가 서서히 증가 | 5ms → 10ms → 15ms |
| Loss (패킷 손실) | LBM 실패가 연속 발생 | FALSE, FALSE, FALSE |
| Jitter (흔들림) | RTT가 불규칙하게 변동 | 5ms → 15ms → 6ms → 16ms |

## 파일 입력 파이프라인 구현

### 1. 파일 읽기

```python
# ocad/data/loaders/file_loader.py

class FileMetricLoader:
    """파일에서 메트릭을 읽는 로더."""

    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """CSV 파일 읽기."""
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        return self._validate_and_normalize(df)

    def load_excel(self, file_path: Path, sheet_name: str = 0) -> pd.DataFrame:
        """Excel 파일 읽기."""
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return self._validate_and_normalize(df)

    def load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Parquet 파일 읽기."""
        df = pd.read_parquet(file_path)
        return self._validate_and_normalize(df)

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 검증 및 정규화."""
        # 1. 필수 컬럼 확인
        required = ['timestamp', 'endpoint_id', 'udp_echo_rtt_ms']
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # 2. 타임스탬프 변환
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 3. Wide → Long 변환 (필요 시)
        # ...

        return df
```

### 2. 변환 유틸리티

```python
# ocad/data/converters/format_converter.py

class FormatConverter:
    """파일 포맷 변환기."""

    @staticmethod
    def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
        """Wide format → Long format 변환."""
        id_vars = ['timestamp', 'endpoint_id', 'site_name', 'zone']
        value_vars = ['udp_echo_rtt_ms', 'ecpri_delay_us', 'lbm_rtt_ms']

        long_df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='metric_name',
            value_name='value'
        )
        return long_df

    @staticmethod
    def csv_to_parquet(csv_path: Path, parquet_path: Path):
        """CSV → Parquet 변환."""
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
```

### 3. CLI 명령

```bash
# CSV 파일을 OCAD로 처리
ocad process-file --input data.csv --format csv

# Excel 파일 처리
ocad process-file --input data.xlsx --format excel --sheet "메트릭 데이터"

# Parquet로 변환
ocad convert --input data.csv --output data.parquet --format parquet

# 파일 검증만 수행
ocad validate-file --input data.csv
```

## 작업 단계

### Step 1: 샘플 파일 생성 및 검토 (현재)

**목표**: CFM 담당자에게 보여줄 샘플 파일 작성

**산출물**:
- [ ] `sample_oran_metrics.xlsx` (Wide format)
- [ ] `data_requirements.md` (요구사항 문서)
- [ ] `field_descriptions.xlsx` (필드 설명서)

**담당자**: OCAD 개발팀

### Step 2: CFM 담당자 협의 (다음 단계)

**목표**: 수집 가능 여부 확인 및 피드백 수집

**질문 목록**:
1. 최소 필수 필드(timestamp, endpoint_id, udp_echo_rtt_ms)는 수집 가능한가?
2. 권장 필드는 몇 개나 수집 가능한가?
3. 수집 주기는? (1초 / 5초 / 10초?)
4. 데이터 형식은? (CSV / Excel / DB export?)
5. 제공 방법은? (파일 전달 / SFTP / API?)

**산출물**:
- [ ] 회의록 (수집 가능 필드 목록)
- [ ] 최종 파일 포맷 (v1.0)

### Step 3: 파일 로더 구현 (Step 2 완료 후)

**목표**: 확정된 포맷의 파일을 읽는 코드 구현

**구현 내용**:
- [ ] CSV 로더
- [ ] Excel 로더
- [ ] Parquet 변환기
- [ ] 데이터 검증기

### Step 4: 파이프라인 통합 (Step 3 완료 후)

**목표**: 파일 → OCAD 파이프라인 연결

**구현 내용**:
- [ ] 파일 읽기 → 피처 추출
- [ ] 피처 추출 → 이상 탐지
- [ ] 이상 탐지 → 알람 생성
- [ ] 결과 파일 출력 (CSV/Excel)

### Step 5: 검증 및 피드백 (Step 4 완료 후)

**목표**: 실제 데이터로 테스트

**테스트 시나리오**:
- [ ] 정상 데이터만 있는 파일
- [ ] 이상 데이터가 섞인 파일
- [ ] 대용량 파일 (10,000+ 행)
- [ ] 누락된 필드가 있는 파일

## 다음 단계 제안

지금 바로 할 수 있는 것:

1. **샘플 Excel 파일 생성** - CFM 담당자에게 보여줄 파일
2. **요구사항 문서 작성** - 어떤 데이터가 필요한지 명시
3. **간단한 파일 로더 구현** - CSV/Excel 읽기

이 중 어떤 것부터 시작할까요?

---

**작성 일시**: 2025-10-22
**작성자**: Claude Code
**상태**: 제안 (승인 대기)
