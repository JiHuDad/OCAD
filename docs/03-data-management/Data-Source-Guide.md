# 데이터 소스 가이드

**목적**: OCAD에 데이터를 제공하는 모든 방법을 설명하는 통합 가이드

**대상**: CFM 담당자, 데이터 제공자, 시스템 운영자

---

## 📊 데이터 입력 방식 개요

OCAD는 두 가지 데이터 입력 방식을 지원합니다:

```
┌──────────────────────────────────────────────────┐
│            OCAD 데이터 입력                        │
├──────────────────────────────────────────────────┤
│                                                  │
│  ✅ 1. 파일 기반 (현재 지원)                      │
│     - CSV, Excel, Parquet                       │
│     - 사람이 읽기 쉬운 형식                       │
│     - 학습/추론 모두 사용                         │
│                                                  │
│  🔄 2. 실시간 스트리밍 (향후 지원)                │
│     - Kafka, WebSocket                          │
│     - NETCONF/YANG 실시간 수집                   │
│     - 추론 전용                                  │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## ✅ 방법 1: 파일 기반 입력 (현재)

### 지원 형식

| 형식 | 확장자 | 권장 용도 | 라이브러리 |
|------|--------|-----------|------------|
| CSV | `.csv` | 소규모 데이터 (<10MB) | pandas |
| Excel | `.xlsx` | 사람이 편집 | openpyxl |
| Parquet | `.parquet` | 대용량 데이터 (>10MB) | pyarrow |

### CSV 파일 예제

```csv
timestamp,endpoint_id,udp_echo_rtt_ms,ecpri_delay_us,lbm_rtt_ms,lbm_success,ccm_interval_ms,ccm_miss_count,label
1760529600000,o-ru-test-001,4.76,93.93,7.33,1,10,0,normal
1760529610000,o-ru-test-001,4.61,110.28,6.3,1,10,0,normal
1760529620000,o-ru-test-001,4.87,92.53,7.14,1,10,0,normal
```

**필수 컬럼**:
- `timestamp` - Unix timestamp (밀리초)
- `endpoint_id` - 엔드포인트 ID
- `udp_echo_rtt_ms` - UDP Echo RTT (밀리초)
- `ecpri_delay_us` - eCPRI delay (마이크로초)
- `lbm_rtt_ms` - LBM RTT (밀리초)

**선택 컬럼**:
- `label` - 라벨 (normal/anomaly) - 테스트용
- `scenario` - 시나리오 이름 - 테스트용
- `site_name` - 사이트 이름
- `zone` - 존 정보

### Excel 파일 예제

Excel 파일도 동일한 구조이지만, 여러 시트를 사용할 수 있습니다:

**Sheet 1: Metrics**
| timestamp | endpoint_id | udp_echo_rtt_ms | ecpri_delay_us | lbm_rtt_ms |
|-----------|-------------|-----------------|----------------|------------|
| 1760529600000 | o-ru-test-001 | 4.76 | 93.93 | 7.33 |

**Sheet 2: Metadata** (선택 사항)
| key | value |
|-----|-------|
| collection_date | 2025-10-27 |
| site | Seoul-01 |

### Parquet 파일 (대용량)

Parquet는 컬럼 기반 압축 형식으로 대용량 데이터에 적합합니다:

```bash
# CSV → Parquet 변환
python3 -c "
import pandas as pd
df = pd.read_csv('data/metrics.csv')
df.to_parquet('data/metrics.parquet', compression='snappy')
"

# 파일 크기 비교
ls -lh data/metrics.csv      # 10 MB
ls -lh data/metrics.parquet  # 2 MB (80% 압축)
```

---

## 📝 데이터 형식 상세

### Wide 형식 (권장)

**한 행에 모든 메트릭 포함**:

```csv
timestamp,endpoint_id,udp_echo_rtt_ms,ecpri_delay_us,lbm_rtt_ms
1760529600000,o-ru-001,5.2,100.5,7.1
1760529610000,o-ru-001,5.3,102.3,7.2
```

**장점**:
- 사람이 읽기 쉬움
- Excel에서 편집 용이
- 시계열 분석에 직관적

### Long 형식 (지원)

**한 행에 하나의 메트릭**:

```csv
timestamp,endpoint_id,metric_name,metric_value
1760529600000,o-ru-001,udp_echo_rtt_ms,5.2
1760529600000,o-ru-001,ecpri_delay_us,100.5
1760529600000,o-ru-001,lbm_rtt_ms,7.1
```

**장점**:
- 메트릭 추가/제거 용이
- 데이터베이스 친화적

**자동 감지**: OCAD는 형식을 자동으로 감지하여 처리합니다.

---

## 🚀 파일 기반 입력 사용법

### Step 1: 데이터 생성

#### Option A: 테스트 데이터 생성

```bash
# 학습용 (정상 데이터만)
python scripts/generate_training_inference_data.py --mode training
# 출력: data/training_normal_only.csv (28,800개)

# 추론 테스트용 (정상 + 이상)
python scripts/generate_training_inference_data.py --mode inference
# 출력: data/inference_test_scenarios.csv (780개)
```

#### Option B: 실제 데이터 수집

CFM 담당자가 ORAN 장비에서 수집:

1. **메트릭 수집** (NETCONF/YANG)
   ```bash
   # UDP Echo RTT
   # eCPRI delay
   # LBM RTT, success rate
   # CCM interval, miss count
   ```

2. **CSV 파일로 저장**
   ```bash
   # 파일명: metrics_YYYYMMDD.csv
   # 예: metrics_20251027.csv
   ```

3. **파일 전달**
   ```bash
   cp metrics_20251027.csv /path/to/ocad/data/
   ```

**상세 요구사항**: [CFM-Data-Requirements.md](CFM-Data-Requirements.md)

### Step 2: 데이터 검증

```bash
# Python으로 데이터 검증
python3 -c "
import pandas as pd

df = pd.read_csv('data/metrics.csv')

# 1. 컬럼 확인
required_cols = ['timestamp', 'endpoint_id', 'udp_echo_rtt_ms', 'ecpri_delay_us', 'lbm_rtt_ms']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f'누락된 컬럼: {missing_cols}')
else:
    print('✅ 모든 필수 컬럼 존재')

# 2. 데이터 타입 확인
print(df.dtypes)

# 3. 결측치 확인
print(df.isnull().sum())

# 4. 통계 요약
print(df.describe())
"
```

### Step 3: 학습 또는 추론 실행

```bash
# 학습
python scripts/train_model.py \
    --data-source data/training_normal_only.csv

# 추론
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --output data/results.csv
```

---

## 🔄 방법 2: 실시간 스트리밍 (향후)

### Kafka 통합 (예정)

```bash
# Kafka에서 실시간 추론
python scripts/run_inference.py \
    --streaming \
    --kafka-broker localhost:9092 \
    --kafka-topic oran-metrics \
    --kafka-group ocad-inference \
    --batch-size 100
```

**Kafka 메시지 형식**:
```json
{
  "timestamp": 1760529600000,
  "endpoint_id": "o-ru-test-001",
  "metrics": {
    "udp_echo_rtt_ms": 5.2,
    "ecpri_delay_us": 100.5,
    "lbm_rtt_ms": 7.1
  }
}
```

### WebSocket 통합 (예정)

```bash
# WebSocket에서 실시간 추론
python scripts/run_inference.py \
    --streaming \
    --websocket-url ws://oran-server:8080/metrics \
    --batch-size 100
```

### NETCONF/YANG 직접 수집 (기본)

```yaml
# config/local.yaml
endpoints:
  - id: o-ru-001
    host: 192.168.1.100
    port: 830
    username: admin
    capabilities:
      udp_echo: true
      ecpri_delay: true
      lbm: true
```

```bash
# 시스템 실행 (자동 수집)
python -m ocad.main
```

---

## 🎨 데이터 소스 추상화

### 통일된 인터페이스

OCAD는 데이터 소스에 관계없이 동일한 방식으로 처리합니다:

```python
from ocad.core.data_source import DataSourceFactory

# 파일 기반
config = {
    "type": "file",
    "file_path": "data/metrics.csv",
    "batch_size": 100
}
data_source = DataSourceFactory.create_from_config(config)

# 스트리밍 (향후)
config = {
    "type": "kafka",
    "kafka": {
        "bootstrap_servers": "localhost:9092",
        "topic": "oran-metrics"
    },
    "batch_size": 100
}
data_source = DataSourceFactory.create_from_config(config)

# 사용 방법은 동일
for batch in data_source:
    for metric in batch.metrics:
        process(metric)
```

**상세 설계**: [Data-Source-Abstraction-Design.md](../05-architecture/Data-Source-Abstraction-Design.md)

---

## 📋 데이터 품질 체크리스트

### ✅ 필수 사항

- [ ] 모든 필수 컬럼 존재
- [ ] timestamp가 Unix timestamp (밀리초)
- [ ] 결측치 없음 (또는 < 1%)
- [ ] 데이터 타입 올바름 (숫자는 float/int)
- [ ] endpoint_id 일관성 유지

### ⚠️ 권장 사항

- [ ] 최소 1시간 이상의 데이터
- [ ] 10초 이하 간격 (실시간성)
- [ ] 여러 엔드포인트 포함 (학습 데이터)
- [ ] 정상 데이터만 포함 (학습용)
- [ ] 정상 + 이상 혼합 (추론 테스트용)

### 🔍 검증 스크립트

```bash
# 데이터 품질 체크
python scripts/validate_data.py data/metrics.csv
```

**예상 출력**:
```
======================================================================
데이터 품질 검증
======================================================================
파일: data/metrics.csv

✅ 필수 컬럼: 모두 존재
✅ 데이터 타입: 올바름
✅ 결측치: 없음
⚠️  시간 간격: 평균 15초 (권장: 10초 이하)
✅ 엔드포인트: 8개

총 레코드: 28,800개
시작 시간: 2025-10-27 12:00:00
종료 시간: 2025-10-27 13:00:00
지속 시간: 1시간

✅ 데이터 품질: 양호
======================================================================
```

---

## 🔧 문제 해결

### 문제 1: "파일을 찾을 수 없음"

**오류**:
```
FileNotFoundError: data/metrics.csv
```

**해결**:
```bash
# 1. 파일 경로 확인
ls -l data/metrics.csv

# 2. 절대 경로 사용
python scripts/run_inference.py \
    --data-source /home/finux/dev/OCAD/data/metrics.csv
```

### 문제 2: "필수 컬럼 누락"

**오류**:
```
KeyError: 'udp_echo_rtt_ms'
```

**해결**:
```python
# 컬럼명 확인
import pandas as pd
df = pd.read_csv('data/metrics.csv')
print(df.columns.tolist())

# 컬럼명 수정
df = df.rename(columns={'udp_rtt': 'udp_echo_rtt_ms'})
df.to_csv('data/metrics_fixed.csv', index=False)
```

### 문제 3: "데이터 타입 오류"

**오류**:
```
TypeError: float() argument must be a string or a number, not 'NoneType'
```

**해결**:
```python
# 결측치 처리
import pandas as pd
df = pd.read_csv('data/metrics.csv')

# 결측치 확인
print(df.isnull().sum())

# 결측치 제거
df = df.dropna()
df.to_csv('data/metrics_clean.csv', index=False)
```

### 문제 4: "메모리 부족"

**오류**:
```
MemoryError: Unable to allocate array
```

**해결**:
```bash
# Option 1: Parquet 변환 (압축)
python3 -c "
import pandas as pd
df = pd.read_csv('data/large_metrics.csv')
df.to_parquet('data/large_metrics.parquet')
"

# Option 2: 파일 분할
split -l 10000 data/large_metrics.csv data/chunk_

# Option 3: 배치 크기 줄이기
python scripts/run_inference.py \
    --data-source data/metrics.csv \
    --batch-size 50
```

---

## 📚 관련 문서

### CFM 담당자
- [CFM-Data-Requirements.md](CFM-Data-Requirements.md) - 수집해야 할 메트릭
- [Data-Format-Specification.md](Data-Format-Specification.md) - 형식 명세

### 사용자
- [Training-Guide.md](../04-training-inference/Training-Guide.md) - 학습 가이드
- [Inference-Guide.md](../04-training-inference/Inference-Guide.md) - 추론 가이드
- [Training-Inference-Workflow.md](../02-user-guides/Training-Inference-Workflow.md) - 전체 워크플로우

### 개발자
- [Data-Source-Abstraction-Design.md](../05-architecture/Data-Source-Abstraction-Design.md) - 아키텍처 상세

---

## ❓ FAQ

### Q1: CSV와 Parquet 중 어느 것을 사용해야 하나요?

**A**:
- **CSV**: 소규모 (<10MB), 사람이 편집 필요
- **Parquet**: 대규모 (>10MB), 압축 효율 중요

### Q2: Excel 파일에 여러 시트가 있으면?

**A**: 첫 번째 시트만 읽습니다. 특정 시트 지정은 향후 추가 예정.

### Q3: 실시간 수집과 파일 입력 중 어느 것이 좋나요?

**A**:
- **학습**: 파일 입력 권장 (재현 가능)
- **추론**: 실시간 수집 권장 (지연 최소화)

### Q4: 데이터 형식을 변환하려면?

**A**:
```bash
# CSV → Excel
python3 -c "
import pandas as pd
df = pd.read_csv('data/metrics.csv')
df.to_excel('data/metrics.xlsx', index=False)
"

# Excel → CSV
python3 -c "
import pandas as pd
df = pd.read_excel('data/metrics.xlsx')
df.to_csv('data/metrics.csv', index=False)
"

# CSV → Parquet
python3 -c "
import pandas as pd
df = pd.read_csv('data/metrics.csv')
df.to_parquet('data/metrics.parquet')
"
```

### Q5: 여러 파일을 합치려면?

**A**:
```bash
# CSV 파일 합치기
python3 -c "
import pandas as pd
import glob

files = glob.glob('data/metrics_*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values('timestamp')
df.to_csv('data/metrics_merged.csv', index=False)
"
```

---

**작성자**: Claude Code
**최종 업데이트**: 2025-10-27
**버전**: 1.0.0
