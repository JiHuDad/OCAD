# 내일 작업 Quick Start Guide

**날짜**: 2025-10-29
**목표**: TCN, Isolation Forest 모델 학습 완료

---

## ⚡ 빠른 시작

### 1단계: 시계열 데이터 준비 (필수!)

현재 데이터는 TCN 학습에 필요한 형식이 아닙니다. 먼저 변환이 필요합니다.

#### 필요한 데이터 형식

```
필요한 컬럼:
- timestamp: 시간
- endpoint_id: 엔드포인트 ID
- metric_type: 메트릭 타입 (udp_echo, ecpri, lbm)
- sequence: [v1, v2, ..., v10] (리스트, 10개 timestep)
- target: v11 (다음 값 예측)
- is_anomaly: False (정상 데이터만)
```

#### 변환 스크립트 작성

```bash
# scripts/prepare_timeseries_data.py 생성
cat > scripts/prepare_timeseries_data.py << 'EOF'
#!/usr/bin/env python3
"""시계열 학습 데이터 준비 스크립트.

슬라이딩 윈도우로 시퀀스 생성:
- Input: [t1, t2, t3, ..., t100]
- Output:
  - sequence: [t1, t2, ..., t10], target: t11
  - sequence: [t2, t3, ..., t11], target: t12
  - ...
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

SEQUENCE_LENGTH = 10  # 입력 시퀀스 길이

def create_sequences(df, metric_col, sequence_length=10):
    """슬라이딩 윈도우로 시퀀스 생성."""
    sequences = []
    targets = []

    # 엔드포인트별로 그룹화
    for endpoint_id, group in df.groupby('endpoint_id'):
        # 시간순 정렬
        group = group.sort_values('timestamp').reset_index(drop=True)
        values = group[metric_col].values

        # 슬라이딩 윈도우
        for i in range(len(values) - sequence_length):
            seq = values[i:i + sequence_length].tolist()
            target = values[i + sequence_length]

            sequences.append({
                'timestamp': group.iloc[i + sequence_length]['timestamp'],
                'endpoint_id': endpoint_id,
                'sequence': seq,
                'target': float(target),
                'is_anomaly': False,  # 정상 데이터만
            })

    return pd.DataFrame(sequences)

def main():
    print("=" * 70)
    print("시계열 학습 데이터 준비")
    print("=" * 70)

    # 원본 데이터 로드
    input_file = Path("data/training_normal_only.csv")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"\n원본 데이터: {len(df):,}개 레코드")

    # 메트릭별 시퀀스 생성
    metrics_config = [
        ('udp_echo', 'udp_echo_rtt_ms'),
        ('ecpri', 'ecpri_delay_us'),
        ('lbm', 'lbm_rtt_ms'),
    ]

    all_sequences = []

    for metric_type, col_name in metrics_config:
        print(f"\n[{metric_type}] 시퀀스 생성 중...")

        seq_df = create_sequences(df, col_name, SEQUENCE_LENGTH)
        seq_df['metric_type'] = metric_type

        print(f"  - 생성된 시퀀스: {len(seq_df):,}개")
        all_sequences.append(seq_df)

    # 모든 메트릭 결합
    combined_df = pd.concat(all_sequences, ignore_index=True)
    print(f"\n전체 시퀀스: {len(combined_df):,}개")

    # 학습/검증/테스트 분할 (80/10/10)
    train_size = int(len(combined_df) * 0.8)
    val_size = int(len(combined_df) * 0.1)

    train_df = combined_df[:train_size]
    val_df = combined_df[train_size:train_size + val_size]
    test_df = combined_df[train_size + val_size:]

    # 저장
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    train_df.to_parquet(output_dir / "timeseries_train.parquet", index=False)
    val_df.to_parquet(output_dir / "timeseries_val.parquet", index=False)
    test_df.to_parquet(output_dir / "timeseries_test.parquet", index=False)

    print(f"\n✅ 저장 완료:")
    print(f"  - 학습: {len(train_df):,}개 ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"  - 검증: {len(val_df):,}개 ({len(val_df)/len(combined_df)*100:.1f}%)")
    print(f"  - 테스트: {len(test_df):,}개 ({len(test_df)/len(combined_df)*100:.1f}%)")

    print(f"\n데이터 샘플:")
    print(train_df.head(2))

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/prepare_timeseries_data.py
```

#### 실행

```bash
source .venv/bin/activate
python scripts/prepare_timeseries_data.py
```

**예상 결과**:
- `data/processed/timeseries_train.parquet` (약 70,000개 시퀀스)
- `data/processed/timeseries_val.parquet` (약 8,700개 시퀀스)
- `data/processed/timeseries_test.parquet` (약 8,700개 시퀀스)

---

### 2단계: TCN 모델 학습 (시간 제한 없음)

#### UDP Echo 학습

```bash
source .venv/bin/activate

python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --train-data data/processed/timeseries_train.parquet \
    --val-data data/processed/timeseries_val.parquet \
    --test-data data/processed/timeseries_test.parquet \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --early-stopping \
    --patience 10 \
    --output-dir ocad/models/tcn \
    --version v2.0.0
```

**예상 소요 시간**: 30-60분 (CPU), 5-10분 (GPU)

#### eCPRI 학습

```bash
python scripts/train_tcn_model.py \
    --metric-type ecpri \
    --epochs 50 \
    --batch-size 64 \
    --output-dir ocad/models/tcn \
    --version v2.0.0
```

#### LBM 학습

```bash
python scripts/train_tcn_model.py \
    --metric-type lbm \
    --epochs 50 \
    --batch-size 64 \
    --output-dir ocad/models/tcn \
    --version v2.0.0
```

#### 병렬 실행 (권장)

터미널 3개를 열어서 동시에 실행:

```bash
# Terminal 1
python scripts/train_tcn_model.py --metric-type udp_echo --epochs 50 --version v2.0.0

# Terminal 2
python scripts/train_tcn_model.py --metric-type ecpri --epochs 50 --version v2.0.0

# Terminal 3
python scripts/train_tcn_model.py --metric-type lbm --epochs 50 --version v2.0.0
```

---

### 3단계: Isolation Forest 학습

#### 다변량 데이터 준비

```bash
cat > scripts/prepare_multivariate_data.py << 'EOF'
#!/usr/bin/env python3
"""다변량 이상 탐지 데이터 준비."""

import pandas as pd
from pathlib import Path

def main():
    print("=" * 70)
    print("다변량 학습 데이터 준비")
    print("=" * 70)

    # 원본 데이터 로드
    df = pd.read_csv("data/training_normal_only.csv")

    # 특징 선택
    features = [
        'udp_echo_rtt_ms',
        'ecpri_delay_us',
        'lbm_rtt_ms',
        'ccm_miss_count',
    ]

    # 필요한 컬럼만 선택
    multi_df = df[['timestamp', 'endpoint_id'] + features].copy()
    multi_df['is_anomaly'] = False

    # 학습/테스트 분할 (80/20)
    train_size = int(len(multi_df) * 0.8)
    train_df = multi_df[:train_size]
    test_df = multi_df[train_size:]

    # 저장
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    train_df.to_parquet(output_dir / "multivariate_train.parquet", index=False)
    test_df.to_parquet(output_dir / "multivariate_test.parquet", index=False)

    print(f"\n✅ 저장 완료:")
    print(f"  - 학습: {len(train_df):,}개")
    print(f"  - 테스트: {len(test_df):,}개")
    print(f"\n특징: {features}")
    print("=" * 70)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/prepare_multivariate_data.py
python scripts/prepare_multivariate_data.py
```

#### Isolation Forest 학습 스크립트 확인

```bash
# 스크립트 존재 확인
ls -lh scripts/train_isolation_forest.py

# 없으면 기존 참고하여 작성 필요
# 또는 ocad/training/trainers/ 에 IsolationForestTrainer 사용
```

#### 학습 실행

```bash
python scripts/train_isolation_forest.py \
    --train-data data/processed/multivariate_train.parquet \
    --test-data data/processed/multivariate_test.parquet \
    --n-estimators 100 \
    --contamination 0.1 \
    --output-dir ocad/models/isolation_forest \
    --version v2.0.0
```

---

## 🔍 학습 진행 확인

### 학습 중 모니터링

```bash
# 실시간 로그 확인
tail -f logs/training_*.log

# 모델 파일 생성 확인
watch -n 5 'ls -lh ocad/models/tcn/'

# GPU 사용량 확인 (GPU 있는 경우)
watch -n 1 nvidia-smi
```

### 학습 완료 확인

```bash
# 모델 파일 확인
ls -lh ocad/models/tcn/*.pth
ls -lh ocad/models/isolation_forest/*.pkl

# 메타데이터 확인
cat ocad/models/tcn/udp_echo_v2.0.0.json
```

**기대 결과**:
```
ocad/models/tcn/
├── udp_echo_v2.0.0.pth   (수십 MB)
├── udp_echo_v2.0.0.json
├── ecpri_v2.0.0.pth
├── ecpri_v2.0.0.json
├── lbm_v2.0.0.pth
└── lbm_v2.0.0.json

ocad/models/isolation_forest/
├── multivariate_v2.0.0.pkl
└── multivariate_v2.0.0.json
```

---

## 🧪 학습 완료 후 검증

### 모델 로드 테스트

```python
# test_models.py
import torch
import joblib

# TCN 모델 로드
tcn_model = torch.load('ocad/models/tcn/udp_echo_v2.0.0.pth')
print("✅ TCN 모델 로드 성공")

# Isolation Forest 로드
if_model = joblib.load('ocad/models/isolation_forest/multivariate_v2.0.0.pkl')
print("✅ Isolation Forest 로드 성공")
```

### 추론 테스트

```bash
# 학습된 모델로 추론 (업데이트 필요)
python scripts/inference_with_report.py \
    --data-source data/inference_anomaly_only.csv \
    --model-path ocad/models/tcn
```

---

## ⚠️ 문제 해결

### 문제 1: "KeyError: 'sequence'"

**원인**: 시계열 데이터 형식 불일치

**해결**: 1단계 (prepare_timeseries_data.py) 반드시 실행

### 문제 2: "CUDA out of memory"

**해결**: 배치 크기 줄이기

```bash
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --batch-size 32 \  # 64 → 32로 감소
    --epochs 50
```

### 문제 3: 학습 시간이 너무 오래 걸림

**해결 옵션**:
1. Early stopping 활성화 (기본 활성화됨)
2. Epoch 수 감소 (50 → 20)
3. GPU 사용 (가능한 경우)

```bash
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --epochs 20 \
    --device cuda  # GPU 사용
```

### 문제 4: "FileNotFoundError: timeseries_train.parquet"

**해결**: 1단계 데이터 준비 먼저 실행

```bash
python scripts/prepare_timeseries_data.py
ls -lh data/processed/
```

---

## 📊 성공 기준

### 학습 완료 체크리스트

- [ ] `data/processed/timeseries_train.parquet` 생성됨 (약 70,000개 시퀀스)
- [ ] `ocad/models/tcn/udp_echo_v2.0.0.pth` 생성됨
- [ ] `ocad/models/tcn/ecpri_v2.0.0.pth` 생성됨
- [ ] `ocad/models/tcn/lbm_v2.0.0.pth` 생성됨
- [ ] `ocad/models/isolation_forest/multivariate_v2.0.0.pkl` 생성됨
- [ ] 모델 로드 테스트 성공

### 성능 기준

- **Training Loss**: 점진적 감소
- **Validation Loss**: 학습 중반부터 안정화
- **Test MSE**: < 1.0 (정규화된 데이터 기준)
- **Isolation Forest Score**: 대부분 음수값 (정상 데이터)

---

## 🚀 다음 단계 (학습 완료 후)

1. **학습된 모델 통합**
   ```bash
   # ResidualDetector, MultivariateDetector 업데이트
   # inference_with_report.py 수정
   ```

2. **ONNX 변환**
   ```bash
   python scripts/convert_to_onnx.py \
       --model-path ocad/models/tcn/udp_echo_v2.0.0.pth \
       --output ocad/models/onnx/udp_echo_v2.0.0.onnx
   ```

3. **성능 비교**
   ```bash
   # 룰 기반 vs 학습된 모델 비교
   python scripts/compare_detection_performance.py
   ```

---

## 💡 유용한 명령어

```bash
# 가상환경 활성화
source .venv/bin/activate

# 모든 학습 스크립트 한번에 실행 (백그라운드)
nohup python scripts/train_tcn_model.py --metric-type udp_echo --epochs 50 > logs/train_udp.log 2>&1 &
nohup python scripts/train_tcn_model.py --metric-type ecpri --epochs 50 > logs/train_ecpri.log 2>&1 &
nohup python scripts/train_tcn_model.py --metric-type lbm --epochs 50 > logs/train_lbm.log 2>&1 &

# 백그라운드 작업 확인
jobs
ps aux | grep train_tcn

# 로그 실시간 확인
tail -f logs/train_*.log

# 디스크 공간 확인
df -h
du -sh ocad/models/*
```

---

**작업 시간 예상**: 총 4-8시간 (CPU 기준)
- 데이터 준비: 30분
- TCN 학습 (3개): 1.5-3시간
- Isolation Forest: 30분
- 검증 및 테스트: 1-2시간

**최종 목표**: 오늘 내로 모든 모델 학습 완료 ✅
