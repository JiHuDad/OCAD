# Phase 1: TCN 모델 1개 학습 (UDP Echo)

**목표**: 가장 중요한 UDP Echo 메트릭에 대한 TCN 모델 학습 완료
**예상 시간**: 2-3시간
**난이도**: ⭐⭐☆☆☆

---

## 📋 Phase 1 범위

### ✅ 이번에 할 일
1. 시계열 데이터 준비 (1시간)
2. UDP Echo TCN 모델 학습 (1-2시간)
3. 학습 결과 검증 (30분)

### ⏭️ 다음 Phase에서 할 일
- Phase 2: eCPRI, LBM 모델 학습
- Phase 3: Isolation Forest 학습
- Phase 4: ONNX 변환

---

## 🚀 단계별 가이드

### Step 1: 환경 준비 (5분)

```bash
cd /home/finux/dev/OCAD
source .venv/bin/activate

# 현재 상태 확인
echo "✅ 가상환경 활성화 완료"
python --version
ls -lh data/training_normal_only.csv
```

---

### Step 2: 시계열 데이터 준비 스크립트 작성 (30분)

TCN 학습에 필요한 **슬라이딩 윈도우 시퀀스 데이터**를 생성합니다.

#### 스크립트 생성

```bash
cat > scripts/prepare_timeseries_data.py << 'SCRIPT_END'
#!/usr/bin/env python3
"""시계열 학습 데이터 준비 (Phase 1: UDP Echo만).

슬라이딩 윈도우로 시퀀스 생성:
- 입력: 10개 연속 값 [t1, t2, ..., t10]
- 출력: 다음 값 t11 예측
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 설정
SEQUENCE_LENGTH = 10  # 10개 timestep으로 다음 값 예측
METRIC_TYPE = 'udp_echo'
METRIC_COLUMN = 'udp_echo_rtt_ms'

def create_sequences(df, metric_col, sequence_length=10):
    """슬라이딩 윈도우로 시퀀스 생성."""
    sequences = []

    # 엔드포인트별로 처리
    print(f"\n엔드포인트별 시퀀스 생성 중...")
    for endpoint_id, group in tqdm(df.groupby('endpoint_id')):
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
                'metric_type': METRIC_TYPE,
                'sequence': seq,
                'target': float(target),
                'is_anomaly': False,  # 정상 데이터만 학습
            })

    return pd.DataFrame(sequences)


def main():
    print("=" * 70)
    print("Phase 1: UDP Echo 시계열 데이터 준비")
    print("=" * 70)

    # 1. 원본 데이터 로드
    print("\n[1/4] 원본 데이터 로드 중...")
    input_file = Path("data/training_normal_only.csv")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  ✅ 로드 완료: {len(df):,}개 레코드")
    print(f"  - 엔드포인트 수: {df['endpoint_id'].nunique()}개")
    print(f"  - 시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # 2. UDP Echo 시퀀스 생성
    print(f"\n[2/4] {METRIC_TYPE} 시퀀스 생성 중...")
    print(f"  - 시퀀스 길이: {SEQUENCE_LENGTH}")
    print(f"  - 메트릭: {METRIC_COLUMN}")

    seq_df = create_sequences(df, METRIC_COLUMN, SEQUENCE_LENGTH)
    print(f"  ✅ 생성 완료: {len(seq_df):,}개 시퀀스")

    # 3. 학습/검증/테스트 분할 (80/10/10)
    print(f"\n[3/4] 데이터 분할 중...")
    train_size = int(len(seq_df) * 0.8)
    val_size = int(len(seq_df) * 0.1)

    train_df = seq_df[:train_size].copy()
    val_df = seq_df[train_size:train_size + val_size].copy()
    test_df = seq_df[train_size + val_size:].copy()

    print(f"  ✅ 분할 완료:")
    print(f"    - 학습: {len(train_df):,}개 ({len(train_df)/len(seq_df)*100:.1f}%)")
    print(f"    - 검증: {len(val_df):,}개 ({len(val_df)/len(seq_df)*100:.1f}%)")
    print(f"    - 테스트: {len(test_df):,}개 ({len(test_df)/len(seq_df)*100:.1f}%)")

    # 4. Parquet 저장
    print(f"\n[4/4] 저장 중...")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    train_df.to_parquet(output_dir / "timeseries_train.parquet", index=False)
    val_df.to_parquet(output_dir / "timeseries_val.parquet", index=False)
    test_df.to_parquet(output_dir / "timeseries_test.parquet", index=False)

    train_size_mb = (output_dir / "timeseries_train.parquet").stat().st_size / 1024 / 1024
    print(f"  ✅ 저장 완료:")
    print(f"    - data/processed/timeseries_train.parquet ({train_size_mb:.2f} MB)")
    print(f"    - data/processed/timeseries_val.parquet")
    print(f"    - data/processed/timeseries_test.parquet")

    # 5. 데이터 샘플 출력
    print(f"\n[샘플 데이터]")
    print(train_df.head(2))

    print("\n" + "=" * 70)
    print("✅ Phase 1 데이터 준비 완료!")
    print("=" * 70)
    print("\n다음 단계: UDP Echo TCN 모델 학습")
    print("  python scripts/train_tcn_model.py --metric-type udp_echo --epochs 30")

if __name__ == "__main__":
    main()
SCRIPT_END

chmod +x scripts/prepare_timeseries_data.py
```

#### 실행

```bash
# tqdm 설치 (진행률 표시용)
pip install tqdm

# 데이터 준비 실행
python scripts/prepare_timeseries_data.py
```

**예상 출력**:
```
======================================================================
Phase 1: UDP Echo 시계열 데이터 준비
======================================================================

[1/4] 원본 데이터 로드 중...
  ✅ 로드 완료: 28,800개 레코드
  - 엔드포인트 수: 5개
  - 시간 범위: 2025-10-01 00:00:00 ~ 2025-10-02 23:59:30

[2/4] udp_echo 시퀀스 생성 중...
  - 시퀀스 길이: 10
  - 메트릭: udp_echo_rtt_ms

엔드포인트별 시퀀스 생성 중...
100%|████████████████████████████████| 5/5 [00:10<00:00]

  ✅ 생성 완료: 28,750개 시퀀스

[3/4] 데이터 분할 중...
  ✅ 분할 완료:
    - 학습: 23,000개 (80.0%)
    - 검증: 2,875개 (10.0%)
    - 테스트: 2,875개 (10.0%)

[4/4] 저장 중...
  ✅ 저장 완료:
    - data/processed/timeseries_train.parquet (X.XX MB)
```

---

### Step 3: TCN 모델 학습 (1-2시간)

#### 학습 실행 (시간 제한 없음)

```bash
# 30 epochs로 학습 (빠른 테스트용)
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --train-data data/processed/timeseries_train.parquet \
    --val-data data/processed/timeseries_val.parquet \
    --test-data data/processed/timeseries_test.parquet \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --early-stopping \
    --patience 5 \
    --output-dir ocad/models/tcn \
    --version v2.0.0

# 또는 더 긴 학습 (더 좋은 성능)
# --epochs 50 --patience 10
```

#### 학습 중 모니터링 (다른 터미널)

```bash
# 터미널 2 (로그 확인)
tail -f logs/training_*.log

# 터미널 3 (모델 파일 확인)
watch -n 5 'ls -lh ocad/models/tcn/'

# CPU 사용률 확인
htop
```

**예상 출력** (학습 진행):
```
2025-10-29 09:00:00 [info] TCN 모델 학습 시작    metric_type=udp_echo version=2.0.0

Epoch 1/30
Train Loss: 0.1234, Val Loss: 0.0987
⏱️ 소요 시간: 1분 30초

Epoch 2/30
Train Loss: 0.0856, Val Loss: 0.0743
⏱️ 소요 시간: 1분 28초

...

Epoch 25/30
Train Loss: 0.0123, Val Loss: 0.0156
⏱️ 소요 시간: 1분 25초

Early stopping triggered! (patience=5)
Best epoch: 20 (Val Loss: 0.0145)

✅ 학습 완료!
   - 모델: ocad/models/tcn/udp_echo_v2.0.0.pth
   - 메타데이터: ocad/models/tcn/udp_echo_v2.0.0.json
   - 총 소요 시간: 38분
```

---

### Step 4: 학습 결과 검증 (30분)

#### 4.1 모델 파일 확인

```bash
ls -lh ocad/models/tcn/

# 예상 출력:
# udp_echo_v2.0.0.pth       (20-50 MB)
# udp_echo_v2.0.0.json      (1 KB)
```

#### 4.2 메타데이터 확인

```bash
cat ocad/models/tcn/udp_echo_v2.0.0.json
```

**예상 내용**:
```json
{
  "model_type": "tcn",
  "metric_type": "udp_echo",
  "version": "v2.0.0",
  "training_date": "2025-10-29",
  "best_epoch": 20,
  "best_val_loss": 0.0145,
  "test_mse": 0.0158,
  "sequence_length": 10,
  "hidden_size": 32,
  "num_epochs": 30,
  "batch_size": 64
}
```

#### 4.3 모델 로드 테스트

```python
# test_udp_echo_model.py
import torch
from pathlib import Path

print("TCN 모델 로드 테스트...")

model_path = Path("ocad/models/tcn/udp_echo_v2.0.0.pth")
model = torch.load(model_path, map_location='cpu')

print(f"✅ 모델 로드 성공!")
print(f"   - 파일 크기: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
print(f"   - 모델 타입: {type(model)}")

# 간단한 추론 테스트
import numpy as np
model.eval()

# 더미 입력 (batch_size=1, seq_len=10, features=1)
dummy_input = torch.randn(1, 10, 1)
with torch.no_grad():
    output = model(dummy_input)

print(f"✅ 추론 테스트 성공!")
print(f"   - 입력 shape: {dummy_input.shape}")
print(f"   - 출력 shape: {output.shape}")
print(f"   - 출력 값: {output.item():.4f}")
```

```bash
python test_udp_echo_model.py
```

---

## 🎯 Phase 1 완료 체크리스트

완료된 항목에 체크하세요:

- [ ] 가상환경 활성화 완료
- [ ] `scripts/prepare_timeseries_data.py` 작성 완료
- [ ] 시계열 데이터 생성 완료 (`data/processed/timeseries_train.parquet`)
- [ ] TCN 모델 학습 실행 완료
- [ ] 모델 파일 생성 확인 (`ocad/models/tcn/udp_echo_v2.0.0.pth`)
- [ ] 메타데이터 확인 (`udp_echo_v2.0.0.json`)
- [ ] 모델 로드 테스트 성공

---

## ⚠️ 문제 해결

### 문제 1: "KeyError: 'sequence'" 또는 "KeyError: 'target'"

**원인**: 데이터 준비 스크립트가 실행되지 않았거나 실패함

**해결**:
```bash
# 데이터 확인
ls -lh data/processed/
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/timeseries_train.parquet')
print('컬럼:', df.columns.tolist())
print('샘플:', df.head(1))
"

# 필요 시 재생성
python scripts/prepare_timeseries_data.py
```

### 문제 2: 학습이 너무 느림

**해결 1**: Epoch 수 줄이기
```bash
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --epochs 10 \
    --batch-size 128  # 배치 크기 증가
```

**해결 2**: CPU 코어 활용
```bash
# num_workers 추가 (데이터 로딩 병렬화)
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --num-workers 4
```

### 문제 3: "CUDA out of memory" (GPU 사용 시)

**해결**:
```bash
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --batch-size 32 \  # 배치 크기 감소
    --device cpu       # CPU로 전환
```

### 문제 4: "ModuleNotFoundError: No module named 'tqdm'"

**해결**:
```bash
pip install tqdm
```

---

## 📊 성공 기준

### 최소 요구사항
- ✅ 모델 파일 생성: `ocad/models/tcn/udp_echo_v2.0.0.pth`
- ✅ Training Loss < 0.1
- ✅ Validation Loss < 0.1
- ✅ 모델 로드 테스트 통과

### 권장 목표
- 🎯 Validation Loss < 0.05
- 🎯 Test MSE < 0.05
- 🎯 Early stopping으로 자동 종료

---

## 🚀 Phase 1 완료 후 다음 단계

### 즉시 가능한 작업

1. **Phase 1 모델로 추론 테스트** (15분)
```bash
# ResidualDetector에 v2.0.0 모델 통합 (코드 수정 필요)
# 이후 추론 실행
python scripts/inference_with_report.py \
    --data-source data/inference_anomaly_only.csv \
    --model-path ocad/models/tcn
```

2. **Phase 2 준비** (내일 또는 시간 남으면)
```bash
# eCPRI, LBM 모델도 학습
python scripts/train_tcn_model.py --metric-type ecpri --epochs 30
python scripts/train_tcn_model.py --metric-type lbm --epochs 30
```

### 전체 로드맵

```
✅ Phase 1 (오늘): UDP Echo TCN 학습
⬜ Phase 2 (내일): eCPRI, LBM TCN 학습
⬜ Phase 3 (모레): Isolation Forest 학습
⬜ Phase 4 (다음주): ONNX 변환 및 배포
```

---

## 💡 유용한 팁

### 백그라운드 실행 (장시간 학습 시)

```bash
# nohup으로 백그라운드 실행
nohup python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --epochs 50 \
    > logs/train_udp_echo.log 2>&1 &

# 작업 ID 확인
echo $!

# 로그 실시간 확인
tail -f logs/train_udp_echo.log

# 작업 종료 (필요 시)
kill -9 <PID>
```

### 학습 중단 후 재개

```bash
# 체크포인트 저장 (구현 필요)
# --resume-from ocad/models/tcn/checkpoint_epoch_15.pth
```

### 하이퍼파라미터 빠른 튜닝

```bash
# 작은 데이터로 빠르게 테스트
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --epochs 5 \
    --batch-size 128 \
    --hidden-size 16  # 작은 모델로 빠른 테스트
```

---

## 📝 Phase 1 체크포인트

작업 진행 상황을 기록하세요:

```
시작 시간: ____:____
데이터 준비 완료: ____:____
학습 시작: ____:____
학습 완료: ____:____
검증 완료: ____:____
총 소요 시간: ____ 시간 ____ 분

모델 파일 크기: ______ MB
Best Val Loss: ______
Test MSE: ______

메모:
-
-
```

---

**예상 총 소요 시간**: 2-3시간
**난이도**: ⭐⭐☆☆☆ (초급-중급)

화이팅! 🚀
