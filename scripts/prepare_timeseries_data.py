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
