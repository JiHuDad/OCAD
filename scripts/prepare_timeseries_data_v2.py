#!/usr/bin/env python3
"""시계열 데이터를 TCN 학습용 시퀀스로 변환 (v2 - 메트릭 타입 파라미터화)."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# 메트릭 타입별 컬럼 매핑
METRIC_COLUMNS = {
    'udp_echo': 'udp_echo_rtt_ms',
    'ecpri': 'ecpri_delay_us',
    'lbm': 'lbm_rtt_ms',
}


def create_sequences(df, metric_col, metric_type, sequence_length=10):
    """시계열 데이터를 슬라이딩 윈도우 시퀀스로 변환.
    
    Args:
        df: 원본 데이터프레임
        metric_col: 메트릭 컬럼 이름
        metric_type: 메트릭 타입 (udp_echo, ecpri, lbm)
        sequence_length: 시퀀스 길이 (기본값: 10)
        
    Returns:
        시퀀스 데이터프레임
    """
    sequences = []
    
    print(f"\n시퀀스 생성 중 ({metric_type})...")
    print(f"  - 메트릭 컬럼: {metric_col}")
    print(f"  - 시퀀스 길이: {sequence_length}")
    
    # endpoint_id별로 그룹화
    for endpoint_id, group in tqdm(df.groupby('endpoint_id'), desc="Endpoints"):
        # 시간순 정렬
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        # 메트릭 값 추출
        values = group[metric_col].values
        
        # 슬라이딩 윈도우: [t1, t2, ..., t10] -> t11
        for i in range(len(values) - sequence_length):
            seq = values[i:i + sequence_length].tolist()
            target = values[i + sequence_length]
            
            sequences.append({
                'timestamp': group.iloc[i + sequence_length]['timestamp'],
                'endpoint_id': endpoint_id,
                'metric_type': metric_type,
                'sequence': seq,
                'target': float(target),
                'is_anomaly': False,  # 정상 데이터로 가정
            })
    
    return pd.DataFrame(sequences)


def split_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """데이터를 train/val/test로 분할.
    
    Args:
        df: 시퀀스 데이터프레임
        train_ratio: 학습 비율
        val_ratio: 검증 비율
        test_ratio: 테스트 비율
        
    Returns:
        (train_df, val_df, test_df)
    """
    # 셔플
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='시계열 데이터를 TCN 학습용 시퀀스로 변환')
    parser.add_argument('--metric-type', type=str, required=True,
                        choices=['udp_echo', 'ecpri', 'lbm'],
                        help='메트릭 타입')
    parser.add_argument('--input', type=str, default='data/samples/01_normal_operation_24h.csv',
                        help='입력 CSV 파일')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='출력 디렉토리')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='시퀀스 길이')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='학습 데이터 비율')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='검증 데이터 비율')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='테스트 데이터 비율')
    
    args = parser.parse_args()
    
    # 메트릭 컬럼 확인
    if args.metric_type not in METRIC_COLUMNS:
        print(f"❌ 지원하지 않는 메트릭 타입: {args.metric_type}")
        return
    
    metric_col = METRIC_COLUMNS[args.metric_type]
    
    print("=" * 70)
    print(f"시계열 데이터 준비 ({args.metric_type})")
    print("=" * 70)
    
    # 1. 데이터 로드
    print(f"\n[1/4] 데이터 로드 중...")
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 입력 파일이 없습니다: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"  ✅ 로드 완료: {len(df):,} 레코드")
    
    # 메트릭 컬럼 확인
    if metric_col not in df.columns:
        print(f"❌ 메트릭 컬럼이 없습니다: {metric_col}")
        print(f"   사용 가능한 컬럼: {list(df.columns)}")
        return
    
    # 2. 시퀀스 생성
    print(f"\n[2/4] 시퀀스 생성 중...")
    seq_df = create_sequences(df, metric_col, args.metric_type, args.sequence_length)
    print(f"  ✅ 생성 완료: {len(seq_df):,} 시퀀스")
    
    # 3. 데이터 분할
    print(f"\n[3/4] 데이터 분할 중...")
    train_df, val_df, test_df = split_data(
        seq_df,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    print(f"  ✅ 분할 완료:")
    print(f"    - Train: {len(train_df):,} ({args.train_ratio*100:.0f}%)")
    print(f"    - Val:   {len(val_df):,} ({args.val_ratio*100:.0f}%)")
    print(f"    - Test:  {len(test_df):,} ({args.test_ratio*100:.0f}%)")
    
    # 4. 저장
    print(f"\n[4/4] 데이터 저장 중...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 메트릭별 파일명
    train_path = output_dir / f"timeseries_{args.metric_type}_train.parquet"
    val_path = output_dir / f"timeseries_{args.metric_type}_val.parquet"
    test_path = output_dir / f"timeseries_{args.metric_type}_test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    train_size_mb = train_path.stat().st_size / 1024 / 1024
    val_size_mb = val_path.stat().st_size / 1024 / 1024
    test_size_mb = test_path.stat().st_size / 1024 / 1024
    
    print(f"  ✅ 저장 완료:")
    print(f"    - Train: {train_path} ({train_size_mb:.2f} MB)")
    print(f"    - Val:   {val_path} ({val_size_mb:.2f} MB)")
    print(f"    - Test:  {test_path} ({test_size_mb:.2f} MB)")
    
    print("\n" + "=" * 70)
    print(f"✅ {args.metric_type} 시계열 데이터 준비 완료!")
    print("=" * 70)


if __name__ == '__main__':
    main()
