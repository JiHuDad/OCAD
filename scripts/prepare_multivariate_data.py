#!/usr/bin/env python3
"""다변량 이상 탐지를 위한 데이터 준비."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

def prepare_multivariate_data(input_path, output_dir, window_size=10):
    """Wide 형식 다변량 데이터 생성.
    
    Args:
        input_path: 입력 CSV 파일 (예: 01_normal_operation_24h.csv)
        output_dir: 출력 디렉토리
        window_size: 윈도우 크기 (통계 계산용)
    
    Returns:
        생성된 parquet 파일 경로
    """
    print("="*70)
    print("다변량 데이터 준비")
    print("="*70)
    print(f"\n입력: {input_path}")
    print(f"윈도우 크기: {window_size}")
    
    # 데이터 로드
    df = pd.read_csv(input_path)
    print(f"\n총 레코드: {len(df):,}")
    
    # 필요한 메트릭 컬럼
    metric_cols = [
        'udp_echo_rtt_ms',
        'ecpri_delay_us', 
        'lbm_rtt_ms',
        'ccm_miss_count'
    ]
    
    # 메트릭 컬럼 존재 확인
    for col in metric_cols:
        if col not in df.columns:
            raise ValueError(f"컬럼 없음: {col}")
    
    print(f"\n메트릭 컬럼:")
    for col in metric_cols:
        print(f"  - {col}")
    
    # 타임스탬프 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Endpoint별로 그룹화하여 통계 계산
    features = []
    
    print(f"\n윈도우 기반 피처 생성 중...")
    for endpoint_id, group in tqdm(df.groupby('endpoint_id'), desc="Endpoints"):
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(group) - window_size + 1):
            window = group.iloc[i:i+window_size]
            
            feature_row = {
                'timestamp': window.iloc[-1]['timestamp'],
                'endpoint_id': endpoint_id,
            }
            
            # 각 메트릭의 통계량 계산
            for metric in metric_cols:
                values = window[metric].values
                feature_row[f'{metric}_mean'] = np.mean(values)
                feature_row[f'{metric}_std'] = np.std(values)
                feature_row[f'{metric}_min'] = np.min(values)
                feature_row[f'{metric}_max'] = np.max(values)
                feature_row[f'{metric}_last'] = values[-1]
            
            # 이상 여부 (기본값: False)
            feature_row['is_anomaly'] = False
            
            features.append(feature_row)
    
    result_df = pd.DataFrame(features)
    print(f"\n생성된 샘플: {len(result_df):,}")
    print(f"피처 개수: {len([c for c in result_df.columns if c not in ['timestamp', 'endpoint_id', 'is_anomaly']])}")
    
    # Train/Val/Test 분할
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(result_df)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    
    train_df = result_df[:train_end]
    val_df = result_df[train_end:val_end]
    test_df = result_df[val_end:]
    
    print(f"\n데이터 분할:")
    print(f"  - Train: {len(train_df):,} (80%)")
    print(f"  - Val:   {len(val_df):,} (10%)")
    print(f"  - Test:  {len(test_df):,} (10%)")
    
    # 저장
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'multivariate_train.parquet'
    val_path = output_dir / 'multivariate_val.parquet'
    test_path = output_dir / 'multivariate_test.parquet'
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    train_size_mb = train_path.stat().st_size / 1024 / 1024
    val_size_mb = val_path.stat().st_size / 1024 / 1024
    test_size_mb = test_path.stat().st_size / 1024 / 1024
    
    print(f"\n저장 완료:")
    print(f"  - Train: {train_path.name} ({train_size_mb:.2f} MB)")
    print(f"  - Val:   {val_path.name} ({val_size_mb:.2f} MB)")
    print(f"  - Test:  {test_path.name} ({test_size_mb:.2f} MB)")
    
    # 샘플 데이터 출력
    print(f"\n샘플 데이터 (첫 2행):")
    print(train_df.head(2).to_string())
    
    print("\n" + "="*70)
    print("✅ 다변량 데이터 준비 완료!")
    print("="*70)
    
    return train_path, val_path, test_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/samples/01_normal_operation_24h.csv')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--window-size', type=int, default=10)
    
    args = parser.parse_args()
    prepare_multivariate_data(args.input, args.output_dir, args.window_size)
