#!/usr/bin/env python3
"""데모용 이상 탐지 결과 생성 (리포트 기능 시연용)."""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# 정상 데이터 생성 (100개)
normal_data = []
start_time = datetime(2025, 10, 2, 0, 0, 0)

for i in range(100):
    timestamp = start_time + timedelta(minutes=i)
    normal_data.append({
        'timestamp': timestamp,
        'endpoint_id': 'endpoint-demo',
        'udp_echo_rtt_ms': np.random.normal(5.0, 0.5),  # 정상: 5ms ± 0.5
        'ecpri_delay_us': np.random.normal(100.0, 10.0),  # 정상: 100us ± 10
        'lbm_rtt_ms': np.random.normal(7.0, 0.3),  # 정상: 7ms ± 0.3
        'ccm_miss_count': 0,
        'residual_score': 0.0,
        'residual_anomaly': 0,
        'multivariate_score': np.random.uniform(0.05, 0.15),  # 정상 범위
        'multivariate_anomaly': 0,
        'final_score': np.random.uniform(0.05, 0.15),
        'is_anomaly': 0
    })

# 이상 데이터 생성 (20개) - Spike 패턴
anomaly_data = []
for i in range(20):
    timestamp = start_time + timedelta(minutes=100 + i)

    # UDP Echo RTT가 급격히 증가 (5ms → 25ms)
    udp_rtt = np.random.normal(25.0, 3.0)
    # eCPRI도 약간 증가
    ecpri = np.random.normal(250.0, 30.0)
    # LBM도 증가
    lbm = np.random.normal(18.0, 2.0)

    # 점수 계산
    multivariate_score = np.random.uniform(0.6, 0.9)
    final_score = multivariate_score

    anomaly_data.append({
        'timestamp': timestamp,
        'endpoint_id': 'endpoint-demo',
        'udp_echo_rtt_ms': udp_rtt,
        'ecpri_delay_us': ecpri,
        'lbm_rtt_ms': lbm,
        'ccm_miss_count': 0,
        'residual_score': 0.0,
        'residual_anomaly': 0,
        'multivariate_score': multivariate_score,
        'multivariate_anomaly': 1,
        'final_score': final_score,
        'is_anomaly': 1
    })

# 정상 데이터로 복귀 (50개)
for i in range(50):
    timestamp = start_time + timedelta(minutes=120 + i)
    normal_data.append({
        'timestamp': timestamp,
        'endpoint_id': 'endpoint-demo',
        'udp_echo_rtt_ms': np.random.normal(5.0, 0.5),
        'ecpri_delay_us': np.random.normal(100.0, 10.0),
        'lbm_rtt_ms': np.random.normal(7.0, 0.3),
        'ccm_miss_count': 0,
        'residual_score': 0.0,
        'residual_anomaly': 0,
        'multivariate_score': np.random.uniform(0.05, 0.15),
        'multivariate_anomaly': 0,
        'final_score': np.random.uniform(0.05, 0.15),
        'is_anomaly': 0
    })

# 합치기
all_data = normal_data + anomaly_data
df = pd.DataFrame(all_data)
df = df.sort_values('timestamp').reset_index(drop=True)

# 추론 결과 저장 (모든 컬럼 포함)
inference_path = 'data/results/demo_spike_result.csv'
df.to_csv(inference_path, index=False)
print(f"✅ 데모 이상 탐지 결과 생성 완료: {inference_path}")
print(f"   총 샘플: {len(df)}개")
print(f"   정상: {len(df[df['is_anomaly'] == 0])}개")
print(f"   이상: {len(df[df['is_anomaly'] == 1])}개")
print(f"   이상 비율: {df['is_anomaly'].mean()*100:.1f}%")

# 원본 데이터 저장 (메트릭만 포함, 점수/라벨 제외)
original_df = df[['timestamp', 'endpoint_id', 'udp_echo_rtt_ms', 'ecpri_delay_us', 'lbm_rtt_ms', 'ccm_miss_count']].copy()
original_path = 'data/datasets/demo_spike_original.csv'
original_df.to_csv(original_path, index=False)
print(f"\n✅ 원본 데이터 생성 완료: {original_path}")
