#!/usr/bin/env python3
"""다변량 패턴 이상 데모 데이터 생성.

개별 메트릭은 정상 범위이지만, 메트릭 간 상관관계가 비정상인 케이스를 생성합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# 타임스탬프 생성 (3시간 = 180분)
start_time = datetime(2025, 10, 2, 0, 0, 0)
timestamps = [start_time + timedelta(minutes=i) for i in range(180)]

# 정상 데이터 생성 (2시간 = 120분)
# 특징: UDP Echo RTT와 eCPRI Delay가 강한 정의 상관관계 (0.8)
normal_data = []

for i in range(120):
    # UDP Echo RTT: 평균 5ms, 상관관계의 기준
    udp_rtt = np.random.normal(5.0, 0.5)

    # eCPRI Delay: UDP Echo RTT와 강한 정의 상관관계
    # (RTT 높으면 eCPRI도 높음 - 네트워크 혼잡 시 함께 증가)
    ecpri_delay = 100 + (udp_rtt - 5.0) * 10 + np.random.normal(0, 5)

    # LBM RTT: UDP Echo RTT와 약한 상관관계
    lbm_rtt = 7 + (udp_rtt - 5.0) * 0.3 + np.random.normal(0, 0.3)

    normal_data.append({
        'timestamp': timestamps[i],
        'endpoint_id': 'endpoint-1',
        'udp_echo_rtt_ms': max(0, udp_rtt),
        'ecpri_delay_us': max(0, ecpri_delay),
        'lbm_rtt_ms': max(0, lbm_rtt),
        'ccm_miss_count': 0
    })

# 이상 데이터 생성 (1시간 = 60분)
# 특징: 개별 값은 정상 범위이지만, 상관관계가 역전됨!
# (UDP Echo RTT 높은데 eCPRI Delay 낮음 - 비정상 패턴)
anomaly_data = []

for i in range(120, 180):
    # UDP Echo RTT: 여전히 정상 범위 (4-6ms)
    udp_rtt = np.random.normal(5.0, 0.5)

    # eCPRI Delay: **역의 상관관계**로 변경! (비정상 패턴)
    # (RTT 높으면 eCPRI가 낮아짐 - 학습 데이터와 반대)
    ecpri_delay = 100 - (udp_rtt - 5.0) * 10 + np.random.normal(0, 5)

    # LBM RTT: 여전히 약한 상관관계 유지
    lbm_rtt = 7 + (udp_rtt - 5.0) * 0.3 + np.random.normal(0, 0.3)

    anomaly_data.append({
        'timestamp': timestamps[i],
        'endpoint_id': 'endpoint-1',
        'udp_echo_rtt_ms': max(0, udp_rtt),
        'ecpri_delay_us': max(0, ecpri_delay),
        'lbm_rtt_ms': max(0, lbm_rtt),
        'ccm_miss_count': 0
    })

# 데이터 병합
all_data = normal_data + anomaly_data
df = pd.DataFrame(all_data)

# 원본 데이터 저장 (메트릭만 포함)
original_path = 'data/datasets/demo_multivariate_original.csv'
df.to_csv(original_path, index=False)

print(f"✅ 다변량 패턴 이상 데모 데이터 생성 완료: {original_path}")
print(f"   총 샘플: {len(df)}개")
print(f"   정상 구간: 0-120분 (UDP RTT ↑ → eCPRI Delay ↑)")
print(f"   이상 구간: 120-180분 (UDP RTT ↑ → eCPRI Delay ↓) <- 역의 상관관계!")
print(f"\n📊 정상 구간 통계:")
print(f"   UDP Echo RTT: {df.iloc[:120]['udp_echo_rtt_ms'].mean():.2f} ± {df.iloc[:120]['udp_echo_rtt_ms'].std():.2f} ms")
print(f"   eCPRI Delay:  {df.iloc[:120]['ecpri_delay_us'].mean():.2f} ± {df.iloc[:120]['ecpri_delay_us'].std():.2f} μs")
print(f"   상관계수: {df.iloc[:120][['udp_echo_rtt_ms', 'ecpri_delay_us']].corr().iloc[0, 1]:.3f}")
print(f"\n📊 이상 구간 통계:")
print(f"   UDP Echo RTT: {df.iloc[120:]['udp_echo_rtt_ms'].mean():.2f} ± {df.iloc[120:]['udp_echo_rtt_ms'].std():.2f} ms")
print(f"   eCPRI Delay:  {df.iloc[120:]['ecpri_delay_us'].mean():.2f} ± {df.iloc[120:]['ecpri_delay_us'].std():.2f} μs")
print(f"   상관계수: {df.iloc[120:][['udp_echo_rtt_ms', 'ecpri_delay_us']].corr().iloc[0, 1]:.3f}")
print(f"\n💡 개별 메트릭 값은 거의 동일하지만, 상관관계가 역전되어 다변량 탐지기가 이상을 감지할 것입니다!")
