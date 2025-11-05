#!/usr/bin/env python3
"""테스트용 샘플 데이터 생성 스크립트.

정상 데이터와 다양한 유형의 이상 데이터를 생성합니다.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# 시드 고정 (재현성)
random.seed(42)

# 시작 시간
start_time = datetime.now() - timedelta(hours=2)

# 샘플 수
n_normal = 100
n_anomaly = 30
total_samples = n_normal + n_anomaly

# 엔드포인트 ID
endpoint_id = 'test_endpoint_1'

def normal_random(mean, std):
    """정규분포 난수 생성 (Box-Muller 변환)."""
    u1 = random.random()
    u2 = random.random()
    z0 = (-2 * (u1 if u1 > 0 else 0.000001) ** 0.5) * (2 * 3.14159 * u2) ** 0.5
    import math
    z0 = math.sqrt(-2 * math.log(u1 if u1 > 0 else 0.000001)) * math.cos(2 * math.pi * u2)
    return mean + z0 * std

# 데이터 저장용 리스트
data_rows = []

print("샘플 데이터 생성 중...")

# === 정상 데이터 생성 ===
print(f"  정상 데이터: {n_normal}개")
for i in range(n_normal):
    timestamp = start_time + timedelta(seconds=i*30)
    row = {
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'endpoint_id': endpoint_id,
        'udp_echo_rtt_ms': max(0, normal_random(5.0, 0.5)),
        'ecpri_delay_us': max(0, normal_random(100.0, 10.0)),
        'lbm_rtt_ms': max(0, normal_random(4.0, 0.3)),
        'ccm_miss_count': 0,
    }
    data_rows.append(row)

# === 시나리오 1: 네트워크 지연 급증 ===
print(f"  시나리오 1 (네트워크 지연 급증): 10개")
for i in range(10):
    timestamp = start_time + timedelta(seconds=(n_normal+i)*30)
    row = {
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'endpoint_id': endpoint_id,
        'udp_echo_rtt_ms': max(0, normal_random(15.0, 2.0)),
        'ecpri_delay_us': max(0, normal_random(250.0, 30.0)),
        'lbm_rtt_ms': max(0, normal_random(12.0, 1.0)),
        'ccm_miss_count': random.randint(1, 3),
    }
    data_rows.append(row)

# === 시나리오 2: 간헐적 스파이크 ===
print(f"  시나리오 2 (간헐적 스파이크): 10개")
for i in range(10):
    timestamp = start_time + timedelta(seconds=(n_normal+10+i)*30)
    is_spike = i >= 5
    row = {
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'endpoint_id': endpoint_id,
        'udp_echo_rtt_ms': max(0, normal_random(20.0, 3.0) if is_spike else normal_random(5.0, 0.5)),
        'ecpri_delay_us': max(0, normal_random(100.0, 50.0)),
        'lbm_rtt_ms': max(0, normal_random(2.0, 0.2)),
        'ccm_miss_count': 0,
    }
    data_rows.append(row)

# === 시나리오 3: 다변량 이상 ===
print(f"  시나리오 3 (다변량 이상): 10개")
for i in range(10):
    timestamp = start_time + timedelta(seconds=(n_normal+20+i)*30)
    row = {
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'endpoint_id': endpoint_id,
        'udp_echo_rtt_ms': max(0, normal_random(6.5, 0.3)),
        'ecpri_delay_us': max(0, normal_random(80.0, 5.0)),
        'lbm_rtt_ms': max(0, normal_random(5.5, 0.2)),
        'ccm_miss_count': 0,
    }
    data_rows.append(row)

# === CSV 저장 ===
output_path = Path('data/test_sample_data.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', newline='') as f:
    fieldnames = ['timestamp', 'endpoint_id', 'udp_echo_rtt_ms', 'ecpri_delay_us', 'lbm_rtt_ms', 'ccm_miss_count']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data_rows)

print("\n" + "="*70)
print("✅ 샘플 데이터 생성 완료!")
print("="*70)
print(f"출력 파일: {output_path.absolute()}")
print(f"총 샘플 수: {len(data_rows)}")
print(f"\n구성:")
print(f"  - 정상 데이터: {n_normal}개 (77%)")
print(f"  - 이상 데이터: {n_anomaly}개 (23%)")
print(f"    - 시나리오 1 (네트워크 지연 급증): 10개")
print(f"    - 시나리오 2 (간헐적 스파이크): 10개")
print(f"    - 시나리오 3 (다변량 이상): 10개")

print(f"\n처음 5개 샘플:")
for i, row in enumerate(data_rows[:5]):
    print(f"  {i+1}. {row['timestamp']}: UDP={row['udp_echo_rtt_ms']:.2f}ms, eCPRI={row['ecpri_delay_us']:.2f}us, LBM={row['lbm_rtt_ms']:.2f}ms")

print(f"\n마지막 5개 샘플 (이상 데이터):")
for i, row in enumerate(data_rows[-5:]):
    print(f"  {len(data_rows)-4+i}. {row['timestamp']}: UDP={row['udp_echo_rtt_ms']:.2f}ms, eCPRI={row['ecpri_delay_us']:.2f}us, LBM={row['lbm_rtt_ms']:.2f}ms")

