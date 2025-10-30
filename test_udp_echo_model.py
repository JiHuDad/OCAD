#!/usr/bin/env python3
"""TCN 모델 로드 및 추론 테스트."""

import torch
import numpy as np
from pathlib import Path
import json

print("=" * 70)
print("TCN 모델 로드 테스트")
print("=" * 70)

# 1. 메타데이터 로드
print("\n[1/4] 메타데이터 로드 중...")
metadata_path = Path("ocad/models/tcn/udp_echo_vv2.0.0.json")

with open(metadata_path) as f:
    metadata = json.load(f)

print(f"  ✅ 메타데이터 로드 성공!")
print(f"    - 버전: {metadata['metadata']['version']}")
print(f"    - 메트릭 타입: {metadata['metadata']['metric_type']}")
print(f"    - 학습 날짜: {metadata['metadata']['training_date']}")
print(f"    - 시퀀스 길이: {metadata['metadata']['sequence_length']}")

# 2. 모델 아키텍처 생성
print("\n[2/4] 모델 아키텍처 생성 중...")

# SimpleTCN import
from ocad.detectors.residual import SimpleTCN

# 메타데이터에서 설정 가져오기
model_config = metadata['model_config']
input_size = model_config['input_size']
hidden_size = model_config['hidden_size']
output_size = model_config['output_size']

model = SimpleTCN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size
)

print(f"  ✅ 모델 아키텍처 생성 완료!")
print(f"    - input_size: {input_size}")
print(f"    - hidden_size: {hidden_size}")
print(f"    - output_size: {output_size}")

# 3. 모델 state_dict 로드
print("\n[3/4] 학습된 가중치 로드 중...")
model_path = Path("ocad/models/tcn/udp_echo_vv2.0.0.pth")

if not model_path.exists():
    print(f"❌ 모델 파일이 없습니다: {model_path}")
    exit(1)

# .pth 파일 로드 (메타데이터 포함)
checkpoint = torch.load(model_path, map_location='cpu')

# state_dict 추출
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

file_size_mb = model_path.stat().st_size / 1024 / 1024
print(f"  ✅ 가중치 로드 성공!")
print(f"    - 파일 크기: {file_size_mb:.2f} MB")

# 4. 더미 추론 테스트
print("\n[4/5] 더미 추론 테스트...")
print("  - 입력 형식: [batch, features, sequence]")
print("  - SimpleTCN 기대 shape: (batch=1, features=1, seq=10)")

# 더미 입력 (batch_size=1, features=1, seq_len=10) - Conv1d 형식
dummy_input = torch.randn(1, 1, 10)

with torch.no_grad():
    output = model(dummy_input)

print(f"  ✅ 더미 추론 성공!")
print(f"    - 입력 shape: {dummy_input.shape} (batch, features, sequence)")
print(f"    - 출력 shape: {output.shape}")
print(f"    - 예측 값: {output.item():.4f}")

# 5. 실제 데이터로 추론 테스트
print("\n[5/5] 실제 시퀀스로 추론 테스트...")
import pandas as pd

test_data_path = Path("data/processed/timeseries_test.parquet")
test_df = pd.read_parquet(test_data_path)

# 첫 3개 샘플로 테스트
num_samples = min(3, len(test_df))
errors = []

for i in range(num_samples):
    sample = test_df.iloc[i]
    sequence = sample['sequence']
    target = sample['target']
    
    # 추론 - SimpleTCN은 [batch, features, sequence] 기대
    input_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(0)  # (1, 1, 10)
    with torch.no_grad():
        prediction = model(input_tensor)
    
    pred_value = prediction.item()
    error = abs(pred_value - target)
    errors.append(error)
    
    if i == 0:
        print(f"  샘플 {i+1}:")
        print(f"    - 시퀀스: [{sequence[0]:.2f}, {sequence[1]:.2f}, ..., {sequence[-1]:.2f}]")
        print(f"    - 예측 값: {pred_value:.4f}")
        print(f"    - 실제 값: {target:.4f}")
        print(f"    - 오차(MAE): {error:.4f}")

avg_error = np.mean(errors)
print(f"\n  ✅ {num_samples}개 샘플 평균 오차: {avg_error:.4f}")

# 6. 성능 메트릭 출력
print("\n" + "=" * 70)
print("학습된 모델 성능")
print("=" * 70)
perf = metadata['performance']
print(f"  - Test MSE:  {perf['test_mse']:.4f}")
print(f"  - Test RMSE: {perf['test_rmse']:.4f}")
print(f"  - Test MAE:  {perf['test_mae']:.4f}")
print(f"  - Test R²:   {perf['test_r2']:.4f}")
print(f"  - Best Val Loss: {perf['best_val_loss']:.4f}")
print(f"  - Total Epochs: {perf['total_epochs']}")

print("\n" + "=" * 70)
print("✅ Phase 1 완료!")
print("=" * 70)
print("\nUDP Echo TCN 모델이 정상적으로 학습되고 로드되었습니다.")
print("모델은 추론에 사용할 준비가 되었습니다.")
print("\n다음 단계:")
print("  - Phase 2: eCPRI, LBM TCN 학습")
print("  - Phase 3: Isolation Forest 학습")
print("  - Phase 4: 학습된 모델을 추론 파이프라인에 통합")
