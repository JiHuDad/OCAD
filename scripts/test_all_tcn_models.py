#!/usr/bin/env python3
"""모든 TCN 모델 로드 및 검증."""

import torch
import json
from pathlib import Path
import pandas as pd

def test_model(metric_type, model_path, metadata_path, test_data_path):
    """단일 모델 테스트."""
    print(f"\n{'='*70}")
    print(f"{metric_type.upper()} TCN 모델 테스트")
    print(f"{'='*70}")
    
    # 1. 메타데이터 로드
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"\n[1/3] 메타데이터:")
    print(f"  - 버전: {metadata['metadata']['version']}")
    print(f"  - 학습 날짜: {metadata['metadata']['training_date']}")
    print(f"  - 총 에포크: {metadata['performance']['total_epochs']}")
    
    # 2. 모델 로드
    from ocad.detectors.residual import SimpleTCN
    
    model_config = metadata['model_config']
    model = SimpleTCN(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        output_size=model_config['output_size']
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    file_size_kb = model_path.stat().st_size / 1024
    print(f"\n[2/3] 모델 로드:")
    print(f"  - 파일 크기: {file_size_kb:.1f} KB")
    print(f"  - 아키텍처: SimpleTCN({model_config['input_size']}, {model_config['hidden_size']}, {model_config['output_size']})")
    
    # 3. 추론 테스트
    test_df = pd.read_parquet(test_data_path)
    sample = test_df.iloc[0]
    sequence = sample['sequence']
    target = sample['target']
    
    input_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(0)  # (1, 1, 10)
    with torch.no_grad():
        prediction = model(input_tensor)
    
    pred_value = prediction.item()
    error = abs(pred_value - target)
    
    print(f"\n[3/3] 추론 테스트:")
    print(f"  - 예측 값: {pred_value:.4f}")
    print(f"  - 실제 값: {target:.4f}")
    print(f"  - 오차: {error:.4f}")
    
    # 4. 성능 메트릭
    print(f"\n성능 메트릭:")
    perf = metadata['performance']
    print(f"  - Test MSE:  {perf['test_mse']:.4f}")
    print(f"  - Test MAE:  {perf['test_mae']:.4f}")
    print(f"  - Test RMSE: {perf['test_rmse']:.4f}")
    print(f"  - Test R²:   {perf['test_r2']:.4f}")
    print(f"  - Best Val Loss: {perf['best_val_loss']:.4f}")
    
    return {
        'metric_type': metric_type,
        'version': metadata['metadata']['version'],
        'epochs': metadata['performance']['total_epochs'],
        'test_mse': perf['test_mse'],
        'test_mae': perf['test_mae'],
        'test_r2': perf['test_r2'],
        'file_size_kb': file_size_kb,
    }


def main():
    print("="*70)
    print("모든 TCN 모델 검증")
    print("="*70)
    
    models = [
        {
            'metric_type': 'udp_echo',
            'model_path': Path('ocad/models/tcn/udp_echo_vv2.0.0.pth'),
            'metadata_path': Path('ocad/models/tcn/udp_echo_vv2.0.0.json'),
            'test_data_path': Path('data/processed/timeseries_train.parquet'),  # UDP Echo는 기존 파일
        },
        {
            'metric_type': 'ecpri',
            'model_path': Path('ocad/models/tcn/ecpri_vv2.0.0.pth'),
            'metadata_path': Path('ocad/models/tcn/ecpri_vv2.0.0.json'),
            'test_data_path': Path('data/processed/timeseries_ecpri_test.parquet'),
        },
        {
            'metric_type': 'lbm',
            'model_path': Path('ocad/models/tcn/lbm_vv2.0.0.pth'),
            'metadata_path': Path('ocad/models/tcn/lbm_vv2.0.0.json'),
            'test_data_path': Path('data/processed/timeseries_lbm_test.parquet'),
        },
    ]
    
    results = []
    
    for model_info in models:
        try:
            result = test_model(**model_info)
            results.append(result)
            print(f"\n✅ {model_info['metric_type']} 모델 검증 성공")
        except Exception as e:
            print(f"\n❌ {model_info['metric_type']} 모델 검증 실패: {e}")
    
    # 요약 테이블
    print("\n" + "="*70)
    print("전체 모델 성능 요약")
    print("="*70)
    
    print(f"\n{'메트릭':<12} {'버전':<8} {'에포크':<8} {'Test MSE':<12} {'Test MAE':<12} {'R²':<10} {'크기(KB)':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['metric_type']:<12} {r['version']:<8} {r['epochs']:<8} "
              f"{r['test_mse']:<12.4f} {r['test_mae']:<12.4f} "
              f"{r['test_r2']:<10.4f} {r['file_size_kb']:<10.1f}")
    
    print("\n" + "="*70)
    print(f"✅ Phase 2 완료! ({len(results)}/3 모델 검증 성공)")
    print("="*70)
    
    print("\n다음 단계:")
    print("  - Phase 3: Isolation Forest 다변량 이상 탐지 모델 학습")
    print("  - Phase 4: 학습된 모델을 추론 파이프라인에 통합")


if __name__ == '__main__':
    main()
