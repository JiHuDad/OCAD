#!/usr/bin/env python3
"""학습된 모든 모델로 검증 데이터셋 테스트."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import pickle
import json
from datetime import datetime

from ocad.detectors.residual import SimpleTCN

def load_tcn_model(metric_type, version='v2.0.0'):
    """TCN 모델 로드."""
    model_path = Path(f'ocad/models/tcn/{metric_type}_vv2.0.0.pth')
    metadata_path = Path(f'ocad/models/tcn/{metric_type}_vv2.0.0.json')
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
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
    
    return model, metadata

def load_isolation_forest():
    """Isolation Forest 모델 로드."""
    model_path = Path('ocad/models/isolation_forest/isolation_forest_v1.0.0.pkl')
    scaler_path = Path('ocad/models/isolation_forest/isolation_forest_v1.0.0_scaler.pkl')
    metadata_path = Path('ocad/models/isolation_forest/isolation_forest_v1.0.0.json')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    return model, scaler, metadata

def prepare_multivariate_features(df, window_size=10):
    """다변량 피처 생성."""
    metric_cols = ['udp_echo_rtt_ms', 'ecpri_delay_us', 'lbm_rtt_ms', 'ccm_miss_count']
    features = []
    
    for endpoint_id, group in df.groupby('endpoint_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(group) - window_size + 1):
            window = group.iloc[i:i+window_size]
            feature_row = {}
            
            for metric in metric_cols:
                if metric in window.columns:
                    values = window[metric].values
                    feature_row[f'{metric}_mean'] = np.mean(values)
                    feature_row[f'{metric}_std'] = np.std(values)
                    feature_row[f'{metric}_min'] = np.min(values)
                    feature_row[f'{metric}_max'] = np.max(values)
                    feature_row[f'{metric}_last'] = values[-1]
            
            features.append(feature_row)
    
    return pd.DataFrame(features)

def validate_dataset(dataset_path, dataset_name):
    """단일 데이터셋 검증."""
    print(f"\n{'='*70}")
    print(f"검증 중: {dataset_name}")
    print(f"{'='*70}")
    
    # 데이터 로드
    df = pd.read_csv(dataset_path)
    print(f"  레코드 수: {len(df):,}")
    
    # TCN 모델로 예측
    print(f"\n[TCN 모델 예측]")
    tcn_results = {}
    
    for metric_type in ['udp_echo', 'ecpri', 'lbm']:
        try:
            model, metadata = load_tcn_model(metric_type)
            
            # 간단한 예측 (마지막 10개 값으로)
            metric_col = {
                'udp_echo': 'udp_echo_rtt_ms',
                'ecpri': 'ecpri_delay_us',
                'lbm': 'lbm_rtt_ms'
            }[metric_type]
            
            if metric_col in df.columns:
                values = df[metric_col].values[-10:]
                input_tensor = torch.FloatTensor(values).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model(input_tensor)
                
                actual_next = df[metric_col].values[-1] if len(df) > 10 else values[-1]
                pred_value = prediction.item()
                error = abs(pred_value - actual_next)
                
                tcn_results[metric_type] = {
                    'predicted': pred_value,
                    'actual': actual_next,
                    'error': error
                }
                
                print(f"  {metric_type}: pred={pred_value:.2f}, actual={actual_next:.2f}, error={error:.2f}")
        except Exception as e:
            print(f"  {metric_type}: 스킵 ({e})")
    
    # Isolation Forest 예측
    print(f"\n[Isolation Forest 예측]")
    try:
        model, scaler, metadata = load_isolation_forest()
        
        # 다변량 피처 생성
        features_df = prepare_multivariate_features(df, window_size=10)
        
        if len(features_df) > 0:
            feature_cols = [col for col in features_df.columns]
            X = features_df[feature_cols].values
            X_scaled = scaler.transform(X)
            
            scores = model.decision_function(X_scaled)
            predictions = model.predict(X_scaled)
            
            anomaly_count = (predictions == -1).sum()
            anomaly_rate = anomaly_count / len(predictions) * 100
            
            print(f"  총 샘플: {len(predictions)}")
            print(f"  이상 탐지: {anomaly_count} ({anomaly_rate:.1f}%)")
            print(f"  Score 범위: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Score 평균: {scores.mean():.4f}")
            
            # 가장 이상한 샘플 3개
            if anomaly_count > 0:
                top_anomalies = np.argsort(scores)[:min(3, anomaly_count)]
                print(f"\n  가장 이상한 샘플:")
                for rank, idx in enumerate(top_anomalies, 1):
                    print(f"    #{rank}: score={scores[idx]:.4f}, index={idx}")
        else:
            print(f"  피처 생성 실패 (데이터 부족)")
    except Exception as e:
        print(f"  오류: {e}")

def main():
    print("="*70)
    print("학습된 모델 검증 (모든 데이터셋)")
    print("="*70)
    print(f"\n실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    datasets = {
        'data/samples/01_normal_operation_24h.csv': '정상 운영 (24시간)',
        'data/samples/02_drift_anomaly.csv': 'Drift 이상',
        'data/samples/03_spike_anomaly.csv': 'Spike 이상',
        'data/samples/04_multi_endpoint.csv': '다중 엔드포인트',
    }
    
    for dataset_path, dataset_name in datasets.items():
        if Path(dataset_path).exists():
            try:
                validate_dataset(dataset_path, dataset_name)
            except Exception as e:
                print(f"\n❌ {dataset_name} 검증 실패: {e}")
        else:
            print(f"\n⚠️  {dataset_name}: 파일 없음")
    
    print("\n" + "="*70)
    print("✅ 검증 완료!")
    print("="*70)

if __name__ == '__main__':
    main()
