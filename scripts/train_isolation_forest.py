#!/usr/bin/env python3
"""Isolation Forest 모델 학습."""

import argparse
import json
from pathlib import Path
from datetime import datetime
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def train_isolation_forest(
    train_path,
    val_path,
    test_path,
    output_dir,
    n_estimators=100,
    contamination=0.1,
    random_state=42,
    version='v1.0.0'
):
    """Isolation Forest 모델 학습."""
    
    print("="*70)
    print("Isolation Forest 학습 시작")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n[1/6] 데이터 로드...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # 2. 피처 추출
    print("\n[2/6] 피처 추출...")
    feature_cols = [col for col in train_df.columns 
                    if col not in ['timestamp', 'endpoint_id', 'is_anomaly']]
    
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    y_test = test_df['is_anomaly'].values  # 테스트용
    
    print(f"  피처 개수: {len(feature_cols)}")
    print(f"  피처 목록 (처음 5개): {feature_cols[:5]}")
    
    # 3. 정규화
    print("\n[3/6] 데이터 정규화...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Scaler 학습 완료")
    print(f"    Mean (처음 3개): {scaler.mean_[:3]}")
    print(f"    Std (처음 3개):  {scaler.scale_[:3]}")
    
    # 4. 모델 학습
    print("\n[4/6] Isolation Forest 학습...")
    print(f"  하이퍼파라미터:")
    print(f"    - n_estimators: {n_estimators}")
    print(f"    - contamination: {contamination}")
    print(f"    - random_state: {random_state}")
    
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled)
    print(f"  학습 완료!")
    
    # 5. 평가
    print("\n[5/6] 모델 평가...")
    
    # Anomaly score 계산 (낮을수록 이상)
    train_scores = model.decision_function(X_train_scaled)
    val_scores = model.decision_function(X_val_scaled)
    test_scores = model.decision_function(X_test_scaled)
    
    # Prediction (-1: anomaly, 1: normal)
    test_pred = model.predict(X_test_scaled)
    test_pred_binary = (test_pred == -1).astype(int)
    
    print(f"\n평가 결과:")
    print(f"  Train anomaly score: mean={train_scores.mean():.4f}, std={train_scores.std():.4f}")
    print(f"  Val anomaly score:   mean={val_scores.mean():.4f}, std={val_scores.std():.4f}")
    print(f"  Test anomaly score:  mean={test_scores.mean():.4f}, std={test_scores.std():.4f}")
    print(f"  Test predicted anomalies: {test_pred_binary.sum()} / {len(test_pred_binary)} ({test_pred_binary.sum()/len(test_pred_binary)*100:.1f}%)")
    
    # 6. 모델 저장
    print("\n[6/6] 모델 저장...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f'isolation_forest_{version}.pkl'
    scaler_path = output_dir / f'isolation_forest_{version}_scaler.pkl'
    metadata_path = output_dir / f'isolation_forest_{version}.json'
    
    # 모델 저장
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Scaler 저장
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 메타데이터 저장
    metadata = {
        'model_type': 'scikit-learn',
        'algorithm': 'IsolationForest',
        'metadata': {
            'version': version,
            'training_date': datetime.now().isoformat(),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
        },
        'hyperparameters': {
            'n_estimators': n_estimators,
            'contamination': contamination,
            'random_state': random_state,
        },
        'performance': {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_score_mean': float(train_scores.mean()),
            'train_score_std': float(train_scores.std()),
            'val_score_mean': float(val_scores.mean()),
            'val_score_std': float(val_scores.std()),
            'test_score_mean': float(test_scores.mean()),
            'test_score_std': float(test_scores.std()),
            'test_predicted_anomalies': int(test_pred_binary.sum()),
            'test_anomaly_rate': float(test_pred_binary.sum() / len(test_pred_binary)),
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    model_size_mb = model_path.stat().st_size / 1024 / 1024
    scaler_size_kb = scaler_path.stat().st_size / 1024
    
    print(f"\n저장 완료:")
    print(f"  - 모델: {model_path.name} ({model_size_mb:.2f} MB)")
    print(f"  - Scaler: {scaler_path.name} ({scaler_size_kb:.1f} KB)")
    print(f"  - 메타데이터: {metadata_path.name}")
    
    print("\n" + "="*70)
    print("✅ Isolation Forest 학습 완료!")
    print("="*70)
    
    return model_path, metadata_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', default='data/processed/multivariate_train.parquet')
    parser.add_argument('--val-data', default='data/processed/multivariate_val.parquet')
    parser.add_argument('--test-data', default='data/processed/multivariate_test.parquet')
    parser.add_argument('--output-dir', default='ocad/models/isolation_forest')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--contamination', type=float, default=0.1)
    parser.add_argument('--version', default='v1.0.0')
    
    args = parser.parse_args()
    
    train_isolation_forest(
        args.train_data,
        args.val_data,
        args.test_data,
        args.output_dir,
        args.n_estimators,
        args.contamination,
        version=args.version
    )
