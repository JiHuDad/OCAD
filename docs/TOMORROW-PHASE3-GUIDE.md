# Phase 3 작업 가이드 (내일 진행)

**날짜**: 2025-10-31 예정  
**예상 소요 시간**: 1-2시간  
**목표**: Isolation Forest 다변량 이상 탐지 모델 학습

---

## 📋 작업 전 체크리스트

### 1. 현재 상태 확인

```bash
# 가상환경 활성화
source .venv/bin/activate

# Phase 1-2 완료 확인
ls -lh ocad/models/tcn/*vv2.0.0.*
# 예상 결과: UDP Echo, eCPRI, LBM 모델 파일 6개

# 진행 리포트 확인
cat docs/PROGRESS-REPORT-20251030.md
```

### 2. 필요한 라이브러리 확인

```bash
# Scikit-learn 설치 확인
python -c "import sklearn; print(sklearn.__version__)"

# 없다면 설치
pip install scikit-learn
```

---

## 🎯 Phase 3 작업 단계

### Step 1: 다변량 데이터 준비 (30분)

**목표**: 4개 메트릭을 동시에 고려하는 Wide 형식 데이터 생성

#### 1.1 다변량 데이터 준비 스크립트 작성

**파일**: `scripts/prepare_multivariate_data.py`

```python
#!/usr/bin/env python3
"""다변량 이상 탐지를 위한 데이터 준비."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def prepare_multivariate_data(input_path, output_dir, window_size=10):
    """Wide 형식 다변량 데이터 생성.
    
    Args:
        input_path: 입력 CSV 파일 (예: 01_normal_operation_24h.csv)
        output_dir: 출력 디렉토리
        window_size: 윈도우 크기 (통계 계산용)
    
    Returns:
        생성된 parquet 파일 경로
    """
    print(f"다변량 데이터 준비 중...")
    print(f"  입력: {input_path}")
    
    # 데이터 로드
    df = pd.read_csv(input_path)
    print(f"  총 레코드: {len(df):,}")
    
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
    
    # 타임스탬프 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Endpoint별로 그룹화하여 통계 계산
    features = []
    
    for endpoint_id, group in df.groupby('endpoint_id'):
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
    print(f"  생성된 샘플: {len(result_df):,}")
    
    # Train/Val/Test 분할
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(result_df)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    
    train_df = result_df[:train_end]
    val_df = result_df[train_end:val_end]
    test_df = result_df[val_end:]
    
    # 저장
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'multivariate_train.parquet'
    val_path = output_dir / 'multivariate_val.parquet'
    test_path = output_dir / 'multivariate_test.parquet'
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\n저장 완료:")
    print(f"  Train: {train_path} ({len(train_df):,} samples)")
    print(f"  Val:   {val_path} ({len(val_df):,} samples)")
    print(f"  Test:  {test_path} ({len(test_df):,} samples)")
    
    return train_path, val_path, test_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/samples/01_normal_operation_24h.csv')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--window-size', type=int, default=10)
    
    args = parser.parse_args()
    prepare_multivariate_data(args.input, args.output_dir, args.window_size)
```

#### 1.2 실행

```bash
python scripts/prepare_multivariate_data.py \
  --input data/samples/01_normal_operation_24h.csv \
  --output-dir data/processed \
  --window-size 10
```

**예상 결과**:
- `data/processed/multivariate_train.parquet`
- `data/processed/multivariate_val.parquet`
- `data/processed/multivariate_test.parquet`

---

### Step 2: Isolation Forest 학습 (30분)

**목표**: 다변량 이상 탐지 모델 학습

#### 2.1 학습 스크립트 작성

**파일**: `scripts/train_isolation_forest.py`

```python
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
    print("\n[1/5] 데이터 로드...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # 2. 피처 추출
    print("\n[2/5] 피처 추출...")
    feature_cols = [col for col in train_df.columns 
                    if col not in ['timestamp', 'endpoint_id', 'is_anomaly']]
    
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    y_test = test_df['is_anomaly'].values  # 테스트용
    
    print(f"  피처 개수: {len(feature_cols)}")
    print(f"  피처 목록: {feature_cols[:5]}... (총 {len(feature_cols)}개)")
    
    # 3. 정규화
    print("\n[3/5] 데이터 정규화...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Scaler 학습 완료")
    print(f"    Mean: {scaler.mean_[:3]}...")
    print(f"    Std:  {scaler.scale_[:3]}...")
    
    # 4. 모델 학습
    print("\n[4/5] Isolation Forest 학습...")
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
    print("\n[5/5] 모델 평가...")
    
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
    print(f"  Test predicted anomalies: {test_pred_binary.sum()} / {len(test_pred_binary)}")
    
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
            'test_score_mean': float(test_scores.mean()),
            'test_score_std': float(test_scores.std()),
            'test_predicted_anomalies': int(test_pred_binary.sum()),
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  모델: {model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  메타데이터: {metadata_path}")
    
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
```

#### 2.2 실행

```bash
python scripts/train_isolation_forest.py \
  --train-data data/processed/multivariate_train.parquet \
  --val-data data/processed/multivariate_val.parquet \
  --test-data data/processed/multivariate_test.parquet \
  --output-dir ocad/models/isolation_forest \
  --n-estimators 100 \
  --contamination 0.1 \
  --version v1.0.0
```

**예상 결과**:
- `ocad/models/isolation_forest/isolation_forest_v1.0.0.pkl`
- `ocad/models/isolation_forest/isolation_forest_v1.0.0_scaler.pkl`
- `ocad/models/isolation_forest/isolation_forest_v1.0.0.json`

---

### Step 3: 모델 검증 (20분)

#### 3.1 검증 스크립트 작성

**파일**: `scripts/test_isolation_forest.py`

```python
#!/usr/bin/env python3
"""Isolation Forest 모델 검증."""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np

def test_isolation_forest(model_path, scaler_path, metadata_path, test_data_path):
    """Isolation Forest 모델 검증."""
    
    print("="*70)
    print("Isolation Forest 모델 검증")
    print("="*70)
    
    # 1. 메타데이터 로드
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"\n[1/4] 메타데이터:")
    print(f"  버전: {metadata['metadata']['version']}")
    print(f"  학습 날짜: {metadata['metadata']['training_date']}")
    print(f"  피처 개수: {metadata['metadata']['n_features']}")
    print(f"  N Estimators: {metadata['hyperparameters']['n_estimators']}")
    print(f"  Contamination: {metadata['hyperparameters']['contamination']}")
    
    # 2. 모델 로드
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"\n[2/4] 모델 로드:")
    print(f"  모델 타입: {type(model).__name__}")
    print(f"  Scaler 타입: {type(scaler).__name__}")
    
    # 3. 테스트 데이터 로드
    test_df = pd.read_parquet(test_data_path)
    
    feature_cols = [col for col in test_df.columns 
                    if col not in ['timestamp', 'endpoint_id', 'is_anomaly']]
    
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 추론
    scores = model.decision_function(X_test_scaled)
    predictions = model.predict(X_test_scaled)
    
    print(f"\n[3/4] 추론 테스트:")
    print(f"  테스트 샘플 수: {len(test_df):,}")
    print(f"  Anomaly score 범위: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Anomaly score mean: {scores.mean():.4f}")
    print(f"  예측된 이상 개수: {(predictions == -1).sum()} / {len(predictions)}")
    
    # 샘플 출력
    print(f"\n[4/4] 샘플 예측:")
    for i in range(min(5, len(test_df))):
        pred_label = "이상" if predictions[i] == -1 else "정상"
        print(f"  샘플 {i+1}: score={scores[i]:.4f}, 예측={pred_label}")
    
    print("\n" + "="*70)
    print("✅ 모델 검증 완료!")
    print("="*70)


if __name__ == '__main__':
    model_path = Path('ocad/models/isolation_forest/isolation_forest_v1.0.0.pkl')
    scaler_path = Path('ocad/models/isolation_forest/isolation_forest_v1.0.0_scaler.pkl')
    metadata_path = Path('ocad/models/isolation_forest/isolation_forest_v1.0.0.json')
    test_data_path = Path('data/processed/multivariate_test.parquet')
    
    test_isolation_forest(model_path, scaler_path, metadata_path, test_data_path)
```

#### 3.2 실행

```bash
python scripts/test_isolation_forest.py
```

---

### Step 4: Phase 3 완료 확인 (10분)

```bash
# 생성된 파일 확인
ls -lh data/processed/multivariate_*.parquet
ls -lh ocad/models/isolation_forest/*

# 전체 모델 목록
echo "=== 학습된 모델 목록 ==="
echo ""
echo "TCN 모델 (3개):"
ls -lh ocad/models/tcn/*vv2.0.0.pth
echo ""
echo "Isolation Forest 모델 (1개):"
ls -lh ocad/models/isolation_forest/*.pkl
```

---

## ✅ 완료 체크리스트

Phase 3 완료 시 다음 항목들이 모두 체크되어야 합니다:

- [ ] `scripts/prepare_multivariate_data.py` 작성 완료
- [ ] 다변량 데이터 생성 완료 (train/val/test parquet 파일)
- [ ] `scripts/train_isolation_forest.py` 작성 완료
- [ ] Isolation Forest 모델 학습 완료 (.pkl 파일)
- [ ] Scaler 저장 완료 (_scaler.pkl 파일)
- [ ] 메타데이터 생성 완료 (.json 파일)
- [ ] `scripts/test_isolation_forest.py` 작성 완료
- [ ] 모델 검증 테스트 통과

---

## 📝 다음 단계 (Phase 4)

Phase 3 완료 후, Phase 4에서는 학습된 모델들을 실제 추론 파이프라인에 통합합니다.

**주요 작업**:
1. `ocad/detectors/residual.py` 수정 - TCN 모델 로드 기능
2. `ocad/detectors/multivariate.py` 수정 - Isolation Forest 로드 기능
3. `config/local.yaml` 업데이트 - 모델 경로 설정
4. 통합 테스트 및 성능 측정

**참고 문서**: 
- [docs/TODO.md](./TODO.md)
- [docs/Training-Inference-Separation-Design.md](./Training-Inference-Separation-Design.md)

---

## 🔧 트러블슈팅

### 문제 1: 메모리 부족
**해결**: `n_estimators` 줄이기 (100 → 50)

### 문제 2: 학습 시간 너무 오래 걸림
**해결**: 데이터 샘플링 또는 `n_jobs=-1` 설정 확인

### 문제 3: Contamination 파라미터 선택
**권장값**: 0.05 ~ 0.15 (데이터의 5-15%가 이상치라고 가정)

---

**작성일**: 2025-10-30  
**예정 실행일**: 2025-10-31  
**예상 소요 시간**: 1-2시간
