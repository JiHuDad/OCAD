# OCAD 학습-추론 분리 설계 (Training-Inference Separation Design)

## 📋 목차

1. [현재 시스템 분석](#1-현재-시스템-분석)
2. [학습-추론 분리의 목적과 이점](#2-학습-추론-분리의-목적과-이점)
3. [아키텍처 설계](#3-아키텍처-설계)
4. [데이터셋 요구사항 및 구조](#4-데이터셋-요구사항-및-구조)
5. [구현 계획](#5-구현-계획)
6. [성능 측정 지표](#6-성능-측정-지표)

---

## 1. 현재 시스템 분석

### 1.1 현재 학습-추론 구조

OCAD 시스템은 현재 **온라인 학습 방식**을 사용합니다:

```python
# ResidualDetector (ocad/detectors/residual.py)
class ResidualDetector:
    def detect(self, features):
        # 1. 실시간 데이터 수집
        self.history[metric_type].append(value)

        # 2. 온라인 학습: 50개 샘플 수집 시 자동 훈련
        if len(self.history[metric_type]) >= 50:
            self._train_model(metric_type)  # 추론 중 학습!

        # 3. 예측 (추론)
        prediction = self._predict(metric_type, sequence)
        residual = abs(value - prediction)
        return score

# MultivariateDetector (ocad/detectors/multivariate.py)
class MultivariateDetector:
    def detect(self, features, capabilities):
        # 1. 피처 이력 저장
        self.feature_history[group_key].append(features)

        # 2. 온라인 학습: 50개 샘플 수집 시 자동 훈련
        if len(self.feature_history[group_key]) >= 50:
            self._train_model(group_key, capabilities)  # 추론 중 학습!

        # 3. 예측 (추론)
        score = self._predict_anomaly(group_key, feature_array)
        return score
```

### 1.2 현재 방식의 문제점

| 문제점 | 영향 | 심각도 |
|--------|------|--------|
| **추론 지연** | 학습이 진행되는 동안 탐지 중단 (TCN: 20 epoch, Isolation Forest: 100 trees) | 🔴 높음 |
| **재현성 부족** | 동일한 조건에서도 학습 타이밍에 따라 결과가 달라짐 | 🟡 중간 |
| **평가 불가** | 사전에 성능을 측정하고 검증할 수 없음 | 🔴 높음 |
| **모델 버전 관리 불가** | 어떤 모델 버전이 사용되는지 추적 불가 | 🟡 중간 |
| **하이퍼파라미터 최적화 불가** | 학습률, 히든 유닛 수 등을 실험적으로 튜닝할 수 없음 | 🟠 중간-높음 |
| **메모리 누수 위험** | 무한정 이력 데이터 저장 (현재는 500-1000개로 제한) | 🟡 중간 |

### 1.3 현재 사용 중인 AI 모델

| 모델 | 탐지 방법 | 학습 방식 | 데이터 요구사항 |
|------|----------|----------|----------------|
| **TCN** (SimpleTCN) | 예측-잔차 | 온라인 (20 epoch) | 최소 50개 시계열 샘플 |
| **Isolation Forest** | 다변량 이상 | 온라인 (100 trees) | 최소 50개 다변량 피처 벡터 |
| **CUSUM** | 변화점 | 규칙 기반 (학습 불필요) | N/A |
| **Rule-based** | 임계값 | 규칙 기반 (학습 불필요) | N/A |

---

## 2. 학습-추론 분리의 목적과 이점

### 2.1 설계 목적

**핵심 목표**: 학습(Training)과 추론(Inference)을 완전히 분리하여 **안정적이고 예측 가능한 이상 탐지 시스템** 구축

```
현재 방식:
데이터 수집 → [학습 + 추론 혼재] → 탐지 결과

새로운 방식:
[오프라인 학습] → 훈련된 모델 저장 → [온라인 추론만] → 탐지 결과
```

### 2.2 주요 이점

#### 2.2.1 성능 및 안정성
- ✅ **일관된 추론 속도**: 학습 부담 제거로 예측 가능한 지연 시간 (<30초 목표 달성)
- ✅ **실시간 보장**: 추론만 수행하므로 탐지 중단 없음
- ✅ **메모리 효율**: 추론 시 모델 가중치만 메모리에 로드 (이력 데이터 불필요)

#### 2.2.2 개발 및 운영
- ✅ **모델 검증**: 배포 전 성능 측정 및 벤치마크 가능
- ✅ **버전 관리**: 모델 파일(`.pth`, `.pkl`) Git/MLflow 관리
- ✅ **롤백 가능**: 문제 발생 시 이전 모델로 즉시 복구
- ✅ **A/B 테스팅**: 여러 모델 버전 동시 배포 및 비교

#### 2.2.3 연구 및 개선
- ✅ **하이퍼파라미터 튜닝**: Grid Search, Bayesian Optimization 적용
- ✅ **데이터셋 확보**: 재사용 가능한 표준 데이터셋 구축
- ✅ **벤치마크**: 다양한 알고리즘 성능 정량적 비교
- ✅ **전이 학습**: 다른 네트워크 환경에 모델 재사용

---

## 3. 아키텍처 설계

### 3.1 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     OCAD Training-Inference Separation          │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐          ┌───────────────────────────────┐
│  OFFLINE TRAINING    │          │   ONLINE INFERENCE            │
│  (ocad/training/)    │          │   (ocad/detectors/)           │
└──────────────────────┘          └───────────────────────────────┘

[1. 데이터 수집]                   [1. 모델 로딩]
    ↓                                  ↓
Historical Data                    Trained Models
  - Real Network Logs                - tcn_udp_echo.pth
  - Simulated Data                   - tcn_ecpri.pth
  - Injected Anomalies               - tcn_lbm.pth
    ↓                                 - isolation_forest.pkl
                                       ↓
[2. 데이터셋 생성]                [2. 실시간 추론]
    ↓                                  ↓
ocad/training/datasets/            MetricSample
  - TimeSeriesDataset                  ↓
  - MultivariateDataset            FeatureEngine
  - AnomalyDataset                     ↓
    ↓                              FeatureVector
                                       ↓
[3. 모델 학습]                     Detectors (추론만)
    ↓                                  ↓
Training Scripts                   Anomaly Score
  - train_tcn.py                       ↓
  - train_isolation_forest.py      AlertManager
  - hyperparameter_tuning.py
    ↓

[4. 모델 평가]
    ↓
Metrics & Validation
  - Precision, Recall, F1
  - MTTD, Lead Time
  - ROC-AUC, Confusion Matrix
    ↓

[5. 모델 저장]
    ↓
models/
  - tcn_udp_echo_v1.0.0.pth
  - metadata.json
  - performance_report.json
```

### 3.2 디렉토리 구조

```
ocad/
├── training/                          # 새로 추가: 학습 전용 모듈
│   ├── __init__.py
│   ├── datasets/                      # 데이터셋 관리
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseDataset 추상 클래스
│   │   ├── timeseries_dataset.py      # TimeSeriesDataset (TCN용)
│   │   ├── multivariate_dataset.py    # MultivariateDataset (IF용)
│   │   └── data_loader.py             # 데이터 로딩 유틸리티
│   ├── trainers/                      # 학습 로직
│   │   ├── __init__.py
│   │   ├── base_trainer.py            # BaseTrainer 추상 클래스
│   │   ├── tcn_trainer.py             # TCN 학습기
│   │   └── isolation_forest_trainer.py
│   ├── evaluators/                    # 모델 평가
│   │   ├── __init__.py
│   │   ├── metrics.py                 # 평가 지표 계산
│   │   └── validator.py               # Cross-validation
│   └── utils/
│       ├── __init__.py
│       ├── model_saver.py             # 모델 저장/로드
│       └── hyperparameter_tuning.py   # 하이퍼파라미터 최적화
│
├── detectors/                         # 기존: 추론 전용으로 변경
│   ├── base.py                        # (수정) 추론 인터페이스만
│   ├── residual.py                    # (수정) 학습 코드 제거
│   ├── multivariate.py                # (수정) 학습 코드 제거
│   └── ...
│
├── models/                            # 새로 추가: 훈련된 모델 저장소
│   ├── tcn/
│   │   ├── udp_echo_v1.0.0.pth
│   │   ├── ecpri_v1.0.0.pth
│   │   └── lbm_v1.0.0.pth
│   ├── isolation_forest/
│   │   └── multivariate_v1.0.0.pkl
│   └── metadata/
│       ├── tcn_udp_echo_v1.0.0.json
│       └── performance_reports/
│
├── data/                              # 새로 추가: 학습 데이터셋
│   ├── raw/                           # 원시 데이터
│   │   ├── oran_logs_2024/
│   │   └── simulated_data/
│   ├── processed/                     # 전처리된 데이터
│   │   ├── timeseries_train.parquet
│   │   ├── timeseries_val.parquet
│   │   ├── timeseries_test.parquet
│   │   └── anomaly_labels.csv
│   └── synthetic/                     # 합성 데이터
│       └── injected_anomalies.parquet
│
└── scripts/
    ├── train_tcn_model.py             # 새로 추가: TCN 학습 스크립트
    ├── train_isolation_forest.py      # 새로 추가: IF 학습 스크립트
    ├── evaluate_models.py             # 새로 추가: 모델 평가 스크립트
    └── generate_training_data.py      # 새로 추가: 학습 데이터 생성
```

### 3.3 클래스 설계

#### 3.3.1 학습 파이프라인

```python
# ocad/training/trainers/base_trainer.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from pathlib import Path

class BaseTrainer(ABC):
    """모든 학습기의 기본 클래스"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.best_metrics = {}

    @abstractmethod
    def build_model(self) -> Any:
        """모델 아키텍처 구성"""
        pass

    @abstractmethod
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """1 에포크 학습"""
        pass

    @abstractmethod
    def validate(self, val_loader) -> Dict[str, float]:
        """검증 데이터로 평가"""
        pass

    def train(self, train_loader, val_loader, epochs: int):
        """전체 학습 프로세스"""
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            if self._is_best_model(val_metrics):
                self.save_checkpoint()

    def save_checkpoint(self, path: Path):
        """모델 체크포인트 저장"""
        pass

    def load_checkpoint(self, path: Path):
        """모델 체크포인트 로드"""
        pass


# ocad/training/trainers/tcn_trainer.py
class TCNTrainer(BaseTrainer):
    """TCN 모델 학습기"""

    def build_model(self):
        return SimpleTCN(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            output_size=1
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {"train_loss": total_loss / len(train_loader)}

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                val_loss += loss.item()
                predictions_list.extend(predictions.cpu().numpy())
                targets_list.extend(batch_y.cpu().numpy())

        # 평가 지표 계산
        mse = mean_squared_error(targets_list, predictions_list)
        mae = mean_absolute_error(targets_list, predictions_list)

        return {
            "val_loss": val_loss / len(val_loader),
            "mse": mse,
            "mae": mae
        }
```

#### 3.3.2 추론 파이프라인

```python
# ocad/detectors/residual.py (수정 버전)
class ResidualDetector(BaseDetector):
    """잔차 기반 탐지기 (추론 전용)"""

    def __init__(self, config, model_path: Optional[Path] = None):
        super().__init__(config)

        # 학습 코드 완전 제거
        # self.history = {}  # 삭제
        # self._train_model() # 삭제

        # 사전 훈련된 모델 로드
        self.models = self._load_pretrained_models(model_path)
        self.scalers = self._load_scalers(model_path)

        # 추론을 위한 최소 버퍼만 유지 (sequence_length만큼)
        self.inference_buffer = {
            "udp_echo": deque(maxlen=10),
            "ecpri": deque(maxlen=10),
            "lbm": deque(maxlen=10),
        }

    def _load_pretrained_models(self, model_path: Path) -> Dict:
        """사전 훈련된 모델 로드"""
        models = {}
        model_dir = model_path or Path("ocad/models/tcn/")

        for metric_type in ["udp_echo", "ecpri", "lbm"]:
            model_file = model_dir / f"{metric_type}_v1.0.0.pth"
            if model_file.exists():
                checkpoint = torch.load(model_file)
                model = SimpleTCN(
                    input_size=checkpoint['config']['input_size'],
                    hidden_size=checkpoint['config']['hidden_size'],
                    output_size=1
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()  # 추론 모드
                models[metric_type] = model

                self.logger.info(
                    "Pretrained model loaded",
                    metric_type=metric_type,
                    version=checkpoint['version'],
                    trained_on=checkpoint['metadata']['training_date']
                )

        return models

    def detect(self, features: FeatureVector, capabilities: Capabilities) -> float:
        """이상 탐지 (추론만 수행)"""
        residuals = []

        if capabilities.udp_echo and features.udp_echo_p95 is not None:
            residual = self._calculate_residual(
                "udp_echo",
                features.udp_echo_p95,
                features.endpoint_id
            )
            if residual is not None:
                residuals.append(residual)

        # ... (나머지 메트릭 동일)

        if not residuals:
            return 0.0

        max_residual = max(residuals)
        score = min(1.0, max_residual / self.config.residual_threshold)

        return score

    def _calculate_residual(self, metric_type: str, value: float, endpoint_id: str) -> Optional[float]:
        """잔차 계산 (추론만)"""
        # 추론 버퍼에 추가
        self.inference_buffer[metric_type].append(value)

        # 시퀀스 길이만큼 데이터가 없으면 대기
        if len(self.inference_buffer[metric_type]) < 10:
            return None

        # 모델이 없으면 사용 불가
        if metric_type not in self.models:
            self.logger.warning(
                "No pretrained model available",
                metric_type=metric_type
            )
            return None

        try:
            # 예측 수행 (학습 없음!)
            sequence = list(self.inference_buffer[metric_type])[-10:]
            prediction = self._predict(metric_type, sequence)

            if prediction is not None:
                residual = abs(value - prediction)

                # 정규화
                recent_values = list(self.inference_buffer[metric_type])[-20:]
                if len(recent_values) > 3:
                    std_dev = np.std(recent_values)
                    if std_dev > 0:
                        return residual / std_dev

                return residual

        except Exception as e:
            self.logger.error(
                "Inference failed",
                metric_type=metric_type,
                error=str(e)
            )

        return None

    def _predict(self, metric_type: str, sequence: List[float]) -> Optional[float]:
        """예측 수행 (추론 전용)"""
        model = self.models[metric_type]
        scaler = self.scalers[metric_type]

        # 정규화
        sequence_array = np.array(sequence).reshape(-1, 1)
        sequence_scaled = scaler.transform(sequence_array).flatten()

        # 예측
        with torch.no_grad():
            x_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).unsqueeze(0)
            prediction_scaled = model(x_tensor).item()

        # 역정규화
        prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

        return prediction
```

---

## 4. 데이터셋 요구사항 및 구조

### 4.1 측정 지표별 데이터셋 요구사항

학습-추론 분리를 위해서는 각 탐지 방법론에 맞는 **표준 데이터셋**이 필요합니다.

#### 4.1.1 TCN 시계열 예측 데이터셋

**목적**: 시계열 값을 예측하여 잔차로 이상 탐지

**데이터 구조**:
```python
{
    "metric_type": "udp_echo" | "ecpri" | "lbm",
    "endpoint_id": "o-ru-001",
    "timestamp_ms": 1234567890000,
    "sequence": [5.1, 5.3, 5.2, 5.4, 5.3, 5.5, 5.4, 5.6, 5.5, 5.7],  # 10개 시계열
    "target": 5.8,  # 다음 값
    "is_anomaly": false,  # 라벨 (선택)
    "anomaly_type": null | "spike" | "drift" | "loss"  # 이상 유형
}
```

**수집 방법**:
1. **실제 네트워크 로그** (최우선)
   - 정상 운영 데이터: 80%
   - 실제 장애 데이터: 20%

2. **시뮬레이터 생성 데이터**
   - `ocad/utils/simulator.py` 활용
   - 정상 패턴: 평균 5ms, 표준편차 0.5ms
   - 이상 패턴 주입: 스파이크, 드리프트, 패킷 손실

3. **합성 이상 주입**
   ```python
   # 스파이크: 급격한 지연 증가
   spike_pattern = normal_data.copy()
   spike_pattern[50:60] += 20  # 10ms → 30ms

   # 드리프트: 점진적 성능 저하
   drift_pattern = normal_data.copy()
   drift_pattern += np.linspace(0, 10, len(drift_pattern))

   # 패킷 손실: 불규칙한 값
   loss_pattern = normal_data.copy()
   loss_pattern[::5] = np.nan  # 20% 손실
   ```

**데이터셋 크기**:
- 훈련 데이터: 최소 10,000 시퀀스 (약 100,000 샘플)
- 검증 데이터: 2,000 시퀀스
- 테스트 데이터: 2,000 시퀀스
- **이상 비율**: 5-10% (실제 네트워크 환경 반영)

#### 4.1.2 Isolation Forest 다변량 데이터셋

**목적**: 여러 메트릭 간 복합 패턴으로 이상 탐지

**데이터 구조**:
```python
{
    "endpoint_id": "o-ru-001",
    "timestamp_ms": 1234567890000,
    "features": {
        # UDP Echo 피처
        "udp_echo_p95": 5.2,
        "udp_echo_p99": 5.8,
        "udp_echo_slope": 0.01,
        "cusum_udp_echo": 0.3,

        # eCPRI 피처
        "ecpri_p95": 98.5,
        "ecpri_p99": 102.3,
        "ecpri_slope": -0.02,
        "cusum_ecpri": 0.1,

        # LBM 피처
        "lbm_rtt_p95": 8.1,
        "lbm_rtt_p99": 8.6,
        "lbm_slope": 0.005,
        "cusum_lbm": 0.2,
        "lbm_fail_runlen": 0
    },
    "is_anomaly": false,
    "anomaly_type": null | "concurrent" | "correlated"
}
```

**수집 방법**:
1. **FeatureEngine 출력 수집**
   - `ocad/features/engine.py`에서 생성된 FeatureVector 저장
   - 5분 윈도우마다 1개 피처 벡터 생성

2. **다변량 이상 패턴**
   ```python
   # 동시성 이상: 여러 메트릭이 동시에 증가
   concurrent_anomaly = {
       "udp_echo_p95": 15.0,  # 정상 5.0 → 이상 15.0
       "ecpri_p95": 250.0,    # 정상 100.0 → 이상 250.0
       "lbm_rtt_p95": 20.0,   # 정상 8.0 → 이상 20.0
   }

   # 상관관계 이상: UDP는 정상, eCPRI만 증가 (비정상 패턴)
   correlated_anomaly = {
       "udp_echo_p95": 5.2,   # 정상
       "ecpri_p95": 300.0,    # 이상 (UDP 정상인데 eCPRI만 높음)
       "lbm_rtt_p95": 8.1,    # 정상
   }
   ```

**데이터셋 크기**:
- 훈련 데이터: 최소 5,000 피처 벡터
- 검증 데이터: 1,000 피처 벡터
- 테스트 데이터: 1,000 피처 벡터

### 4.2 데이터셋 파일 포맷

#### 4.2.1 시계열 데이터셋 (Parquet)

```python
# data/processed/timeseries_train.parquet
import pandas as pd

df = pd.DataFrame({
    'endpoint_id': ['o-ru-001', 'o-ru-001', ...],
    'metric_type': ['udp_echo', 'udp_echo', ...],
    'timestamp_ms': [1234567890000, 1234567890100, ...],
    'sequence': [  # List[float] 저장 가능
        [5.1, 5.3, 5.2, 5.4, 5.3, 5.5, 5.4, 5.6, 5.5, 5.7],
        [5.3, 5.2, 5.4, 5.3, 5.5, 5.4, 5.6, 5.5, 5.7, 5.8],
        ...
    ],
    'target': [5.8, 5.9, ...],
    'is_anomaly': [False, False, True, ...],
    'anomaly_type': [None, None, 'spike', ...]
})

df.to_parquet('data/processed/timeseries_train.parquet')
```

**장점**:
- 빠른 읽기/쓰기 속도
- 압축 효율 (CSV 대비 10배)
- List 타입 저장 가능
- Pandas와 완벽 호환

#### 4.2.2 라벨 데이터 (CSV)

```csv
# data/processed/anomaly_labels.csv
endpoint_id,timestamp_ms,is_anomaly,anomaly_type,severity,description
o-ru-001,1234567890000,true,spike,critical,급격한 UDP Echo 지연 증가 (5ms → 30ms)
o-ru-002,1234567891000,true,drift,warning,점진적 eCPRI 지연 증가 (100μs → 200μs)
o-du-001,1234567892000,true,loss,critical,LBM 패킷 손실 50%
```

### 4.3 데이터셋 생성 스크립트

```python
# scripts/generate_training_data.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from ocad.utils.simulator import VirtualEndpoint
from ocad.features.engine import FeatureEngine
from ocad.core.config import Settings

def generate_timeseries_dataset(
    num_endpoints: int = 10,
    duration_hours: int = 24,
    anomaly_rate: float = 0.1,
    output_dir: Path = Path("data/processed")
):
    """시계열 학습 데이터셋 생성"""

    sequences = []

    for endpoint_id in range(num_endpoints):
        # 가상 엔드포인트 생성
        endpoint = VirtualEndpoint(
            endpoint_id=f"o-ru-{endpoint_id:03d}",
            role="o-ru"
        )

        # 정상 데이터 생성
        for hour in range(duration_hours):
            for minute in range(0, 60, 5):  # 5분마다
                # 메트릭 수집
                samples = []
                for _ in range(10):  # 10개 시퀀스
                    sample = endpoint.collect_metrics()
                    samples.append(sample['udp_echo_rtt_ms'])

                target = endpoint.collect_metrics()['udp_echo_rtt_ms']

                # 이상 주입
                is_anomaly = np.random.random() < anomaly_rate
                if is_anomaly:
                    anomaly_type = np.random.choice(['spike', 'drift', 'loss'])
                    target = inject_anomaly(target, anomaly_type)
                else:
                    anomaly_type = None

                sequences.append({
                    'endpoint_id': endpoint.endpoint_id,
                    'metric_type': 'udp_echo',
                    'timestamp_ms': int(time.time() * 1000),
                    'sequence': samples,
                    'target': target,
                    'is_anomaly': is_anomaly,
                    'anomaly_type': anomaly_type
                })

    # DataFrame 저장
    df = pd.DataFrame(sequences)

    # Train/Val/Test 분할 (70/15/15)
    train_df = df.sample(frac=0.7, random_state=42)
    remaining = df.drop(train_df.index)
    val_df = remaining.sample(frac=0.5, random_state=42)
    test_df = remaining.drop(val_df.index)

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "timeseries_train.parquet")
    val_df.to_parquet(output_dir / "timeseries_val.parquet")
    test_df.to_parquet(output_dir / "timeseries_test.parquet")

    print(f"✅ Generated {len(df)} sequences")
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"   Anomaly rate: {df['is_anomaly'].mean():.1%}")

def inject_anomaly(value: float, anomaly_type: str) -> float:
    """이상 패턴 주입"""
    if anomaly_type == 'spike':
        return value + np.random.uniform(15, 25)  # 급격한 증가
    elif anomaly_type == 'drift':
        return value + np.random.uniform(5, 10)   # 점진적 증가
    elif anomaly_type == 'loss':
        return np.nan if np.random.random() < 0.5 else value
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", type=int, default=10)
    parser.add_argument("--duration-hours", type=int, default=24)
    parser.add_argument("--anomaly-rate", type=float, default=0.1)
    args = parser.parse_args()

    generate_timeseries_dataset(
        num_endpoints=args.endpoints,
        duration_hours=args.duration_hours,
        anomaly_rate=args.anomaly_rate
    )
```

---

## 5. 구현 계획

### 5.1 Phase 1: 인프라 구축 (Week 1-2)

**목표**: 학습-추론 분리를 위한 기본 인프라 구축

#### Task 1.1: 디렉토리 구조 생성
```bash
mkdir -p ocad/training/{datasets,trainers,evaluators,utils}
mkdir -p ocad/models/{tcn,isolation_forest,metadata}
mkdir -p data/{raw,processed,synthetic}
touch ocad/training/__init__.py
touch ocad/training/datasets/__init__.py
touch ocad/training/trainers/__init__.py
```

#### Task 1.2: BaseTrainer 구현
- 파일: `ocad/training/trainers/base_trainer.py`
- 내용: 추상 클래스, 학습 루프, 체크포인트 저장/로드
- 테스트: `tests/unit/test_base_trainer.py`

#### Task 1.3: 데이터셋 생성 스크립트
- 파일: `scripts/generate_training_data.py`
- 기능:
  - 시뮬레이터로 10개 엔드포인트 × 24시간 데이터 생성
  - 이상 패턴 주입 (spike, drift, loss)
  - Train/Val/Test 분할 (70/15/15)
  - Parquet 형식 저장

#### Task 1.4: 모델 저장/로드 유틸리티
- 파일: `ocad/training/utils/model_saver.py`
- 기능:
  ```python
  class ModelSaver:
      def save_model(self, model, path, metadata):
          """모델 + 메타데이터 저장"""
          torch.save({
              'model_state_dict': model.state_dict(),
              'config': model.config,
              'version': metadata['version'],
              'training_date': metadata['training_date'],
              'performance': metadata['performance']
          }, path)

      def load_model(self, model_class, path):
          """모델 로드"""
          checkpoint = torch.load(path)
          model = model_class(**checkpoint['config'])
          model.load_state_dict(checkpoint['model_state_dict'])
          return model, checkpoint['metadata']
  ```

**산출물**:
- ✅ 학습 인프라 코드
- ✅ 샘플 데이터셋 (10,000 시퀀스)
- ✅ 단위 테스트

---

### 5.2 Phase 2: TCN 모델 학습-추론 분리 (Week 3-4)

**목표**: ResidualDetector를 학습-추론 분리 아키텍처로 전환

#### Task 2.1: TCNTrainer 구현
- 파일: `ocad/training/trainers/tcn_trainer.py`
- 기능:
  - SimpleTCN 학습
  - 하이퍼파라미터: learning_rate, hidden_size, num_epochs
  - Early stopping (validation loss 기준)
  - 모델 저장 (`.pth` 형식)

#### Task 2.2: TCN 학습 스크립트
- 파일: `scripts/train_tcn_model.py`
- 사용법:
  ```bash
  python scripts/train_tcn_model.py \
      --metric-type udp_echo \
      --data-path data/processed/timeseries_train.parquet \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate 0.001 \
      --hidden-size 32 \
      --output models/tcn/udp_echo_v1.0.0.pth
  ```

#### Task 2.3: ResidualDetector 추론 전용 변환
- 파일: `ocad/detectors/residual.py`
- 변경사항:
  - ❌ `self.history` 삭제
  - ❌ `_train_model()` 메서드 삭제
  - ✅ `_load_pretrained_models()` 추가
  - ✅ `inference_buffer` (sequence_length만큼만 유지)
  - ✅ `detect()` 메서드에서 학습 로직 제거

#### Task 2.4: 모델 평가
- 파일: `scripts/evaluate_models.py`
- 평가 지표:
  - MSE, MAE (예측 정확도)
  - Precision, Recall, F1 (이상 탐지 성능)
  - Lead Time (조기 경고 시간)
  - Inference Latency (추론 지연)

**산출물**:
- ✅ 훈련된 TCN 모델 (udp_echo, ecpri, lbm)
- ✅ 성능 리포트 (performance_report.json)
- ✅ 추론 전용 ResidualDetector
- ✅ 벤치마크 결과

---

### 5.3 Phase 3: Isolation Forest 학습-추론 분리 (Week 5)

**목표**: MultivariateDetector를 학습-추론 분리 아키텍처로 전환

#### Task 3.1: 다변량 데이터셋 생성
- 파일: `scripts/generate_multivariate_data.py`
- 기능:
  - FeatureEngine으로 피처 벡터 생성
  - 다변량 이상 패턴 주입 (concurrent, correlated)
  - 5,000개 피처 벡터 생성

#### Task 3.2: IsolationForestTrainer 구현
- 파일: `ocad/training/trainers/isolation_forest_trainer.py`
- 기능:
  - Scikit-learn IsolationForest 학습
  - Cross-validation (5-fold)
  - 모델 저장 (`.pkl` 형식)

#### Task 3.3: MultivariateDetector 추론 전용 변환
- 파일: `ocad/detectors/multivariate.py`
- 변경사항:
  - ❌ `feature_history` 삭제
  - ❌ `_train_model()` 메서드 삭제
  - ✅ `_load_pretrained_model()` 추가
  - ✅ 추론만 수행

**산출물**:
- ✅ 훈련된 Isolation Forest 모델
- ✅ 추론 전용 MultivariateDetector

---

### 5.4 Phase 4: 통합 및 검증 (Week 6)

**목표**: 전체 시스템 통합 및 성능 검증

#### Task 4.1: 모델 버전 관리
- Git LFS 설정 (대용량 모델 파일)
  ```bash
  git lfs install
  git lfs track "*.pth"
  git lfs track "*.pkl"
  ```

#### Task 4.2: 하이퍼파라미터 튜닝
- 파일: `ocad/training/utils/hyperparameter_tuning.py`
- 라이브러리: Optuna
- 튜닝 대상:
  - TCN: learning_rate, hidden_size, num_layers
  - Isolation Forest: n_estimators, contamination

#### Task 4.3: A/B 테스팅 프레임워크
- 여러 모델 버전 동시 배포
- 성능 비교 (MTTD, FAR, Lead Time)

#### Task 4.4: 문서 업데이트
- CLAUDE.md에 학습-추론 분리 설명 추가
- README.md에 학습 명령어 추가
- API.md에 모델 관리 API 추가

**산출물**:
- ✅ 최적화된 모델 (v1.1.0)
- ✅ A/B 테스트 결과
- ✅ 완전한 문서화

---

### 5.5 Phase 5: 자동화 및 MLOps (Week 7-8)

**목표**: 학습-배포 자동화 파이프라인 구축

#### Task 5.1: MLflow 통합 (선택)
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("hidden_size", 32)

    # 학습
    trainer.train(train_loader, val_loader, epochs=50)

    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("f1_score", f1)

    # 모델 저장
    mlflow.pytorch.log_model(model, "tcn_model")
```

#### Task 5.2: CI/CD 파이프라인
```yaml
# .github/workflows/train_models.yml
name: Train Models

on:
  schedule:
    - cron: '0 0 * * 0'  # 매주 일요일

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Generate training data
        run: python scripts/generate_training_data.py

      - name: Train TCN models
        run: |
          python scripts/train_tcn_model.py --metric-type udp_echo
          python scripts/train_tcn_model.py --metric-type ecpri
          python scripts/train_tcn_model.py --metric-type lbm

      - name: Evaluate models
        run: python scripts/evaluate_models.py

      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: trained-models
          path: ocad/models/
```

**산출물**:
- ✅ MLflow 실험 추적
- ✅ 자동 학습 파이프라인
- ✅ 모델 성능 대시보드

---

## 6. 성능 측정 지표

### 6.1 모델 평가 지표

#### 6.1.1 예측 정확도 (TCN)

| 지표 | 설명 | 목표값 |
|------|------|--------|
| **MSE** (Mean Squared Error) | 예측 오차 제곱의 평균 | < 1.0 ms² |
| **MAE** (Mean Absolute Error) | 예측 오차 절대값의 평균 | < 0.5 ms |
| **RMSE** (Root MSE) | MSE의 제곱근 | < 1.0 ms |
| **R² Score** | 결정 계수 (설명력) | > 0.85 |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_tcn(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2
    }
```

#### 6.1.2 이상 탐지 성능

| 지표 | 설명 | 목표값 |
|------|------|--------|
| **Precision** | 정밀도 (탐지 중 실제 이상 비율) | > 0.90 |
| **Recall** | 재현율 (실제 이상 중 탐지 비율) | > 0.85 |
| **F1-Score** | Precision과 Recall의 조화평균 | > 0.87 |
| **FAR** (False Alarm Rate) | 오탐률 | < 0.06 |
| **ROC-AUC** | ROC 곡선 아래 면적 | > 0.92 |

```python
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def evaluate_anomaly_detection(y_true, y_pred, y_scores):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    far = (y_pred == 1).sum() / len(y_pred) - precision
    roc_auc = roc_auc_score(y_true, y_scores)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alarm_rate": far,
        "roc_auc": roc_auc
    }
```

#### 6.1.3 운영 지표

| 지표 | 설명 | 목표값 |
|------|------|--------|
| **MTTD** (Mean Time To Detect) | 평균 탐지 시간 | < 20초 (룰 대비 30% 단축) |
| **Lead Time** | 조기 경고 시간 | ≥ 4분 (P50) |
| **Inference Latency** | 추론 지연 시간 | < 100ms (P95) |
| **Model Size** | 모델 파일 크기 | < 10MB |
| **Memory Usage** | 추론 시 메모리 사용량 | < 100MB |

```python
import time
import psutil

def benchmark_inference(detector, feature_vectors):
    latencies = []

    for features in feature_vectors:
        start = time.time()
        score = detector.detect(features)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

    return {
        "mean_latency_ms": np.mean(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "memory_usage_mb": memory_mb
    }
```

### 6.2 성능 리포트 생성

```python
# scripts/evaluate_models.py
def generate_performance_report(model_path, test_data):
    """성능 리포트 생성"""

    report = {
        "model_info": {
            "name": "TCN UDP Echo Predictor",
            "version": "v1.0.0",
            "training_date": "2024-01-15",
            "architecture": "SimpleTCN (3 layers, 32 hidden units)",
            "parameters": 3456
        },
        "prediction_accuracy": {
            "mse": 0.85,
            "mae": 0.42,
            "rmse": 0.92,
            "r2_score": 0.89
        },
        "anomaly_detection": {
            "precision": 0.92,
            "recall": 0.87,
            "f1_score": 0.89,
            "false_alarm_rate": 0.05,
            "roc_auc": 0.94
        },
        "operational_metrics": {
            "mttd_seconds": 18,
            "lead_time_minutes": 4.2,
            "inference_latency_p95_ms": 85,
            "model_size_mb": 0.3,
            "memory_usage_mb": 45
        },
        "dataset_info": {
            "train_samples": 7000,
            "val_samples": 1500,
            "test_samples": 1500,
            "anomaly_rate": 0.08
        }
    }

    # JSON 저장
    with open(f"{model_path}/performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
```

---

## 7. 마이그레이션 전략

### 7.1 점진적 전환 (Gradual Migration)

기존 시스템을 중단하지 않고 단계적으로 전환합니다.

```python
# ocad/detectors/residual.py
class ResidualDetector(BaseDetector):
    def __init__(self, config, model_path: Optional[Path] = None, use_pretrained: bool = False):
        """
        Args:
            use_pretrained: True이면 사전 훈련 모델 사용, False이면 기존 온라인 학습
        """
        super().__init__(config)

        if use_pretrained and model_path:
            # 새로운 방식: 사전 훈련된 모델 로드
            self.models = self._load_pretrained_models(model_path)
            self.mode = "inference_only"
        else:
            # 기존 방식: 온라인 학습
            self.models = {"udp_echo": None, "ecpri": None, "lbm": None}
            self.history = {"udp_echo": [], "ecpri": [], "lbm": []}
            self.mode = "online_training"

        self.logger.info(f"ResidualDetector initialized in {self.mode} mode")
```

**설정 플래그**:
```yaml
# config/local.yaml
detection:
  residual:
    use_pretrained_models: true  # false로 설정하면 기존 방식 유지
    model_path: "ocad/models/tcn/"
```

### 7.2 비교 테스트

```python
# scripts/compare_modes.py
def compare_online_vs_pretrained():
    """온라인 학습 vs 사전 훈련 모델 비교"""

    # 테스트 데이터 로드
    test_data = load_test_data("data/processed/timeseries_test.parquet")

    # 두 방식으로 탐지
    detector_online = ResidualDetector(config, use_pretrained=False)
    detector_pretrained = ResidualDetector(config, use_pretrained=True)

    results = {
        "online": evaluate(detector_online, test_data),
        "pretrained": evaluate(detector_pretrained, test_data)
    }

    # 비교 리포트
    print(f"Online Learning - F1: {results['online']['f1']:.3f}, Latency: {results['online']['latency_p95']:.1f}ms")
    print(f"Pretrained - F1: {results['pretrained']['f1']:.3f}, Latency: {results['pretrained']['latency_p95']:.1f}ms")

    return results
```

---

## 8. 요약

### 8.1 핵심 변화

| 구분 | 기존 (온라인 학습) | 개선 (학습-추론 분리) |
|------|-------------------|---------------------|
| **학습 시점** | 추론 중 (50개 샘플마다) | 사전 오프라인 학습 |
| **추론 지연** | 불규칙 (학습 시 증가) | 일정 (< 100ms) |
| **모델 검증** | 불가능 | 배포 전 완전 검증 |
| **재현성** | 낮음 (타이밍 의존) | 높음 (동일 모델) |
| **하이퍼파라미터** | 고정 (실험 불가) | 최적화 가능 |
| **메모리 사용** | 높음 (이력 저장) | 낮음 (모델만) |
| **배포 유연성** | 낮음 | 높음 (버전 관리) |

### 8.2 기대 효과

1. **성능 향상**
   - 추론 지연 50% 감소 (학습 제거)
   - 메모리 사용 70% 감소 (이력 데이터 불필요)

2. **품질 보증**
   - 배포 전 성능 측정 가능
   - 표준 데이터셋으로 일관된 평가

3. **운영 효율**
   - 모델 롤백으로 빠른 장애 복구
   - A/B 테스팅으로 최적 모델 선택

4. **연구 활성화**
   - 하이퍼파라미터 최적화
   - 새로운 알고리즘 실험 용이

---

## 9. 참고 자료

### 9.1 관련 문서
- `docs/AI-Models-Guide.md`: 현재 사용 중인 AI 모델 설명
- `CLAUDE.md`: 프로젝트 전체 가이드
- `README.md`: 시스템 개요

### 9.2 주요 코드 파일
- `ocad/detectors/residual.py`: TCN 기반 잔차 탐지기
- `ocad/detectors/multivariate.py`: Isolation Forest 다변량 탐지기
- `ocad/features/engine.py`: 피처 추출 엔진
- `ocad/utils/simulator.py`: 데이터 생성용 시뮬레이터

### 9.3 외부 라이브러리
- PyTorch: https://pytorch.org/
- Scikit-learn: https://scikit-learn.org/
- Optuna (하이퍼파라미터 튜닝): https://optuna.org/
- MLflow (실험 추적): https://mlflow.org/

---

**작성일**: 2024-01-15
**버전**: 1.0.0
**작성자**: OCAD Development Team
