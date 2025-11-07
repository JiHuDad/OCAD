# OCAD 플러그인 개발 가이드

> **최종 업데이트**: 2025-11-05 (Phase 4)
> **대상 독자**: 플러그인 개발자, 시스템 확장 담당자
> **소요 시간**: 30-45분

## 목차

1. [개발 환경 설정](#개발-환경-설정)
2. [새 프로토콜 어댑터 개발](#새-프로토콜-어댑터-개발)
3. [새 탐지기 개발](#새-탐지기-개발)
4. [테스트 작성](#테스트-작성)
5. [플러그인 배포](#플러그인-배포)
6. [성능 최적화](#성능-최적화)
7. [예제 코드](#예제-코드)

---

## 개발 환경 설정

### 필수 요구사항

```bash
# Python 3.9+
python --version

# 가상 환경 활성화
source .venv/bin/activate

# 개발 의존성 설치
pip install -r requirements-dev.txt

# 프로젝트 루트에서 작업
cd /home/user/OCAD
```

### 디렉토리 구조

```
ocad/plugins/
├── base.py                    # 기본 인터페이스
├── registry.py                # 플러그인 레지스트리
├── protocol_adapters/         # 프로토콜 어댑터
│   ├── cfm/
│   │   ├── __init__.py
│   │   └── (implementation files)
│   ├── bfd/
│   ├── bgp/
│   └── ptp/
└── detectors/                 # 탐지기
    ├── lstm/
    │   ├── __init__.py
    │   └── (implementation files)
    ├── hmm/
    ├── gnn/
    └── tcn/
```

---

## 새 프로토콜 어댑터 개발

### 1단계: 인터페이스 이해

모든 프로토콜 어댑터는 `ProtocolAdapter` 추상 클래스를 상속해야 합니다:

```python
from ocad.plugins.base import ProtocolAdapter
from typing import AsyncIterator, Dict, Any, List

class MyProtocolAdapter(ProtocolAdapter):
    @property
    def name(self) -> str:
        """프로토콜 이름 (예: 'ospf', 'isis')"""
        return "my-protocol"

    @property
    def version(self) -> str:
        """버전 (semantic versioning)"""
        return "1.0.0"

    @property
    def supported_metrics(self) -> List[str]:
        """지원하는 메트릭 목록"""
        return ["metric1", "metric2", "metric3"]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """설정 검증"""
        required_keys = ["endpoints", "interval_sec"]
        return all(key in config for key in required_keys)

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """메트릭 수집 (async generator)"""
        while True:
            # 메트릭 수집 로직
            yield {
                "timestamp": datetime.utcnow(),
                "source_id": "endpoint-1",
                "metric_name": "metric1",
                "value": 10.5,
                "metadata": {}
            }
            await asyncio.sleep(config["interval_sec"])

    def get_recommended_models(self) -> List[str]:
        """권장 AI 모델"""
        return ["lstm", "tcn"]
```

### 2단계: 실제 구현 예제 - OSPF 어댑터

**파일**: `ocad/plugins/protocol_adapters/ospf/__init__.py`

```python
"""OSPF Protocol Adapter for OCAD.

Monitors OSPF routing protocol:
- Neighbor state changes
- LSA updates
- Route flapping
- Link state database changes
"""

import asyncio
from datetime import datetime
from typing import AsyncIterator, Dict, Any, List
import logging

from ocad.plugins.base import ProtocolAdapter

logger = logging.getLogger(__name__)


class OSPFAdapter(ProtocolAdapter):
    """OSPF protocol adapter."""

    @property
    def name(self) -> str:
        return "ospf"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "neighbor_state",      # 0=Down, 1=Init, 2=2-Way, 3=ExStart, 4=Exchange, 5=Loading, 6=Full
            "hello_interval_sec",  # Hello interval
            "dead_interval_sec",   # Dead interval
            "priority",            # Router priority
            "lsa_count",           # Number of LSAs
            "route_count",         # Number of routes
            "flap_count",          # Neighbor flap count
        ]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate OSPF adapter configuration.

        Required config:
        {
            "routers": [
                {
                    "id": "router-1",
                    "ip": "192.168.1.1",
                    "area": "0.0.0.0",
                    "router_id": "1.1.1.1"
                }
            ],
            "interval_sec": 1
        }
        """
        if "routers" not in config:
            raise ValueError("Missing 'routers' in config")

        if "interval_sec" not in config:
            raise ValueError("Missing 'interval_sec' in config")

        for router in config["routers"]:
            required = ["id", "ip", "area", "router_id"]
            if not all(key in router for key in required):
                raise ValueError(f"Router config missing required keys: {required}")

        return True

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Collect OSPF metrics.

        Yields:
            Metric dictionaries with timestamp, source_id, metric_name, value, metadata
        """
        self.validate_config(config)

        interval_sec = config["interval_sec"]
        routers = config["routers"]

        logger.info(f"Starting OSPF collection for {len(routers)} routers")

        # Track state for flap detection
        neighbor_states = {}

        while True:
            for router in routers:
                router_id = router["id"]

                # TODO: Replace with actual NETCONF/SNMP collection
                # For now, simulate metric collection
                metrics = await self._collect_from_router(router)

                # Detect flapping
                current_state = metrics.get("neighbor_state", 0)
                if router_id in neighbor_states:
                    if neighbor_states[router_id] != current_state:
                        metrics["flap_count"] = metrics.get("flap_count", 0) + 1
                neighbor_states[router_id] = current_state

                # Yield each metric
                timestamp = datetime.utcnow()
                for metric_name, value in metrics.items():
                    yield {
                        "timestamp": timestamp,
                        "source_id": router_id,
                        "metric_name": metric_name,
                        "value": float(value),
                        "metadata": {
                            "router_ip": router["ip"],
                            "area": router["area"],
                            "protocol": "ospf",
                        }
                    }

            await asyncio.sleep(interval_sec)

    async def _collect_from_router(self, router: Dict[str, Any]) -> Dict[str, float]:
        """Collect metrics from a single router.

        In production, this would:
        1. Connect via NETCONF/SNMP
        2. Query OSPF neighbor state
        3. Parse LSA database
        4. Return metrics

        For now, returns simulated data.
        """
        # TODO: Implement actual collection
        # Example: Use ncclient for NETCONF
        import random

        return {
            "neighbor_state": random.choice([0, 2, 6]),  # Down, 2-Way, Full
            "hello_interval_sec": 10.0,
            "dead_interval_sec": 40.0,
            "priority": 1.0,
            "lsa_count": random.randint(50, 100),
            "route_count": random.randint(100, 200),
            "flap_count": 0.0,
        }

    def get_recommended_models(self) -> List[str]:
        """OSPF benefits from HMM (state transitions) and LSTM (time series)."""
        return ["hmm", "lstm"]


# Register the adapter
adapter = OSPFAdapter()
```

### 3단계: 플러그인 등록

`__init__.py`에 `adapter` 또는 `detector` 변수를 정의하면 자동으로 레지스트리에 등록됩니다:

```python
# ocad/plugins/protocol_adapters/ospf/__init__.py
adapter = OSPFAdapter()  # 이 변수 이름 중요!
```

### 4단계: 설정 파일 추가

`config/plugins.yaml`에 설정 예제 추가:

```yaml
protocol_adapters:
  ospf:
    enabled: true
    config:
      routers:
        - id: "core-router-1"
          ip: "192.168.1.1"
          area: "0.0.0.0"
          router_id: "1.1.1.1"

        - id: "core-router-2"
          ip: "192.168.1.2"
          area: "0.0.0.0"
          router_id: "2.2.2.2"

      interval_sec: 1
```

---

## 새 탐지기 개발

### 1단계: 인터페이스 이해

모든 탐지기는 `DetectorPlugin` 추상 클래스를 상속해야 합니다:

```python
from ocad.plugins.base import DetectorPlugin
from typing import Dict, Any, List

class MyDetector(DetectorPlugin):
    @property
    def name(self) -> str:
        return "my-detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_protocols(self) -> List[str]:
        return ["ospf", "isis"]

    def train(self, data: Any, **kwargs) -> None:
        """모델 학습 (offline)"""
        pass

    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """이상 탐지 (online inference)"""
        return {
            "score": 0.5,
            "is_anomaly": False,
            "confidence": 0.9,
            "detector_name": self.name
        }

    def save_model(self, path: str) -> None:
        """모델 저장"""
        pass

    def load_model(self, path: str) -> None:
        """모델 로드"""
        pass
```

### 2단계: 실제 구현 예제 - 간단한 통계 기반 탐지기

**파일**: `ocad/plugins/detectors/statistical/__init__.py`

```python
"""Statistical Anomaly Detector.

Simple z-score based anomaly detection.
No ML training required - purely statistical.
"""

import numpy as np
from collections import deque
from typing import Dict, Any, List
import pickle
import logging

from ocad.plugins.base import DetectorPlugin

logger = logging.getLogger(__name__)


class StatisticalDetector(DetectorPlugin):
    """Z-score based statistical anomaly detector."""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.metric_windows = {}  # metric_name -> deque of values

    @property
    def name(self) -> str:
        return "statistical"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_protocols(self) -> List[str]:
        # Works with any protocol
        return ["cfm", "bfd", "bgp", "ptp", "ospf", "isis"]

    def train(self, data: Any, **kwargs) -> None:
        """No training needed for statistical detector.

        Optionally, can use data to initialize windows.
        """
        logger.info("Statistical detector doesn't require training")
        pass

    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies using z-score.

        Args:
            features: {
                "timestamp": datetime,
                "source_id": str,
                "metric_name": str,
                "value": float,
                "metadata": dict
            }

        Returns:
            {
                "score": float,  # 0.0-1.0
                "is_anomaly": bool,
                "confidence": float,
                "z_score": float,
                "detector_name": str
            }
        """
        metric_name = features["metric_name"]
        value = features["value"]

        # Initialize window for this metric
        if metric_name not in self.metric_windows:
            self.metric_windows[metric_name] = deque(maxlen=self.window_size)

        window = self.metric_windows[metric_name]

        # Need at least 10 samples to detect
        if len(window) < 10:
            window.append(value)
            return {
                "score": 0.0,
                "is_anomaly": False,
                "confidence": 0.0,
                "z_score": 0.0,
                "detector_name": self.name,
            }

        # Calculate z-score
        mean = np.mean(window)
        std = np.std(window)

        if std == 0:
            z_score = 0.0
        else:
            z_score = abs((value - mean) / std)

        # Update window
        window.append(value)

        # Determine anomaly
        is_anomaly = z_score > self.z_threshold

        # Convert z-score to 0-1 score (sigmoid-like)
        score = min(z_score / (2 * self.z_threshold), 1.0)

        # Confidence based on window size
        confidence = min(len(window) / self.window_size, 1.0)

        return {
            "score": float(score),
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "z_score": float(z_score),
            "mean": float(mean),
            "std": float(std),
            "detector_name": self.name,
        }

    def save_model(self, path: str) -> None:
        """Save detector state."""
        state = {
            "window_size": self.window_size,
            "z_threshold": self.z_threshold,
            "metric_windows": {
                k: list(v) for k, v in self.metric_windows.items()
            }
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Statistical detector state saved to {path}")

    def load_model(self, path: str) -> None:
        """Load detector state."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.window_size = state["window_size"]
        self.z_threshold = state["z_threshold"]
        self.metric_windows = {
            k: deque(v, maxlen=self.window_size)
            for k, v in state["metric_windows"].items()
        }

        logger.info(f"Statistical detector state loaded from {path}")


# Register the detector
detector = StatisticalDetector()
```

### 3단계: PyTorch 기반 딥러닝 탐지기 (고급)

**파일**: `ocad/plugins/detectors/autoencoder/__init__.py`

```python
"""Autoencoder Anomaly Detector.

Unsupervised anomaly detection using autoencoders.
Reconstruction error indicates anomaly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

from ocad.plugins.base import DetectorPlugin

logger = logging.getLogger(__name__)


class AutoencoderNetwork(nn.Module):
    """Autoencoder neural network."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector(DetectorPlugin):
    """Autoencoder-based anomaly detector."""

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.input_dim = config.get("input_dim", 10)
        self.hidden_dims = config.get("hidden_dims", [128, 64, 32])
        self.latent_dim = config.get("latent_dim", 16)
        self.threshold = config.get("threshold", 2.0)
        self.learning_rate = config.get("learning_rate", 0.001)

        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    @property
    def name(self) -> str:
        return "autoencoder"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_protocols(self) -> List[str]:
        return ["cfm", "bfd", "bgp", "ptp"]

    def train(self, data: Any, **kwargs) -> None:
        """Train autoencoder model.

        Args:
            data: pandas DataFrame with numerical features
            **kwargs: epochs, batch_size, etc.
        """
        epochs = kwargs.get("epochs", 100)
        batch_size = kwargs.get("batch_size", 64)

        logger.info(f"Training autoencoder (epochs={epochs}, batch={batch_size})")

        # Convert to numpy
        if isinstance(data, pd.DataFrame):
            X = data.select_dtypes(include=[np.number]).values
        else:
            X = np.array(data)

        # Normalize
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0) + 1e-8
        X_norm = (X - self.scaler_mean) / self.scaler_std

        # Create model
        self.input_dim = X_norm.shape[1]
        self.model = AutoencoderNetwork(
            self.input_dim,
            self.hidden_dims,
            self.latent_dim
        )

        # Prepare data
        X_tensor = torch.FloatTensor(X_norm)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        logger.info("Autoencoder training completed")

    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies using reconstruction error."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load_model()")

        # Extract numerical features
        # Assume features has "value" and other metrics
        # TODO: Customize feature extraction for your use case
        feature_vector = np.array([features.get("value", 0.0)])

        # Normalize
        feature_norm = (feature_vector - self.scaler_mean) / self.scaler_std

        # Reconstruct
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(feature_norm).unsqueeze(0)
            reconstructed = self.model(X_tensor)

        # Calculate reconstruction error
        reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2).item()

        # Determine anomaly
        is_anomaly = reconstruction_error > self.threshold
        score = min(reconstruction_error / (2 * self.threshold), 1.0)

        return {
            "score": float(score),
            "is_anomaly": bool(is_anomaly),
            "confidence": 0.9,
            "reconstruction_error": float(reconstruction_error),
            "detector_name": self.name,
        }

    def save_model(self, path: str) -> None:
        """Save model and scaler."""
        if self.model is None:
            raise ValueError("No model to save")

        state = {
            "model_state": self.model.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "latent_dim": self.latent_dim,
                "threshold": self.threshold,
            },
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
        }

        torch.save(state, path)
        logger.info(f"Autoencoder model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model and scaler."""
        state = torch.load(path)

        config = state["config"]
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.latent_dim = config["latent_dim"]
        self.threshold = config["threshold"]

        self.model = AutoencoderNetwork(
            self.input_dim,
            self.hidden_dims,
            self.latent_dim
        )
        self.model.load_state_dict(state["model_state"])
        self.model.eval()

        self.scaler_mean = state["scaler_mean"]
        self.scaler_std = state["scaler_std"]

        logger.info(f"Autoencoder model loaded from {path}")


# Register the detector
detector = AutoencoderDetector()
```

---

## 테스트 작성

### 단위 테스트

**파일**: `tests/plugins/test_ospf_adapter.py`

```python
import pytest
import asyncio
from ocad.plugins.protocol_adapters.ospf import adapter as ospf_adapter


@pytest.mark.asyncio
async def test_ospf_adapter_basic():
    """Test OSPF adapter basic functionality."""
    # Test properties
    assert ospf_adapter.name == "ospf"
    assert ospf_adapter.version == "1.0.0"
    assert len(ospf_adapter.supported_metrics) > 0

    # Test configuration validation
    valid_config = {
        "routers": [
            {
                "id": "test-router",
                "ip": "192.168.1.1",
                "area": "0.0.0.0",
                "router_id": "1.1.1.1"
            }
        ],
        "interval_sec": 1
    }

    assert ospf_adapter.validate_config(valid_config) is True

    # Test metric collection
    collected = []
    async for metric in ospf_adapter.collect(valid_config):
        collected.append(metric)
        if len(collected) >= 7:  # 7 metrics per router
            break

    assert len(collected) == 7

    # Verify metric structure
    for metric in collected:
        assert "timestamp" in metric
        assert "source_id" in metric
        assert "metric_name" in metric
        assert "value" in metric
        assert "metadata" in metric


@pytest.mark.asyncio
async def test_ospf_adapter_invalid_config():
    """Test OSPF adapter with invalid configuration."""
    invalid_config = {"invalid": "config"}

    with pytest.raises(ValueError):
        ospf_adapter.validate_config(invalid_config)
```

### 통합 테스트

**파일**: `tests/integration/test_ospf_end_to_end.py`

```python
import pytest
from ocad.plugins.registry import PluginRegistry


@pytest.mark.asyncio
async def test_ospf_with_statistical_detector():
    """Test OSPF adapter with statistical detector."""
    registry = PluginRegistry()
    registry.discover_plugins("ocad/plugins")

    # Get plugins
    adapter = registry.get_protocol_adapter("ospf")
    detector = registry.get_detector("statistical")

    assert adapter is not None
    assert detector is not None

    # Collect and detect
    config = {
        "routers": [{"id": "test", "ip": "192.168.1.1", "area": "0.0.0.0", "router_id": "1.1.1.1"}],
        "interval_sec": 1
    }

    collected = 0
    async for metric in adapter.collect(config):
        result = detector.detect(metric)

        assert "score" in result
        assert "is_anomaly" in result
        assert result["detector_name"] == "statistical"

        collected += 1
        if collected >= 20:
            break
```

---

## 플러그인 배포

### 1. 패키징

```bash
# 플러그인을 별도 패키지로 만들기
cd ocad/plugins/protocol_adapters/ospf

# setup.py 작성
cat > setup.py <<EOF
from setuptools import setup

setup(
    name="ocad-plugin-ospf",
    version="1.0.0",
    author="Your Name",
    description="OSPF protocol adapter for OCAD",
    py_modules=["__init__"],
    install_requires=[
        "ocad>=1.0.0",
    ],
)
EOF

# 빌드
python setup.py sdist bdist_wheel
```

### 2. 설치

```bash
# 로컬 설치
pip install -e ocad/plugins/protocol_adapters/ospf

# 또는 wheel 설치
pip install dist/ocad_plugin_ospf-1.0.0-py3-none-any.whl
```

### 3. 버전 관리

```python
# ocad/plugins/protocol_adapters/ospf/__init__.py
__version__ = "1.0.0"

class OSPFAdapter(ProtocolAdapter):
    @property
    def version(self) -> str:
        return __version__
```

---

## 성능 최적화

### 1. 비동기 처리

```python
# ❌ Bad: Synchronous blocking
def collect_metric():
    time.sleep(1)  # Blocks entire event loop
    return metric

# ✅ Good: Async non-blocking
async def collect_metric():
    await asyncio.sleep(1)  # Allows other tasks to run
    return metric
```

### 2. 배치 처리

```python
# ❌ Bad: One at a time
async def collect(self, config):
    for endpoint in endpoints:
        metric = await collect_from_endpoint(endpoint)
        yield metric

# ✅ Good: Batch collection
async def collect(self, config):
    batch = []
    for endpoint in endpoints:
        metric = await collect_from_endpoint(endpoint)
        batch.append(metric)

        if len(batch) >= 10:
            for m in batch:
                yield m
            batch = []
```

### 3. 메모리 관리

```python
# ✅ Good: Use deque for fixed-size windows
from collections import deque

self.metric_window = deque(maxlen=100)  # Auto-drops old values

# ✅ Good: Cleanup in destructor
def __del__(self):
    self.cleanup_resources()
```

### 4. 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(value):
    # Cache results for repeated inputs
    return complex_calculation(value)
```

---

## 예제 코드

### 전체 예제: 간단한 Ping 어댑터

**파일**: `ocad/plugins/protocol_adapters/ping/__init__.py`

```python
"""Ping Protocol Adapter."""

import asyncio
import subprocess
from datetime import datetime
from typing import AsyncIterator, Dict, Any, List

from ocad.plugins.base import ProtocolAdapter


class PingAdapter(ProtocolAdapter):
    """Simple ICMP ping adapter."""

    @property
    def name(self) -> str:
        return "ping"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_metrics(self) -> List[str]:
        return ["rtt_ms", "packet_loss_rate"]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        required = ["targets", "interval_sec"]
        return all(key in config for key in required)

    async def collect(self, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Collect ping metrics."""
        self.validate_config(config)

        targets = config["targets"]  # List of IPs
        interval = config["interval_sec"]

        while True:
            for target in targets:
                rtt, loss = await self._ping(target)

                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": target,
                    "metric_name": "rtt_ms",
                    "value": rtt,
                    "metadata": {"target": target}
                }

                yield {
                    "timestamp": datetime.utcnow(),
                    "source_id": target,
                    "metric_name": "packet_loss_rate",
                    "value": loss,
                    "metadata": {"target": target}
                }

            await asyncio.sleep(interval)

    async def _ping(self, target: str) -> tuple:
        """Execute ping command."""
        proc = await asyncio.create_subprocess_exec(
            "ping", "-c", "4", target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, _ = await proc.communicate()
        output = stdout.decode()

        # Parse output (simplified)
        rtt = 0.0
        loss = 0.0

        for line in output.split("\n"):
            if "time=" in line:
                rtt = float(line.split("time=")[1].split()[0])
            if "packet loss" in line:
                loss = float(line.split("%")[0].split()[-1])

        return rtt, loss / 100.0

    def get_recommended_models(self) -> List[str]:
        return ["statistical", "lstm"]


adapter = PingAdapter()
```

---

## 다음 단계

1. **실제 프로토콜 통합**: NETCONF, SNMP, gRPC 등을 사용하여 실제 장비 연동
2. **고급 AI 모델**: Transformer, Attention 메커니즘 추가
3. **분산 처리**: 여러 노드에서 플러그인 실행
4. **모니터링**: 플러그인 성능 및 헬스 체크 추가

---

## 참고 자료

- [Plugin-User-Guide.md](../06-plugins/Plugin-User-Guide.md): 사용자 가이드
- [Plugin-Architecture.md](../05-architecture/Plugin-Architecture.md): 아키텍처 설계
- [ocad/plugins/base.py](../../ocad/plugins/base.py): 기본 인터페이스
- [tests/plugins/](../../tests/plugins/): 플러그인 테스트 예제
