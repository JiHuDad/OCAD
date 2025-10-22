# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Status

**프로젝트 현황**: OCAD 시스템은 기본 아키텍처가 완성되어 있으며, 핵심 파이프라인(수집 → 피처 엔지니어링 → 탐지 → 알람)이 정상 동작합니다. 실제 ORAN 장비 없이도 시뮬레이터를 통한 완전한 검증이 가능합니다.

**최근 작업**:

- 구조화된 다층 로깅 시스템 구현 (debug/summary/alerts 분리)
- 사람 친화적 알람 분석 보고서 자동 생성 기능 추가
- 플랫폼 호환성 검증 스크립트 추가
- AI 모델 통합 가이드 문서화

**다음 단계**: 다변량 탐지 알고리즘 고도화, 실시간 대시보드 UI 개선, 성능 최적화

## Communication Rules

**IMPORTANT**: Always respond in Korean (한글) when communicating with users in this repository. This is a mandatory rule for all interactions.

## Python Environment

**IMPORTANT**: This project uses Python virtual environment (venv). Always activate the virtual environment before running any Python commands:

```bash
source .venv/bin/activate
```

When running Python commands, scripts, or tests, ensure the virtual environment is activated first. All Python-related commands in this document assume the venv is activated.

## Project Overview

OCAD (ORAN CFM-Lite AI Anomaly Detection System) is a hybrid anomaly detection system for ORAN networks that uses reduced CFM functionality. It provides capability-driven monitoring with rule-based, changepoint (CUSUM/PELT), prediction-residual (TCN/LSTM), and multivariate detection methods.

## Architecture

The system follows a pipeline architecture:
```
O-RU/O-DU → Capability Detector → Collectors → Feature Engine → Detectors → Alerts
```

### Core Components

- **SystemOrchestrator** (`ocad/system/orchestrator.py`): Main coordinator managing all components and data flow
- **CapabilityDetector** (`ocad/capability/detector.py`): Auto-detects equipment capabilities via NETCONF/YANG
- **Collectors** (`ocad/collectors/`): Data collection for UDP Echo, eCPRI delay, LBM
- **FeatureEngine** (`ocad/features/engine.py`): Extracts features from time series data
- **Detectors** (`ocad/detectors/`): Multiple detection algorithms (rule-based, changepoint, residual, multivariate)
- **AlertManager** (`ocad/alerts/manager.py`): Manages alert generation with evidence-based filtering

### Directory Structure

- `ocad/`: Main Python package
  - `api/`: FastAPI REST endpoints
  - `core/`: Configuration, logging, models
  - `system/`: System orchestration
  - `capability/`: Equipment capability detection
  - `collectors/`: Data collection modules
  - `features/`: Feature engineering
  - `detectors/`: Anomaly detection algorithms
  - `alerts/`: Alert management
  - `utils/`: Utilities and simulators
- `scripts/`: Utility scripts for testing and setup
- `config/`: Configuration files
- `logs/`: Structured logging output
- `tests/`: Unit and integration tests
- `docs/`: Documentation

## Common Development Commands

### Setup and Installation
```bash
# Quick start (automated setup)
./scripts/start.sh

# Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/example.yaml config/local.yaml
```

### Running the System
```bash
# API server (main mode)
python -m ocad.api.main

# Background system only
python -m ocad.main

# CLI interface
python -m ocad.cli --help
```

### Testing and Validation
```bash
# Platform compatibility check (30 seconds)
python3 scripts/compatibility_check.py

# Quick system validation (2 minutes)
python3 scripts/quick_test.py

# Scenario testing (10 minutes)
python3 scripts/scenario_test.py

# Dashboard testing
python3 scripts/dashboard_test.py

# Generate sample reports
python3 scripts/generate_sample_report.py

# Run single test file
pytest tests/unit/test_collectors.py
pytest tests/integration/test_pipeline.py -v

# Run single test function
pytest tests/unit/test_collectors.py::test_udp_echo_collector -v
```

### Code Quality
```bash
# Linting and formatting
ruff check .
ruff format .

# Type checking
mypy ocad/

# Run all tests
pytest tests/

# Run specific test suites
pytest tests/unit/
pytest tests/integration/

# Coverage report
pytest --cov=ocad --cov-report=html
pytest --cov=ocad --cov-report=term-missing

# Watch mode for development
pytest --watch
```

### CLI Operations
```bash
# Add endpoint for monitoring
python -m ocad.cli add-endpoint 192.168.1.100 --role o-ru

# System status
python -m ocad.cli status

# List endpoints
python -m ocad.cli list-endpoints

# Run simulation
python -m ocad.cli simulate --count 10 --duration 300
```

## Configuration

Configuration is managed through YAML files and environment variables:

- `config/local.yaml`: Main configuration file
- `config/example.yaml`: Template configuration
- Environment variables use `__` delimiter for nested configs (e.g., `DATABASE__URL`)

### Log Directory Configuration
```bash
# Default: project root/logs
export OCAD_LOG_DIR=/var/log/ocad  # Custom log directory
```

## Logging System

OCAD uses a structured multi-layer logging system with timestamped directories:

```
logs/test_YYYYMMDD_HHMMSS/
├── debug/detailed.log              # All debug information
├── summary/summary.log             # Important events only
└── alerts/
    ├── alert_details.log           # Technical alert information
    └── human_readable_analysis.txt # User-friendly analysis
```

## Key Development Patterns

### Adding New Collectors
1. Inherit from `BaseCollector` in `ocad/collectors/base.py`
2. Implement `collect_data()` method returning metrics dictionary
3. Register in `SystemOrchestrator.collectors` list in `ocad/system/orchestrator.py`
4. Add capability detection in `CapabilityDetector` in `ocad/capability/detector.py`
5. Add corresponding unit tests in `tests/unit/test_collectors.py`

Example collector structure:
```python
from ocad.collectors.base import BaseCollector

class MyCollector(BaseCollector):
    async def collect_data(self, endpoint) -> dict:
        # Return metrics: {"metric_name": value, ...}
        pass
```

### Adding New Detectors
1. Inherit from `BaseDetector` in `ocad/detectors/base.py`
2. Implement `detect()` method returning anomaly score (0.0-1.0)
3. Add to `CompositeDetector` in `ocad/system/orchestrator.py`
4. Configure weights in `DetectionConfig` in `ocad/core/config.py`
5. Add unit tests in `tests/unit/test_detectors.py`

Example detector structure:
```python
from ocad.detectors.base import BaseDetector

class MyDetector(BaseDetector):
    def detect(self, features: dict) -> float:
        # Return score between 0.0 (normal) and 1.0 (anomalous)
        pass
```

### Configuration Management
- Main config: `config/local.yaml` (gitignored, created from `config/example.yaml`)
- Settings classes in `ocad/core/config.py` using Pydantic
- Environment variables override YAML: use `__` for nested keys (e.g., `DATABASE__URL`)
- Always update `config/example.yaml` when adding new config options
- Maintain backward compatibility with existing environment variables

### Testing Strategy
- **Unit tests** (`tests/unit/`): Individual components in isolation
- **Integration tests** (`tests/integration/`): Full pipeline validation
- **Simulation-based**: Use `ocad/utils/simulator.py` for virtual endpoints
- **Validation scripts** (`scripts/`): End-to-end system validation
- Always test without real ORAN equipment using simulators

## API Endpoints

The system provides REST API at http://localhost:8080 (configured in `ocad/api/main.py`):
- `GET /health` - Health status check
- `GET /endpoints` - List monitored endpoints
- `POST /endpoints` - Add new endpoint
- `DELETE /endpoints/{id}` - Remove endpoint
- `GET /alerts` - List active alerts
- `GET /alerts/{id}` - Get alert details
- `POST /alerts/{id}/acknowledge` - Acknowledge alert
- `GET /stats` - System statistics and KPIs
- `GET /debug/windows` - Debug sliding window data

## Dependencies

### Core Technologies
- **FastAPI**: REST API framework
- **Pydantic**: Data validation and settings
- **NumPy/Pandas**: Data processing
- **Scikit-learn**: Machine learning algorithms
- **PyTorch/TensorFlow**: Deep learning models
- **Ruptures**: Changepoint detection
- **ncclient**: NETCONF communication

### Development Tools
- **pytest**: Testing framework
- **ruff**: Linting and formatting
- **mypy**: Type checking
- **structlog**: Structured logging

## Performance Targets

- Processing latency < 30 seconds (95th percentile)
- False alarm rate ≤ 6%
- Early warning lead time ≥ 4 minutes (p50)
- Support for thousands of endpoints

## Data Flow Architecture

Understanding the complete data pipeline is critical for development:

```
1. Endpoint Registration
   → CLI/API adds endpoint → SystemOrchestrator tracks it

2. Capability Detection (once per endpoint)
   → CapabilityDetector queries via NETCONF/YANG
   → Determines which collectors to activate (UDP Echo, eCPRI, LBM, CCM)

3. Data Collection (periodic, e.g., every 10s)
   → Active collectors gather metrics from endpoint
   → Returns dict: {"rtt_ms": 5.2, "loss_rate": 0.01, ...}

4. Feature Engineering (per window, e.g., 5 min)
   → FeatureEngine computes: percentiles (p95, p99), EWMA, gradients, CUSUM
   → Sliding window maintains historical data

5. Anomaly Detection (per window)
   → RuleBasedDetector: threshold violations → score 0-1
   → ChangepointDetector: CUSUM/PELT → score 0-1
   → ResidualDetector: TCN/LSTM prediction error → score 0-1
   → MultivariateDetector: cross-metric patterns → score 0-1
   → CompositeDetector: weighted ensemble → final score

6. Alert Generation (when score > threshold)
   → AlertManager applies evidence rules (≥2 of: drift, spike, concurrency)
   → Hold-down timer prevents alert flooding
   → Deduplication merges similar alerts
   → Severity classification: INFO/WARNING/CRITICAL

7. Output
   → REST API: real-time alerts and stats
   → Logs: structured logs in logs/test_*/
   → Human-readable analysis: alerts/human_readable_analysis.txt
```

Key files for each stage:
- `ocad/system/orchestrator.py` - Main pipeline coordinator
- `ocad/capability/detector.py` - Stage 2
- `ocad/collectors/*.py` - Stage 3
- `ocad/features/engine.py` - Stage 4
- `ocad/detectors/*.py` - Stage 5
- `ocad/alerts/manager.py` - Stage 6

## Testing Without Real Equipment

The system supports complete validation without actual ORAN equipment through simulation:
- **Virtual endpoint simulation**: `ocad/utils/simulator.py` creates mock O-RU/O-DU endpoints
- **Synthetic metric generation**: Realistic time series data with configurable patterns
- **Anomaly injection**: Inject spikes, drifts, packet loss for validation
- **Complete pipeline testing**: All stages from collection to alerting work in simulation mode

## Training-Inference Separation (NEW)

**Important Architecture Change**: The system is transitioning from online learning to separated training-inference architecture.

### Current State (Online Learning)

- ResidualDetector (TCN) and MultivariateDetector (Isolation Forest) perform training during inference
- Models are trained automatically when 50+ samples are collected
- This causes unpredictable inference latency and makes performance validation impossible

### New Architecture (Separated)

- **Offline Training**: Models are trained separately using dedicated training pipelines
- **Online Inference**: Detectors load pre-trained models and perform inference only
- **Benefits**: Consistent latency, reproducible results, model versioning, A/B testing

### Directory Structure for Training

```
ocad/
├── training/              # NEW: Training modules
│   ├── datasets/          # Dataset management
│   ├── trainers/          # Training logic (TCNTrainer, IsolationForestTrainer)
│   ├── evaluators/        # Model evaluation
│   └── utils/             # Model saving/loading, hyperparameter tuning
├── models/                # NEW: Trained model storage
│   ├── tcn/               # TCN models (.pth files)
│   ├── isolation_forest/  # Isolation Forest models (.pkl files)
│   └── metadata/          # Model metadata and performance reports
└── data/                  # NEW: Training datasets
    ├── raw/               # Raw ORAN logs or simulated data
    ├── processed/         # Preprocessed datasets (Parquet format)
    └── synthetic/         # Synthetic anomaly data
```

### Training Commands

```bash
# Generate training dataset
python scripts/generate_training_data.py \
    --endpoints 10 \
    --duration-hours 24 \
    --anomaly-rate 0.1

# Train TCN models
python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --data-path data/processed/timeseries_train.parquet \
    --epochs 50 \
    --batch-size 32 \
    --output models/tcn/udp_echo_v1.0.0.pth

# Train Isolation Forest
python scripts/train_isolation_forest.py \
    --data-path data/processed/multivariate_train.parquet \
    --output models/isolation_forest/multivariate_v1.0.0.pkl

# Evaluate models
python scripts/evaluate_models.py \
    --model-path models/tcn/udp_echo_v1.0.0.pth \
    --test-data data/processed/timeseries_test.parquet
```

### Using Pre-trained Models

```yaml
# config/local.yaml
detection:
  residual:
    use_pretrained_models: true  # Enable pre-trained mode
    model_path: "ocad/models/tcn/"
  multivariate:
    use_pretrained_models: true
    model_path: "ocad/models/isolation_forest/"
```

### Implementation Plan

See detailed design and implementation plan in [docs/Training-Inference-Separation-Design.md](docs/Training-Inference-Separation-Design.md)

**Phases**:

1. **Phase 1 (Week 1-2)**: Infrastructure setup (directories, BaseTrainer, dataset generation)
2. **Phase 2 (Week 3-4)**: TCN training-inference separation
3. **Phase 3 (Week 5)**: Isolation Forest training-inference separation
4. **Phase 4 (Week 6)**: Integration and validation
5. **Phase 5 (Week 7-8)**: Automation and MLOps (MLflow, CI/CD)

**Migration Strategy**: Gradual transition with backward compatibility flag to keep existing online learning mode during transition period.
