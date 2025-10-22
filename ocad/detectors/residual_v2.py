"""잔차 기반 이상 탐지기 (추론 전용 버전).

이 버전은 사전 훈련된 TCN 모델을 로드하여 추론만 수행합니다.
온라인 학습 코드가 완전히 제거되었습니다.
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

from ..core.models import Capabilities, FeatureVector
from .base import BaseDetector
from ..core.logging import get_logger


logger = get_logger(__name__)


class SimpleTCN(nn.Module):
    """Simple Temporal Convolutional Network for time series prediction."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, output_size: int = 1):
        """Initialize TCN.

        Args:
            input_size: Input feature size
            hidden_size: Hidden layer size
            output_size: Output size
        """
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, output_size, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor [batch, features, sequence]

        Returns:
            Output tensor
        """
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.conv3(x)
        return x[:, :, -1]  # Return last timestep


class ResidualDetectorV2(BaseDetector):
    """잔차 기반 탐지기 (추론 전용).

    사전 훈련된 TCN 모델을 사용하여 시계열 예측을 수행하고,
    실제 값과 예측 값의 잔차를 기반으로 이상을 탐지합니다.

    **변경 사항**:
    - ❌ 온라인 학습 코드 제거 (_train_model 삭제)
    - ❌ 무한 이력 저장 제거 (self.history 삭제)
    - ✅ 사전 훈련 모델 로드 추가
    - ✅ 추론 버퍼만 유지 (sequence_length만큼)
    - ✅ 메모리 효율성 향상
    """

    def __init__(
        self,
        config,
        model_dir: Optional[Path] = None,
        use_pretrained: bool = True,
        device: str = "cpu",
    ):
        """잔차 탐지기를 초기화합니다.

        Args:
            config: 탐지 설정
            model_dir: 사전 훈련된 모델 디렉토리 (기본값: ocad/models/tcn/)
            use_pretrained: 사전 훈련 모델 사용 여부
            device: 추론 디바이스 ("cpu", "cuda", "mps")
        """
        super().__init__(config)

        self.use_pretrained = use_pretrained
        self.device = device
        self.sequence_length = 10  # 예측을 위한 시퀀스 길이

        # 사전 훈련된 모델 및 scaler
        self.models: Dict[str, Optional[SimpleTCN]] = {}
        self.scalers: Dict[str, Optional[StandardScaler]] = {}

        # 추론을 위한 최소 버퍼 (sequence_length만큼만 유지)
        self.inference_buffers: Dict[str, deque] = {
            "udp_echo": deque(maxlen=self.sequence_length),
            "ecpri": deque(maxlen=self.sequence_length),
            "lbm": deque(maxlen=self.sequence_length),
        }

        # 사전 훈련 모델 로드
        if use_pretrained:
            model_dir = model_dir or Path("ocad/models/tcn")
            self._load_pretrained_models(model_dir)
        else:
            self.logger.warning(
                "사전 훈련 모델을 사용하지 않습니다. "
                "추론만 가능하며 학습은 불가능합니다."
            )

    def _load_pretrained_models(self, model_dir: Path) -> None:
        """사전 훈련된 모델과 scaler를 로드합니다.

        Args:
            model_dir: 모델 디렉토리
        """
        model_dir = Path(model_dir)

        if not model_dir.exists():
            self.logger.warning(
                "모델 디렉토리가 없습니다",
                path=str(model_dir),
            )
            return

        # 각 메트릭별 모델 로드
        for metric_type in ["udp_echo", "ecpri", "lbm"]:
            # 모델 파일 찾기 (버전 무관하게 가장 최신 파일)
            model_files = list(model_dir.glob(f"{metric_type}_v*.pth"))

            if not model_files:
                self.logger.debug(
                    "모델 파일이 없습니다",
                    metric_type=metric_type,
                    path=str(model_dir),
                )
                continue

            # 가장 최신 모델 선택
            model_file = sorted(model_files)[-1]

            try:
                # 모델 로드
                checkpoint = torch.load(model_file, map_location=self.device)

                # 모델 초기화
                model_config = checkpoint.get("model_config", {})
                model = SimpleTCN(
                    input_size=model_config.get("input_size", 1),
                    hidden_size=model_config.get("hidden_size", 32),
                    output_size=model_config.get("output_size", 1),
                )

                # 가중치 로드
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()  # 추론 모드

                self.models[metric_type] = model

                # Scaler 로드 (모델과 함께 저장되어 있을 수 있음)
                scaler_file = model_file.with_suffix(".scaler.pkl")
                if scaler_file.exists():
                    self.scalers[metric_type] = joblib.load(scaler_file)
                else:
                    # Scaler가 없으면 기본 생성 (훈련 데이터 통계 사용)
                    metadata = checkpoint.get("metadata", {})
                    if "scaler_mean" in metadata and "scaler_std" in metadata:
                        scaler = StandardScaler()
                        scaler.mean_ = np.array([metadata["scaler_mean"]])
                        scaler.var_ = np.array([metadata["scaler_std"] ** 2])
                        scaler.scale_ = np.array([metadata["scaler_std"]])
                        scaler.n_features_in_ = 1
                        self.scalers[metric_type] = scaler

                self.logger.info(
                    "사전 훈련 모델 로드 완료",
                    metric_type=metric_type,
                    model_file=model_file.name,
                    version=checkpoint.get("metadata", {}).get("version", "unknown"),
                    device=self.device,
                )

            except Exception as e:
                self.logger.error(
                    "모델 로드 실패",
                    metric_type=metric_type,
                    model_file=str(model_file),
                    error=str(e),
                )

    def can_detect(self, capabilities: Capabilities) -> bool:
        """잔차 탐지가 가능한지 확인합니다.

        Args:
            capabilities: 엔드포인트 기능

        Returns:
            사용 가능한 메트릭이 있으면 True
        """
        return any([
            capabilities.udp_echo,
            capabilities.ecpri_delay,
            capabilities.lbm,
        ])

    def detect(self, features: FeatureVector, capabilities: Capabilities) -> float:
        """잔차를 사용하여 이상을 탐지합니다 (추론만 수행).

        Args:
            features: 분석할 피처 벡터
            capabilities: 엔드포인트 기능

        Returns:
            0.0에서 1.0 사이의 이상 점수
        """
        residuals = []

        # 각 메트릭별로 잔차 계산
        if capabilities.udp_echo and features.udp_echo_p95 is not None:
            residual = self._calculate_residual(
                "udp_echo",
                features.udp_echo_p95,
                features.endpoint_id
            )
            if residual is not None:
                residuals.append(residual)

        if capabilities.ecpri_delay and features.ecpri_p95 is not None:
            # 마이크로초를 밀리초로 변환
            ecpri_ms = features.ecpri_p95 / 1000.0
            residual = self._calculate_residual(
                "ecpri",
                ecpri_ms,
                features.endpoint_id
            )
            if residual is not None:
                residuals.append(residual)

        if capabilities.lbm and features.lbm_rtt_p95 is not None:
            residual = self._calculate_residual(
                "lbm",
                features.lbm_rtt_p95,
                features.endpoint_id
            )
            if residual is not None:
                residuals.append(residual)

        if not residuals:
            return 0.0

        # 최대 정규화된 잔차
        max_residual = max(residuals)

        # 임계값을 사용하여 점수 정규화
        score = min(1.0, max_residual / self.config.residual_threshold)

        if score > 0.5:
            self.logger.debug(
                "높은 예측 잔차 탐지",
                endpoint_id=features.endpoint_id,
                max_residual=max_residual,
                residuals=residuals,
            )

        return score

    def _calculate_residual(
        self,
        metric_type: str,
        value: float,
        endpoint_id: str
    ) -> Optional[float]:
        """메트릭의 예측 잔차를 계산합니다 (추론만 수행).

        Args:
            metric_type: 메트릭 유형 (udp_echo, ecpri, lbm)
            value: 현재 메트릭 값
            endpoint_id: 엔드포인트 식별자

        Returns:
            정규화된 잔차 또는 None
        """
        # 추론 버퍼에 추가
        self.inference_buffers[metric_type].append(value)

        # 시퀀스 길이만큼 데이터가 없으면 대기
        if len(self.inference_buffers[metric_type]) < self.sequence_length:
            return None

        # 모델이 없으면 사용 불가
        if metric_type not in self.models or self.models[metric_type] is None:
            return None

        try:
            # 예측 수행 (학습 없음!)
            sequence = list(self.inference_buffers[metric_type])
            prediction = self._predict(metric_type, sequence)

            if prediction is not None:
                residual = abs(value - prediction)

                # 표준편차로 정규화
                recent_values = list(self.inference_buffers[metric_type])
                if len(recent_values) > 3:
                    std_dev = np.std(recent_values)
                    if std_dev > 0:
                        return residual / std_dev

                return residual

        except Exception as e:
            self.logger.debug(
                "추론 실패",
                metric_type=metric_type,
                endpoint_id=endpoint_id,
                error=str(e),
            )

        return None

    def _predict(self, metric_type: str, sequence: List[float]) -> Optional[float]:
        """예측을 수행합니다 (추론 전용).

        Args:
            metric_type: 메트릭 유형
            sequence: 입력 시퀀스

        Returns:
            예측 값 또는 None
        """
        model = self.models[metric_type]
        scaler = self.scalers.get(metric_type)

        if model is None:
            return None

        try:
            # 정규화
            sequence_array = np.array(sequence, dtype=np.float32).reshape(-1, 1)

            if scaler is not None:
                sequence_scaled = scaler.transform(sequence_array).flatten()
            else:
                # Scaler가 없으면 원본 값 사용
                sequence_scaled = sequence_array.flatten()

            # 예측
            with torch.no_grad():
                x_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).unsqueeze(0)
                x_tensor = x_tensor.to(self.device)
                prediction_scaled = model(x_tensor).item()

            # 역정규화
            if scaler is not None:
                prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]
            else:
                prediction = prediction_scaled

            return float(prediction)

        except Exception as e:
            self.logger.debug(
                "예측 실패",
                metric_type=metric_type,
                error=str(e),
            )
            return None

    def get_evidence(self, features: FeatureVector) -> Dict[str, float]:
        """잔차 탐지에 대한 증거 세부사항을 반환합니다.

        Args:
            features: 피처 벡터

        Returns:
            증거 세부사항 딕셔너리
        """
        evidence = {}

        # 예측 vs 실제 값 추가 (사용 가능한 경우)
        if features.udp_echo_residual is not None:
            evidence["udp_echo_residual"] = features.udp_echo_residual

        if features.ecpri_residual is not None:
            evidence["ecpri_residual"] = features.ecpri_residual

        if features.lbm_residual is not None:
            evidence["lbm_residual"] = features.lbm_residual

        return evidence

    def get_model_info(self) -> Dict[str, Dict]:
        """로드된 모델 정보를 반환합니다.

        Returns:
            모델 정보 딕셔너리
        """
        info = {}

        for metric_type in ["udp_echo", "ecpri", "lbm"]:
            if self.models.get(metric_type) is not None:
                info[metric_type] = {
                    "loaded": True,
                    "has_scaler": self.scalers.get(metric_type) is not None,
                    "buffer_size": len(self.inference_buffers[metric_type]),
                    "sequence_length": self.sequence_length,
                }
            else:
                info[metric_type] = {
                    "loaded": False,
                }

        return info
