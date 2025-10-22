"""TCN 모델 학습기."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base_trainer import BaseTrainer, TrainingConfig
from ...detectors.residual import SimpleTCN
from ...core.logging import get_logger


logger = get_logger(__name__)


class TCNTrainer(BaseTrainer):
    """TCN(Temporal Convolutional Network) 모델 학습기.

    시계열 예측을 위한 TCN 모델을 학습합니다.
    """

    def __init__(
        self,
        config: TrainingConfig,
        input_size: int = 1,
        hidden_size: int = 32,
        output_size: int = 1,
        sequence_length: int = 10,
    ):
        """TCN Trainer를 초기화합니다.

        Args:
            config: 학습 설정
            input_size: 입력 피처 크기
            hidden_size: 히든 레이어 크기
            output_size: 출력 크기
            sequence_length: 입력 시퀀스 길이
        """
        super().__init__(config)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length

        # 모델 초기화
        self.model = self.build_model()
        self.model.to(self.config.device)

        # 옵티마이저 초기화
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 손실 함수
        self.criterion = nn.MSELoss()

        self.logger.info(
            "TCN Trainer 초기화",
            input_size=input_size,
            hidden_size=hidden_size,
            sequence_length=sequence_length,
            device=self.config.device,
        )

    def build_model(self) -> SimpleTCN:
        """TCN 모델을 구성합니다.

        Returns:
            SimpleTCN 인스턴스
        """
        return SimpleTCN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """1 에포크 학습을 수행합니다.

        Args:
            train_loader: 학습 데이터 로더

        Returns:
            학습 지표 딕셔너리
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        for batch_x, batch_y in train_loader:
            # 디바이스로 이동
            batch_x = batch_x.to(self.config.device)
            batch_y = batch_y.to(self.config.device)

            # TCN은 [batch, features, sequence] 형태를 기대
            # batch_x shape: [batch, sequence] -> [batch, 1, sequence]
            batch_x = batch_x.unsqueeze(1)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)

            # Loss 계산
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 예측값 저장 (추가 지표 계산용)
            all_predictions.extend(outputs.detach().cpu().numpy().flatten())
            all_targets.extend(batch_y.detach().cpu().numpy().flatten())

        # 평균 loss
        avg_loss = total_loss / max(num_batches, 1)

        # 추가 지표 계산
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)

        return {
            "train_loss": avg_loss,
            "train_mse": mse,
            "train_mae": mae,
            "train_rmse": rmse,
        }

    def validate(self, val_loader) -> Dict[str, float]:
        """검증 데이터로 평가합니다.

        Args:
            val_loader: 검증 데이터 로더

        Returns:
            검증 지표 딕셔너리
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # 디바이스로 이동
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                # TCN input shape: [batch, 1, sequence]
                batch_x = batch_x.unsqueeze(1)

                # Forward pass
                outputs = self.model(batch_x)

                # Loss 계산
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                num_batches += 1

                # 예측값 저장
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())

        # 평균 loss
        avg_loss = total_loss / max(num_batches, 1)

        # 추가 지표 계산
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)

        return {
            "val_loss": avg_loss,
            "val_mse": mse,
            "val_mae": mae,
            "val_rmse": rmse,
            "val_r2": r2,
        }

    def save_model(
        self,
        save_path: Path,
        metadata: Optional[Dict] = None,
        performance: Optional[Dict] = None,
    ) -> None:
        """TCN 모델을 저장합니다.

        Args:
            save_path: 저장 경로
            metadata: 추가 메타데이터
            performance: 성능 지표
        """
        from ..utils.model_saver import ModelSaver

        model_config = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        }

        # 메타데이터에 시퀀스 길이 추가
        if metadata is None:
            metadata = {}
        metadata["sequence_length"] = self.sequence_length

        # 성능 지표 추가
        if performance is None:
            performance = {}
        performance.update({
            "best_val_loss": self.best_metric,
            "total_epochs": self.current_epoch + 1,
        })

        ModelSaver.save_pytorch_model(
            model=self.model,
            save_path=save_path,
            model_config=model_config,
            metadata=metadata,
            performance=performance,
        )

        self.logger.info("TCN 모델 저장 완료", path=str(save_path))

    def evaluate(self, test_loader) -> Dict[str, float]:
        """테스트 데이터로 최종 평가합니다.

        Args:
            test_loader: 테스트 데이터 로더

        Returns:
            평가 지표 딕셔너리
        """
        self.logger.info("TCN 모델 평가 시작")

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                # TCN input shape
                batch_x = batch_x.unsqueeze(1)

                # 예측
                outputs = self.model(batch_x)

                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())

        # 평가 지표 계산
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)

        # 잔차 분석
        residuals = all_targets - all_predictions
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)

        metrics = {
            "test_mse": float(mse),
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "test_r2": float(r2),
            "residual_mean": float(residual_mean),
            "residual_std": float(residual_std),
        }

        self.logger.info(
            "TCN 모델 평가 완료",
            **metrics,
        )

        return metrics

    def get_model_config(self) -> Dict:
        """모델 설정을 반환합니다.

        Returns:
            모델 설정 딕셔너리
        """
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "sequence_length": self.sequence_length,
        }
