"""BaseTrainer 단위 테스트."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ocad.training.trainers.base_trainer import BaseTrainer, TrainingConfig


class SimpleModel(nn.Module):
    """테스트용 간단한 모델."""

    def __init__(self, input_size: int = 10, hidden_size: int = 5, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MockTrainer(BaseTrainer):
    """테스트용 Mock Trainer."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()

    def build_model(self):
        return SimpleModel()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {"train_loss": total_loss / max(num_batches, 1)}

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1

        return {"val_loss": total_loss / max(num_batches, 1)}


def create_dummy_dataloader(num_samples: int = 100, batch_size: int = 32):
    """더미 데이터 로더 생성."""
    from torch.utils.data import TensorDataset, DataLoader

    # 랜덤 데이터 생성
    x = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestTrainingConfig:
    """TrainingConfig 클래스 테스트."""

    def test_default_config(self):
        """기본 설정 테스트."""
        config = TrainingConfig()

        assert config.batch_size == 32
        assert config.epochs == 50
        assert config.learning_rate == 0.001
        assert config.early_stopping is True
        assert config.patience == 10

    def test_custom_config(self):
        """커스텀 설정 테스트."""
        config = TrainingConfig(
            batch_size=64,
            epochs=100,
            learning_rate=0.01,
            early_stopping=False,
        )

        assert config.batch_size == 64
        assert config.epochs == 100
        assert config.learning_rate == 0.01
        assert config.early_stopping is False

    def test_to_dict(self):
        """설정 딕셔너리 변환 테스트."""
        config = TrainingConfig(batch_size=16, epochs=25)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["batch_size"] == 16
        assert config_dict["epochs"] == 25
        assert "learning_rate" in config_dict


class TestBaseTrainer:
    """BaseTrainer 클래스 테스트."""

    def test_trainer_initialization(self):
        """Trainer 초기화 테스트."""
        config = TrainingConfig(epochs=10)
        trainer = MockTrainer(config)

        assert trainer.config.epochs == 10
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.current_epoch == 0
        assert trainer.best_metric == float("inf")

    def test_train_single_epoch(self):
        """1 에포크 학습 테스트."""
        config = TrainingConfig(epochs=1, log_interval=1)
        trainer = MockTrainer(config)

        train_loader = create_dummy_dataloader(num_samples=50, batch_size=16)
        val_loader = create_dummy_dataloader(num_samples=20, batch_size=16)

        # 1 에포크만 학습
        history = trainer.train(train_loader, val_loader)

        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1
        assert isinstance(history["train_loss"][0], float)
        assert isinstance(history["val_loss"][0], float)

    def test_train_multiple_epochs(self):
        """여러 에포크 학습 테스트."""
        config = TrainingConfig(epochs=5, log_interval=2, early_stopping=False)
        trainer = MockTrainer(config)

        train_loader = create_dummy_dataloader(num_samples=50, batch_size=16)
        val_loader = create_dummy_dataloader(num_samples=20, batch_size=16)

        history = trainer.train(train_loader, val_loader)

        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5
        assert trainer.current_epoch == 4  # 0-indexed

    def test_early_stopping(self):
        """조기 종료 테스트."""
        config = TrainingConfig(
            epochs=100,
            early_stopping=True,
            patience=3,
            min_delta=0.001,
        )
        trainer = MockTrainer(config)

        # 개선이 없는 더미 검증 손실 시뮬레이션
        trainer.history["val_loss"] = [1.0, 1.0, 1.0, 1.0]  # 4번 연속 개선 없음

        train_loader = create_dummy_dataloader(num_samples=50, batch_size=16)
        val_loader = create_dummy_dataloader(num_samples=20, batch_size=16)

        history = trainer.train(train_loader, val_loader)

        # patience=3이므로 조기 종료되어야 함
        assert len(history["train_loss"]) < 100

    def test_save_and_load_checkpoint(self):
        """체크포인트 저장 및 로드 테스트."""
        config = TrainingConfig(epochs=2)
        trainer = MockTrainer(config)

        train_loader = create_dummy_dataloader(num_samples=50, batch_size=16)
        val_loader = create_dummy_dataloader(num_samples=20, batch_size=16)

        # 학습
        trainer.train(train_loader, val_loader)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"

            # 저장
            trainer.save_checkpoint(checkpoint_path)
            assert checkpoint_path.exists()

            # 새로운 Trainer 생성 및 로드
            new_trainer = MockTrainer(config)
            new_trainer.load_checkpoint(checkpoint_path)

            # 상태 검증
            assert new_trainer.current_epoch == trainer.current_epoch
            assert new_trainer.best_metric == trainer.best_metric
            assert len(new_trainer.history["train_loss"]) == len(trainer.history["train_loss"])

    def test_is_best_model(self):
        """최고 모델 판별 테스트."""
        config = TrainingConfig(min_delta=0.01)
        trainer = MockTrainer(config)

        trainer.best_metric = 1.0

        # 충분히 개선된 경우
        assert trainer._is_best_model(0.9) is True

        # min_delta보다 작게 개선된 경우
        assert trainer._is_best_model(0.995) is False

        # 악화된 경우
        assert trainer._is_best_model(1.1) is False

    def test_training_summary(self):
        """학습 요약 정보 테스트."""
        config = TrainingConfig(epochs=3)
        trainer = MockTrainer(config)

        train_loader = create_dummy_dataloader(num_samples=50, batch_size=16)
        val_loader = create_dummy_dataloader(num_samples=20, batch_size=16)

        trainer.train(train_loader, val_loader)

        summary = trainer.get_training_summary()

        assert "total_epochs" in summary
        assert "best_val_loss" in summary
        assert "final_train_loss" in summary
        assert "final_val_loss" in summary
        assert "config" in summary

        assert summary["total_epochs"] == 3
        assert isinstance(summary["best_val_loss"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
