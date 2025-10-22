"""학습기 기본 클래스."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from ...core.logging import get_logger


logger = get_logger(__name__)


class TrainingConfig:
    """학습 설정 클래스.

    모든 학습 관련 하이퍼파라미터와 설정을 관리합니다.
    """

    def __init__(
        self,
        # 데이터 관련
        batch_size: int = 32,
        num_workers: int = 4,

        # 학습 관련
        epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,

        # 조기 종료
        early_stopping: bool = True,
        patience: int = 10,
        min_delta: float = 0.001,

        # 모델 저장
        save_best_only: bool = True,
        save_frequency: int = 5,

        # 로깅
        log_interval: int = 10,

        # 디바이스
        device: str = "cpu",

        # 재현성
        random_seed: int = 42,
    ):
        """학습 설정을 초기화합니다.

        Args:
            batch_size: 배치 크기
            num_workers: 데이터 로딩 워커 수
            epochs: 학습 에포크 수
            learning_rate: 학습률
            weight_decay: 가중치 감쇠 (L2 정규화)
            early_stopping: 조기 종료 사용 여부
            patience: 조기 종료 인내 에포크 수
            min_delta: 조기 종료 최소 개선폭
            save_best_only: 최고 성능 모델만 저장
            save_frequency: 체크포인트 저장 주기
            log_interval: 로그 출력 간격
            device: 학습 디바이스 ("cpu", "cuda", "mps")
            random_seed: 랜덤 시드
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        self.save_best_only = save_best_only
        self.save_frequency = save_frequency

        self.log_interval = log_interval

        self.device = device
        self.random_seed = random_seed

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환합니다.

        Returns:
            설정 딕셔너리
        """
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "save_best_only": self.save_best_only,
            "save_frequency": self.save_frequency,
            "log_interval": self.log_interval,
            "device": self.device,
            "random_seed": self.random_seed,
        }


class BaseTrainer(ABC):
    """모든 학습기의 기본 추상 클래스.

    학습 루프, 검증, 체크포인트 저장 등의 공통 기능을 제공합니다.
    """

    def __init__(self, config: TrainingConfig):
        """학습기를 초기화합니다.

        Args:
            config: 학습 설정
        """
        self.config = config
        self.logger = logger.bind(component="trainer")

        # 모델과 옵티마이저는 서브클래스에서 초기화
        self.model: Optional[Any] = None
        self.optimizer: Optional[Any] = None
        self.criterion: Optional[Any] = None

        # 학습 상태 추적
        self.current_epoch = 0
        self.best_metric = float("inf")  # 낮을수록 좋은 지표 가정
        self.epochs_without_improvement = 0

        # 학습 이력
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
        }

    @abstractmethod
    def build_model(self) -> Any:
        """모델 아키텍처를 구성합니다.

        Returns:
            구성된 모델 인스턴스
        """
        pass

    @abstractmethod
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """1 에포크 학습을 수행합니다.

        Args:
            train_loader: 학습 데이터 로더

        Returns:
            학습 지표 딕셔너리 (예: {"train_loss": 0.123})
        """
        pass

    @abstractmethod
    def validate(self, val_loader) -> Dict[str, float]:
        """검증 데이터로 평가합니다.

        Args:
            val_loader: 검증 데이터 로더

        Returns:
            검증 지표 딕셔너리 (예: {"val_loss": 0.456, "accuracy": 0.89})
        """
        pass

    def train(self, train_loader, val_loader, output_dir: Optional[Path] = None):
        """전체 학습 프로세스를 실행합니다.

        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            output_dir: 모델 저장 디렉토리
        """
        self.logger.info(
            "학습 시작",
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # 1 에포크 학습
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["train_loss"])

            # 검증
            val_metrics = self.validate(val_loader)
            self.history["val_loss"].append(val_metrics["val_loss"])

            # 로그 출력
            if epoch % self.config.log_interval == 0:
                self.logger.info(
                    "에포크 완료",
                    epoch=epoch + 1,
                    **train_metrics,
                    **val_metrics,
                )

            # 최고 성능 모델 체크
            val_loss = val_metrics["val_loss"]
            is_best = self._is_best_model(val_loss)

            if is_best:
                self.best_metric = val_loss
                self.epochs_without_improvement = 0

                if output_dir and self.config.save_best_only:
                    self.save_checkpoint(output_dir / "best_model.pth")
                    self.logger.info("최고 성능 모델 저장", val_loss=val_loss)
            else:
                self.epochs_without_improvement += 1

            # 주기적 체크포인트 저장
            if (
                output_dir
                and not self.config.save_best_only
                and (epoch + 1) % self.config.save_frequency == 0
            ):
                self.save_checkpoint(output_dir / f"checkpoint_epoch_{epoch+1}.pth")

            # 조기 종료 체크
            if self.config.early_stopping:
                if self.epochs_without_improvement >= self.config.patience:
                    self.logger.info(
                        "조기 종료",
                        epoch=epoch + 1,
                        patience=self.config.patience,
                    )
                    break

        self.logger.info(
            "학습 완료",
            total_epochs=self.current_epoch + 1,
            best_val_loss=self.best_metric,
        )

        return self.history

    def _is_best_model(self, current_metric: float) -> bool:
        """현재 모델이 최고 성능인지 확인합니다.

        Args:
            current_metric: 현재 검증 지표 (낮을수록 좋음)

        Returns:
            최고 성능이면 True
        """
        return current_metric < (self.best_metric - self.config.min_delta)

    def save_checkpoint(self, path: Path) -> None:
        """모델 체크포인트를 저장합니다.

        Args:
            path: 저장 경로
        """
        if self.model is None:
            self.logger.warning("모델이 초기화되지 않아 저장할 수 없습니다")
            return

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "best_metric": self.best_metric,
            "config": self.config.to_dict(),
            "history": self.history,
        }

        import torch
        torch.save(checkpoint, path)

        self.logger.debug("체크포인트 저장", path=str(path))

    def load_checkpoint(self, path: Path) -> None:
        """모델 체크포인트를 로드합니다.

        Args:
            path: 체크포인트 경로
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        import torch
        checkpoint = torch.load(path)

        if self.model is None:
            raise RuntimeError("모델을 먼저 초기화해야 합니다")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        self.history = checkpoint["history"]

        self.logger.info(
            "체크포인트 로드",
            path=str(path),
            epoch=self.current_epoch,
            best_metric=self.best_metric,
        )

    def get_training_summary(self) -> Dict[str, Any]:
        """학습 요약 정보를 반환합니다.

        Returns:
            학습 요약 딕셔너리
        """
        return {
            "total_epochs": self.current_epoch + 1,
            "best_val_loss": self.best_metric,
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
            "config": self.config.to_dict(),
        }
