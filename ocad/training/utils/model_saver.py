"""모델 저장 및 로드 유틸리티."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from sklearn.ensemble import IsolationForest
import joblib

from ...core.logging import get_logger


logger = get_logger(__name__)


class ModelSaver:
    """모델 저장 및 로드를 관리하는 유틸리티 클래스.

    PyTorch 모델(.pth), Scikit-learn 모델(.pkl) 모두 지원하며,
    모델과 함께 메타데이터와 성능 리포트를 저장합니다.
    """

    @staticmethod
    def save_pytorch_model(
        model: torch.nn.Module,
        save_path: Path,
        model_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
    ) -> None:
        """PyTorch 모델을 저장합니다.

        Args:
            model: 저장할 PyTorch 모델
            save_path: 저장 경로 (.pth)
            model_config: 모델 설정 (아키텍처 재구성용)
            metadata: 추가 메타데이터
            performance: 성능 지표
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 모델 체크포인트 구성
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "metadata": metadata or {},
            "performance": performance or {},
            "save_date": datetime.now().isoformat(),
        }

        # 버전 정보 추가
        if "version" not in checkpoint["metadata"]:
            checkpoint["metadata"]["version"] = "1.0.0"

        # 모델 저장
        torch.save(checkpoint, save_path)

        # 메타데이터를 별도 JSON 파일로 저장
        metadata_path = save_path.with_suffix(".json")
        metadata_dict = {
            "model_type": "pytorch",
            "model_path": str(save_path),
            "model_config": model_config,
            "metadata": checkpoint["metadata"],
            "performance": checkpoint["performance"],
            "save_date": checkpoint["save_date"],
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        logger.info(
            "PyTorch 모델 저장 완료",
            path=str(save_path),
            version=checkpoint["metadata"].get("version"),
        )

    @staticmethod
    def load_pytorch_model(
        model_class: type,
        load_path: Path,
        device: str = "cpu",
    ) -> tuple[torch.nn.Module, Dict[str, Any]]:
        """PyTorch 모델을 로드합니다.

        Args:
            model_class: 모델 클래스 (예: SimpleTCN)
            load_path: 모델 파일 경로 (.pth)
            device: 디바이스 ("cpu", "cuda", "mps")

        Returns:
            (로드된 모델, 메타데이터) 튜플
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        # 체크포인트 로드
        checkpoint = torch.load(load_path, map_location=device)

        # 모델 초기화
        model_config = checkpoint["model_config"]
        model = model_class(**model_config)

        # 가중치 로드
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()  # 추론 모드

        logger.info(
            "PyTorch 모델 로드 완료",
            path=str(load_path),
            version=checkpoint["metadata"].get("version"),
            device=device,
        )

        return model, checkpoint

    @staticmethod
    def save_sklearn_model(
        model: Any,
        save_path: Path,
        model_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Scikit-learn 모델을 저장합니다.

        Args:
            model: 저장할 Scikit-learn 모델
            save_path: 저장 경로 (.pkl)
            model_config: 모델 설정
            metadata: 추가 메타데이터
            performance: 성능 지표
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 모델 저장 (joblib 사용)
        joblib.dump(model, save_path)

        # 메타데이터를 별도 JSON 파일로 저장
        metadata_path = save_path.with_suffix(".json")
        metadata_dict = {
            "model_type": "sklearn",
            "model_class": type(model).__name__,
            "model_path": str(save_path),
            "model_config": model_config,
            "metadata": metadata or {},
            "performance": performance or {},
            "save_date": datetime.now().isoformat(),
        }

        # 버전 정보 추가
        if "version" not in metadata_dict["metadata"]:
            metadata_dict["metadata"]["version"] = "1.0.0"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        logger.info(
            "Scikit-learn 모델 저장 완료",
            path=str(save_path),
            model_class=type(model).__name__,
            version=metadata_dict["metadata"].get("version"),
        )

    @staticmethod
    def load_sklearn_model(load_path: Path) -> tuple[Any, Dict[str, Any]]:
        """Scikit-learn 모델을 로드합니다.

        Args:
            load_path: 모델 파일 경로 (.pkl)

        Returns:
            (로드된 모델, 메타데이터) 튜플
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        # 모델 로드
        model = joblib.load(load_path)

        # 메타데이터 로드
        metadata_path = load_path.with_suffix(".json")
        metadata_dict = {}

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

        logger.info(
            "Scikit-learn 모델 로드 완료",
            path=str(load_path),
            model_class=type(model).__name__,
            version=metadata_dict.get("metadata", {}).get("version"),
        )

        return model, metadata_dict

    @staticmethod
    def save_performance_report(
        save_path: Path,
        model_name: str,
        version: str,
        performance_metrics: Dict[str, Any],
        dataset_info: Dict[str, Any],
        training_config: Dict[str, Any],
    ) -> None:
        """성능 리포트를 저장합니다.

        Args:
            save_path: 저장 경로
            model_name: 모델 이름
            version: 모델 버전
            performance_metrics: 성능 지표
            dataset_info: 데이터셋 정보
            training_config: 학습 설정
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "model_name": model_name,
            "version": version,
            "report_date": datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "dataset_info": dataset_info,
            "training_config": training_config,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("성능 리포트 저장 완료", path=str(save_path))

    @staticmethod
    def get_model_metadata(model_path: Path) -> Dict[str, Any]:
        """모델 메타데이터를 조회합니다.

        Args:
            model_path: 모델 파일 경로

        Returns:
            메타데이터 딕셔너리
        """
        metadata_path = model_path.with_suffix(".json")

        if not metadata_path.exists():
            logger.warning("메타데이터 파일 없음", path=str(metadata_path))
            return {}

        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def list_models(models_dir: Path, model_type: Optional[str] = None) -> list[Dict[str, Any]]:
        """모델 디렉토리의 모든 모델을 나열합니다.

        Args:
            models_dir: 모델 디렉토리
            model_type: 필터링할 모델 타입 ("pytorch", "sklearn", None=전체)

        Returns:
            모델 정보 리스트
        """
        models_dir = Path(models_dir)

        if not models_dir.exists():
            logger.warning("모델 디렉토리 없음", path=str(models_dir))
            return []

        models_list = []

        # PyTorch 모델 검색
        if model_type in (None, "pytorch"):
            for pth_file in models_dir.rglob("*.pth"):
                metadata = ModelSaver.get_model_metadata(pth_file)
                if metadata:
                    models_list.append({
                        "type": "pytorch",
                        "path": str(pth_file),
                        "name": pth_file.stem,
                        **metadata,
                    })

        # Scikit-learn 모델 검색
        if model_type in (None, "sklearn"):
            for pkl_file in models_dir.rglob("*.pkl"):
                metadata = ModelSaver.get_model_metadata(pkl_file)
                if metadata:
                    models_list.append({
                        "type": "sklearn",
                        "path": str(pkl_file),
                        "name": pkl_file.stem,
                        **metadata,
                    })

        return models_list
