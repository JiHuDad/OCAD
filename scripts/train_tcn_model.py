#!/usr/bin/env python3
"""TCN 모델 학습 스크립트.

시계열 예측을 위한 TCN 모델을 학습하고 저장합니다.
"""

import argparse
from datetime import datetime
from pathlib import Path

# OCAD 모듈 import를 위한 경로 설정
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.training.datasets import create_dataloaders
from ocad.training.trainers import TCNTrainer, TrainingConfig
from ocad.core.logging import get_logger


logger = get_logger(__name__)


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="TCN 모델 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 데이터 관련
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/processed/timeseries_train.parquet"),
        help="학습 데이터 경로",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/processed/timeseries_val.parquet"),
        help="검증 데이터 경로",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/timeseries_test.parquet"),
        help="테스트 데이터 경로",
    )
    parser.add_argument(
        "--metric-type",
        choices=["udp_echo", "ecpri", "lbm"],
        required=True,
        help="학습할 메트릭 타입",
    )

    # 모델 관련
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="TCN 히든 레이어 크기",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="입력 시퀀스 길이",
    )

    # 학습 관련
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="학습 에포크 수",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="학습률",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="가중치 감쇠 (L2 정규화)",
    )

    # 조기 종료
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="조기 종료 사용",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="조기 종료 인내 에포크 수",
    )

    # 출력 관련
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ocad/models/tcn"),
        help="모델 저장 디렉토리",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="모델 버전",
    )

    # 기타
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="학습 디바이스",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="데이터 로딩 워커 수",
    )

    args = parser.parse_args()

    # 로깅 설정

    logger.info(
        "TCN 모델 학습 시작",
        metric_type=args.metric_type,
        version=args.version,
    )

    # 데이터 존재 확인
    if not args.train_data.exists():
        logger.error("학습 데이터 파일이 없습니다", path=str(args.train_data))
        print(f"\n❌ 오류: 학습 데이터 파일이 없습니다: {args.train_data}")
        print(f"\n먼저 데이터셋을 생성하세요:")
        print(f"  python scripts/generate_training_data.py --dataset-type timeseries")
        return 1

    # DataLoader 생성
    logger.info("DataLoader 생성 중...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        metric_type=args.metric_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True,
    )

    print(f"\n{'='*60}")
    print(f"데이터셋 로딩 완료")
    print(f"{'='*60}")
    print(f"Train: {len(train_loader.dataset):,} 샘플")
    print(f"Val:   {len(val_loader.dataset):,} 샘플")
    print(f"Test:  {len(test_loader.dataset):,} 샘플")
    print(f"{'='*60}\n")

    # 학습 설정
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        patience=args.patience,
        device=args.device,
        log_interval=5,  # 5 에포크마다 로그 출력
    )

    # Trainer 생성
    trainer = TCNTrainer(
        config=config,
        input_size=1,
        hidden_size=args.hidden_size,
        output_size=1,
        sequence_length=args.sequence_length,
    )

    print(f"{'='*60}")
    print(f"TCN 모델 학습 설정")
    print(f"{'='*60}")
    print(f"메트릭 타입: {args.metric_type}")
    print(f"히든 크기: {args.hidden_size}")
    print(f"시퀀스 길이: {args.sequence_length}")
    print(f"에포크: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"학습률: {args.learning_rate}")
    print(f"디바이스: {args.device}")
    print(f"{'='*60}\n")

    # 학습 시작
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"학습 시작...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
    )

    # 최종 평가
    print(f"\n{'='*60}")
    print(f"테스트 데이터 평가 중...")
    print(f"{'='*60}")
    test_metrics = trainer.evaluate(test_loader)

    print(f"\n테스트 결과:")
    print(f"  MSE:  {test_metrics['test_mse']:.4f}")
    print(f"  MAE:  {test_metrics['test_mae']:.4f}")
    print(f"  RMSE: {test_metrics['test_rmse']:.4f}")
    print(f"  R²:   {test_metrics['test_r2']:.4f}")
    print(f"  Residual Mean: {test_metrics['residual_mean']:.4f}")
    print(f"  Residual Std:  {test_metrics['residual_std']:.4f}")

    # 최종 모델 저장
    model_filename = f"{args.metric_type}_v{args.version}.pth"
    model_path = output_dir / model_filename

    metadata = {
        "version": args.version,
        "metric_type": args.metric_type,
        "training_date": datetime.now().isoformat(),
        "sequence_length": args.sequence_length,
    }

    performance = {
        **test_metrics,
        "best_val_loss": trainer.best_metric,
        "total_epochs": trainer.current_epoch + 1,
    }

    trainer.save_model(
        save_path=model_path,
        metadata=metadata,
        performance=performance,
    )

    print(f"\n{'='*60}")
    print(f"모델 저장 완료")
    print(f"{'='*60}")
    print(f"경로: {model_path.absolute()}")
    print(f"메타데이터: {model_path.with_suffix('.json')}")
    print(f"{'='*60}\n")

    # 성능 리포트 저장
    from ocad.training.utils.model_saver import ModelSaver

    report_path = output_dir.parent / "metadata" / "performance_reports" / f"{model_filename.replace('.pth', '_report.json')}"

    ModelSaver.save_performance_report(
        save_path=report_path,
        model_name=f"TCN_{args.metric_type}",
        version=args.version,
        performance_metrics=test_metrics,
        dataset_info={
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "metric_type": args.metric_type,
        },
        training_config=config.to_dict(),
    )

    print(f"성능 리포트 저장 완료: {report_path}\n")

    logger.info(
        "TCN 모델 학습 완료",
        model_path=str(model_path),
        test_r2=test_metrics["test_r2"],
        test_mae=test_metrics["test_mae"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
