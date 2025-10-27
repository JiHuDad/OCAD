#!/usr/bin/env python3
"""모델 학습 스크립트 (데이터 소스 추상화 버전).

데이터 소스를 선택하여 모델을 학습합니다.
파일 기반 / 실시간 스트리밍 모두 지원합니다.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.data_source import DataSourceFactory
from ocad.core.logging import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """모델 학습 클래스."""

    def __init__(self, model_type: str, config: dict):
        """초기화.

        Args:
            model_type: 모델 타입 ("tcn", "isolation_forest" 등)
            config: 학습 설정
        """
        self.model_type = model_type
        self.config = config
        self.model = None

        logger.info("ModelTrainer 초기화", model_type=model_type, config=config)

    def train(self, data_source):
        """모델 학습.

        Args:
            data_source: 데이터 소스 (DataSource 인터페이스)
        """
        print(f"\n{'=' * 70}")
        print(f"모델 학습 시작: {self.model_type}")
        print(f"{'=' * 70}")

        # 데이터 소스 메타데이터 출력
        metadata = data_source.get_metadata()
        print(f"\n데이터 소스 정보:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        # 학습 데이터 수집
        all_data = []
        batch_count = 0

        print(f"\n데이터 로드 중...")
        try:
            for batch in data_source:
                all_data.extend(batch.metrics)
                batch_count += 1

                if batch_count % 10 == 0:
                    print(f"  배치 {batch_count}: {len(all_data)}개 레코드 로드됨")

            print(f"\n✅ 총 {len(all_data)}개 레코드 로드 완료")

        except Exception as e:
            logger.error("데이터 로드 실패", error=str(e))
            raise

        finally:
            data_source.close()

        # 라벨 분포 확인 (정상 데이터만 있어야 함)
        labels = [d.get("label", "unknown") for d in all_data]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\n라벨 분포:")
        for label, count in label_counts.items():
            percentage = count / len(all_data) * 100
            print(f"  {label}: {count}개 ({percentage:.1f}%)")

        # 정상 데이터만 있는지 확인
        if "anomaly" in label_counts:
            print(f"\n⚠️  경고: 학습 데이터에 이상 데이터가 포함되어 있습니다!")
            print(f"   이상 탐지 모델은 정상 데이터로만 학습해야 합니다.")
            user_input = input("\n계속 진행하시겠습니까? (y/n): ")
            if user_input.lower() != 'y':
                print("학습 중단")
                return

        # 실제 학습 수행
        print(f"\n모델 학습 중...")
        print(f"  모델 타입: {self.model_type}")
        print(f"  학습 데이터: {len(all_data)}개")
        print(f"  에폭: {self.config.get('epochs', 50)}")
        print(f"  배치 크기: {self.config.get('batch_size', 32)}")

        # TODO: 실제 모델 학습 로직
        # 현재는 데이터 소스 인터페이스만 검증
        print(f"\n⚠️  실제 학습 로직은 아직 통합되지 않았습니다.")
        print(f"   기존 train_tcn_model.py를 참고하여 통합 예정")

        print(f"\n{'=' * 70}")
        print(f"학습 완료")
        print(f"{'=' * 70}")


def parse_args():
    """명령줄 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="모델 학습 (데이터 소스 선택 가능)"
    )

    # 데이터 소스 선택
    parser.add_argument(
        "--data-source",
        type=str,
        required=True,
        help="데이터 소스 경로 (CSV/Excel/Parquet 파일)"
    )

    # 모델 설정
    parser.add_argument(
        "--model-type",
        type=str,
        default="tcn",
        choices=["tcn", "isolation_forest"],
        help="모델 타입 (기본값: tcn)"
    )

    parser.add_argument(
        "--metric-type",
        type=str,
        default="udp_echo",
        choices=["udp_echo", "ecpri", "lbm"],
        help="메트릭 타입 (기본값: udp_echo)"
    )

    # 학습 파라미터
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="학습 에폭 수 (기본값: 50)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기 (기본값: 32)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="학습률 (기본값: 0.001)"
    )

    # 데이터 소스 설정
    parser.add_argument(
        "--data-batch-size",
        type=int,
        default=1000,
        help="데이터 소스 배치 크기 (기본값: 1000)"
    )

    # 출력 설정
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ocad/models",
        help="모델 출력 디렉토리 (기본값: ocad/models)"
    )

    return parser.parse_args()


def main():
    """메인 함수."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("OCAD 모델 학습 (데이터 소스 추상화)")
    print("=" * 70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n설정:")
    print(f"  데이터 소스: {args.data_source}")
    print(f"  모델 타입: {args.model_type}")
    print(f"  메트릭 타입: {args.metric_type}")
    print(f"  에폭: {args.epochs}")
    print(f"  배치 크기: {args.batch_size}")
    print("=" * 70)

    # 데이터 소스 생성
    try:
        data_source_config = {
            "type": "file",
            "file_path": args.data_source,
            "batch_size": args.data_batch_size
        }

        data_source = DataSourceFactory.create_from_config(data_source_config)
        print(f"\n✅ 데이터 소스 생성 완료")

    except FileNotFoundError:
        print(f"\n❌ 파일 없음: {args.data_source}")
        print(f"\n사용 가능한 학습 데이터:")
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.glob("**/*.csv"):
                if "training" in f.name or "normal" in f.name:
                    print(f"  - {f}")
        print(f"\n데이터 생성:")
        print(f"  python scripts/generate_training_inference_data.py")
        return 1

    except Exception as e:
        print(f"\n❌ 데이터 소스 생성 실패: {e}")
        return 1

    # 학습 설정
    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "metric_type": args.metric_type,
        "output_dir": args.output_dir,
    }

    # 모델 학습
    try:
        trainer = ModelTrainer(
            model_type=args.model_type,
            config=training_config
        )

        trainer.train(data_source)

        print(f"\n종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 학습을 중단했습니다.")
        return 0

    except Exception as e:
        print(f"\n❌ 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
