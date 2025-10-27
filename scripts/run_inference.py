#!/usr/bin/env python3
"""추론 실행 스크립트 (데이터 소스 추상화 버전).

데이터 소스를 선택하여 추론을 실행합니다.
파일 기반 / 실시간 스트리밍 모두 지원합니다.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.data_source import DataSourceFactory
from ocad.core.logging import get_logger

logger = get_logger(__name__)


class InferenceRunner:
    """추론 실행 클래스."""

    def __init__(self, model_path: Path, config: dict):
        """초기화.

        Args:
            model_path: 학습된 모델 경로
            config: 추론 설정
        """
        self.model_path = model_path
        self.config = config
        logger.info("InferenceRunner 초기화 완료", config=config)

    def run(self, data_source, output_path: Path = None):
        """추론 실행.

        Args:
            data_source: 데이터 소스 (DataSource 인터페이스)
            output_path: 결과 출력 경로 (선택)
        """
        print(f"\n{'=' * 70}")
        print(f"추론 실행")
        print(f"{'=' * 70}")

        # 데이터 소스 메타데이터 출력
        metadata = data_source.get_metadata()
        print(f"\n데이터 소스 정보:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        # 추론 결과 수집
        all_results = []
        batch_count = 0
        total_processed = 0

        print(f"\n추론 실행 중...")
        try:
            for batch in data_source:
                batch_results = []

                for metric_data in batch.metrics:
                    # 간소화된 룰 기반 탐지 (임계값 초과 여부)
                    scores = {}

                    # UDP Echo RTT 체크
                    udp_rtt = metric_data.get("udp_echo_rtt_ms", metric_data.get("udp_echo_rtt", 0))
                    if udp_rtt > self.config.get("rule_threshold", 10.0):
                        scores["rule_based_score"] = 1.0
                    else:
                        scores["rule_based_score"] = 0.0

                    # eCPRI delay 체크 (us → ms 변환)
                    ecpri_delay_us = metric_data.get("ecpri_delay_us", metric_data.get("ecpri_delay", 0))
                    ecpri_delay_ms = ecpri_delay_us / 1000.0
                    if ecpri_delay_ms > self.config.get("ecpri_threshold", 0.2):  # 200us = 0.2ms
                        scores["ecpri_score"] = 1.0
                    else:
                        scores["ecpri_score"] = 0.0

                    # LBM RTT 체크
                    lbm_rtt = metric_data.get("lbm_rtt_ms", metric_data.get("lbm_rtt", 0))
                    if lbm_rtt > self.config.get("lbm_threshold", 15.0):
                        scores["lbm_score"] = 1.0
                    else:
                        scores["lbm_score"] = 0.0

                    # 종합 점수 (평균)
                    composite_score = sum(scores.values()) / len(scores) if scores else 0.0

                    # 예측 (임계값)
                    threshold = self.config.get("threshold", 0.5)
                    predicted_label = "anomaly" if composite_score > threshold else "normal"

                    # 결과 저장
                    result = {
                        **metric_data,
                        **scores,
                        "composite_score": composite_score,
                        "predicted_label": predicted_label,
                    }

                    batch_results.append(result)

                all_results.extend(batch_results)
                batch_count += 1
                total_processed += len(batch_results)

                if batch_count % 5 == 0:
                    print(f"  배치 {batch_count}: {total_processed}개 처리됨")

            print(f"\n✅ 총 {total_processed}개 레코드 처리 완료")

        except Exception as e:
            logger.error("추론 실패", error=str(e))
            raise

        finally:
            data_source.close()

        # 결과 분석
        self._analyze_results(all_results)

        # 결과 저장
        if output_path:
            self._save_results(all_results, output_path)

        return all_results

    def _analyze_results(self, results: list):
        """결과 분석 및 출력."""
        print(f"\n{'=' * 70}")
        print(f"결과 분석")
        print(f"{'=' * 70}")

        df = pd.DataFrame(results)

        # 예측 분포
        pred_dist = df["predicted_label"].value_counts()
        print(f"\n예측 분포:")
        for label, count in pred_dist.items():
            percentage = count / len(df) * 100
            print(f"  {label}: {count}개 ({percentage:.1f}%)")

        # 실제 라벨이 있으면 정확도 계산
        if "label" in df.columns and df["label"].notna().any():
            accuracy = (df["predicted_label"] == df["label"]).mean()
            print(f"\n정확도: {accuracy * 100:.2f}%")

            # Confusion Matrix
            confusion = pd.crosstab(
                df["label"],
                df["predicted_label"],
                rownames=["실제"],
                colnames=["예측"]
            )
            print(f"\nConfusion Matrix:")
            print(confusion)

        # 탐지기별 평균 점수
        score_columns = [col for col in df.columns if col.endswith("_score")]
        if score_columns:
            print(f"\n탐지기별 평균 점수:")
            for col in score_columns:
                detector_name = col.replace("_score", "")
                avg_score = df[col].mean()
                print(f"  {detector_name:15}: {avg_score:.3f}")

    def _save_results(self, results: list, output_path: Path):
        """결과 저장."""
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n✅ 결과 저장: {output_path}")
        print(f"   파일 크기: {output_path.stat().st_size / 1024:.2f} KB")


def parse_args():
    """명령줄 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="추론 실행 (데이터 소스 선택 가능)"
    )

    # 데이터 소스 선택
    parser.add_argument(
        "--data-source",
        type=str,
        required=True,
        help="데이터 소스 경로 (CSV/Excel/Parquet 파일)"
    )

    # 모델 경로
    parser.add_argument(
        "--model-path",
        type=str,
        default="ocad/models/tcn",
        help="학습된 모델 경로 (기본값: ocad/models/tcn)"
    )

    # 추론 파라미터
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="이상 판단 임계값 (기본값: 0.5)"
    )

    parser.add_argument(
        "--rule-threshold",
        type=float,
        default=10.0,
        help="룰 기반 임계값 - UDP RTT (기본값: 10.0 ms)"
    )

    # 데이터 소스 설정
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="데이터 소스 배치 크기 (기본값: 100)"
    )

    # 출력 설정
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 출력 경로 (기본값: data/inference_results_{timestamp}.csv)"
    )

    # 실시간 모드 (향후 구현)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="실시간 스트리밍 모드 (향후 구현)"
    )

    return parser.parse_args()


def main():
    """메인 함수."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("OCAD 추론 실행 (데이터 소스 추상화)")
    print("=" * 70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n설정:")
    print(f"  데이터 소스: {args.data_source}")
    print(f"  모델 경로: {args.model_path}")
    print(f"  임계값: {args.threshold}")
    print(f"  배치 크기: {args.batch_size}")
    if args.streaming:
        print(f"  모드: 실시간 스트리밍 (아직 미구현)")
    print("=" * 70)

    # 데이터 소스 생성
    try:
        if args.streaming:
            print(f"\n⚠️  실시간 스트리밍 모드는 아직 구현되지 않았습니다.")
            print(f"   Kafka/WebSocket 연동이 필요합니다.")
            return 1

        data_source_config = {
            "type": "file",
            "file_path": args.data_source,
            "batch_size": args.batch_size
        }

        data_source = DataSourceFactory.create_from_config(data_source_config)
        print(f"\n✅ 데이터 소스 생성 완료")

    except FileNotFoundError:
        print(f"\n❌ 파일 없음: {args.data_source}")
        print(f"\n사용 가능한 테스트 데이터:")
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.glob("**/*.csv"):
                if "inference" in f.name or "test" in f.name:
                    print(f"  - {f}")
        print(f"\n데이터 생성:")
        print(f"  python scripts/generate_training_inference_data.py")
        return 1

    except Exception as e:
        print(f"\n❌ 데이터 소스 생성 실패: {e}")
        return 1

    # 출력 경로 설정
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/inference_results_{timestamp}.csv")

    # 추론 설정
    inference_config = {
        "threshold": args.threshold,
        "rule_threshold": args.rule_threshold,
    }

    # 추론 실행
    try:
        runner = InferenceRunner(
            model_path=Path(args.model_path),
            config=inference_config
        )

        runner.run(data_source, output_path=output_path)

        print(f"\n{'=' * 70}")
        print(f"✅ 추론 완료!")
        print(f"{'=' * 70}")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n결과 파일: {output_path}")
        print(f"\n다음 단계:")
        print(f"  - 결과 확인: head -20 {output_path}")
        print(f"  - Excel로 열기: open {output_path}")
        print("=" * 70 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 추론을 중단했습니다.")
        return 0

    except Exception as e:
        print(f"\n❌ 추론 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
