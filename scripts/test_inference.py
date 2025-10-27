#!/usr/bin/env python3
"""추론 테스트 스크립트.

학습된 모델로 정상/비정상 데이터를 추론하여 이상 탐지 성능을 평가합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.loaders import CSVLoader
from ocad.detectors.residual import ResidualDetectorV2
from ocad.detectors.rule_based import RuleBasedDetector
from ocad.detectors.changepoint import ChangepointDetector
from ocad.core.config import DetectionConfig
from ocad.core.logging import get_logger

logger = get_logger(__name__)


class InferenceTest:
    """추론 테스트 클래스."""

    def __init__(self, test_data_path: Path, model_path: Path):
        """초기화.

        Args:
            test_data_path: 테스트 데이터 경로
            model_path: 학습된 모델 경로
        """
        self.test_data_path = test_data_path
        self.model_path = model_path

        # 탐지기 초기화
        self.config = DetectionConfig()
        self.detectors = self._initialize_detectors()

    def _initialize_detectors(self):
        """탐지기 초기화."""
        detectors = {}

        # 1. 룰 기반 탐지기
        detectors["rule_based"] = RuleBasedDetector(
            udp_echo_threshold=10.0,
            ecpri_threshold=200.0,
            lbm_threshold=15.0
        )

        # 2. 변화점 탐지기
        detectors["changepoint"] = ChangepointDetector(
            method="cusum",
            threshold=5.0
        )

        # 3. 잔차 기반 탐지기 (TCN) - 학습된 모델 사용
        try:
            detectors["residual"] = ResidualDetectorV2(
                model_path=self.model_path,
                use_pretrained=True
            )
            print(f"✅ 학습된 모델 로드: {self.model_path}")
        except Exception as e:
            print(f"⚠️  학습된 모델 로드 실패: {e}")
            print(f"   룰 기반 및 변화점 탐지기만 사용합니다.")
            detectors.pop("residual", None)

        return detectors

    def load_test_data(self) -> pd.DataFrame:
        """테스트 데이터 로드."""
        print(f"\n테스트 데이터 로드 중...")
        print(f"  경로: {self.test_data_path}")

        df = pd.read_csv(self.test_data_path)
        print(f"  레코드 수: {len(df):,}개")

        # 라벨 분포 확인
        label_dist = df['label'].value_counts()
        print(f"\n  라벨 분포:")
        for label, count in label_dist.items():
            percentage = count / len(df) * 100
            print(f"    {label}: {count:,}개 ({percentage:.1f}%)")

        return df

    def run_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """추론 실행.

        Args:
            df: 테스트 데이터

        Returns:
            DataFrame: 추론 결과 (예측 레이블 포함)
        """
        print(f"\n추론 실행 중...")

        results = []

        for idx, row in df.iterrows():
            # 메트릭 딕셔너리 생성
            metrics = {
                "udp_echo_rtt": row["udp_echo_rtt_ms"],
                "ecpri_delay": row["ecpri_delay_us"],
                "lbm_rtt": row["lbm_rtt_ms"],
            }

            # 각 탐지기로 이상 점수 계산
            scores = {}
            for name, detector in self.detectors.items():
                try:
                    score = detector.detect(metrics)
                    scores[f"{name}_score"] = score
                except Exception as e:
                    logger.warning(f"탐지기 {name} 오류: {e}")
                    scores[f"{name}_score"] = 0.0

            # 종합 점수 (평균)
            composite_score = np.mean(list(scores.values()))

            # 예측 (임계값 0.5)
            predicted_label = "anomaly" if composite_score > 0.5 else "normal"

            results.append({
                **row.to_dict(),
                **scores,
                "composite_score": composite_score,
                "predicted_label": predicted_label,
                "correct": predicted_label == row["label"]
            })

            # 진행 상황 표시 (100개마다)
            if (idx + 1) % 100 == 0:
                print(f"  진행: {idx + 1}/{len(df)} ({(idx + 1) / len(df) * 100:.1f}%)")

        return pd.DataFrame(results)

    def evaluate_results(self, results_df: pd.DataFrame):
        """결과 평가 및 출력.

        Args:
            results_df: 추론 결과 DataFrame
        """
        print("\n" + "=" * 70)
        print("추론 결과 평가")
        print("=" * 70)

        # 1. 전체 정확도
        accuracy = results_df["correct"].mean()
        print(f"\n전체 정확도: {accuracy * 100:.2f}%")

        # 2. Confusion Matrix
        print(f"\nConfusion Matrix:")
        confusion = pd.crosstab(
            results_df["label"],
            results_df["predicted_label"],
            rownames=["실제"],
            colnames=["예측"]
        )
        print(confusion)

        # 3. 클래스별 정확도
        print(f"\n클래스별 정확도:")
        for label in ["normal", "anomaly"]:
            subset = results_df[results_df["label"] == label]
            if len(subset) > 0:
                acc = subset["correct"].mean()
                print(f"  {label:8}: {acc * 100:.2f}% ({subset['correct'].sum()}/{len(subset)})")

        # 4. 시나리오별 정확도
        print(f"\n시나리오별 정확도:")
        scenario_acc = results_df.groupby("scenario")["correct"].agg(["mean", "count"])
        for scenario, row in scenario_acc.iterrows():
            print(f"  {scenario:30}: {row['mean'] * 100:5.1f}% ({int(row['count'])}개)")

        # 5. 탐지기별 평균 점수
        print(f"\n탐지기별 평균 점수:")
        score_columns = [col for col in results_df.columns if col.endswith("_score")]
        for col in score_columns:
            detector_name = col.replace("_score", "")
            avg_normal = results_df[results_df["label"] == "normal"][col].mean()
            avg_anomaly = results_df[results_df["label"] == "anomaly"][col].mean()
            print(f"  {detector_name:15}: 정상 {avg_normal:.3f}, 이상 {avg_anomaly:.3f}")

        # 6. 오탐 (False Positive) 분석
        false_positives = results_df[(results_df["label"] == "normal") & (results_df["predicted_label"] == "anomaly")]
        if len(false_positives) > 0:
            print(f"\n⚠️  오탐 (False Positive): {len(false_positives)}개")
            print(f"   오탐률: {len(false_positives) / len(results_df[results_df['label'] == 'normal']) * 100:.2f}%")

        # 7. 미탐 (False Negative) 분석
        false_negatives = results_df[(results_df["label"] == "anomaly") & (results_df["predicted_label"] == "normal")]
        if len(false_negatives) > 0:
            print(f"\n❌ 미탐 (False Negative): {len(false_negatives)}개")
            print(f"   미탐률: {len(false_negatives) / len(results_df[results_df['label'] == 'anomaly']) * 100:.2f}%")

    def save_results(self, results_df: pd.DataFrame, output_path: Path):
        """결과 저장.

        Args:
            results_df: 추론 결과
            output_path: 출력 경로
        """
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ 결과 저장: {output_path}")
        print(f"   파일 크기: {output_path.stat().st_size / 1024:.2f} KB")


def main():
    """메인 함수."""
    print("\n" + "=" * 70)
    print("OCAD 추론 테스트")
    print("=" * 70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 경로 설정
    project_root = Path(__file__).parent.parent
    test_data_path = project_root / "data" / "inference_test_scenarios.csv"
    model_path = project_root / "ocad" / "models" / "tcn"  # TCN 모델 디렉토리
    output_path = project_root / "data" / "inference_results.csv"

    # 파일 존재 확인
    if not test_data_path.exists():
        print(f"\n❌ 테스트 데이터 없음: {test_data_path}")
        print(f"   먼저 데이터를 생성하세요:")
        print(f"   python scripts/generate_training_inference_data.py")
        sys.exit(1)

    # 추론 테스트 실행
    tester = InferenceTest(test_data_path, model_path)

    # 1. 데이터 로드
    df_test = tester.load_test_data()

    # 2. 추론 실행
    results_df = tester.run_inference(df_test)

    # 3. 결과 평가
    tester.evaluate_results(results_df)

    # 4. 결과 저장
    tester.save_results(results_df, output_path)

    print("\n" + "=" * 70)
    print("✅ 추론 테스트 완료!")
    print("=" * 70)
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n결과 파일: {output_path}")
    print(f"\n다음 단계:")
    print(f"  - 결과 분석: head -20 {output_path}")
    print(f"  - Excel로 열기: open {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
