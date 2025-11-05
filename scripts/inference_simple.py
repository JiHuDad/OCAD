#!/usr/bin/env python3
"""간단한 OCAD 추론 스크립트.

학습된 모델을 사용하여 입력 데이터에 대한 이상 탐지를 수행합니다.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle
import torch
import sys

# OCAD 모듈 import를 위한 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.detectors.residual import ResidualDetector
from ocad.detectors.multivariate import MultivariateDetector
from ocad.core.config import DetectionConfig
from ocad.core.logging import get_logger
from ocad.core.models import FeatureVector, Capabilities


logger = get_logger(__name__)


def load_input_data(input_path: Path) -> pd.DataFrame:
    """입력 데이터 로드 (CSV, Excel, Parquet 지원)."""
    suffix = input_path.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(input_path)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(input_path)
    elif suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info(f"Loaded {len(df)} records from {input_path}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[list[FeatureVector], list[datetime]]:
    """데이터프레임을 FeatureVector 리스트로 변환."""
    features = []
    timestamps = []

    for _, row in df.iterrows():
        # timestamp 파싱
        if 'timestamp' in row:
            ts = pd.to_datetime(row['timestamp'])
        else:
            ts = datetime.now()

        timestamps.append(ts)

        # FeatureVector 생성
        feature = FeatureVector(
            ts_ms=int(ts.timestamp() * 1000),
            endpoint_id=row.get('endpoint_id', 'unknown'),
            window_size_ms=60000,  # 1분 윈도우

            # UDP Echo 메트릭
            udp_echo_p95=float(row.get('udp_echo_rtt_ms', 0)),
            udp_echo_p99=float(row.get('udp_echo_rtt_ms', 0)),
            udp_echo_mean=float(row.get('udp_echo_rtt_ms', 0)),
            udp_echo_ewma=float(row.get('udp_echo_rtt_ms', 0)),
            udp_echo_gradient=0.0,
            udp_echo_cusum=0.0,

            # eCPRI 메트릭
            ecpri_p95=float(row.get('ecpri_delay_us', 0)),
            ecpri_p99=float(row.get('ecpri_delay_us', 0)),
            ecpri_mean=float(row.get('ecpri_delay_us', 0)),
            ecpri_ewma=float(row.get('ecpri_delay_us', 0)),
            ecpri_gradient=0.0,
            ecpri_cusum=0.0,

            # LBM 메트릭
            lbm_rtt_p95=float(row.get('lbm_rtt_ms', 0)),
            lbm_rtt_p99=float(row.get('lbm_rtt_ms', 0)),
            lbm_mean=float(row.get('lbm_rtt_ms', 0)),
            lbm_ewma=float(row.get('lbm_rtt_ms', 0)),
            lbm_gradient=0.0,
            lbm_cusum=0.0,

            # CCM 메트릭
            ccm_miss_count=int(row.get('ccm_miss_count', 0)),
            ccm_miss_rate=float(row.get('ccm_miss_count', 0)),
        )
        features.append(feature)

    return features, timestamps


def run_inference(
    input_path: Path,
    model_dir: Path,
    output_path: Path | None = None,
    use_residual: bool = True,
    use_multivariate: bool = True,
) -> pd.DataFrame:
    """추론 실행."""
    # 입력 데이터 로드
    print(f"\n{'='*70}")
    print("OCAD 추론 실행")
    print(f"{'='*70}")
    print(f"\n📂 입력 데이터: {input_path}")
    print(f"📁 모델 디렉토리: {model_dir}")

    df = load_input_data(input_path)
    print(f"✅ {len(df)}개 레코드 로드 완료")

    # FeatureVector로 변환
    print("\n⚙️  피처 준비 중...")
    features, timestamps = prepare_features(df)

    # 설정 생성
    config = DetectionConfig(
        use_pretrained_models=True,
        model_path=str(model_dir / "tcn"),
        multivariate_model_path=str(model_dir / "isolation_forest"),
    )

    # Detector 초기화
    detectors = {}
    if use_residual:
        print("🔧 ResidualDetector 초기화 중...")
        detectors['residual'] = ResidualDetector(config)

    if use_multivariate:
        print("🔧 MultivariateDetector 초기화 중...")
        detectors['multivariate'] = MultivariateDetector(config)

    # 기본 capabilities (모든 기능 활성화)
    capabilities = Capabilities(
        ccm_min=True,
        lbm=True,
        udp_echo=True,
        ecpri_delay=True,
        lldp=True,
    )

    # 추론 실행
    print(f"\n🚀 {len(features)}개 샘플에 대한 추론 실행 중...")
    results = []

    for i, (feature, ts) in enumerate(zip(features, timestamps)):
        if i % 100 == 0 and i > 0:
            print(f"   진행: {i}/{len(features)} ({i/len(features)*100:.1f}%)")

        result = {
            'timestamp': ts,
            'endpoint_id': feature.endpoint_id,
        }

        # 각 detector 실행 (상세 정보 포함)
        for name, detector in detectors.items():
            try:
                # detect_detailed() 사용하여 상세 정보 얻기
                detection_result = detector.detect_detailed(feature, capabilities)

                # 기본 점수
                result[f'{name}_score'] = detection_result.score
                result[f'{name}_anomaly'] = 1 if detection_result.is_anomaly else 0

                # 상세 정보 추가
                if name == 'residual':
                    # ResidualDetector: 각 메트릭의 예측값, 실제값, 오차
                    for metric_name, detail in detection_result.metric_details.items():
                        prefix = f'residual_{metric_name}'
                        result[f'{prefix}_actual'] = detail.actual_value
                        result[f'{prefix}_predicted'] = detail.predicted_value
                        result[f'{prefix}_error'] = detail.error
                        result[f'{prefix}_normalized_error'] = detail.normalized_error

                    result['residual_explanation'] = detection_result.explanation

                elif name == 'multivariate':
                    # MultivariateDetector: 다변량 패턴 정보
                    result['multivariate_explanation'] = detection_result.explanation
                    if detection_result.dominant_metric:
                        result['multivariate_dominant_metric'] = detection_result.dominant_metric

            except Exception as e:
                logger.error(f"Error in {name} detector: {e}")
                result[f'{name}_score'] = 0.0
                result[f'{name}_anomaly'] = 0

        # 최종 이상 여부 (OR 조합)
        anomaly_scores = [result.get(f'{name}_score', 0) for name in detectors.keys()]
        result['final_score'] = max(anomaly_scores) if anomaly_scores else 0.0
        result['is_anomaly'] = 1 if result['final_score'] > 0.5 else 0

        results.append(result)

    # 결과 DataFrame 생성
    results_df = pd.DataFrame(results)

    # 통계 출력
    print("\n" + "="*70)
    print("📊 추론 결과 요약")
    print("="*70)
    print(f"총 샘플 수: {len(results_df)}")
    print(f"이상 탐지 수: {results_df['is_anomaly'].sum()}")
    print(f"이상 탐지율: {results_df['is_anomaly'].mean():.2%}")

    for name in detectors.keys():
        score_col = f'{name}_score'
        if score_col in results_df.columns:
            print(f"\n{name.upper()} Detector:")
            print(f"  평균 점수: {results_df[score_col].mean():.4f}")
            print(f"  최대 점수: {results_df[score_col].max():.4f}")
            print(f"  이상 수: {results_df[f'{name}_anomaly'].sum()}")

    print("="*70)

    # 결과 저장
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix == '.csv':
            results_df.to_csv(output_path, index=False)
        elif suffix == '.parquet':
            results_df.to_parquet(output_path, index=False)
        else:
            results_df.to_csv(output_path.with_suffix('.csv'), index=False)

        print(f"\n💾 결과 저장 완료: {output_path}")

    return results_df


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="OCAD 추론 실행 (간단 버전)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 필수 인자
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="입력 데이터 파일 (CSV, Excel, Parquet)",
    )

    # 선택적 인자
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="결과 저장 경로 (미지정 시 자동 생성)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("ocad/models"),
        help="학습된 모델 디렉토리",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="ResidualDetector 비활성화",
    )
    parser.add_argument(
        "--no-multivariate",
        action="store_true",
        help="MultivariateDetector 비활성화",
    )

    args = parser.parse_args()

    # 출력 경로 자동 생성
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"data/results/inference_{timestamp}.csv")

    # 추론 실행
    run_inference(
        input_path=args.input,
        model_dir=args.model_dir,
        output_path=args.output,
        use_residual=not args.no_residual,
        use_multivariate=not args.no_multivariate,
    )


if __name__ == "__main__":
    main()
