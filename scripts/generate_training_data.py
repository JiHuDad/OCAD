#!/usr/bin/env python3
"""학습용 데이터셋 생성 스크립트.

시뮬레이터를 사용하여 시계열 및 다변량 학습 데이터를 생성합니다.
"""

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# OCAD 모듈 import를 위한 경로 설정
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.utils.simulator import SyntheticEndpoint
from ocad.core.logging import get_logger


logger = get_logger(__name__)


def inject_anomaly(value: float, anomaly_type: str) -> float:
    """이상 패턴을 주입합니다.

    Args:
        value: 정상 값
        anomaly_type: 이상 유형 ("spike", "drift", "loss")

    Returns:
        이상이 주입된 값
    """
    if anomaly_type == "spike":
        # 급격한 지연 증가 (3-5배)
        return value + np.random.uniform(15, 25)
    elif anomaly_type == "drift":
        # 점진적 성능 저하 (1.5-2배)
        return value + np.random.uniform(5, 10)
    elif anomaly_type == "loss":
        # 패킷 손실 (50% 확률로 NaN)
        return np.nan if np.random.random() < 0.5 else value
    return value


def generate_timeseries_dataset(
    num_endpoints: int = 10,
    duration_hours: int = 24,
    sample_interval_sec: int = 10,
    window_size: int = 10,
    anomaly_rate: float = 0.1,
    output_dir: Path = Path("data/processed"),
) -> None:
    """시계열 학습 데이터셋을 생성합니다.

    Args:
        num_endpoints: 가상 엔드포인트 수
        duration_hours: 데이터 수집 기간 (시간)
        sample_interval_sec: 샘플링 간격 (초)
        window_size: 시퀀스 길이
        anomaly_rate: 이상 비율 (0.0-1.0)
        output_dir: 출력 디렉토리
    """
    logger.info(
        "시계열 데이터셋 생성 시작",
        endpoints=num_endpoints,
        duration_hours=duration_hours,
        anomaly_rate=anomaly_rate,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sequences = []

    for endpoint_idx in range(num_endpoints):
        endpoint_id = f"o-ru-{endpoint_idx:03d}"

        logger.info(f"엔드포인트 데이터 생성 중: {endpoint_id}")

        # 가상 엔드포인트 생성
        endpoint = SyntheticEndpoint(
            endpoint_id=endpoint_id,
            role="o-ru",
        )

        # 메트릭별 시계열 버퍼
        metric_buffers = {
            "udp_echo": [],
            "ecpri": [],
            "lbm": [],
        }

        # 데이터 수집
        total_samples = (duration_hours * 3600) // sample_interval_sec

        for sample_idx in range(total_samples):
            # 메트릭 수집
            sample = endpoint.generate_sample()

            metric_buffers["udp_echo"].append(sample.udp_echo_rtt_ms if sample.udp_echo_rtt_ms is not None else 5.0)
            metric_buffers["ecpri"].append(sample.ecpri_ow_us if sample.ecpri_ow_us is not None else 100.0 / 1000.0)  # μs → ms
            metric_buffers["lbm"].append(sample.lbm_rtt_ms if sample.lbm_rtt_ms is not None else 8.0)

        # 시퀀스 생성 (슬라이딩 윈도우)
        for metric_type, values in metric_buffers.items():
            for i in range(len(values) - window_size):
                sequence = values[i:i + window_size]
                target = values[i + window_size]

                # 이상 주입
                is_anomaly = np.random.random() < anomaly_rate
                anomaly_type = None

                if is_anomaly:
                    anomaly_type = np.random.choice(["spike", "drift", "loss"])
                    target = inject_anomaly(target, anomaly_type)

                all_sequences.append({
                    "endpoint_id": endpoint_id,
                    "metric_type": metric_type,
                    "timestamp_ms": int(time.time() * 1000) + i * sample_interval_sec * 1000,
                    "sequence": sequence,
                    "target": target,
                    "is_anomaly": is_anomaly,
                    "anomaly_type": anomaly_type,
                })

    # DataFrame 생성
    df = pd.DataFrame(all_sequences)

    logger.info(
        "전체 시퀀스 생성 완료",
        total_sequences=len(df),
        anomaly_count=df["is_anomaly"].sum(),
        anomaly_rate_actual=df["is_anomaly"].mean(),
    )

    # Train/Val/Test 분할 (70/15/15)
    np.random.seed(42)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    # Parquet 저장
    train_df.to_parquet(output_dir / "timeseries_train.parquet", index=False)
    val_df.to_parquet(output_dir / "timeseries_val.parquet", index=False)
    test_df.to_parquet(output_dir / "timeseries_test.parquet", index=False)

    logger.info(
        "데이터셋 저장 완료",
        train_samples=len(train_df),
        val_samples=len(val_df),
        test_samples=len(test_df),
        output_dir=str(output_dir),
    )

    # 통계 출력
    print("\n" + "=" * 60)
    print("시계열 데이터셋 생성 완료")
    print("=" * 60)
    print(f"총 시퀀스: {len(df):,}")
    print(f"  - Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\n이상 비율: {df['is_anomaly'].mean():.1%}")
    print(f"  - Spike: {(df['anomaly_type'] == 'spike').sum():,}")
    print(f"  - Drift: {(df['anomaly_type'] == 'drift').sum():,}")
    print(f"  - Loss:  {(df['anomaly_type'] == 'loss').sum():,}")
    print(f"\n출력 위치: {output_dir.absolute()}")
    print("=" * 60 + "\n")


def generate_multivariate_dataset(
    num_endpoints: int = 10,
    duration_hours: int = 24,
    window_interval_min: int = 5,
    anomaly_rate: float = 0.1,
    output_dir: Path = Path("data/processed"),
) -> None:
    """다변량 학습 데이터셋을 생성합니다.

    Args:
        num_endpoints: 가상 엔드포인트 수
        duration_hours: 데이터 수집 기간 (시간)
        window_interval_min: 윈도우 간격 (분)
        anomaly_rate: 이상 비율
        output_dir: 출력 디렉토리
    """
    logger.info(
        "다변량 데이터셋 생성 시작",
        endpoints=num_endpoints,
        duration_hours=duration_hours,
        anomaly_rate=anomaly_rate,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []

    for endpoint_idx in range(num_endpoints):
        endpoint_id = f"o-ru-{endpoint_idx:03d}"

        logger.info(f"엔드포인트 피처 생성 중: {endpoint_id}")

        # 가상 엔드포인트 생성
        endpoint = SyntheticEndpoint(
            endpoint_id=endpoint_id,
            role="o-ru",
        )

        # 윈도우별 피처 생성
        num_windows = (duration_hours * 60) // window_interval_min

        for window_idx in range(num_windows):
            # 윈도우 내 샘플 수집 (30개)
            samples = []
            for _ in range(30):
                sample = endpoint.generate_sample()
                samples.append(metrics)

            # 피처 계산
            udp_values = [s.get("udp_echo_rtt_ms", 5.0) for s in samples]
            ecpri_values = [s.get("ecpri_delay_us", 100.0) for s in samples]
            lbm_values = [s.get("lbm_rtt_ms", 8.0) for s in samples]

            # 이상 주입
            is_anomaly = np.random.random() < anomaly_rate
            anomaly_type = None

            if is_anomaly:
                anomaly_type = np.random.choice(["concurrent", "correlated"])

                if anomaly_type == "concurrent":
                    # 동시 다발적 이상
                    udp_values = [v + np.random.uniform(10, 20) for v in udp_values]
                    ecpri_values = [v + np.random.uniform(100, 200) for v in ecpri_values]
                    lbm_values = [v + np.random.uniform(5, 15) for v in lbm_values]
                else:
                    # 상관관계 이상 (UDP는 정상, eCPRI만 증가)
                    ecpri_values = [v + np.random.uniform(150, 250) for v in ecpri_values]

            # 통계 피처
            features = {
                "endpoint_id": endpoint_id,
                "timestamp_ms": int(time.time() * 1000) + window_idx * window_interval_min * 60 * 1000,
                # UDP Echo
                "udp_echo_p95": np.percentile(udp_values, 95),
                "udp_echo_p99": np.percentile(udp_values, 99),
                "udp_echo_mean": np.mean(udp_values),
                "udp_echo_std": np.std(udp_values),
                # eCPRI
                "ecpri_p95": np.percentile(ecpri_values, 95),
                "ecpri_p99": np.percentile(ecpri_values, 99),
                "ecpri_mean": np.mean(ecpri_values),
                "ecpri_std": np.std(ecpri_values),
                # LBM
                "lbm_rtt_p95": np.percentile(lbm_values, 95),
                "lbm_rtt_p99": np.percentile(lbm_values, 99),
                "lbm_rtt_mean": np.mean(lbm_values),
                "lbm_rtt_std": np.std(lbm_values),
                # 라벨
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_type,
            }

            all_features.append(features)

    # DataFrame 생성
    df = pd.DataFrame(all_features)

    logger.info(
        "전체 피처 벡터 생성 완료",
        total_features=len(df),
        anomaly_count=df["is_anomaly"].sum(),
    )

    # Train/Val/Test 분할
    np.random.seed(42)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    # Parquet 저장
    train_df.to_parquet(output_dir / "multivariate_train.parquet", index=False)
    val_df.to_parquet(output_dir / "multivariate_val.parquet", index=False)
    test_df.to_parquet(output_dir / "multivariate_test.parquet", index=False)

    logger.info(
        "데이터셋 저장 완료",
        train_samples=len(train_df),
        val_samples=len(val_df),
        test_samples=len(test_df),
    )

    # 통계 출력
    print("\n" + "=" * 60)
    print("다변량 데이터셋 생성 완료")
    print("=" * 60)
    print(f"총 피처 벡터: {len(df):,}")
    print(f"  - Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\n이상 비율: {df['is_anomaly'].mean():.1%}")
    print(f"  - Concurrent: {(df['anomaly_type'] == 'concurrent').sum():,}")
    print(f"  - Correlated: {(df['anomaly_type'] == 'correlated').sum():,}")
    print(f"\n출력 위치: {output_dir.absolute()}")
    print("=" * 60 + "\n")


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="학습용 데이터셋 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-type",
        choices=["timeseries", "multivariate", "both"],
        default="both",
        help="생성할 데이터셋 타입",
    )
    parser.add_argument(
        "--endpoints",
        type=int,
        default=10,
        help="가상 엔드포인트 수",
    )
    parser.add_argument(
        "--duration-hours",
        type=int,
        default=24,
        help="데이터 수집 기간 (시간)",
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.1,
        help="이상 비율 (0.0-1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="출력 디렉토리",
    )

    args = parser.parse_args()

    # 데이터셋 생성
    if args.dataset_type in ("timeseries", "both"):
        generate_timeseries_dataset(
            num_endpoints=args.endpoints,
            duration_hours=args.duration_hours,
            anomaly_rate=args.anomaly_rate,
            output_dir=args.output_dir,
        )

    if args.dataset_type in ("multivariate", "both"):
        generate_multivariate_dataset(
            num_endpoints=args.endpoints,
            duration_hours=args.duration_hours,
            anomaly_rate=args.anomaly_rate,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
