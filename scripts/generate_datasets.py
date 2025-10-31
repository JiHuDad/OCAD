#!/usr/bin/env python3
"""학습 및 검증 데이터셋 생성 스크립트.

사용 목적:
1. 학습용 정상 데이터 생성 (Training set)
2. 검증용 정상 데이터 생성 (Validation/Test set)
3. 검증용 비정상 데이터 생성 (Anomaly test set)
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class DatasetGenerator:
    """데이터셋 생성기."""

    def __init__(self, output_dir: Path, random_seed: int = 42):
        """초기화.

        Args:
            output_dir: 출력 디렉토리
            random_seed: 재현성을 위한 랜덤 시드
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(random_seed)

    def generate_normal_data(
        self,
        duration_hours: int = 24,
        interval_seconds: int = 60,
        endpoint_id: str = "endpoint-1",
    ) -> pd.DataFrame:
        """정상 운영 데이터 생성 (학습/검증용).

        Args:
            duration_hours: 생성할 데이터 시간 (시간)
            interval_seconds: 데이터 수집 간격 (초)
            endpoint_id: 엔드포인트 ID

        Returns:
            DataFrame: 정상 데이터
        """
        num_samples = int(duration_hours * 3600 / interval_seconds)
        base_time = datetime(2025, 10, 1, 0, 0, 0)

        print(f"\n📊 정상 데이터 생성 중...")
        print(f"   - 기간: {duration_hours}시간")
        print(f"   - 샘플 수: {num_samples}개")
        print(f"   - 간격: {interval_seconds}초")

        data = []

        # 정상 범위의 기준값
        udp_rtt_base = 5.0  # ms
        ecpri_base = 100.0  # us
        lbm_rtt_base = 7.0  # ms

        for i in range(num_samples):
            timestamp = base_time + timedelta(seconds=i * interval_seconds)

            # 정상 범위 내에서 약간의 변동 (가우시안 노이즈)
            udp_rtt = udp_rtt_base + np.random.normal(0, 0.3)
            ecpri = ecpri_base + np.random.normal(0, 5.0)
            lbm_rtt = lbm_rtt_base + np.random.normal(0, 0.4)

            # 시간대별 약간의 변화 (일중 패턴)
            hour = timestamp.hour
            if 8 <= hour <= 18:  # 낮 시간대 약간 높음
                udp_rtt += 0.2
                ecpri += 3.0
                lbm_rtt += 0.1

            data.append({
                "timestamp": timestamp,
                "endpoint_id": endpoint_id,
                "udp_echo_rtt_ms": round(max(0, udp_rtt), 3),
                "ecpri_delay_us": round(max(0, ecpri), 3),
                "lbm_rtt_ms": round(max(0, lbm_rtt), 3),
                "ccm_miss_count": 0,
            })

        df = pd.DataFrame(data)
        print(f"✅ 정상 데이터 생성 완료: {len(df)}개 레코드")
        return df

    def generate_drift_anomaly(
        self,
        duration_hours: int = 12,
        interval_seconds: int = 60,
        endpoint_id: str = "endpoint-2",
    ) -> pd.DataFrame:
        """Drift (점진적 증가) 이상 데이터 생성.

        Args:
            duration_hours: 생성할 데이터 시간 (시간)
            interval_seconds: 데이터 수집 간격 (초)
            endpoint_id: 엔드포인트 ID

        Returns:
            DataFrame: Drift 이상 데이터
        """
        num_samples = int(duration_hours * 3600 / interval_seconds)
        base_time = datetime(2025, 10, 2, 0, 0, 0)

        print(f"\n📈 Drift 이상 데이터 생성 중...")
        print(f"   - 기간: {duration_hours}시간")
        print(f"   - 샘플 수: {num_samples}개")

        data = []

        # 정상 범위에서 시작
        udp_rtt_base = 5.0
        ecpri_base = 100.0
        lbm_rtt_base = 7.0

        for i in range(num_samples):
            timestamp = base_time + timedelta(seconds=i * interval_seconds)

            # 점진적 증가 (시간에 비례)
            drift_factor = (i / num_samples) * 3.0  # 최대 3배까지 증가

            udp_rtt = (udp_rtt_base + drift_factor) + np.random.normal(0, 0.3)
            ecpri = (ecpri_base + drift_factor * 20) + np.random.normal(0, 5.0)
            lbm_rtt = (lbm_rtt_base + drift_factor * 0.5) + np.random.normal(0, 0.4)

            data.append({
                "timestamp": timestamp,
                "endpoint_id": endpoint_id,
                "udp_echo_rtt_ms": round(max(0, udp_rtt), 3),
                "ecpri_delay_us": round(max(0, ecpri), 3),
                "lbm_rtt_ms": round(max(0, lbm_rtt), 3),
                "ccm_miss_count": 0,
            })

        df = pd.DataFrame(data)
        print(f"✅ Drift 이상 데이터 생성 완료: {len(df)}개 레코드")
        return df

    def generate_spike_anomaly(
        self,
        duration_hours: int = 6,
        interval_seconds: int = 60,
        endpoint_id: str = "endpoint-3",
    ) -> pd.DataFrame:
        """Spike (급격한 증가) 이상 데이터 생성.

        Args:
            duration_hours: 생성할 데이터 시간 (시간)
            interval_seconds: 데이터 수집 간격 (초)
            endpoint_id: 엔드포인트 ID

        Returns:
            DataFrame: Spike 이상 데이터
        """
        num_samples = int(duration_hours * 3600 / interval_seconds)
        base_time = datetime(2025, 10, 3, 0, 0, 0)

        print(f"\n⚡ Spike 이상 데이터 생성 중...")
        print(f"   - 기간: {duration_hours}시간")
        print(f"   - 샘플 수: {num_samples}개")

        data = []

        # 정상 범위
        udp_rtt_base = 5.0
        ecpri_base = 100.0
        lbm_rtt_base = 7.0

        # 스파이크 발생 지점 (중간 지점)
        spike_start = int(num_samples * 0.4)
        spike_end = int(num_samples * 0.6)

        for i in range(num_samples):
            timestamp = base_time + timedelta(seconds=i * interval_seconds)

            # 스파이크 구간에서 급격히 증가
            if spike_start <= i <= spike_end:
                spike_factor = 5.0  # 5배 증가
            else:
                spike_factor = 0.0

            udp_rtt = (udp_rtt_base + spike_factor) + np.random.normal(0, 0.5)
            ecpri = (ecpri_base + spike_factor * 50) + np.random.normal(0, 10.0)
            lbm_rtt = (lbm_rtt_base + spike_factor) + np.random.normal(0, 0.8)

            data.append({
                "timestamp": timestamp,
                "endpoint_id": endpoint_id,
                "udp_echo_rtt_ms": round(max(0, udp_rtt), 3),
                "ecpri_delay_us": round(max(0, ecpri), 3),
                "lbm_rtt_ms": round(max(0, lbm_rtt), 3),
                "ccm_miss_count": 0,
            })

        df = pd.DataFrame(data)
        print(f"✅ Spike 이상 데이터 생성 완료: {len(df)}개 레코드")
        return df

    def generate_packet_loss_anomaly(
        self,
        duration_hours: int = 6,
        interval_seconds: int = 60,
        endpoint_id: str = "endpoint-4",
    ) -> pd.DataFrame:
        """패킷 손실 이상 데이터 생성.

        Args:
            duration_hours: 생성할 데이터 시간 (시간)
            interval_seconds: 데이터 수집 간격 (초)
            endpoint_id: 엔드포인트 ID

        Returns:
            DataFrame: 패킷 손실 이상 데이터
        """
        num_samples = int(duration_hours * 3600 / interval_seconds)
        base_time = datetime(2025, 10, 4, 0, 0, 0)

        print(f"\n📉 패킷 손실 이상 데이터 생성 중...")
        print(f"   - 기간: {duration_hours}시간")
        print(f"   - 샘플 수: {num_samples}개")

        data = []

        udp_rtt_base = 5.0
        ecpri_base = 100.0
        lbm_rtt_base = 7.0

        # 패킷 손실 시작 지점
        loss_start = int(num_samples * 0.5)

        for i in range(num_samples):
            timestamp = base_time + timedelta(seconds=i * interval_seconds)

            udp_rtt = udp_rtt_base + np.random.normal(0, 0.3)
            ecpri = ecpri_base + np.random.normal(0, 5.0)
            lbm_rtt = lbm_rtt_base + np.random.normal(0, 0.4)

            # 패킷 손실 발생
            ccm_miss = 0
            if i >= loss_start:
                # 10-30% 확률로 패킷 손실
                if np.random.random() < 0.2:
                    ccm_miss = np.random.randint(1, 5)

            data.append({
                "timestamp": timestamp,
                "endpoint_id": endpoint_id,
                "udp_echo_rtt_ms": round(max(0, udp_rtt), 3),
                "ecpri_delay_us": round(max(0, ecpri), 3),
                "lbm_rtt_ms": round(max(0, lbm_rtt), 3),
                "ccm_miss_count": ccm_miss,
            })

        df = pd.DataFrame(data)
        print(f"✅ 패킷 손실 이상 데이터 생성 완료: {len(df)}개 레코드")
        return df

    def save_dataset(self, df: pd.DataFrame, filename: str, formats: list[str] = None):
        """데이터셋 저장.

        Args:
            df: DataFrame
            filename: 파일명 (확장자 제외)
            formats: 저장 형식 리스트 (기본: ['csv', 'parquet'])
        """
        if formats is None:
            formats = ['csv', 'parquet']

        for fmt in formats:
            if fmt == 'csv':
                path = self.output_dir / f"{filename}.csv"
                df.to_csv(path, index=False)
                print(f"   💾 CSV 저장: {path}")
            elif fmt == 'parquet':
                path = self.output_dir / f"{filename}.parquet"
                df.to_parquet(path, index=False)
                print(f"   💾 Parquet 저장: {path}")
            elif fmt == 'excel':
                path = self.output_dir / f"{filename}.xlsx"
                df.to_excel(path, index=False)
                print(f"   💾 Excel 저장: {path}")


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="학습 및 검증 데이터셋 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets"),
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--training-hours",
        type=int,
        default=24,
        help="학습 데이터 생성 시간 (시간)",
    )
    parser.add_argument(
        "--validation-hours",
        type=int,
        default=12,
        help="검증 정상 데이터 생성 시간 (시간)",
    )
    parser.add_argument(
        "--anomaly-hours",
        type=int,
        default=6,
        help="비정상 데이터 생성 시간 (시간)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="데이터 수집 간격 (초)",
    )
    parser.add_argument(
        "--formats",
        nargs='+',
        default=['csv', 'parquet'],
        choices=['csv', 'parquet', 'excel'],
        help="저장 형식",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (재현성)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("📦 OCAD 데이터셋 생성기")
    print("=" * 70)
    print(f"\n출력 디렉토리: {args.output_dir}")
    print(f"저장 형식: {', '.join(args.formats)}")
    print(f"랜덤 시드: {args.seed}")

    # 생성기 초기화
    generator = DatasetGenerator(args.output_dir, random_seed=args.seed)

    # 1. 학습용 정상 데이터 생성
    print("\n" + "=" * 70)
    print("1️⃣ 학습용 정상 데이터 (Training Set)")
    print("=" * 70)
    training_normal = generator.generate_normal_data(
        duration_hours=args.training_hours,
        interval_seconds=args.interval,
        endpoint_id="training-endpoint",
    )
    generator.save_dataset(training_normal, "01_training_normal", args.formats)

    # 2. 검증용 정상 데이터 생성
    print("\n" + "=" * 70)
    print("2️⃣ 검증용 정상 데이터 (Validation Normal Set)")
    print("=" * 70)
    validation_normal = generator.generate_normal_data(
        duration_hours=args.validation_hours,
        interval_seconds=args.interval,
        endpoint_id="validation-endpoint",
    )
    generator.save_dataset(validation_normal, "02_validation_normal", args.formats)

    # 3. 검증용 비정상 데이터 - Drift
    print("\n" + "=" * 70)
    print("3️⃣ 검증용 비정상 데이터 - Drift (점진적 증가)")
    print("=" * 70)
    validation_drift = generator.generate_drift_anomaly(
        duration_hours=args.anomaly_hours,
        interval_seconds=args.interval,
    )
    generator.save_dataset(validation_drift, "03_validation_drift_anomaly", args.formats)

    # 4. 검증용 비정상 데이터 - Spike
    print("\n" + "=" * 70)
    print("4️⃣ 검증용 비정상 데이터 - Spike (급격한 증가)")
    print("=" * 70)
    validation_spike = generator.generate_spike_anomaly(
        duration_hours=args.anomaly_hours,
        interval_seconds=args.interval,
    )
    generator.save_dataset(validation_spike, "04_validation_spike_anomaly", args.formats)

    # 5. 검증용 비정상 데이터 - 패킷 손실
    print("\n" + "=" * 70)
    print("5️⃣ 검증용 비정상 데이터 - 패킷 손실")
    print("=" * 70)
    validation_loss = generator.generate_packet_loss_anomaly(
        duration_hours=args.anomaly_hours,
        interval_seconds=args.interval,
    )
    generator.save_dataset(validation_loss, "05_validation_packet_loss_anomaly", args.formats)

    # 요약 출력
    print("\n" + "=" * 70)
    print("✅ 데이터셋 생성 완료!")
    print("=" * 70)
    print(f"\n생성된 파일 목록:")
    print(f"   1. 학습용 정상 데이터: {len(training_normal)}개 레코드")
    print(f"   2. 검증용 정상 데이터: {len(validation_normal)}개 레코드")
    print(f"   3. Drift 이상 데이터: {len(validation_drift)}개 레코드")
    print(f"   4. Spike 이상 데이터: {len(validation_spike)}개 레코드")
    print(f"   5. 패킷 손실 이상 데이터: {len(validation_loss)}개 레코드")

    print(f"\n💡 다음 단계:")
    print(f"   1. 학습 데이터로 모델 학습:")
    print(f"      python scripts/prepare_timeseries_data_v2.py \\")
    print(f"          --input-csv {args.output_dir}/01_training_normal.csv \\")
    print(f"          --output-dir data/processed --metric-type udp_echo")
    print(f"      python scripts/train_tcn_model.py --metric-type udp_echo")
    print(f"")
    print(f"   2. 검증 데이터로 추론 실행:")
    print(f"      python scripts/inference_simple.py \\")
    print(f"          --input {args.output_dir}/02_validation_normal.csv \\")
    print(f"          --output results_normal.csv")
    print(f"      python scripts/inference_simple.py \\")
    print(f"          --input {args.output_dir}/03_validation_drift_anomaly.csv \\")
    print(f"          --output results_drift.csv")


if __name__ == "__main__":
    main()
