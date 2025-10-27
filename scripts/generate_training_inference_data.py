#!/usr/bin/env python3
"""학습용 및 추론용 데이터 생성 스크립트.

이상 탐지 워크플로우:
1. 학습: 정상 데이터로만 학습 (normal pattern 학습)
2. 추론: 정상/비정상 데이터 모두 테스트 (이상 여부 판단)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class TrainingInferenceDataGenerator:
    """학습/추론 데이터 생성기."""

    def __init__(self, output_dir: Path):
        """초기화."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_normal_training_data(
        self,
        num_endpoints: int = 5,
        duration_hours: int = 48,
        interval_seconds: int = 30
    ) -> pd.DataFrame:
        """정상 데이터 생성 (학습용).

        이상 탐지 모델은 정상 패턴만 학습합니다.

        Args:
            num_endpoints: 엔드포인트 수
            duration_hours: 생성할 데이터 기간 (시간)
            interval_seconds: 측정 간격 (초)

        Returns:
            DataFrame: 정상 데이터
        """
        base_time = datetime(2025, 10, 1, 0, 0, 0)
        num_samples = int(duration_hours * 3600 / interval_seconds)

        data = []

        # 엔드포인트별 기준값 (약간씩 다름)
        endpoints_config = {
            "o-ru-001": {"udp": 5.0, "ecpri": 100.0, "lbm": 7.0, "site": "Tower-A", "zone": "Urban"},
            "o-ru-002": {"udp": 4.8, "ecpri": 98.0, "lbm": 6.8, "site": "Tower-B", "zone": "Rural"},
            "o-ru-003": {"udp": 5.2, "ecpri": 102.0, "lbm": 7.2, "site": "Tower-C", "zone": "Suburban"},
            "o-du-001": {"udp": 3.5, "ecpri": 85.0, "lbm": 5.0, "site": "DC-A", "zone": "Urban"},
            "o-du-002": {"udp": 3.3, "ecpri": 83.0, "lbm": 4.8, "site": "DC-B", "zone": "Urban"},
        }

        selected_endpoints = list(endpoints_config.keys())[:num_endpoints]

        for endpoint_id in selected_endpoints:
            config = endpoints_config[endpoint_id]

            for i in range(num_samples):
                timestamp = base_time + timedelta(seconds=i * interval_seconds)

                # 정상 범위 내 변동 (±10%, 가우시안 노이즈)
                udp_rtt = config["udp"] + np.random.normal(0, config["udp"] * 0.05)
                ecpri = config["ecpri"] + np.random.normal(0, config["ecpri"] * 0.05)
                lbm_rtt = config["lbm"] + np.random.normal(0, config["lbm"] * 0.05)

                # 일일 주기 반영 (밤에는 약간 낮음)
                hour = timestamp.hour
                if 0 <= hour < 6:  # 야간
                    daily_factor = 0.95
                elif 9 <= hour < 18:  # 주간 (트래픽 많음)
                    daily_factor = 1.05
                else:  # 출퇴근 시간
                    daily_factor = 1.0

                udp_rtt *= daily_factor
                ecpri *= daily_factor
                lbm_rtt *= daily_factor

                data.append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "endpoint_id": endpoint_id,
                    "site_name": config["site"],
                    "zone": config["zone"],
                    "udp_echo_rtt_ms": round(max(0, udp_rtt), 2),
                    "ecpri_delay_us": round(max(0, ecpri), 2),
                    "lbm_rtt_ms": round(max(0, lbm_rtt), 2),
                    "lbm_success": True,
                    "ccm_interval_ms": 1000,
                    "ccm_miss_count": 0,
                    "label": "normal"  # 학습용 라벨
                })

        return pd.DataFrame(data)

    def generate_inference_test_data(self) -> pd.DataFrame:
        """추론 테스트용 데이터 생성 (정상 + 비정상).

        다양한 이상 시나리오를 포함합니다.

        Returns:
            DataFrame: 테스트 데이터 (정상 + 비정상)
        """
        base_time = datetime(2025, 10, 15, 12, 0, 0)
        data = []

        # ========================================
        # Scenario 1: 정상 데이터 (30분)
        # ========================================
        print("  [1/6] 정상 데이터 생성 중...")
        for i in range(180):  # 10초 간격, 30분
            timestamp = base_time + timedelta(seconds=i * 10)
            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": "o-ru-test-001",
                "site_name": "Test-Site-A",
                "zone": "Urban",
                "udp_echo_rtt_ms": round(5.0 + np.random.normal(0, 0.3), 2),
                "ecpri_delay_us": round(100.0 + np.random.normal(0, 5.0), 2),
                "lbm_rtt_ms": round(7.0 + np.random.normal(0, 0.4), 2),
                "lbm_success": True,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 0,
                "label": "normal",
                "scenario": "정상 운영"
            })

        base_time += timedelta(minutes=30)

        # ========================================
        # Scenario 2: Drift (점진적 증가) - 30분
        # ========================================
        print("  [2/6] Drift 이상 패턴 생성 중...")
        for i in range(180):
            timestamp = base_time + timedelta(seconds=i * 10)
            progress = i / 180.0  # 0 to 1

            # 점진적으로 증가
            udp_rtt = 5.0 + (progress * 20.0) + np.random.normal(0, 0.5)
            ecpri = 100.0 + (progress * 250.0) + np.random.normal(0, 10.0)
            lbm_rtt = 7.0 + (progress * 18.0) + np.random.normal(0, 0.6)

            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": "o-ru-test-001",
                "site_name": "Test-Site-A",
                "zone": "Urban",
                "udp_echo_rtt_ms": round(udp_rtt, 2),
                "ecpri_delay_us": round(ecpri, 2),
                "lbm_rtt_ms": round(lbm_rtt, 2),
                "lbm_success": progress < 0.8,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 1 if progress > 0.7 else 0,
                "label": "anomaly",
                "scenario": "Drift (점진적 증가)"
            })

        base_time += timedelta(minutes=30)

        # ========================================
        # Scenario 3: Spike (급격한 일시적 증가) - 20분
        # ========================================
        print("  [3/6] Spike 이상 패턴 생성 중...")
        for i in range(120):
            timestamp = base_time + timedelta(seconds=i * 10)

            # 매 2분마다 spike
            is_spike = (i % 12 == 6)

            if is_spike:
                udp_rtt = 25.0 + np.random.normal(0, 2.0)
                ecpri = 350.0 + np.random.normal(0, 20.0)
                lbm_rtt = 23.0 + np.random.normal(0, 2.0)
                lbm_success = False
                label = "anomaly"
            else:
                udp_rtt = 5.0 + np.random.normal(0, 0.3)
                ecpri = 100.0 + np.random.normal(0, 5.0)
                lbm_rtt = 7.0 + np.random.normal(0, 0.4)
                lbm_success = True
                label = "normal"

            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": "o-ru-test-001",
                "site_name": "Test-Site-A",
                "zone": "Urban",
                "udp_echo_rtt_ms": round(udp_rtt, 2),
                "ecpri_delay_us": round(ecpri, 2),
                "lbm_rtt_ms": round(lbm_rtt, 2),
                "lbm_success": lbm_success,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 1 if is_spike else 0,
                "label": label,
                "scenario": "Spike (일시적 급증)"
            })

        base_time += timedelta(minutes=20)

        # ========================================
        # Scenario 4: Jitter (불안정) - 20분
        # ========================================
        print("  [4/6] Jitter 이상 패턴 생성 중...")
        for i in range(120):
            timestamp = base_time + timedelta(seconds=i * 10)

            # 불규칙한 변동 (진폭 큼)
            if i % 3 == 0:
                udp_rtt = 5.0 + np.random.uniform(-2, 15)
                ecpri = 100.0 + np.random.uniform(-20, 200)
                label = "anomaly" if udp_rtt > 10 else "normal"
            else:
                udp_rtt = 5.0 + np.random.normal(0, 0.3)
                ecpri = 100.0 + np.random.normal(0, 5.0)
                label = "normal"

            lbm_rtt = 7.0 + np.random.normal(0, 1.0)

            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": "o-ru-test-001",
                "site_name": "Test-Site-A",
                "zone": "Urban",
                "udp_echo_rtt_ms": round(max(0, udp_rtt), 2),
                "ecpri_delay_us": round(max(0, ecpri), 2),
                "lbm_rtt_ms": round(max(0, lbm_rtt), 2),
                "lbm_success": True,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 0,
                "label": label,
                "scenario": "Jitter (불안정)"
            })

        base_time += timedelta(minutes=20)

        # ========================================
        # Scenario 5: 복합 장애 (여러 메트릭 동시 이상) - 15분
        # ========================================
        print("  [5/6] 복합 장애 패턴 생성 중...")
        for i in range(90):
            timestamp = base_time + timedelta(seconds=i * 10)

            # 모든 메트릭이 동시에 나빠짐
            udp_rtt = 30.0 + np.random.normal(0, 3.0)
            ecpri = 400.0 + np.random.normal(0, 30.0)
            lbm_rtt = 28.0 + np.random.normal(0, 3.0)

            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": "o-ru-test-001",
                "site_name": "Test-Site-A",
                "zone": "Urban",
                "udp_echo_rtt_ms": round(udp_rtt, 2),
                "ecpri_delay_us": round(ecpri, 2),
                "lbm_rtt_ms": round(lbm_rtt, 2),
                "lbm_success": False,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 2,
                "label": "anomaly",
                "scenario": "복합 장애 (Multi-metric)"
            })

        base_time += timedelta(minutes=15)

        # ========================================
        # Scenario 6: 정상 복구 - 15분
        # ========================================
        print("  [6/6] 정상 복구 패턴 생성 중...")
        for i in range(90):
            timestamp = base_time + timedelta(seconds=i * 10)

            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": "o-ru-test-001",
                "site_name": "Test-Site-A",
                "zone": "Urban",
                "udp_echo_rtt_ms": round(5.0 + np.random.normal(0, 0.3), 2),
                "ecpri_delay_us": round(100.0 + np.random.normal(0, 5.0), 2),
                "lbm_rtt_ms": round(7.0 + np.random.normal(0, 0.4), 2),
                "lbm_success": True,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 0,
                "label": "normal",
                "scenario": "정상 복구"
            })

        return pd.DataFrame(data)


def main():
    """메인 함수."""
    print("\n" + "=" * 70)
    print("학습/추론 데이터 생성")
    print("=" * 70)
    print("\n이상 탐지 워크플로우:")
    print("  1. 학습: 정상 데이터로만 학습 (정상 패턴 학습)")
    print("  2. 추론: 정상/비정상 데이터로 테스트 (이상 여부 판단)")
    print("=" * 70)

    output_dir = Path(__file__).parent.parent / "data"
    generator = TrainingInferenceDataGenerator(output_dir)

    # ========================================
    # 1. 학습 데이터 (정상만)
    # ========================================
    print("\n[1/2] 학습용 데이터 생성 중 (정상 데이터만)...")
    print("  - 엔드포인트 수: 5개")
    print("  - 기간: 48시간")
    print("  - 간격: 30초")

    df_training = generator.generate_normal_training_data(
        num_endpoints=5,
        duration_hours=48,
        interval_seconds=30
    )

    # CSV 저장
    training_csv = output_dir / "training_normal_only.csv"
    df_training.to_csv(training_csv, index=False)
    print(f"\n✅ 학습 데이터 저장: {training_csv}")
    print(f"   레코드 수: {len(df_training):,}개")
    print(f"   파일 크기: {training_csv.stat().st_size / 1024:.2f} KB")
    print(f"   모든 레코드: 정상 (label=normal)")

    # Parquet 저장 (학습 시 사용)
    training_parquet = output_dir / "training" / "normal_data.parquet"
    training_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_training.to_parquet(training_parquet, engine="pyarrow", compression="snappy")
    print(f"   Parquet: {training_parquet}")

    # ========================================
    # 2. 추론 테스트 데이터 (정상 + 비정상)
    # ========================================
    print("\n[2/2] 추론 테스트용 데이터 생성 중 (정상 + 비정상)...")

    df_inference = generator.generate_inference_test_data()

    # CSV 저장
    inference_csv = output_dir / "inference_test_scenarios.csv"
    df_inference.to_csv(inference_csv, index=False)
    print(f"\n✅ 추론 테스트 데이터 저장: {inference_csv}")
    print(f"   레코드 수: {len(df_inference):,}개")
    print(f"   파일 크기: {inference_csv.stat().st_size / 1024:.2f} KB")

    # 라벨 분포
    label_dist = df_inference['label'].value_counts()
    print(f"\n   라벨 분포:")
    for label, count in label_dist.items():
        percentage = count / len(df_inference) * 100
        print(f"     {label}: {count:,}개 ({percentage:.1f}%)")

    # 시나리오별 분포
    print(f"\n   시나리오별:")
    scenario_dist = df_inference.groupby(['scenario', 'label']).size()
    for (scenario, label), count in scenario_dist.items():
        print(f"     [{label:7}] {scenario}: {count:,}개")

    print("\n" + "=" * 70)
    print("✅ 모든 데이터 생성 완료!")
    print("=" * 70)

    print("\n📁 생성된 파일:")
    print(f"  1. {training_csv.name}")
    print(f"     - 용도: 모델 학습")
    print(f"     - 내용: 정상 데이터만 ({len(df_training):,}개)")
    print(f"     - 명령: python scripts/train_models.py")

    print(f"\n  2. {inference_csv.name}")
    print(f"     - 용도: 추론 테스트")
    print(f"     - 내용: 정상 + 비정상 ({len(df_inference):,}개)")
    print(f"     - 명령: python scripts/test_inference.py")

    print("\n💡 다음 단계:")
    print("  1. 학습: python scripts/train_models.py")
    print("  2. 추론: python scripts/test_inference.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
