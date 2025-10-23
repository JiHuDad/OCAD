#!/usr/bin/env python3
"""사람이 읽을 수 있는 다양한 샘플 데이터 생성 스크립트.

다양한 시나리오와 형식으로 샘플 데이터를 생성합니다:
- 일간/주간/월간 데이터
- 정상 운영 vs 장애 시나리오
- 단일 장비 vs 여러 장비
- CSV, Excel, Parquet 형식
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class SampleDataGenerator:
    """샘플 데이터 생성기."""

    def __init__(self, output_dir: Path):
        """초기화.

        Args:
            output_dir: 출력 디렉토리
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_normal_operation(
        self,
        endpoint_id: str = "o-ru-001",
        site_name: str = "Tower-A",
        zone: str = "Urban",
        duration_hours: int = 24,
        interval_seconds: int = 10
    ) -> pd.DataFrame:
        """정상 운영 데이터 생성.

        Args:
            endpoint_id: 엔드포인트 ID
            site_name: 사이트 이름
            zone: 지역
            duration_hours: 생성할 데이터 시간 (시간)
            interval_seconds: 데이터 수집 간격 (초)

        Returns:
            DataFrame: 생성된 데이터
        """
        num_samples = int(duration_hours * 3600 / interval_seconds)
        base_time = datetime(2025, 10, 22, 0, 0, 0)

        data = []

        # 정상 범위 값
        udp_rtt_base = 5.0  # ms
        ecpri_base = 100.0  # us
        lbm_rtt_base = 7.0  # ms

        for i in range(num_samples):
            timestamp = base_time + timedelta(seconds=i * interval_seconds)

            # 정상 범위 내에서 약간의 변동 (±10%)
            udp_rtt = udp_rtt_base + np.random.normal(0, 0.3)
            ecpri = ecpri_base + np.random.normal(0, 5.0)
            lbm_rtt = lbm_rtt_base + np.random.normal(0, 0.4)

            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": endpoint_id,
                "site_name": site_name,
                "zone": zone,
                "udp_echo_rtt_ms": round(max(0, udp_rtt), 2),
                "ecpri_delay_us": round(max(0, ecpri), 2),
                "lbm_rtt_ms": round(max(0, lbm_rtt), 2),
                "lbm_success": True,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 0,
                "notes": "정상 운영"
            })

        return pd.DataFrame(data)

    def generate_drift_anomaly(
        self,
        endpoint_id: str = "o-ru-002",
        site_name: str = "Tower-B",
        zone: str = "Rural"
    ) -> pd.DataFrame:
        """Drift (점진적 증가) 이상 패턴 생성.

        Args:
            endpoint_id: 엔드포인트 ID
            site_name: 사이트 이름
            zone: 지역

        Returns:
            DataFrame: 생성된 데이터
        """
        base_time = datetime(2025, 10, 22, 12, 0, 0)
        data = []

        # Phase 1: 정상 (30분)
        for i in range(180):  # 10초 간격
            timestamp = base_time + timedelta(seconds=i * 10)
            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": endpoint_id,
                "site_name": site_name,
                "zone": zone,
                "udp_echo_rtt_ms": round(5.0 + np.random.normal(0, 0.3), 2),
                "ecpri_delay_us": round(100.0 + np.random.normal(0, 5.0), 2),
                "lbm_rtt_ms": round(7.0 + np.random.normal(0, 0.4), 2),
                "lbm_success": True,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 0,
                "notes": "정상 운영"
            })

        # Phase 2: Drift 시작 (1시간)
        for i in range(360):
            timestamp = base_time + timedelta(seconds=(180 + i) * 10)
            progress = i / 360.0  # 0 to 1

            # 점진적으로 증가
            udp_rtt = 5.0 + (progress * 15.0) + np.random.normal(0, 0.5)
            ecpri = 100.0 + (progress * 200.0) + np.random.normal(0, 10.0)
            lbm_rtt = 7.0 + (progress * 15.0) + np.random.normal(0, 0.6)

            status = "Drift 진행 중" if progress > 0.3 else "약간 증가"

            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": endpoint_id,
                "site_name": site_name,
                "zone": zone,
                "udp_echo_rtt_ms": round(udp_rtt, 2),
                "ecpri_delay_us": round(ecpri, 2),
                "lbm_rtt_ms": round(lbm_rtt, 2),
                "lbm_success": progress < 0.8,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 1 if progress > 0.7 else 0,
                "notes": status
            })

        # Phase 3: 복구 (30분)
        for i in range(180):
            timestamp = base_time + timedelta(seconds=(540 + i) * 10)
            progress = 1.0 - (i / 180.0)  # 1 to 0

            udp_rtt = 5.0 + (progress * 15.0) + np.random.normal(0, 0.5)
            ecpri = 100.0 + (progress * 200.0) + np.random.normal(0, 10.0)
            lbm_rtt = 7.0 + (progress * 15.0) + np.random.normal(0, 0.6)

            status = "복구 중" if progress > 0.2 else "정상 복구"

            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": endpoint_id,
                "site_name": site_name,
                "zone": zone,
                "udp_echo_rtt_ms": round(udp_rtt, 2),
                "ecpri_delay_us": round(ecpri, 2),
                "lbm_rtt_ms": round(lbm_rtt, 2),
                "lbm_success": True,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 0,
                "notes": status
            })

        return pd.DataFrame(data)

    def generate_spike_anomaly(
        self,
        endpoint_id: str = "o-ru-003",
        site_name: str = "Tower-C",
        zone: str = "Suburban"
    ) -> pd.DataFrame:
        """Spike (급격한 일시적 증가) 이상 패턴 생성."""
        base_time = datetime(2025, 10, 22, 15, 0, 0)
        data = []

        for i in range(360):  # 1시간
            timestamp = base_time + timedelta(seconds=i * 10)

            # 매 10분마다 spike 발생
            is_spike = (i % 60 == 30)

            if is_spike:
                udp_rtt = 20.0 + np.random.normal(0, 2.0)
                ecpri = 300.0 + np.random.normal(0, 20.0)
                lbm_rtt = 22.0 + np.random.normal(0, 2.0)
                notes = "🚨 Spike 발생"
            else:
                udp_rtt = 5.0 + np.random.normal(0, 0.3)
                ecpri = 100.0 + np.random.normal(0, 5.0)
                lbm_rtt = 7.0 + np.random.normal(0, 0.4)
                notes = "정상"

            data.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint_id": endpoint_id,
                "site_name": site_name,
                "zone": zone,
                "udp_echo_rtt_ms": round(udp_rtt, 2),
                "ecpri_delay_us": round(ecpri, 2),
                "lbm_rtt_ms": round(lbm_rtt, 2),
                "lbm_success": not is_spike,
                "ccm_interval_ms": 1000,
                "ccm_miss_count": 1 if is_spike else 0,
                "notes": notes
            })

        return pd.DataFrame(data)

    def generate_multi_endpoint_data(self) -> pd.DataFrame:
        """여러 엔드포인트 데이터 생성."""
        endpoints = [
            ("o-ru-001", "Tower-A", "Urban"),
            ("o-ru-002", "Tower-B", "Rural"),
            ("o-ru-003", "Tower-C", "Suburban"),
            ("o-du-001", "Datacenter-A", "Urban"),
        ]

        all_data = []

        for endpoint_id, site_name, zone in endpoints:
            # 각 엔드포인트마다 1시간 데이터
            df = self.generate_normal_operation(
                endpoint_id=endpoint_id,
                site_name=site_name,
                zone=zone,
                duration_hours=1,
                interval_seconds=30
            )
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True).sort_values("timestamp")

    def save_as_csv(self, df: pd.DataFrame, filename: str):
        """CSV로 저장."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ CSV 저장: {output_path}")
        print(f"   레코드 수: {len(df):,}개")
        print(f"   파일 크기: {output_path.stat().st_size / 1024:.2f} KB")

    def save_as_excel(self, df: pd.DataFrame, filename: str, sheet_name: str = "메트릭 데이터"):
        """Excel로 저장."""
        output_path = self.output_dir / filename

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"✅ Excel 저장: {output_path}")
        print(f"   레코드 수: {len(df):,}개")
        print(f"   파일 크기: {output_path.stat().st_size / 1024:.2f} KB")

    def save_as_parquet(self, df: pd.DataFrame, filename: str):
        """Parquet로 저장."""
        output_path = self.output_dir / filename
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")
        print(f"✅ Parquet 저장: {output_path}")
        print(f"   레코드 수: {len(df):,}개")
        print(f"   파일 크기: {output_path.stat().st_size / 1024:.2f} KB")


def main():
    """메인 함수."""
    print("\n" + "=" * 60)
    print("사람이 읽을 수 있는 샘플 데이터 생성")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "samples"
    generator = SampleDataGenerator(output_dir)

    # 1. 정상 운영 데이터 (24시간)
    print("\n[1/6] 정상 운영 데이터 생성 중 (24시간)...")
    df_normal = generator.generate_normal_operation(
        duration_hours=24,
        interval_seconds=60  # 1분 간격
    )
    generator.save_as_csv(df_normal, "01_normal_operation_24h.csv")

    # 2. Drift 이상 패턴
    print("\n[2/6] Drift 이상 패턴 생성 중...")
    df_drift = generator.generate_drift_anomaly()
    generator.save_as_csv(df_drift, "02_drift_anomaly.csv")
    generator.save_as_excel(df_drift, "02_drift_anomaly.xlsx")

    # 3. Spike 이상 패턴
    print("\n[3/6] Spike 이상 패턴 생성 중...")
    df_spike = generator.generate_spike_anomaly()
    generator.save_as_csv(df_spike, "03_spike_anomaly.csv")

    # 4. 여러 엔드포인트 데이터
    print("\n[4/6] 여러 엔드포인트 데이터 생성 중...")
    df_multi = generator.generate_multi_endpoint_data()
    generator.save_as_csv(df_multi, "04_multi_endpoint.csv")
    generator.save_as_parquet(df_multi, "04_multi_endpoint.parquet")

    # 5. 주간 데이터 (1주일, 5분 간격)
    print("\n[5/6] 주간 데이터 생성 중 (1주일)...")
    df_weekly = generator.generate_normal_operation(
        duration_hours=24 * 7,
        interval_seconds=300  # 5분 간격
    )
    generator.save_as_parquet(df_weekly, "05_weekly_data.parquet")

    # 6. 종합 예제 (정상 + Drift + Spike)
    print("\n[6/6] 종합 예제 데이터 생성 중...")
    df_normal_short = generator.generate_normal_operation(
        endpoint_id="o-ru-001",
        duration_hours=2,
        interval_seconds=30
    )
    df_comprehensive = pd.concat([
        df_normal_short,
        df_drift,
        df_spike
    ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    generator.save_as_excel(df_comprehensive, "06_comprehensive_example.xlsx")

    print("\n" + "=" * 60)
    print("✅ 모든 샘플 데이터 생성 완료!")
    print("=" * 60)
    print(f"\n📁 출력 디렉토리: {output_dir}")
    print("\n생성된 파일:")
    print("  1. 01_normal_operation_24h.csv      - 정상 운영 (24시간)")
    print("  2. 02_drift_anomaly.csv/.xlsx       - Drift 이상 패턴")
    print("  3. 03_spike_anomaly.csv             - Spike 이상 패턴")
    print("  4. 04_multi_endpoint.csv/.parquet   - 여러 엔드포인트")
    print("  5. 05_weekly_data.parquet           - 주간 데이터 (7일)")
    print("  6. 06_comprehensive_example.xlsx   - 종합 예제")
    print("\n💡 사용 방법:")
    print("  - CSV 파일: Excel, 텍스트 에디터로 열람")
    print("  - Excel 파일: Microsoft Excel, LibreOffice로 열람")
    print("  - Parquet 파일: 파이썬 pandas로 열람")
    print("    >>> import pandas as pd")
    print("    >>> df = pd.read_parquet('05_weekly_data.parquet')")
    print("    >>> print(df.head())")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
