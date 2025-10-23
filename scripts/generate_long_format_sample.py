#!/usr/bin/env python3
"""Long Format 샘플 데이터 생성 스크립트.

Wide Format과 대비되는 Long Format 예시를 생성합니다.
각 메트릭이 별도의 행으로 표현됩니다.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_long_format_data():
    """Long Format 메트릭 데이터 생성."""

    data = []
    base_time = datetime(2025, 10, 22, 9, 0, 0)

    # o-ru-001: 정상 → 이상 → 복구 패턴
    scenarios = [
        # 정상
        (0, "o-ru-001", "Tower-A", "Urban", [
            ("udp_echo_rtt", 5.2, "ms", "OK"),
            ("ecpri_delay", 102.3, "us", "OK"),
            ("lbm_rtt", 7.1, "ms", "OK"),
            ("lbm_success", 1, "bool", "OK"),
            ("ccm_miss_count", 0, "count", "OK"),
        ]),
        # Drift 시작
        (5, "o-ru-001", "Tower-A", "Urban", [
            ("udp_echo_rtt", 8.2, "ms", "WARNING"),
            ("ecpri_delay", 158.3, "us", "WARNING"),
            ("lbm_rtt", 10.5, "ms", "WARNING"),
            ("lbm_success", 1, "bool", "OK"),
            ("ccm_miss_count", 0, "count", "OK"),
        ]),
        # Critical
        (8, "o-ru-001", "Tower-A", "Urban", [
            ("udp_echo_rtt", 25.8, "ms", "CRITICAL"),
            ("ecpri_delay", 350.1, "us", "CRITICAL"),
            ("lbm_rtt", 25.5, "ms", "CRITICAL"),
            ("lbm_success", 0, "bool", "CRITICAL"),
            ("ccm_miss_count", 1, "count", "WARNING"),
        ]),
        # 복구
        (13, "o-ru-001", "Tower-A", "Urban", [
            ("udp_echo_rtt", 5.3, "ms", "OK"),
            ("ecpri_delay", 103.1, "us", "OK"),
            ("lbm_rtt", 7.2, "ms", "OK"),
            ("lbm_success", 1, "bool", "OK"),
            ("ccm_miss_count", 0, "count", "OK"),
        ]),
        # o-ru-002: Spike 패턴
        (0, "o-ru-002", "Tower-B", "Rural", [
            ("udp_echo_rtt", 4.8, "ms", "OK"),
            ("ecpri_delay", 98.5, "us", "OK"),
            ("lbm_rtt", 6.9, "ms", "OK"),
        ]),
        (4, "o-ru-002", "Tower-B", "Rural", [
            ("udp_echo_rtt", 22.5, "ms", "CRITICAL"),
            ("ecpri_delay", 320.5, "us", "CRITICAL"),
            ("lbm_rtt", 21.8, "ms", "CRITICAL"),
            ("lbm_success", 0, "bool", "CRITICAL"),
        ]),
        (5, "o-ru-002", "Tower-B", "Rural", [
            ("udp_echo_rtt", 5.0, "ms", "OK"),
            ("ecpri_delay", 100.1, "us", "OK"),
            ("lbm_rtt", 7.1, "ms", "OK"),
            ("lbm_success", 1, "bool", "OK"),
        ]),
    ]

    for offset, endpoint_id, site_name, zone, metrics in scenarios:
        timestamp = base_time + pd.Timedelta(seconds=offset)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        for metric_name, value, unit, status in metrics:
            data.append({
                "timestamp": timestamp_str,
                "endpoint_id": endpoint_id,
                "site_name": site_name,
                "zone": zone,
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "status": status
            })

    return pd.DataFrame(data)


def main():
    """메인 함수."""
    print("\n" + "=" * 60)
    print("Long Format 샘플 데이터 생성")
    print("=" * 60)

    # 출력 경로
    output_dir = Path(__file__).parent.parent / "data" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sample_oran_metrics_long.csv"

    print(f"\n출력 파일: {output_file}")

    # 데이터 생성
    print("\n데이터 생성 중...")
    df = create_long_format_data()

    # CSV로 저장
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("✅ Long Format CSV 파일 생성 완료!")
    print("=" * 60)
    print(f"\n📁 파일 위치: {output_file}")
    print(f"📊 총 {len(df)}개 레코드")
    print(f"\n특징:")
    print(f"  - Wide Format과 달리 각 메트릭이 별도의 행으로 표현됨")
    print(f"  - 새로운 메트릭 추가 시 열이 아닌 행만 추가하면 됨 (확장성 우수)")
    print(f"  - 프로그래밍 처리에 유리 (pivot, groupby 등)")
    print(f"\n비교:")
    print(f"  Wide Format: {output_dir / 'sample_oran_metrics_wide.csv'}")
    print(f"  Long Format: {output_file}")
    print("\n💡 CFM 담당자와 협의하여 어느 형식이 더 적합한지 결정하세요.")
    print("=" * 60 + "\n")

    # 샘플 출력
    print("\n샘플 데이터 (처음 10개 행):")
    print("=" * 60)
    print(df.head(10).to_string(index=False))
    print("\n...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
