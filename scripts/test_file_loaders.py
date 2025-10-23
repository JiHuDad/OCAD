#!/usr/bin/env python3
"""파일 로더 테스트 스크립트.

CSV, Excel, Parquet 로더와 형식 변환기를 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.loaders import CSVLoader, ExcelLoader, ParquetLoader, FormatConverter
from ocad.core.logging import get_logger

logger = get_logger(__name__)


def test_csv_loader_wide():
    """CSV Loader 테스트 (Wide Format)."""
    print("\n" + "=" * 60)
    print("Test 1: CSV Loader (Wide Format)")
    print("=" * 60)

    csv_path = Path("data/samples/sample_oran_metrics_wide.csv")

    if not csv_path.exists():
        print(f"❌ 파일 없음: {csv_path}")
        return False

    try:
        loader = CSVLoader(strict_mode=False)
        result = loader.load(csv_path)

        print(f"\n결과:")
        print(f"  성공: {result.success}")
        print(f"  전체 레코드: {result.total_records}")
        print(f"  유효 레코드: {result.valid_records}")
        print(f"  무효 레코드: {result.invalid_records}")
        print(f"  성공률: {result.success_rate * 100:.1f}%")

        if result.metrics:
            print(f"\n첫 번째 메트릭:")
            metric = result.metrics[0]
            print(f"  endpoint_id: {metric.endpoint_id}")
            print(f"  metric_type: {metric.metric_type}")
            print(f"  value: {metric.value} {metric.unit}")
            print(f"  timestamp: {metric.timestamp}")
            print(f"  labels: {metric.labels}")

        if result.errors:
            print(f"\n에러 (최대 3개):")
            for error in result.errors[:3]:
                print(f"  - {error}")

        print("\n✅ CSV Loader (Wide) 테스트 완료")
        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_loader_long():
    """CSV Loader 테스트 (Long Format)."""
    print("\n" + "=" * 60)
    print("Test 2: CSV Loader (Long Format)")
    print("=" * 60)

    csv_path = Path("data/samples/sample_oran_metrics_long.csv")

    if not csv_path.exists():
        print(f"❌ 파일 없음: {csv_path}")
        return False

    try:
        loader = CSVLoader(strict_mode=False, format_type="auto")
        result = loader.load(csv_path)

        print(f"\n결과:")
        print(f"  성공: {result.success}")
        print(f"  전체 레코드: {result.total_records}")
        print(f"  유효 레코드: {result.valid_records}")
        print(f"  무효 레코드: {result.invalid_records}")
        print(f"  성공률: {result.success_rate * 100:.1f}%")

        if result.metrics:
            print(f"\n첫 번째 메트릭:")
            metric = result.metrics[0]
            print(f"  endpoint_id: {metric.endpoint_id}")
            print(f"  metric_type: {metric.metric_type}")
            print(f"  value: {metric.value} {metric.unit}")

        print("\n✅ CSV Loader (Long) 테스트 완료")
        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_excel_loader():
    """Excel Loader 테스트."""
    print("\n" + "=" * 60)
    print("Test 3: Excel Loader")
    print("=" * 60)

    excel_path = Path("data/samples/sample_oran_metrics.xlsx")

    if not excel_path.exists():
        print(f"❌ 파일 없음: {excel_path}")
        return False

    try:
        # Sheet "메트릭 데이터" 로드
        loader = ExcelLoader(strict_mode=False, sheet_name="메트릭 데이터")
        result = loader.load(excel_path)

        print(f"\n결과:")
        print(f"  성공: {result.success}")
        print(f"  전체 레코드: {result.total_records}")
        print(f"  유효 레코드: {result.valid_records}")
        print(f"  무효 레코드: {result.invalid_records}")
        print(f"  성공률: {result.success_rate * 100:.1f}%")

        if result.metrics:
            print(f"\n첫 번째 메트릭:")
            metric = result.metrics[0]
            print(f"  endpoint_id: {metric.endpoint_id}")
            print(f"  metric_type: {metric.metric_type}")
            print(f"  value: {metric.value} {metric.unit}")

        print("\n✅ Excel Loader 테스트 완료")
        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_to_parquet_conversion():
    """CSV → Parquet 변환 테스트."""
    print("\n" + "=" * 60)
    print("Test 4: CSV → Parquet 변환")
    print("=" * 60)

    csv_path = Path("data/samples/sample_oran_metrics_wide.csv")
    parquet_path = Path("data/samples/sample_oran_metrics_wide.parquet")

    if not csv_path.exists():
        print(f"❌ 파일 없음: {csv_path}")
        return False

    try:
        # 변환
        result_path = FormatConverter.csv_to_parquet(csv_path, parquet_path)

        print(f"\n변환 완료:")
        print(f"  입력: {csv_path}")
        print(f"  출력: {result_path}")
        print(f"  파일 크기: {result_path.stat().st_size / 1024:.2f} KB")

        # Parquet 파일 로드 테스트
        loader = ParquetLoader(strict_mode=False)
        load_result = loader.load(result_path)

        print(f"\nParquet 로드 결과:")
        print(f"  성공: {load_result.success}")
        print(f"  유효 레코드: {load_result.valid_records}")

        print("\n✅ CSV → Parquet 변환 테스트 완료")
        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wide_to_long_conversion():
    """Wide → Long 변환 테스트."""
    print("\n" + "=" * 60)
    print("Test 5: Wide → Long 변환")
    print("=" * 60)

    input_path = Path("data/samples/sample_oran_metrics_wide.csv")
    output_path = Path("data/samples/sample_oran_metrics_wide_to_long.csv")

    if not input_path.exists():
        print(f"❌ 파일 없음: {input_path}")
        return False

    try:
        # 변환
        result_path = FormatConverter.wide_to_long(input_path, output_path)

        print(f"\n변환 완료:")
        print(f"  입력: {input_path}")
        print(f"  출력: {result_path}")

        # Long Format 로드 테스트
        loader = CSVLoader(strict_mode=False, format_type="long")
        load_result = loader.load(result_path)

        print(f"\nLong Format 로드 결과:")
        print(f"  성공: {load_result.success}")
        print(f"  유효 레코드: {load_result.valid_records}")

        print("\n✅ Wide → Long 변환 테스트 완료")
        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_long_to_wide_conversion():
    """Long → Wide 변환 테스트."""
    print("\n" + "=" * 60)
    print("Test 6: Long → Wide 변환")
    print("=" * 60)

    input_path = Path("data/samples/sample_oran_metrics_long.csv")
    output_path = Path("data/samples/sample_oran_metrics_long_to_wide.csv")

    if not input_path.exists():
        print(f"❌ 파일 없음: {input_path}")
        return False

    try:
        # 변환
        result_path = FormatConverter.long_to_wide(input_path, output_path)

        print(f"\n변환 완료:")
        print(f"  입력: {input_path}")
        print(f"  출력: {result_path}")

        # Wide Format 로드 테스트
        loader = CSVLoader(strict_mode=False, format_type="wide")
        load_result = loader.load(result_path)

        print(f"\nWide Format 로드 결과:")
        print(f"  성공: {load_result.success}")
        print(f"  유효 레코드: {load_result.valid_records}")

        print("\n✅ Long → Wide 변환 테스트 완료")
        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수."""
    print("\n" + "=" * 60)
    print("파일 로더 테스트")
    print("=" * 60)

    results = []

    # 테스트 실행
    results.append(("CSV Loader (Wide)", test_csv_loader_wide()))
    results.append(("CSV Loader (Long)", test_csv_loader_long()))
    results.append(("Excel Loader", test_excel_loader()))
    results.append(("CSV → Parquet", test_csv_to_parquet_conversion()))
    results.append(("Wide → Long", test_wide_to_long_conversion()))
    results.append(("Long → Wide", test_long_to_wide_conversion()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:<25} {status}")

        if result:
            passed += 1
        else:
            failed += 1

    print("\n" + "-" * 60)
    print(f"총 {len(results)}개 테스트")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
