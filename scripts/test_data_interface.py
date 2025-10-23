#!/usr/bin/env python3
"""데이터 인터페이스 테스트 스크립트.

새로운 데이터 인터페이스 API를 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import time
from typing import List, Dict

# API 베이스 URL
BASE_URL = "http://localhost:8080/api/v1"


def test_single_metric():
    """단일 메트릭 전송 테스트."""
    print("\n" + "=" * 60)
    print("단일 메트릭 전송 테스트")
    print("=" * 60)

    metric = {
        "endpoint_id": "o-ru-001",
        "timestamp": int(time.time() * 1000),
        "metric_type": "udp_echo_rtt",
        "value": 5.2,
        "unit": "ms",
        "labels": {
            "site": "tower-A",
            "zone": "urban"
        },
        "quality": {
            "source_reliability": 0.98,
            "measurement_error": 0.1
        }
    }

    print(f"\n전송할 메트릭:")
    print(f"  endpoint_id: {metric['endpoint_id']}")
    print(f"  metric_type: {metric['metric_type']}")
    print(f"  value: {metric['value']} {metric['unit']}")

    try:
        response = requests.post(
            f"{BASE_URL}/metrics/",
            json=metric,
            timeout=5
        )

        if response.status_code == 202:
            result = response.json()
            print(f"\n✅ 성공!")
            print(f"  상태: {result['status']}")
            print(f"  metric_id: {result['metric_id']}")
            print(f"  received_at: {result['received_at']}")
            return True
        else:
            print(f"\n❌ 실패!")
            print(f"  상태 코드: {response.status_code}")
            print(f"  응답: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 연결 실패!")
        print(f"  API 서버가 실행 중인지 확인하세요: python -m ocad.api.main")
        return False
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False


def test_batch_metrics():
    """배치 메트릭 전송 테스트."""
    print("\n" + "=" * 60)
    print("배치 메트릭 전송 테스트")
    print("=" * 60)

    batch = {
        "metrics": [
            {
                "endpoint_id": "o-ru-001",
                "timestamp": int(time.time() * 1000) + i * 1000,
                "metric_type": "udp_echo_rtt",
                "value": 5.0 + i * 0.1,
                "unit": "ms"
            }
            for i in range(10)
        ],
        "batch_id": f"batch_{int(time.time())}",
        "source": "test-script-v1.0.0"
    }

    print(f"\n전송할 메트릭 수: {len(batch['metrics'])}")
    print(f"배치 ID: {batch['batch_id']}")

    try:
        response = requests.post(
            f"{BASE_URL}/metrics/batch",
            json=batch,
            timeout=5
        )

        if response.status_code == 202:
            result = response.json()
            print(f"\n✅ 성공!")
            print(f"  상태: {result['status']}")
            print(f"  batch_id: {result['batch_id']}")
            print(f"  accepted: {result['accepted_count']}")
            print(f"  rejected: {result['rejected_count']}")
            return True
        else:
            print(f"\n❌ 실패!")
            print(f"  상태 코드: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 연결 실패!")
        return False
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False


def test_get_metrics():
    """메트릭 조회 테스트."""
    print("\n" + "=" * 60)
    print("메트릭 조회 테스트")
    print("=" * 60)

    try:
        # 필터 없이 조회
        response = requests.get(
            f"{BASE_URL}/metrics/",
            params={"limit": 5},
            timeout=5
        )

        if response.status_code == 200:
            metrics = response.json()
            print(f"\n✅ 성공!")
            print(f"  조회된 메트릭 수: {len(metrics)}")

            if metrics:
                print(f"\n  첫 번째 메트릭:")
                m = metrics[0]
                print(f"    endpoint_id: {m['endpoint_id']}")
                print(f"    metric_type: {m['metric_type']}")
                print(f"    value: {m['value']} {m['unit']}")

            return True
        else:
            print(f"\n❌ 실패!")
            print(f"  상태 코드: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 연결 실패!")
        return False
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False


def test_invalid_metric():
    """잘못된 메트릭 전송 테스트 (검증 확인)."""
    print("\n" + "=" * 60)
    print("잘못된 메트릭 전송 테스트 (검증)")
    print("=" * 60)

    # 잘못된 metric_type
    invalid_metric = {
        "endpoint_id": "o-ru-001",
        "timestamp": int(time.time() * 1000),
        "metric_type": "invalid_metric_type",  # 잘못된 타입
        "value": 5.2,
        "unit": "ms"
    }

    print(f"\n잘못된 metric_type 전송: {invalid_metric['metric_type']}")

    try:
        response = requests.post(
            f"{BASE_URL}/metrics/",
            json=invalid_metric,
            timeout=5
        )

        if response.status_code == 422:  # Validation Error
            print(f"\n✅ 검증 성공 (에러 발생 예상됨)")
            print(f"  상태 코드: {response.status_code}")
            error = response.json()
            print(f"  에러 메시지: {error['detail'][0]['msg']}")
            return True
        else:
            print(f"\n❌ 검증 실패 (에러가 발생해야 함)")
            print(f"  상태 코드: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 연결 실패!")
        return False
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False


def test_alert_stats():
    """알람 통계 조회 테스트."""
    print("\n" + "=" * 60)
    print("알람 통계 조회 테스트")
    print("=" * 60)

    try:
        response = requests.get(
            f"{BASE_URL}/alerts/stats/summary",
            timeout=5
        )

        if response.status_code == 200:
            stats = response.json()
            print(f"\n✅ 성공!")
            print(f"  총 알람 수: {stats['total']}")
            print(f"  심각도별:")
            for severity, count in stats['by_severity'].items():
                print(f"    {severity}: {count}")
            print(f"  상태별:")
            for status, count in stats['by_status'].items():
                print(f"    {status}: {count}")
            return True
        else:
            print(f"\n❌ 실패!")
            print(f"  상태 코드: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 연결 실패!")
        return False
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False


def main():
    """메인 함수."""
    print("\n" + "=" * 60)
    print("OCAD 데이터 인터페이스 테스트")
    print("=" * 60)
    print(f"\nAPI 서버: {BASE_URL}")
    print("\n주의: API 서버가 실행 중이어야 합니다.")
    print("  실행 명령: python -m ocad.api.main")

    results = []

    # 테스트 실행
    results.append(("단일 메트릭 전송", test_single_metric()))
    results.append(("배치 메트릭 전송", test_batch_metrics()))
    results.append(("메트릭 조회", test_get_metrics()))
    results.append(("잘못된 메트릭 검증", test_invalid_metric()))
    results.append(("알람 통계 조회", test_alert_stats()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:<30} {status}")

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
