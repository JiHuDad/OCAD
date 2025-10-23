"""알람 API 라우터.

OCAD에서 생성된 알람을 외부 시스템으로 전달하기 위한 API 엔드포인트입니다.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import time

from ...core.schemas import AlertData, Severity, AlertStatus
from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])

# 메모리 내 알람 저장소 (데모용)
# 실제 구현에서는 데이터베이스 사용
alert_store: List[AlertData] = []


@router.get("/", response_model=List[AlertData])
async def get_alerts(
    severity: Optional[Severity] = Query(None, description="심각도 필터"),
    endpoint_id: Optional[str] = Query(None, description="엔드포인트 ID 필터"),
    status: Optional[AlertStatus] = Query(None, description="상태 필터"),
    start_time: Optional[int] = Query(None, description="시작 시간 (Unix timestamp ms)"),
    end_time: Optional[int] = Query(None, description="종료 시간 (Unix timestamp ms)"),
    limit: int = Query(100, ge=1, le=1000, description="최대 반환 개수"),
    offset: int = Query(0, ge=0, description="오프셋")
):
    """알람 목록 조회.

    필터 조건에 맞는 알람을 조회합니다.

    Args:
        severity: 심각도 필터 (INFO/WARNING/CRITICAL)
        endpoint_id: 엔드포인트 ID 필터
        status: 상태 필터 (ACTIVE/ACKNOWLEDGED/RESOLVED)
        start_time: 시작 시간
        end_time: 종료 시간
        limit: 최대 반환 개수 (1-1000)
        offset: 오프셋

    Returns:
        List[AlertData]: 알람 리스트
    """
    # 필터링
    filtered_alerts = []

    for alert in alert_store:
        # Severity 필터
        if severity and alert.severity != severity:
            continue

        # Endpoint ID 필터
        if endpoint_id and alert.endpoint_id != endpoint_id:
            continue

        # Status 필터
        if status and alert.status != status:
            continue

        # 시간 필터
        if start_time and alert.timestamp < start_time:
            continue
        if end_time and alert.timestamp > end_time:
            continue

        filtered_alerts.append(alert)

    # 정렬 (최신순)
    filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)

    # 페이지네이션
    paginated = filtered_alerts[offset:offset + limit]

    logger.info(
        "알람 목록 조회",
        total=len(alert_store),
        filtered=len(filtered_alerts),
        returned=len(paginated),
        filters={
            "severity": severity,
            "endpoint_id": endpoint_id,
            "status": status,
        }
    )

    return paginated


@router.get("/{alert_id}", response_model=AlertData)
async def get_alert(alert_id: str):
    """알람 상세 조회.

    특정 알람의 상세 정보를 조회합니다.

    Args:
        alert_id: 알람 ID

    Returns:
        AlertData: 알람 상세 정보

    Raises:
        HTTPException: 알람을 찾을 수 없는 경우 404 에러
    """
    for alert in alert_store:
        if alert.alert_id == alert_id:
            logger.info("알람 상세 조회", alert_id=alert_id)
            return alert

    logger.warning("알람 찾을 수 없음", alert_id=alert_id)
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "code": "ALERT_NOT_FOUND",
                "message": f"Alert with ID '{alert_id}' not found",
                "timestamp": int(time.time() * 1000)
            }
        }
    )


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Query(..., description="확인한 사용자")
):
    """알람 확인.

    알람을 확인하여 상태를 ACKNOWLEDGED로 변경합니다.

    Args:
        alert_id: 알람 ID
        acknowledged_by: 확인한 사용자

    Returns:
        dict: 확인 결과

    Raises:
        HTTPException: 알람을 찾을 수 없는 경우 404 에러
    """
    for alert in alert_store:
        if alert.alert_id == alert_id:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = int(time.time() * 1000)

            logger.info(
                "알람 확인",
                alert_id=alert_id,
                acknowledged_by=acknowledged_by,
            )

            return {
                "status": "success",
                "message": f"Alert {alert_id} acknowledged by {acknowledged_by}",
                "alert_id": alert_id,
                "acknowledged_at": alert.acknowledged_at
            }

    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "code": "ALERT_NOT_FOUND",
                "message": f"Alert with ID '{alert_id}' not found",
                "timestamp": int(time.time() * 1000)
            }
        }
    )


@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolved_by: str = Query(..., description="해결한 사용자"),
    resolution_note: Optional[str] = Query(None, description="해결 내용")
):
    """알람 해결.

    알람을 해결하여 상태를 RESOLVED로 변경합니다.

    Args:
        alert_id: 알람 ID
        resolved_by: 해결한 사용자
        resolution_note: 해결 내용

    Returns:
        dict: 해결 결과

    Raises:
        HTTPException: 알람을 찾을 수 없는 경우 404 에러
    """
    for alert in alert_store:
        if alert.alert_id == alert_id:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = int(time.time() * 1000)

            logger.info(
                "알람 해결",
                alert_id=alert_id,
                resolved_by=resolved_by,
                note=resolution_note,
            )

            return {
                "status": "success",
                "message": f"Alert {alert_id} resolved by {resolved_by}",
                "alert_id": alert_id,
                "resolved_at": alert.resolved_at,
                "resolution_note": resolution_note
            }

    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "code": "ALERT_NOT_FOUND",
                "message": f"Alert with ID '{alert_id}' not found",
                "timestamp": int(time.time() * 1000)
            }
        }
    )


@router.get("/stats/summary")
async def get_alert_stats():
    """알람 통계 조회.

    전체 알람의 통계 정보를 조회합니다.

    Returns:
        dict: 알람 통계
    """
    total = len(alert_store)

    by_severity = {
        "INFO": 0,
        "WARNING": 0,
        "CRITICAL": 0
    }

    by_status = {
        "ACTIVE": 0,
        "ACKNOWLEDGED": 0,
        "RESOLVED": 0
    }

    by_endpoint = {}

    for alert in alert_store:
        by_severity[alert.severity.value] += 1
        by_status[alert.status.value] += 1

        if alert.endpoint_id not in by_endpoint:
            by_endpoint[alert.endpoint_id] = 0
        by_endpoint[alert.endpoint_id] += 1

    # 상위 10개 엔드포인트
    top_endpoints = sorted(
        by_endpoint.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return {
        "total": total,
        "by_severity": by_severity,
        "by_status": by_status,
        "top_endpoints": [
            {"endpoint_id": ep, "count": count}
            for ep, count in top_endpoints
        ],
        "timestamp": int(time.time() * 1000)
    }


# ============================================================================
# Helper Functions (내부 사용)
# ============================================================================

def add_alert(alert: AlertData):
    """알람 추가 (내부 함수).

    OCAD 시스템에서 생성된 알람을 저장소에 추가합니다.

    Args:
        alert: 알람 데이터
    """
    alert_store.append(alert)

    logger.info(
        "알람 생성",
        alert_id=alert.alert_id,
        severity=alert.severity,
        endpoint_id=alert.endpoint_id,
        anomaly_score=alert.anomaly_score,
    )

    # TODO: WebHook 발송
    # TODO: Kafka 토픽 발행
    # TODO: WebSocket 브로드캐스트
