"""메트릭 API 라우터.

외부 시스템에서 메트릭 데이터를 OCAD로 전송하기 위한 API 엔드포인트입니다.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import uuid
import time
from collections import deque

from ...core.schemas import (
    MetricData,
    MetricBatch,
    MetricResponse,
    MetricBatchResponse,
    ErrorResponse,
)
from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])

# 메모리 내 메트릭 버퍼 (실제 구현에서는 Kafka/DB 등 사용)
# 최근 10,000개 메트릭만 유지
metric_buffer = deque(maxlen=10000)


@router.post("/", response_model=MetricResponse, status_code=202)
async def create_metric(
    metric: MetricData,
    background_tasks: BackgroundTasks
):
    """단일 메트릭 데이터 수신.

    외부 시스템에서 하나의 메트릭 데이터를 OCAD로 전송합니다.

    Args:
        metric: 메트릭 데이터
        background_tasks: 백그라운드 작업 큐

    Returns:
        MetricResponse: 수신 확인 응답

    Raises:
        HTTPException: 데이터 검증 실패 시 400 에러
    """
    try:
        # 메트릭 ID 생성
        metric_id = f"met_{uuid.uuid4().hex[:12]}"

        # 로깅
        logger.info(
            "메트릭 수신",
            metric_id=metric_id,
            endpoint_id=metric.endpoint_id,
            metric_type=metric.metric_type,
            value=metric.value,
        )

        # 백그라운드에서 메트릭 처리
        background_tasks.add_task(process_metric, metric, metric_id)

        return MetricResponse(
            status="accepted",
            metric_id=metric_id,
            received_at=int(time.time() * 1000)
        )

    except Exception as e:
        logger.error("메트릭 수신 실패", error=str(e), endpoint_id=metric.endpoint_id)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": f"Failed to process metric: {str(e)}",
                    "timestamp": int(time.time() * 1000)
                }
            }
        )


@router.post("/batch", response_model=MetricBatchResponse, status_code=202)
async def create_metric_batch(
    batch: MetricBatch,
    background_tasks: BackgroundTasks
):
    """배치 메트릭 데이터 수신.

    외부 시스템에서 여러 메트릭 데이터를 한 번에 OCAD로 전송합니다.
    최대 10,000개의 메트릭을 한 번에 전송할 수 있습니다.

    Args:
        batch: 배치 메트릭 데이터
        background_tasks: 백그라운드 작업 큐

    Returns:
        MetricBatchResponse: 배치 수신 확인 응답
    """
    try:
        accepted_count = 0
        rejected_count = 0
        errors = []

        for idx, metric in enumerate(batch.metrics):
            try:
                metric_id = f"met_{uuid.uuid4().hex[:12]}"
                background_tasks.add_task(process_metric, metric, metric_id)
                accepted_count += 1

            except Exception as e:
                rejected_count += 1
                errors.append({
                    "index": idx,
                    "endpoint_id": metric.endpoint_id,
                    "error": str(e)
                })

        logger.info(
            "배치 메트릭 수신",
            batch_id=batch.batch_id,
            total=len(batch.metrics),
            accepted=accepted_count,
            rejected=rejected_count,
        )

        return MetricBatchResponse(
            status="accepted" if rejected_count == 0 else "partial",
            batch_id=batch.batch_id,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            received_at=int(time.time() * 1000),
            errors=errors if errors else None
        )

    except Exception as e:
        logger.error("배치 메트릭 수신 실패", error=str(e), batch_id=batch.batch_id)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": f"Failed to process batch: {str(e)}",
                    "timestamp": int(time.time() * 1000)
                }
            }
        )


@router.post("/upload/csv", status_code=202)
async def upload_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """CSV 파일 업로드.

    CSV 형식의 메트릭 데이터를 업로드합니다.

    Required CSV format:
    timestamp,endpoint_id,metric_type,value,unit
    1729584000000,o-ru-001,udp_echo_rtt,5.2,ms

    Args:
        file: CSV 파일
        background_tasks: 백그라운드 작업 큐

    Returns:
        dict: 업로드 결과
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "INVALID_FILE_TYPE",
                    "message": "File must be CSV format (.csv extension)",
                    "timestamp": int(time.time() * 1000)
                }
            }
        )

    try:
        import csv
        import io

        # CSV 파일 읽기
        contents = await file.read()
        csv_data = contents.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_data))

        accepted_count = 0
        rejected_count = 0
        errors = []

        for row_num, row in enumerate(csv_reader, start=2):  # 헤더 제외
            try:
                metric = MetricData(
                    endpoint_id=row['endpoint_id'],
                    timestamp=int(row['timestamp']),
                    metric_type=row['metric_type'],
                    value=float(row['value']),
                    unit=row['unit'],
                )

                metric_id = f"met_{uuid.uuid4().hex[:12]}"
                background_tasks.add_task(process_metric, metric, metric_id)
                accepted_count += 1

            except Exception as e:
                rejected_count += 1
                errors.append({
                    "row": row_num,
                    "error": str(e)
                })

        logger.info(
            "CSV 업로드 완료",
            filename=file.filename,
            accepted=accepted_count,
            rejected=rejected_count,
        )

        return {
            "status": "accepted" if rejected_count == 0 else "partial",
            "filename": file.filename,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "errors": errors[:10] if errors else None,  # 최대 10개 에러만 반환
        }

    except Exception as e:
        logger.error("CSV 업로드 실패", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": f"Failed to process CSV: {str(e)}",
                    "timestamp": int(time.time() * 1000)
                }
            }
        )


@router.get("/", response_model=List[dict])
async def get_metrics(
    endpoint_id: str = None,
    metric_type: str = None,
    start_time: int = None,
    end_time: int = None,
    limit: int = 100
):
    """메트릭 조회.

    필터 조건에 맞는 메트릭 데이터를 조회합니다.

    Args:
        endpoint_id: 엔드포인트 ID 필터
        metric_type: 메트릭 타입 필터
        start_time: 시작 시간 (Unix timestamp ms)
        end_time: 종료 시간 (Unix timestamp ms)
        limit: 최대 반환 개수 (기본: 100, 최대: 1000)

    Returns:
        List[dict]: 메트릭 리스트
    """
    limit = min(limit, 1000)  # 최대 1000개로 제한

    # 필터링
    filtered_metrics = []
    for metric in list(metric_buffer):
        # endpoint_id 필터
        if endpoint_id and metric.endpoint_id != endpoint_id:
            continue

        # metric_type 필터
        if metric_type and metric.metric_type != metric_type:
            continue

        # 시간 필터
        if start_time and metric.timestamp < start_time:
            continue
        if end_time and metric.timestamp > end_time:
            continue

        filtered_metrics.append(metric.dict())

        if len(filtered_metrics) >= limit:
            break

    return filtered_metrics


# ============================================================================
# Helper Functions
# ============================================================================

async def process_metric(metric: MetricData, metric_id: str):
    """메트릭 처리 (백그라운드 작업).

    실제 구현에서는:
    1. Kafka로 발행
    2. 데이터베이스 저장
    3. 실시간 파이프라인으로 전달

    Args:
        metric: 메트릭 데이터
        metric_id: 메트릭 ID
    """
    try:
        # 메모리 버퍼에 저장 (데모용)
        metric_buffer.append(metric)

        logger.debug(
            "메트릭 처리 완료",
            metric_id=metric_id,
            endpoint_id=metric.endpoint_id,
            buffer_size=len(metric_buffer),
        )

        # TODO: 실제 구현
        # - await kafka_producer.send("ocad.metrics.raw", metric.dict())
        # - await db.save_metric(metric)
        # - await pipeline.process(metric)

    except Exception as e:
        logger.error(
            "메트릭 처리 실패",
            metric_id=metric_id,
            error=str(e),
        )
