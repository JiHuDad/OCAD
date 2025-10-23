"""데이터 스키마 정의.

외부 시스템과의 데이터 교환을 위한 표준 스키마를 정의합니다.
모든 스키마는 Pydantic을 사용하여 자동 검증됩니다.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any, Literal
from enum import Enum
from datetime import datetime
import time


# ============================================================================
# Enums
# ============================================================================

class MetricType(str, Enum):
    """지원하는 메트릭 타입."""

    UDP_ECHO_RTT = "udp_echo_rtt"
    ECPRI_DELAY = "ecpri_delay"
    LBM_RTT = "lbm_rtt"
    CCM_INTERVAL = "ccm_interval"


class MetricUnit(str, Enum):
    """메트릭 단위."""

    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    COUNT = "count"
    PERCENT = "percent"


class Severity(str, Enum):
    """알람 심각도."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    """알람 상태."""

    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


# ============================================================================
# Ingress Schemas (메트릭 수집)
# ============================================================================

class MetricQuality(BaseModel):
    """메트릭 품질 정보."""

    source_reliability: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="데이터 소스 신뢰도 (0-1)"
    )
    measurement_error: Optional[float] = Field(
        ge=0.0,
        default=None,
        description="측정 오차 범위"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "source_reliability": 0.98,
                "measurement_error": 0.1
            }
        }


class MetricData(BaseModel):
    """메트릭 데이터 스키마 v1.0.0.

    외부 시스템에서 OCAD로 전송하는 메트릭 데이터 포맷입니다.
    """

    # 스키마 버전
    schema_version: Literal["1.0.0"] = "1.0.0"

    # 필수 필드
    endpoint_id: str = Field(
        ...,
        pattern=r'^[a-zA-Z0-9_-]+$',
        max_length=64,
        description="엔드포인트 고유 식별자"
    )
    timestamp: int = Field(
        ...,
        ge=0,
        description="Unix timestamp in milliseconds"
    )
    metric_type: MetricType = Field(..., description="메트릭 타입")
    value: float = Field(..., description="메트릭 값")
    unit: MetricUnit = Field(..., description="측정 단위")

    # 선택 필드
    labels: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="추가 메타데이터 (key-value)"
    )
    quality: Optional[MetricQuality] = Field(
        default=None,
        description="데이터 품질 정보"
    )

    @validator('timestamp')
    def validate_timestamp(cls, v):
        """타임스탬프 유효성 검증.

        - 과거 1년 이내
        - 미래 1시간 이내
        """
        current_time_ms = int(time.time() * 1000)
        one_year_ago = current_time_ms - 365 * 24 * 3600 * 1000
        one_hour_future = current_time_ms + 3600 * 1000

        if not (one_year_ago < v < one_hour_future):
            raise ValueError(
                f'Timestamp out of acceptable range. '
                f'Must be between {one_year_ago} and {one_hour_future}, got {v}'
            )
        return v

    @validator('labels')
    def validate_labels(cls, v):
        """레이블 검증 - 최대 20개, 키/값 최대 길이 100자."""
        if v is None:
            return {}

        if len(v) > 20:
            raise ValueError(f'Too many labels: {len(v)} (max: 20)')

        for key, value in v.items():
            if len(key) > 100:
                raise ValueError(f'Label key too long: {key} (max: 100)')
            if len(value) > 100:
                raise ValueError(f'Label value too long: {value} (max: 100)')

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "schema_version": "1.0.0",
                "endpoint_id": "o-ru-001",
                "timestamp": 1729584000000,
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
        }


class MetricBatch(BaseModel):
    """배치 메트릭 데이터 스키마 v1.0.0."""

    metrics: List[MetricData] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="메트릭 리스트 (최대 10,000개)"
    )
    batch_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="배치 고유 식별자"
    )
    source: Optional[str] = Field(
        default=None,
        max_length=256,
        description="데이터 소스 정보"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "metrics": [
                    {
                        "endpoint_id": "o-ru-001",
                        "timestamp": 1729584000000,
                        "metric_type": "udp_echo_rtt",
                        "value": 5.2,
                        "unit": "ms"
                    },
                    {
                        "endpoint_id": "o-ru-001",
                        "timestamp": 1729584001000,
                        "metric_type": "udp_echo_rtt",
                        "value": 5.4,
                        "unit": "ms"
                    }
                ],
                "batch_id": "batch_xyz789",
                "source": "collector-service-v1.2.3"
            }
        }


# ============================================================================
# Egress Schemas (알람 발행)
# ============================================================================

class DetectionMethod(BaseModel):
    """탐지 알고리즘 정보."""

    method: str = Field(..., description="탐지 알고리즘 이름")
    score: float = Field(..., ge=0.0, le=1.0, description="탐지 점수")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")

    class Config:
        json_schema_extra = {
            "example": {
                "method": "ResidualDetectorV2",
                "score": 0.9,
                "confidence": 0.95
            }
        }


class AlertEvidence(BaseModel):
    """알람 증거 데이터."""

    affected_metrics: List[str] = Field(
        ...,
        min_items=1,
        description="영향받은 메트릭들"
    )
    duration_ms: int = Field(..., ge=0, description="이상 지속 시간 (ms)")
    peak_value: Optional[float] = Field(None, description="최고값")
    baseline_value: Optional[float] = Field(None, description="기준값")
    deviation_percent: Optional[float] = Field(None, description="편차 (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "affected_metrics": ["udp_echo_rtt", "ecpri_delay"],
                "duration_ms": 15000,
                "peak_value": 25.3,
                "baseline_value": 5.0,
                "deviation_percent": 406.0
            }
        }


class AlertData(BaseModel):
    """알람 데이터 스키마 v1.0.0.

    OCAD에서 외부 시스템으로 전송하는 알람 데이터 포맷입니다.
    """

    # 스키마 버전
    schema_version: Literal["1.0.0"] = "1.0.0"

    # 필수 필드
    alert_id: str = Field(
        ...,
        pattern=r'^alt_[a-zA-Z0-9]+$',
        description="알람 고유 ID (alt_ 접두사)"
    )
    timestamp: int = Field(..., ge=0, description="알람 생성 시각 (Unix timestamp ms)")
    severity: Severity = Field(..., description="심각도")
    endpoint_id: str = Field(..., description="문제 발생 엔드포인트 ID")
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="이상 점수 (0-1)"
    )
    detection_methods: List[DetectionMethod] = Field(
        ...,
        min_items=1,
        description="탐지한 알고리즘들"
    )
    evidence: AlertEvidence = Field(..., description="증거 데이터")

    # 선택 필드
    message: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="사람이 읽을 수 있는 메시지"
    )
    suggested_actions: List[str] = Field(
        default_factory=list,
        description="권장 조치 사항"
    )
    status: AlertStatus = Field(
        default=AlertStatus.ACTIVE,
        description="알람 상태"
    )
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="확인한 사용자"
    )
    acknowledged_at: Optional[int] = Field(
        default=None,
        description="확인 시각"
    )
    resolved_at: Optional[int] = Field(
        default=None,
        description="해결 시각"
    )
    details_url: Optional[str] = Field(
        default=None,
        description="상세 정보 URL"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "schema_version": "1.0.0",
                "alert_id": "alt_123abc",
                "timestamp": 1729584100000,
                "severity": "CRITICAL",
                "endpoint_id": "o-ru-001",
                "anomaly_score": 0.85,
                "detection_methods": [
                    {
                        "method": "ResidualDetectorV2",
                        "score": 0.9,
                        "confidence": 0.95
                    },
                    {
                        "method": "RuleBasedDetector",
                        "score": 0.8,
                        "confidence": 1.0
                    }
                ],
                "evidence": {
                    "affected_metrics": ["udp_echo_rtt"],
                    "duration_ms": 15000,
                    "peak_value": 25.3,
                    "baseline_value": 5.0,
                    "deviation_percent": 406.0
                },
                "message": "o-ru-001에서 높은 RTT 이상 탐지됨 (25.3 ms)",
                "suggested_actions": [
                    "엔드포인트 o-ru-001의 네트워크 연결 확인",
                    "최근 설정 변경 검토"
                ],
                "status": "ACTIVE",
                "details_url": "https://ocad.example.com/api/v1/alerts/alt_123abc"
            }
        }


# ============================================================================
# API Response Schemas
# ============================================================================

class APIResponse(BaseModel):
    """표준 API 응답 포맷."""

    status: str = Field(..., description="응답 상태 (success/error)")
    message: Optional[str] = Field(None, description="응답 메시지")
    data: Optional[Any] = Field(None, description="응답 데이터")
    request_id: Optional[str] = Field(None, description="요청 ID")
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ErrorDetail(BaseModel):
    """에러 상세 정보."""

    field: Optional[str] = Field(None, description="에러 발생 필드")
    message: str = Field(..., description="에러 메시지")
    code: Optional[str] = Field(None, description="에러 코드")


class ErrorResponse(BaseModel):
    """표준 에러 응답 포맷."""

    error: Dict[str, Any] = Field(..., description="에러 정보")

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": [
                        {
                            "field": "metric_type",
                            "message": "Value 'invalid_type' is not valid"
                        }
                    ],
                    "request_id": "req_abc123",
                    "timestamp": 1729584100000
                }
            }
        }


class MetricResponse(BaseModel):
    """메트릭 전송 응답."""

    status: str = "accepted"
    metric_id: Optional[str] = None
    received_at: int = Field(default_factory=lambda: int(time.time() * 1000))


class MetricBatchResponse(BaseModel):
    """배치 메트릭 전송 응답."""

    status: str = "accepted"
    batch_id: Optional[str] = None
    accepted_count: int
    rejected_count: int = 0
    received_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    errors: Optional[List[ErrorDetail]] = None
