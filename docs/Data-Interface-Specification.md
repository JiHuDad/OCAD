# OCAD 데이터 인터페이스 명세서

## 문서 정보

- **버전**: 1.0.0
- **작성일**: 2025-10-22
- **상태**: 초안 (Draft)
- **대상 독자**: 시스템 통합 엔지니어, 외부 시스템 개발자

## 1. 개요

### 1.1 목적

이 문서는 OCAD (ORAN CFM-Lite AI Anomaly Detection) 시스템과 외부 시스템 간의 데이터 교환 인터페이스를 정의합니다.

### 1.2 범위

- ✅ 메트릭 데이터 수집 인터페이스 (Ingress)
- ✅ 알람 데이터 발행 인터페이스 (Egress)
- ✅ 배치 데이터 가져오기/내보내기
- ✅ 실시간 스트리밍 인터페이스

### 1.3 용어 정의

| 용어 | 설명 |
|------|------|
| **Endpoint** | 모니터링 대상 ORAN 장비 (O-RU, O-DU 등) |
| **Metric** | 성능 측정값 (RTT, 지연, 손실률 등) |
| **Feature** | 메트릭으로부터 추출된 고급 통계 정보 |
| **Alert** | 이상 탐지 결과 알람 |
| **Ingress** | OCAD로 들어오는 데이터 |
| **Egress** | OCAD에서 나가는 데이터 |

## 2. 데이터 교환 패턴

### 2.1 전체 아키텍처

```
┌──────────────┐         Ingress          ┌──────────────┐
│              │  (1) PULL: NETCONF/REST  │              │
│ ORAN 장비    │ ◄─────────────────────── │     OCAD     │
│ (O-RU/O-DU)  │                          │    System    │
│              │  (2) PUSH: Kafka/gRPC    │              │
│              │ ─────────────────────────►│              │
└──────────────┘                          └──────────────┘
                                                 │
                                                 │ Egress
                                                 │ (3) Alerts
                                                 ▼
                                          ┌──────────────┐
                                          │  Monitoring  │
                                          │   System     │
                                          │ (Grafana 등) │
                                          └──────────────┘
```

### 2.2 지원하는 프로토콜

| 프로토콜 | 방향 | 용도 | 우선순위 |
|---------|------|------|----------|
| **REST API** | Bidirectional | 메트릭 수신, 알람 조회 | ⭐⭐⭐ 필수 |
| **Kafka** | Bidirectional | 대용량 스트리밍 | ⭐⭐ 권장 |
| **gRPC** | Bidirectional | 고성능 스트리밍 | ⭐ 선택 |
| **WebSocket** | Bidirectional | 실시간 대시보드 | ⭐ 선택 |
| **File (CSV/Parquet)** | Bidirectional | 배치 처리 | ⭐⭐ 권장 |
| **NETCONF/YANG** | Pull | ORAN 장비 쿼리 | ⭐⭐⭐ 필수 |

## 3. Ingress: 메트릭 데이터 수집

### 3.1 REST API - 단일 메트릭 전송

**엔드포인트:**
```
POST /api/v1/metrics
Content-Type: application/json
```

**요청 본문:**
```json
{
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
```

**응답:**
```json
{
  "status": "accepted",
  "metric_id": "met_abc123",
  "received_at": 1729584000100
}
```

**에러 응답:**
```json
{
  "status": "error",
  "error_code": "INVALID_METRIC_TYPE",
  "message": "Unknown metric type: 'invalid_metric'",
  "details": {
    "field": "metric_type",
    "allowed_values": ["udp_echo_rtt", "ecpri_delay", "lbm_rtt", "ccm_interval"]
  }
}
```

### 3.2 REST API - 배치 메트릭 전송

**엔드포인트:**
```
POST /api/v1/metrics/batch
Content-Type: application/json
```

**요청 본문:**
```json
{
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
```

**응답:**
```json
{
  "status": "accepted",
  "batch_id": "batch_xyz789",
  "accepted_count": 2,
  "rejected_count": 0,
  "received_at": 1729584001100
}
```

### 3.3 Kafka - 메트릭 스트리밍

**토픽:** `ocad.metrics.raw`

**메시지 키:** `{endpoint_id}`

**메시지 값 (JSON):**
```json
{
  "schema_version": "1.0.0",
  "endpoint_id": "o-ru-001",
  "timestamp": 1729584000000,
  "metric_type": "udp_echo_rtt",
  "value": 5.2,
  "unit": "ms",
  "producer_id": "collector-001",
  "sequence_number": 12345
}
```

**Kafka 헤더:**
```
schema-version: 1.0.0
content-type: application/json
producer-id: collector-001
```

### 3.4 파일 기반 - CSV 업로드

**엔드포인트:**
```
POST /api/v1/metrics/upload/csv
Content-Type: multipart/form-data
```

**CSV 형식:**
```csv
timestamp,endpoint_id,metric_type,value,unit
1729584000000,o-ru-001,udp_echo_rtt,5.2,ms
1729584001000,o-ru-001,udp_echo_rtt,5.4,ms
1729584002000,o-ru-002,ecpri_delay,102.3,us
```

**CSV 요구사항:**
- ✅ 헤더 행 필수
- ✅ UTF-8 인코딩
- ✅ 쉼표(,) 구분자
- ✅ 최대 파일 크기: 100 MB
- ✅ 최대 행 수: 1,000,000

### 3.5 파일 기반 - Parquet 업로드

**엔드포인트:**
```
POST /api/v1/metrics/upload/parquet
Content-Type: application/octet-stream
```

**Parquet 스키마:**
```python
schema = pa.schema([
    ('timestamp', pa.int64()),           # Unix timestamp (ms)
    ('endpoint_id', pa.string()),        # 엔드포인트 ID
    ('metric_type', pa.string()),        # 메트릭 타입
    ('value', pa.float64()),             # 메트릭 값
    ('unit', pa.string()),               # 단위
    ('labels', pa.map_(pa.string(), pa.string())),  # 레이블 (옵션)
])
```

## 4. Egress: 알람 데이터 발행

### 4.1 REST API - 알람 조회

**엔드포인트:**
```
GET /api/v1/alerts?severity=CRITICAL&start_time=1729584000000&limit=100
```

**응답:**
```json
{
  "alerts": [
    {
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
        "affected_metrics": ["udp_echo_rtt", "ecpri_delay"],
        "duration_ms": 15000,
        "peak_value": 25.3
      },
      "suggested_actions": [
        "엔드포인트 o-ru-001의 네트워크 연결 확인",
        "최근 설정 변경 검토"
      ],
      "status": "ACTIVE",
      "acknowledged_by": null,
      "acknowledged_at": null
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 100
}
```

### 4.2 WebHook - 알람 푸시

**설정 (config.yaml):**
```yaml
alerts:
  webhooks:
    - url: "https://monitoring.example.com/api/alerts"
      method: "POST"
      headers:
        Authorization: "Bearer ${WEBHOOK_TOKEN}"
      severity_filter: ["WARNING", "CRITICAL"]
      retry:
        max_attempts: 3
        backoff: "exponential"
```

**WebHook 페이로드:**
```json
{
  "event_type": "alert.created",
  "event_id": "evt_xyz789",
  "timestamp": 1729584100000,
  "alert": {
    "alert_id": "alt_123abc",
    "severity": "CRITICAL",
    "endpoint_id": "o-ru-001",
    "anomaly_score": 0.85,
    "message": "o-ru-001에서 높은 RTT 이상 탐지됨 (25.3 ms)",
    "details_url": "https://ocad.example.com/api/v1/alerts/alt_123abc"
  }
}
```

### 4.3 Kafka - 알람 스트리밍

**토픽:** `ocad.alerts.critical`

**메시지 키:** `{endpoint_id}`

**메시지 값:**
```json
{
  "schema_version": "1.0.0",
  "alert_id": "alt_123abc",
  "timestamp": 1729584100000,
  "severity": "CRITICAL",
  "endpoint_id": "o-ru-001",
  "anomaly_score": 0.85,
  "detection_summary": {
    "methods_count": 2,
    "consensus_score": 0.85
  },
  "evidence": {
    "affected_metrics": ["udp_echo_rtt"],
    "duration_ms": 15000
  }
}
```

### 4.4 WebSocket - 실시간 알람 스트리밍

**연결:**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/v1/alerts');

ws.onopen = () => {
  // 구독 메시지
  ws.send(JSON.stringify({
    action: 'subscribe',
    filters: {
      severity: ['WARNING', 'CRITICAL'],
      endpoint_ids: ['o-ru-001', 'o-ru-002']
    }
  }));
};

ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  console.log('New alert:', alert);
};
```

**수신 메시지:**
```json
{
  "type": "alert",
  "data": {
    "alert_id": "alt_123abc",
    "timestamp": 1729584100000,
    "severity": "CRITICAL",
    "endpoint_id": "o-ru-001",
    "message": "높은 RTT 이상 탐지"
  }
}
```

## 5. 데이터 스키마 정의

### 5.1 메트릭 스키마 (Pydantic)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict
from enum import Enum

class MetricType(str, Enum):
    UDP_ECHO_RTT = "udp_echo_rtt"
    ECPRI_DELAY = "ecpri_delay"
    LBM_RTT = "lbm_rtt"
    CCM_INTERVAL = "ccm_interval"

class MetricUnit(str, Enum):
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    COUNT = "count"
    PERCENT = "percent"

class MetricQuality(BaseModel):
    source_reliability: float = Field(ge=0.0, le=1.0, default=1.0)
    measurement_error: Optional[float] = Field(ge=0.0, default=None)

class MetricData(BaseModel):
    """메트릭 데이터 스키마 v1.0.0"""

    endpoint_id: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$', max_length=64)
    timestamp: int = Field(..., ge=0, description="Unix timestamp in milliseconds")
    metric_type: MetricType
    value: float
    unit: MetricUnit
    labels: Optional[Dict[str, str]] = Field(default_factory=dict)
    quality: Optional[MetricQuality] = None

    @validator('timestamp')
    def validate_timestamp(cls, v):
        import time
        current_time_ms = int(time.time() * 1000)
        # 과거 1년 이내, 미래 1시간 이내
        if not (current_time_ms - 365*24*3600*1000 < v < current_time_ms + 3600*1000):
            raise ValueError('Timestamp out of acceptable range')
        return v

    class Config:
        schema_extra = {
            "example": {
                "endpoint_id": "o-ru-001",
                "timestamp": 1729584000000,
                "metric_type": "udp_echo_rtt",
                "value": 5.2,
                "unit": "ms",
                "labels": {"site": "tower-A"},
                "quality": {"source_reliability": 0.98}
            }
        }
```

### 5.2 알람 스키마 (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"

class DetectionMethod(BaseModel):
    method: str = Field(..., description="탐지 알고리즘 이름")
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)

class AlertEvidence(BaseModel):
    affected_metrics: List[str]
    duration_ms: int = Field(..., ge=0)
    peak_value: Optional[float] = None
    baseline_value: Optional[float] = None
    deviation_percent: Optional[float] = None

class AlertData(BaseModel):
    """알람 데이터 스키마 v1.0.0"""

    alert_id: str = Field(..., regex=r'^alt_[a-zA-Z0-9]+$')
    timestamp: int = Field(..., ge=0)
    severity: Severity
    endpoint_id: str
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    detection_methods: List[DetectionMethod]
    evidence: AlertEvidence
    suggested_actions: List[str] = Field(default_factory=list)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[int] = None
    resolved_at: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
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
                    }
                ],
                "evidence": {
                    "affected_metrics": ["udp_echo_rtt"],
                    "duration_ms": 15000,
                    "peak_value": 25.3
                },
                "suggested_actions": ["네트워크 연결 확인"],
                "status": "ACTIVE"
            }
        }
```

## 6. 에러 처리

### 6.1 표준 에러 응답

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "metric_type",
        "message": "Value 'invalid_type' is not valid. Allowed: [udp_echo_rtt, ecpri_delay, lbm_rtt, ccm_interval]"
      }
    ],
    "request_id": "req_abc123",
    "timestamp": 1729584100000
  }
}
```

### 6.2 에러 코드 목록

| 코드 | HTTP 상태 | 설명 |
|------|-----------|------|
| `VALIDATION_ERROR` | 400 | 요청 데이터 검증 실패 |
| `INVALID_METRIC_TYPE` | 400 | 지원하지 않는 메트릭 타입 |
| `INVALID_TIMESTAMP` | 400 | 타임스탬프 범위 오류 |
| `AUTHENTICATION_REQUIRED` | 401 | 인증 필요 |
| `UNAUTHORIZED` | 403 | 권한 없음 |
| `ENDPOINT_NOT_FOUND` | 404 | 엔드포인트 없음 |
| `RATE_LIMIT_EXCEEDED` | 429 | 요청 한도 초과 |
| `INTERNAL_ERROR` | 500 | 서버 내부 오류 |
| `SERVICE_UNAVAILABLE` | 503 | 서비스 일시 중단 |

## 7. 버전 관리 및 호환성

### 7.1 API 버전 관리

- **URL 기반 버전 관리**: `/api/v1/`, `/api/v2/`
- **하위 호환성 보장**: v1 API는 최소 1년간 유지
- **Deprecation 공지**: 변경 3개월 전 사전 공지

### 7.2 스키마 버전 관리

```json
{
  "schema_version": "1.0.0",
  "data": { /* ... */ }
}
```

**버전 업그레이드 규칙:**
- **Major (1.0.0 → 2.0.0)**: 하위 호환성 없는 변경
- **Minor (1.0.0 → 1.1.0)**: 새 필드 추가 (하위 호환)
- **Patch (1.0.0 → 1.0.1)**: 버그 수정, 문서 개선

## 8. 보안 및 인증

### 8.1 인증 방식

**API Key 인증:**
```
Authorization: Bearer <API_KEY>
```

**JWT 토큰 인증:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 8.2 Rate Limiting

| API 엔드포인트 | 제한 |
|---------------|------|
| `POST /api/v1/metrics` | 1000 req/min |
| `POST /api/v1/metrics/batch` | 100 req/min |
| `GET /api/v1/alerts` | 300 req/min |

## 9. 테스트 및 검증

### 9.1 샘플 코드 (Python)

```python
import requests

# 메트릭 전송
metric = {
    "endpoint_id": "o-ru-001",
    "timestamp": int(time.time() * 1000),
    "metric_type": "udp_echo_rtt",
    "value": 5.2,
    "unit": "ms"
}

response = requests.post(
    "http://localhost:8080/api/v1/metrics",
    json=metric,
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

print(response.json())
```

### 9.2 샘플 코드 (JavaScript)

```javascript
// 메트릭 전송
const metric = {
  endpoint_id: "o-ru-001",
  timestamp: Date.now(),
  metric_type: "udp_echo_rtt",
  value: 5.2,
  unit: "ms"
};

const response = await fetch("http://localhost:8080/api/v1/metrics", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
  },
  body: JSON.stringify(metric)
});

console.log(await response.json());
```

## 10. 부록

### 10.1 완전한 OpenAPI 명세

별도 파일 참조: [openapi-v1.yaml](../api/openapi-v1.yaml)

### 10.2 Kafka Schema Registry

별도 파일 참조: [kafka-schemas/](../api/kafka-schemas/)

### 10.3 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|-----------|
| 1.0.0 | 2025-10-22 | 초안 작성 |

---

**문의**: OCAD 개발팀
**Email**: ocad-dev@example.com
