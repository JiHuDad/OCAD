# OCAD API Documentation

OCAD 시스템은 RESTful API를 제공하여 외부 시스템과의 통합을 지원합니다.

## Base URL

```
http://localhost:8080
```

## Endpoints

### Health Check

#### GET /
기본 상태 확인

**Response:**
```json
{
  "service": "ORAN CFM-Lite AI Anomaly Detection",
  "version": "0.1.0",
  "status": "operational"
}
```

#### GET /health
상세 상태 확인

**Response:**
```json
{
  "healthy": true,
  "components": {
    "orchestrator": true,
    "collectors": 3,
    "endpoints": 5,
    "active_alerts": 2
  },
  "uptime_seconds": 3600
}
```

### Endpoint Management

#### GET /endpoints
등록된 엔드포인트 목록 조회

**Response:**
```json
[
  {
    "id": "192.168.1.100:830",
    "host": "192.168.1.100",
    "port": 830,
    "role": "o-ru",
    "capabilities": {
      "udp_echo": true,
      "ecpri_delay": false,
      "lbm": true,
      "ccm_min": false,
      "lldp": true
    },
    "active": true,
    "last_seen": "2025-09-29T10:30:00Z"
  }
]
```

#### POST /endpoints
새 엔드포인트 추가

**Request:**
```json
{
  "host": "192.168.1.101",
  "port": 830,
  "role": "o-du"
}
```

**Response:**
```json
{
  "message": "Endpoint added successfully",
  "endpoint_id": "192.168.1.101:830"
}
```

#### DELETE /endpoints/{endpoint_id}
엔드포인트 제거

**Response:**
```json
{
  "message": "Endpoint removed successfully"
}
```

### Alert Management

#### GET /alerts
활성 알람 목록 조회

**Query Parameters:**
- `severity` (optional): critical, warning, info
- `endpoint_id` (optional): 특정 엔드포인트 필터
- `limit` (optional): 최대 결과 수 (기본값: 100)
- `offset` (optional): 페이지네이션 오프셋 (기본값: 0)

**Response:**
```json
{
  "alerts": [
    {
      "id": "alert-123",
      "endpoint_id": "192.168.1.100:830",
      "ts_ms": 1632900000000,
      "severity": "warning",
      "title": "Warning anomaly detected on 192.168.1.100:830",
      "description": "Anomaly detected with evidence: drift, spike",
      "evidence": [
        {
          "type": "drift",
          "value": 0.75,
          "description": "Performance drift detected (residual score: 0.75)",
          "confidence": 0.7
        }
      ],
      "acknowledged": false,
      "resolved": false,
      "created_at": "2025-09-29T10:30:00Z"
    }
  ],
  "total": 1
}
```

#### POST /alerts/{alert_id}/acknowledge
알람 확인

**Response:**
```json
{
  "message": "Alert acknowledged successfully"
}
```

#### POST /alerts/{alert_id}/resolve
알람 해결

**Response:**
```json
{
  "message": "Alert resolved successfully"
}
```

#### POST /endpoints/{endpoint_id}/suppress
엔드포인트 알람 억제

**Query Parameters:**
- `duration_minutes` (optional): 억제 시간 (기본값: 60분)

**Response:**
```json
{
  "message": "Endpoint suppressed for 60 minutes"
}
```

### Statistics and Monitoring

#### GET /stats
시스템 통계

**Response:**
```json
{
  "endpoints": 5,
  "active_alerts": 2,
  "capability_coverage": 85.5,
  "processing_latency_p95": 0.023
}
```

#### GET /kpi
KPI 메트릭

**Response:**
```json
{
  "period_start": "2025-09-28T10:30:00Z",
  "period_end": "2025-09-29T10:30:00Z",
  "total_alerts": 15,
  "true_positives": 12,
  "false_positives": 3,
  "false_negatives": 1,
  "capability_coverage": 85.5,
  "false_alarm_rate": 0.2,
  "precision": 0.8,
  "recall": 0.923,
  "f1_score": 0.857
}
```

#### GET /debug/windows
디버그 정보 (피처 윈도우 상태)

**Response:**
```json
{
  "feature_windows": {
    "192.168.1.100:830": 25,
    "192.168.1.101:830": 30
  },
  "collector_status": {
    "UdpEchoCollector": "active",
    "EcpriDelayCollector": "active",
    "LbmCollector": "active"
  },
  "processing_metrics": {
    "samples_processed": 1500,
    "features_extracted": 750,
    "alerts_generated": 15
  }
}
```

## Error Responses

모든 에러 응답은 다음 형식을 따릅니다:

```json
{
  "detail": "Error message description"
}
```

**HTTP Status Codes:**
- `400` - Bad Request (잘못된 요청)
- `404` - Not Found (리소스 없음)
- `500` - Internal Server Error (내부 서버 오류)
- `503` - Service Unavailable (서비스 사용 불가)

## Authentication

현재 버전에서는 인증이 구현되어 있지 않습니다. 프로덕션 환경에서는 적절한 인증 및 권한 부여 메커니즘을 구현해야 합니다.

## Rate Limiting

현재 API에는 rate limiting이 적용되어 있지 않습니다. 필요에 따라 구현할 수 있습니다.

## WebSocket Support

실시간 알람 스트리밍을 위한 WebSocket 엔드포인트는 향후 버전에서 제공될 예정입니다.
