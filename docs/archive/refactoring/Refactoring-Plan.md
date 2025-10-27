# OCAD 프로젝트 리팩토링 계획

## 현재 구조 문제점 분석

### 1. 폴더 구조 문제

#### 현재 구조
```
ocad/
├── core/              # 핵심 설정, 모델, 로깅
├── capability/        # 기능 탐지
├── collectors/        # 데이터 수집
├── features/          # 피처 추출
├── detectors/         # 이상 탐지
├── alerts/            # 알람 관리
├── api/               # REST API
├── utils/             # 유틸리티
├── system/            # 오케스트레이터
├── training/          # 학습 관련 (새로 추가됨)
│   ├── datasets/
│   ├── trainers/
│   ├── evaluators/
│   └── utils/
├── cli.py             # CLI 인터페이스
└── main.py            # 메인 엔트리포인트
```

**문제점:**
1. ❌ `training/` 모듈이 운영 코드와 섞여있음 (학습 ≠ 운영)
2. ❌ `data/` 폴더가 루트에 있어 구조가 불명확
3. ❌ `models/` (저장된 모델)이 `ocad/` 패키지 밖에 있음
4. ❌ `scripts/`에 중요한 로직이 있어 재사용 어려움
5. ❌ 테스트 코드 구조가 소스 코드 구조와 불일치

### 2. 데이터 인터페이스 문제

#### 현재 데이터 흐름
```
외부 시스템 → ??? → OCAD
OCAD → ??? → 외부 시스템
```

**문제점:**
1. ❌ 외부 시스템과의 데이터 교환 명세 부재
2. ❌ 데이터 포맷 검증 로직 없음
3. ❌ 버전 관리 메커니즘 없음
4. ❌ 실시간 데이터 스트리밍 인터페이스 부재
5. ❌ 배치 데이터 가져오기/내보내기 표준 없음

### 3. 모델 관리 문제

**현재:**
```
ocad/models/tcn/
├── udp_echo_v1.0.0.pth
├── udp_echo_v1.0.0.json
└── ...
```

**문제점:**
1. ❌ 모델 레지스트리 없음
2. ❌ 모델 메타데이터가 JSON 파일로 분산
3. ❌ 모델 버전 비교 기능 없음
4. ❌ A/B 테스팅 불가능

## 리팩토링 목표

### 목표 1: 명확한 관심사 분리
- **운영 코드** vs **학습/개발 코드** 분리
- **핵심 비즈니스 로직** vs **인프라 코드** 분리

### 목표 2: 표준화된 데이터 인터페이스
- 외부 시스템과의 명확한 계약 (Contract)
- 데이터 검증 및 변환 레이어
- 버전 관리 및 호환성 보장

### 목표 3: 확장 가능한 아키텍처
- 새로운 메트릭 추가 용이
- 새로운 탐지 알고리즘 추가 용이
- 플러그인 아키텍처

## 제안하는 새로운 구조

### 1. 폴더 구조 재설계

```
ocad/
├── src/ocad/                    # 운영 패키지 (배포용)
│   ├── core/                    # 핵심 도메인
│   │   ├── domain/              # 도메인 모델
│   │   │   ├── models.py        # 엔티티 (Endpoint, Alert 등)
│   │   │   ├── value_objects.py # 값 객체 (MetricValue 등)
│   │   │   └── events.py        # 도메인 이벤트
│   │   ├── ports/               # 포트 (인터페이스)
│   │   │   ├── collectors.py    # 수집기 인터페이스
│   │   │   ├── detectors.py     # 탐지기 인터페이스
│   │   │   └── repositories.py  # 저장소 인터페이스
│   │   └── usecases/            # 유즈케이스
│   │       ├── detect_anomaly.py
│   │       ├── manage_endpoint.py
│   │       └── generate_alert.py
│   │
│   ├── adapters/                # 어댑터 (구현체)
│   │   ├── collectors/          # 데이터 수집 구현
│   │   │   ├── netconf/         # NETCONF 기반
│   │   │   ├── streaming/       # 실시간 스트리밍
│   │   │   └── batch/           # 배치 수집
│   │   ├── detectors/           # 탐지 알고리즘 구현
│   │   │   ├── rule_based.py
│   │   │   ├── statistical.py   # CUSUM, PELT
│   │   │   ├── ml_based.py      # TCN, Isolation Forest
│   │   │   └── ensemble.py      # 앙상블
│   │   ├── storage/             # 저장소 구현
│   │   │   ├── memory.py        # 인메모리
│   │   │   ├── file.py          # 파일 기반
│   │   │   └── database.py      # DB 기반 (선택)
│   │   └── external/            # 외부 시스템 연동
│   │       ├── kafka.py         # Kafka 프로듀서/컨슈머
│   │       ├── rest_api.py      # REST API 클라이언트
│   │       └── grpc.py          # gRPC 클라이언트
│   │
│   ├── services/                # 애플리케이션 서비스
│   │   ├── orchestration.py     # 전체 파이프라인 조정
│   │   ├── feature_extraction.py
│   │   ├── alert_management.py
│   │   └── model_management.py  # 모델 로딩/관리
│   │
│   ├── interfaces/              # 인터페이스 레이어
│   │   ├── api/                 # REST API
│   │   │   ├── v1/              # API v1
│   │   │   │   ├── endpoints.py
│   │   │   │   ├── alerts.py
│   │   │   │   └── metrics.py
│   │   │   └── v2/              # API v2 (미래)
│   │   ├── cli/                 # CLI
│   │   │   ├── commands/
│   │   │   └── main.py
│   │   └── grpc/                # gRPC (선택)
│   │       └── service.proto
│   │
│   ├── infrastructure/          # 인프라 관련
│   │   ├── config.py            # 설정 관리
│   │   ├── logging.py           # 로깅
│   │   ├── monitoring.py        # 모니터링
│   │   └── health.py            # 헬스체크
│   │
│   └── __main__.py              # 엔트리포인트
│
├── training/                    # 학습 패키지 (분리)
│   ├── data/                    # 데이터 관리
│   │   ├── generators/          # 데이터 생성기
│   │   │   ├── synthetic.py     # 합성 데이터
│   │   │   └── augmentation.py  # 데이터 증강
│   │   ├── loaders/             # 데이터 로더
│   │   │   ├── timeseries.py
│   │   │   └── multivariate.py
│   │   └── validators/          # 데이터 검증
│   │       └── quality_check.py
│   │
│   ├── models/                  # 모델 아키텍처
│   │   ├── tcn.py               # TCN
│   │   ├── lstm.py              # LSTM
│   │   └── isolation_forest.py  # Isolation Forest
│   │
│   ├── trainers/                # 학습 로직
│   │   ├── base.py
│   │   ├── tcn_trainer.py
│   │   └── ensemble_trainer.py
│   │
│   ├── evaluation/              # 평가
│   │   ├── metrics.py           # 평가 메트릭
│   │   ├── validators.py        # 모델 검증
│   │   └── reporters.py         # 리포트 생성
│   │
│   ├── experiments/             # 실험 관리
│   │   ├── tracking.py          # 실험 추적 (MLflow)
│   │   └── comparison.py        # 모델 비교
│   │
│   └── pipelines/               # 학습 파이프라인
│       ├── data_preparation.py
│       ├── training.py
│       └── deployment.py
│
├── models/                      # 저장된 모델 (artifacts)
│   ├── registry/                # 모델 레지스트리
│   │   ├── registry.db          # SQLite 레지스트리
│   │   └── metadata/            # 모델 메타데이터
│   ├── tcn/
│   │   ├── production/          # 프로덕션 모델
│   │   ├── staging/             # 스테이징 모델
│   │   └── archive/             # 아카이브
│   └── isolation_forest/
│       ├── production/
│       └── staging/
│
├── data/                        # 데이터 저장소
│   ├── raw/                     # 원시 데이터
│   │   ├── oran_logs/           # ORAN 로그
│   │   └── external/            # 외부 데이터
│   ├── processed/               # 전처리된 데이터
│   │   ├── timeseries/
│   │   └── features/
│   ├── training/                # 학습용 데이터
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── schemas/                 # 데이터 스키마
│       ├── metric_v1.json       # 메트릭 스키마 v1
│       ├── alert_v1.json        # 알람 스키마 v1
│       └── README.md            # 스키마 문서
│
├── tests/                       # 테스트 (소스 구조 반영)
│   ├── unit/
│   │   ├── core/
│   │   ├── adapters/
│   │   └── services/
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_external_api.py
│   └── e2e/
│       └── test_scenarios.py
│
├── scripts/                     # 유틸리티 스크립트
│   ├── setup/                   # 설치 스크립트
│   ├── maintenance/             # 유지보수
│   └── migration/               # 데이터/모델 마이그레이션
│
├── docs/                        # 문서
│   ├── architecture/            # 아키텍처
│   │   ├── overview.md
│   │   ├── data-flow.md
│   │   └── deployment.md
│   ├── api/                     # API 문서
│   │   ├── rest-api-v1.md
│   │   └── data-contracts.md   # 데이터 계약
│   ├── guides/                  # 가이드
│   │   ├── user-guide.md
│   │   ├── developer-guide.md
│   │   └── operator-guide.md
│   └── specifications/          # 명세
│       ├── data-interface-spec.md
│       └── model-spec.md
│
├── config/                      # 설정 파일
│   ├── default.yaml             # 기본 설정
│   ├── development.yaml         # 개발 환경
│   ├── staging.yaml             # 스테이징
│   └── production.yaml          # 프로덕션
│
├── deployments/                 # 배포 관련
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── kubernetes/
│       ├── deployment.yaml
│       └── service.yaml
│
├── pyproject.toml               # 프로젝트 메타데이터
├── setup.py                     # 설치 스크립트
└── README.md
```

### 2. 데이터 인터페이스 명세

#### 2.1 외부 시스템 연동 타입

OCAD는 다음 세 가지 방식으로 데이터를 교환합니다:

```
┌─────────────────────────────────────────────────────────┐
│          외부 시스템과의 데이터 교환 패턴                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. PULL (수집): OCAD → ORAN 장비                       │
│     - NETCONF/YANG 쿼리                                 │
│     - REST API 호출                                     │
│     - 주기적 폴링                                       │
│                                                         │
│  2. PUSH (스트리밍): ORAN 장비 → OCAD                   │
│     - Kafka 토픽 구독                                   │
│     - gRPC 스트리밍                                     │
│     - WebSocket                                         │
│                                                         │
│  3. EXPORT (알람/리포트): OCAD → 외부 모니터링 시스템   │
│     - Webhook                                           │
│     - Kafka 토픽 발행                                   │
│     - REST API 콜백                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 2.2 데이터 계약 (Data Contract)

**핵심 원칙:**
1. **버전 관리**: 모든 데이터 포맷은 버전을 명시
2. **검증**: Pydantic/JSON Schema로 자동 검증
3. **호환성**: 하위 호환성 보장
4. **문서화**: OpenAPI/AsyncAPI 자동 생성

#### 2.3 메트릭 데이터 스키마 (Ingress)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://ocad.example.com/schemas/metric/v1",
  "title": "ORAN Metric Data",
  "version": "1.0.0",
  "type": "object",
  "required": ["endpoint_id", "timestamp", "metric_type", "value"],
  "properties": {
    "endpoint_id": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_-]+$",
      "description": "엔드포인트 고유 식별자"
    },
    "timestamp": {
      "type": "integer",
      "description": "Unix timestamp (밀리초)",
      "minimum": 0
    },
    "metric_type": {
      "type": "string",
      "enum": ["udp_echo_rtt", "ecpri_delay", "lbm_rtt", "ccm_interval"],
      "description": "메트릭 타입"
    },
    "value": {
      "type": "number",
      "description": "메트릭 값"
    },
    "unit": {
      "type": "string",
      "description": "단위 (ms, us, count 등)"
    },
    "labels": {
      "type": "object",
      "description": "추가 메타데이터 (key-value)",
      "additionalProperties": {"type": "string"}
    },
    "quality": {
      "type": "object",
      "properties": {
        "source_reliability": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "데이터 신뢰도 (0-1)"
        },
        "measurement_error": {
          "type": "number",
          "description": "측정 오차 범위"
        }
      }
    }
  }
}
```

#### 2.4 알람 데이터 스키마 (Egress)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://ocad.example.com/schemas/alert/v1",
  "title": "OCAD Alert Event",
  "version": "1.0.0",
  "type": "object",
  "required": ["alert_id", "timestamp", "severity", "endpoint_id", "anomaly_score"],
  "properties": {
    "alert_id": {
      "type": "string",
      "format": "uuid",
      "description": "알람 고유 ID"
    },
    "timestamp": {
      "type": "integer",
      "description": "알람 생성 시각 (Unix timestamp ms)"
    },
    "severity": {
      "type": "string",
      "enum": ["INFO", "WARNING", "CRITICAL"],
      "description": "심각도"
    },
    "endpoint_id": {
      "type": "string",
      "description": "문제가 발생한 엔드포인트 ID"
    },
    "anomaly_score": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "이상 점수 (0-1)"
    },
    "detection_methods": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "method": {"type": "string"},
          "score": {"type": "number"},
          "confidence": {"type": "number"}
        }
      },
      "description": "탐지한 알고리즘들"
    },
    "evidence": {
      "type": "object",
      "description": "증거 데이터",
      "properties": {
        "affected_metrics": {
          "type": "array",
          "items": {"type": "string"}
        },
        "duration_ms": {"type": "integer"},
        "peak_value": {"type": "number"}
      }
    },
    "suggested_actions": {
      "type": "array",
      "items": {"type": "string"},
      "description": "권장 조치 사항"
    }
  }
}
```

#### 2.5 배치 데이터 교환 포맷

**CSV 형식** (간단한 데이터 교환):
```csv
timestamp,endpoint_id,metric_type,value,unit
1729584000000,o-ru-001,udp_echo_rtt,5.2,ms
1729584001000,o-ru-001,udp_echo_rtt,5.4,ms
```

**Parquet 형식** (대용량 데이터):
```python
# 스키마
schema = pa.schema([
    ('timestamp', pa.int64()),
    ('endpoint_id', pa.string()),
    ('metric_type', pa.string()),
    ('value', pa.float64()),
    ('unit', pa.string()),
    ('labels', pa.map_(pa.string(), pa.string())),
])
```

**JSON Lines** (스트리밍):
```jsonl
{"timestamp": 1729584000000, "endpoint_id": "o-ru-001", "metric_type": "udp_echo_rtt", "value": 5.2}
{"timestamp": 1729584001000, "endpoint_id": "o-ru-001", "metric_type": "udp_echo_rtt", "value": 5.4}
```

### 3. API 인터페이스 설계

#### 3.1 REST API v1 엔드포인트

```
# 메트릭 수집 (PUSH)
POST   /api/v1/metrics              # 메트릭 데이터 수신
POST   /api/v1/metrics/batch        # 배치 메트릭 수신

# 메트릭 조회
GET    /api/v1/metrics              # 메트릭 조회 (필터링)
GET    /api/v1/metrics/{metric_id}  # 특정 메트릭 조회

# 엔드포인트 관리
GET    /api/v1/endpoints            # 엔드포인트 목록
POST   /api/v1/endpoints            # 엔드포인트 등록
GET    /api/v1/endpoints/{id}       # 엔드포인트 상세
PUT    /api/v1/endpoints/{id}       # 엔드포인트 수정
DELETE /api/v1/endpoints/{id}       # 엔드포인트 삭제

# 알람 관리
GET    /api/v1/alerts               # 알람 목록
GET    /api/v1/alerts/{id}          # 알람 상세
POST   /api/v1/alerts/{id}/ack      # 알람 확인
POST   /api/v1/alerts/{id}/resolve  # 알람 해결

# 모델 관리
GET    /api/v1/models               # 모델 목록
GET    /api/v1/models/{id}          # 모델 상세
POST   /api/v1/models/{id}/deploy   # 모델 배포
POST   /api/v1/models/{id}/rollback # 모델 롤백

# 시스템 상태
GET    /api/v1/health               # 헬스체크
GET    /api/v1/stats                # 통계
GET    /api/v1/metrics/system       # 시스템 메트릭
```

#### 3.2 Kafka 토픽 설계

```
# 입력 토픽 (데이터 수집)
ocad.metrics.raw                    # 원시 메트릭 데이터
ocad.metrics.processed              # 전처리된 메트릭

# 출력 토픽 (알람 발행)
ocad.alerts.info                    # INFO 레벨 알람
ocad.alerts.warning                 # WARNING 레벨 알람
ocad.alerts.critical                # CRITICAL 레벨 알람

# 내부 토픽
ocad.features                       # 추출된 피처
ocad.detections                     # 탐지 결과
```

#### 3.3 WebSocket (실시간 스트리밍)

```javascript
// 연결
ws://localhost:8080/ws/v1/stream

// 구독
{
  "action": "subscribe",
  "topics": ["alerts", "metrics", "detections"]
}

// 메시지 수신
{
  "topic": "alerts",
  "data": { /* Alert 객체 */ },
  "timestamp": 1729584000000
}
```

## 구현 진행 상황

### ✅ Phase 0: 파일 기반 데이터 입력 (완료 - 2025-10-23)

**목적**: CFM 담당자와 협의를 위해 먼저 파일 기반 입력 시스템 구현

**완료된 작업:**
1. ✅ 샘플 데이터 파일 생성 (Excel, CSV Wide/Long Format)
2. ✅ 파일 로더 구현 (CSV, Excel, Parquet)
3. ✅ 파일 형식 변환기 (CSV ↔ Parquet, Wide ↔ Long)
4. ✅ Pydantic 스키마 기반 자동 검증
5. ✅ 테스트 스크립트 및 문서화

**상세 문서**: [File-Based-Input-Implementation-Summary.md](File-Based-Input-Implementation-Summary.md)

**다음 단계**: CFM 담당자 미팅 → 데이터 수집 가능 여부 확인 → 요구사항 조정

---

### 🔄 Phase A: 데이터 인터페이스 구현 (부분 완료)

**우선순위**: 높음

**완료:**
1. ✅ 데이터 스키마 정의 및 검증 로직 구현 (Pydantic v2)
2. ✅ REST API 엔드포인트 추가 (메트릭 수신, 알람 관리)
3. ✅ 알람 발행 인터페이스 구현
4. ✅ 문서 자동 생성 (OpenAPI/Swagger)

**연기:**
- REST API 실제 배포 (파일 기반 입력 우선)
- Kafka/WebSocket 스트리밍 (Phase D로 이동)

**상세 문서**: [Refactoring-Summary.md](Refactoring-Summary.md)

---

### 📋 Phase B: 폴더 구조 리팩토링 (계획 중)

**우선순위**: 중간 (파일 기반 입력 완료 후)

**계획:**
1. `ocad/loaders/` 모듈 추가 완료 ✅
2. `training/` 와 운영 코드 분리 (필요 시)
3. 더 명확한 관심사 분리

**현재 구조 개선 사항:**
```
ocad/
├── loaders/           # NEW - 파일 기반 입력
│   ├── base.py
│   ├── csv_loader.py
│   ├── excel_loader.py
│   ├── parquet_loader.py
│   └── converter.py
├── core/
│   └── schemas.py     # 업데이트 - Pydantic v2
├── api/
│   └── v1/            # NEW - REST API (구현됨, 배포 연기)
└── training/          # 학습-추론 분리 완료
```

---

### 🔮 Phase C: 모델 레지스트리 구현 (미정)

**우선순위**: 낮음

**계획:**
1. 모델 메타데이터 DB
2. 모델 버전 관리
3. A/B 테스팅 기능

**보류 이유**: 현재 파일 기반 모델 관리로 충분

---

### 🌊 Phase D: 실시간 스트리밍 인터페이스 (미정)

**우선순위**: 낮음 (CFM 협의 완료 후 결정)

**계획:**
1. Kafka 프로듀서/컨슈머
2. WebSocket 실시간 스트리밍
3. gRPC (선택)

**보류 이유**:
- 파일 기반 입력으로 먼저 요구사항 확정 필요
- CFM 담당자와 실시간 데이터 수집 가능 여부 협의 후 결정

---

## 리팩토링 전략 변경

### 기존 계획 (대규모 리팩토링)
```
1. Hexagonal Architecture 전면 적용
2. REST API/Kafka/gRPC 모든 인터페이스 구현
3. 모델 레지스트리 DB 구축
```

### 현재 전략 (점진적, 실용적 접근)
```
1. ✅ 파일 기반 입력 먼저 구현 (실제 데이터 확인)
2. ✅ CFM 담당자와 협의 (수집 가능한 데이터 확정)
3. 📋 요구사항 조정 후 파이프라인 통합
4. 🔮 필요 시 REST API/Kafka 등 실시간 인터페이스 추가
```

**장점:**
- 빠른 검증 및 피드백
- 실제 수집 가능한 데이터에 맞춰 시스템 조정
- 과도한 사전 설계 방지 (YAGNI 원칙)

---

## 다음 단계 (우선순위 순)

### 1. 즉시 수행 (이번 주)
- [ ] CFM 담당자 미팅 일정 잡기
- [ ] Excel 샘플 및 요구사항 문서 전달
- [ ] 데이터 수집 가능 여부 피드백 받기

### 2. 단기 (1-2주)
- [ ] CFM 피드백 반영하여 스키마 수정
- [ ] 파일 로더를 SystemOrchestrator에 통합
- [ ] 배치 처리 스케줄러 구현

### 3. 중기 (1개월)
- [ ] REST API 배포 (필요 시)
- [ ] 대시보드 UI 개선
- [ ] 성능 최적화

### 4. 장기 (3개월+)
- [ ] Kafka 스트리밍 (실시간 수집 가능 시)
- [ ] 모델 레지스트리 (모델 수가 많아질 때)
- [ ] Hexagonal Architecture (대규모 확장 필요 시)

---

**최종 업데이트**: 2025-10-23
**작성자**: Claude Code
**상태**: ✅ Phase 0 완료, Phase A 부분 완료, 나머지 단계별 진행
