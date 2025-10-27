# 문서 재구조화 계획

**날짜**: 2025-10-27
**목적**: OCAD 문서를 목적별로 명확하게 분류하여 사용자가 쉽게 찾을 수 있도록 재구조화

---

## 📊 현재 문서 현황 분석

### 전체 문서 목록 (21개)

**사용자 가이드** (5개):
- Quick-Start-Guide.md (11K) - 빠른 시작
- Operations-Guide.md (8.1K) - 운영 가이드
- Logging-Guide.md (9.7K) - 로깅 시스템
- API.md (4.6K) - REST API
- CFM-Data-Requirements.md (9.7K) - CFM 담당자용 데이터 요구사항

**학습/추론 관련** (5개):
- Training-Inference-Separation-Design.md (40K) - 학습-추론 분리 설계 (상세)
- Training-Inference-Workflow.md (9.3K) - 학습-추론 워크플로우
- Model-Training-Guide.md (11K) - 모델 학습 가이드
- AI-Models-Guide.md (14K) - AI 모델 가이드
- Data-Source-Abstraction-Design.md (15K) - 데이터 소스 추상화 설계
- Data-Source-Abstraction-Summary.md (13K) - 데이터 소스 추상화 요약

**데이터 입력 관련** (4개):
- File-Based-Data-Input-Plan.md (18K) - 파일 입력 계획
- File-Based-Input-Implementation-Summary.md (13K) - 파일 입력 구현 요약
- Data-Interface-Specification.md (17K) - 데이터 인터페이스 명세

**구현 히스토리** (4개):
- Phase1-Implementation-Summary.md (11K) - Phase 1 구현 요약
- Phase2-Implementation-Summary.md (9.5K) - Phase 2 구현 요약
- Phase3-Implementation-Summary.md (7.6K) - Phase 3 구현 요약
- Phase4-Implementation-Summary.md (12K) - Phase 4 구현 요약

**리팩토링 관련** (3개):
- Refactoring-Plan.md (23K) - 리팩토링 계획
- Refactoring-Summary.md (11K) - 리팩토링 요약
- Directory-Structure-Cleanup.md (4.6K) - 디렉토리 정리

---

## 🎯 재구조화 목표

### 1. 사용자 중심 구조
- **신규 사용자**: Quick Start → 학습/추론 → 운영
- **운영자**: Operations → Logging → API
- **CFM 담당자**: 데이터 요구사항 → 파일 입력
- **개발자**: 아키텍처 → 구현 히스토리

### 2. 명확한 분류
- 현재: 모든 문서가 평탄한 구조 (21개 파일이 한 폴더에)
- 개선: 카테고리별 하위 디렉토리 구조

### 3. 최신 상태 유지
- 오래된 문서는 archive로 이동
- 중복된 내용은 통합
- 최신 아키텍처 반영

---

## 📁 제안하는 새 구조

```
docs/
├── README.md                           # 📖 문서 인덱스 (NEW)
│
├── 01-getting-started/                 # 🚀 시작하기
│   ├── Quick-Start-Guide.md           # 빠른 시작 (기존)
│   ├── Installation.md                # 설치 가이드 (NEW)
│   └── First-Steps.md                 # 첫 단계 (NEW)
│
├── 02-user-guides/                     # 👥 사용자 가이드
│   ├── Training-Inference-Workflow.md # 학습-추론 워크플로우 (기존, 업데이트)
│   ├── Operations-Guide.md            # 운영 가이드 (기존)
│   ├── Logging-Guide.md               # 로깅 가이드 (기존)
│   └── API-Reference.md               # API 참조 (기존 API.md 확장)
│
├── 03-data-management/                 # 💾 데이터 관리
│   ├── Data-Source-Guide.md           # 데이터 소스 가이드 (NEW, 통합)
│   │   # - 파일 기반 (CSV/Excel/Parquet)
│   │   # - 스트리밍 (향후 Kafka/WebSocket)
│   ├── CFM-Data-Requirements.md       # CFM 데이터 요구사항 (기존)
│   └── Data-Format-Specification.md   # 데이터 형식 명세 (기존 Data-Interface-Specification.md 간소화)
│
├── 04-training-inference/              # 🤖 학습 및 추론
│   ├── Overview.md                    # 개요 (NEW)
│   ├── Training-Guide.md              # 학습 가이드 (기존 Model-Training-Guide.md 업데이트)
│   ├── Inference-Guide.md             # 추론 가이드 (NEW)
│   ├── Model-Architecture.md          # 모델 아키텍처 (기존 AI-Models-Guide.md 업데이트)
│   └── Performance-Tuning.md          # 성능 튜닝 (NEW)
│
├── 05-architecture/                    # 🏗️ 아키텍처
│   ├── System-Architecture.md         # 시스템 아키텍처 (NEW)
│   ├── Training-Inference-Separation.md # 학습-추론 분리 (기존 Training-Inference-Separation-Design.md 요약)
│   ├── Data-Source-Abstraction.md     # 데이터 소스 추상화 (기존 Data-Source-Abstraction-Design.md)
│   └── Component-Design.md            # 컴포넌트 설계 (NEW)
│
├── 06-development/                     # 🛠️ 개발
│   ├── Contributing.md                # 기여 가이드 (NEW)
│   ├── Testing-Guide.md               # 테스팅 가이드 (NEW)
│   └── Code-Style.md                  # 코드 스타일 (NEW)
│
└── archive/                            # 📦 아카이브 (히스토리)
    ├── implementation-history/
    │   ├── Phase1-Implementation-Summary.md
    │   ├── Phase2-Implementation-Summary.md
    │   ├── Phase3-Implementation-Summary.md
    │   └── Phase4-Implementation-Summary.md
    ├── refactoring/
    │   ├── Refactoring-Plan.md
    │   ├── Refactoring-Summary.md
    │   └── Directory-Structure-Cleanup.md
    └── legacy/
        ├── File-Based-Data-Input-Plan.md
        ├── File-Based-Input-Implementation-Summary.md
        └── Data-Source-Abstraction-Summary.md
```

---

## 🔄 문서 매핑 (기존 → 새)

### 통합 및 간소화

**데이터 소스 관련 (6개 → 1개)**:
```
현재:
  - File-Based-Data-Input-Plan.md (18K)
  - File-Based-Input-Implementation-Summary.md (13K)
  - Data-Source-Abstraction-Design.md (15K)
  - Data-Source-Abstraction-Summary.md (13K)
  - Data-Interface-Specification.md (17K)

→ 통합:
  03-data-management/Data-Source-Guide.md (NEW, 20K)
    # 파일 기반 + 스트리밍 모두 다룸
    # 실용적 사용 예제 중심
```

**학습/추론 관련 (4개 → 4개, 재구성)**:
```
현재:
  - Training-Inference-Separation-Design.md (40K) - 너무 상세
  - Training-Inference-Workflow.md (9.3K)
  - Model-Training-Guide.md (11K)
  - AI-Models-Guide.md (14K)

→ 재구성:
  04-training-inference/Overview.md (NEW, 5K)
    # 학습-추론 분리 개념 요약
  04-training-inference/Training-Guide.md (15K)
    # Model-Training-Guide.md + 실습
  04-training-inference/Inference-Guide.md (NEW, 10K)
    # 추론 전용 가이드
  04-training-inference/Model-Architecture.md (12K)
    # AI-Models-Guide.md 업데이트
```

**아카이브 이동 (11개)**:
```
archive/implementation-history/
  - Phase1-4 Implementation Summary (4개)

archive/refactoring/
  - Refactoring-Plan.md
  - Refactoring-Summary.md
  - Directory-Structure-Cleanup.md

archive/legacy/
  - File-Based-Data-Input-Plan.md (계획 문서)
  - File-Based-Input-Implementation-Summary.md (완료 요약)
  - Data-Source-Abstraction-Summary.md (완료 요약)
  - Training-Inference-Separation-Design.md (40K, 너무 상세)
```

---

## ✅ 실행 계획

### Phase 1: 디렉토리 생성 및 문서 이동
1. 새 디렉토리 구조 생성
2. 기존 문서를 새 위치로 이동
3. 파일명 정규화 (일관된 명명 규칙)

### Phase 2: 문서 통합 및 간소화
1. **Data-Source-Guide.md** 작성
   - 파일 기반 + 스트리밍 통합
   - 실용적 예제 중심

2. **Training-Inference Overview** 작성
   - 학습-추론 분리 개념 요약
   - 5-10분 내 읽을 수 있는 분량

3. **Inference-Guide.md** 작성
   - 추론 전용 가이드
   - 데이터 소스 선택부터 결과 분석까지

### Phase 3: 인덱스 및 링크 업데이트
1. **docs/README.md** 작성
   - 전체 문서 인덱스
   - 사용자 유형별 추천 경로

2. **루트 README.md** 업데이트
   - 새 문서 구조 반영
   - 링크 업데이트

3. **CLAUDE.md** 업데이트
   - 새 문서 위치 반영

### Phase 4: 검증 및 정리
1. 모든 링크 검증
2. 중복 내용 제거
3. 오래된 정보 업데이트

---

## 📋 새로운 문서 작성 필요

### 긴급 (사용자가 많이 찾을 문서)
1. **docs/README.md** - 문서 인덱스
2. **04-training-inference/Overview.md** - 학습-추론 개요
3. **04-training-inference/Inference-Guide.md** - 추론 가이드
4. **03-data-management/Data-Source-Guide.md** - 데이터 소스 통합 가이드

### 중요 (운영 및 개발)
5. **01-getting-started/Installation.md** - 설치 가이드
6. **05-architecture/System-Architecture.md** - 시스템 아키텍처
7. **06-development/Testing-Guide.md** - 테스팅 가이드

### 나중 (완성도 향상)
8. **04-training-inference/Performance-Tuning.md** - 성능 튜닝
9. **06-development/Contributing.md** - 기여 가이드
10. **06-development/Code-Style.md** - 코드 스타일

---

## 🎯 기대 효과

### 1. 사용자 경험 개선
- **신규 사용자**: Quick Start → Training → Inference (명확한 경로)
- **CFM 담당자**: CFM-Data-Requirements.md 바로 찾기
- **개발자**: Architecture 폴더에서 설계 문서 찾기

### 2. 유지보수 용이
- 카테고리별 분리로 관련 문서 한눈에 파악
- 중복 제거로 일관성 유지
- 아카이브로 히스토리 보존

### 3. 문서 품질 향상
- 통합으로 내용 일관성 확보
- 최신 아키텍처 반영 (학습-추론 분리)
- 실용적 예제 강화

---

## 🚧 주의사항

### 1. 기존 링크 깨짐 방지
- README.md의 모든 링크 업데이트
- CLAUDE.md의 참조 업데이트
- Git blame 히스토리 유지 (git mv 사용)

### 2. 점진적 마이그레이션
- 한번에 모든 문서 이동 X
- Phase별로 단계적 진행
- 각 Phase 완료 후 검증

### 3. 아카이브 관리
- 히스토리 문서는 삭제하지 않고 archive로 이동
- 참조가 필요할 수 있으므로 보존

---

## 📝 다음 단계

**즉시 실행**:
1. 사용자 승인 확인
2. Phase 1 시작: 디렉토리 구조 생성
3. 가장 중요한 4개 문서 먼저 작성
   - docs/README.md
   - 04-training-inference/Overview.md
   - 04-training-inference/Inference-Guide.md
   - 03-data-management/Data-Source-Guide.md

**승인 필요 사항**:
- 디렉토리 구조 승인
- 문서 통합 방식 승인
- 아카이브 이동 승인

---

**작성자**: Claude Code
**버전**: 1.0.0
**상태**: 제안 중 (사용자 승인 대기)
