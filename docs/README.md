# OCAD 문서 인덱스

**ORAN CFM-Lite AI Anomaly Detection System** 문서 모음입니다.

---

## 📚 사용자 유형별 가이드

### 🆕 신규 사용자
OCAD를 처음 사용하신다면 다음 순서로 읽어보세요:

1. [빠른 시작 가이드](01-getting-started/Quick-Start-Guide.md) - 5분 내 시작
2. [학습-추론 워크플로우](02-user-guides/Training-Inference-Workflow.md) - 학습과 추론의 전체 흐름
3. [학습 및 추론 개요](04-training-inference/Overview.md) - 학습-추론 분리 아키텍처 이해

### 👨‍💼 CFM 담당자
실제 ORAN 장비에서 데이터를 수집하시려면:

1. [CFM 데이터 요구사항](03-data-management/CFM-Data-Requirements.md) - 수집해야 할 메트릭
2. [데이터 형식 명세](03-data-management/Data-Format-Specification.md) - CSV/Excel/Parquet 형식
3. [데이터 소스 가이드](03-data-management/Data-Source-Guide.md) - 파일 제공 방법

### 🤖 모델 학습/추론 사용자
AI 모델을 학습하고 추론하시려면:

1. [학습 가이드](04-training-inference/Training-Guide.md) - 모델 학습 방법
2. [추론 가이드](04-training-inference/Inference-Guide.md) - 이상 탐지 실행
3. [모델 아키텍처](04-training-inference/Model-Architecture.md) - TCN, LSTM, Isolation Forest

### 🛠️ 운영자
시스템을 운영하고 모니터링하시려면:

1. [운영 가이드](02-user-guides/Operations-Guide.md) - 시스템 운영
2. [로깅 가이드](02-user-guides/Logging-Guide.md) - 로그 분석
3. [API 참조](02-user-guides/API.md) - REST API 사용

### 🏗️ 개발자
OCAD 아키텍처와 설계를 이해하시려면:

1. [시스템 아키텍처](05-architecture/System-Architecture.md) - 전체 아키텍처 (향후 작성)
2. [학습-추론 분리 설계](05-architecture/Training-Inference-Separation-Design.md) - 온라인/오프라인 분리
3. [데이터 소스 추상화](05-architecture/Data-Source-Abstraction-Design.md) - 파일/스트리밍 지원

---

## 📂 문서 구조

### 01-getting-started/ (시작하기)
- [Quick-Start-Guide.md](01-getting-started/Quick-Start-Guide.md) - 빠른 시작 가이드

### 02-user-guides/ (사용자 가이드)
- [Training-Inference-Workflow.md](02-user-guides/Training-Inference-Workflow.md) - 학습-추론 워크플로우
- [Operations-Guide.md](02-user-guides/Operations-Guide.md) - 운영 가이드
- [Logging-Guide.md](02-user-guides/Logging-Guide.md) - 로깅 시스템
- [API.md](02-user-guides/API.md) - REST API 참조

### 03-data-management/ (데이터 관리)
- [Data-Source-Guide.md](03-data-management/Data-Source-Guide.md) - 데이터 소스 가이드 ⭐ (향후 작성)
- [CFM-Data-Requirements.md](03-data-management/CFM-Data-Requirements.md) - CFM 데이터 요구사항
- [Data-Format-Specification.md](03-data-management/Data-Format-Specification.md) - 데이터 형식 명세

### 04-training-inference/ (학습 및 추론)
- [Overview.md](04-training-inference/Overview.md) - 학습-추론 개요 ⭐ (향후 작성)
- [Training-Guide.md](04-training-inference/Training-Guide.md) - 학습 가이드
- [Inference-Guide.md](04-training-inference/Inference-Guide.md) - 추론 가이드 ⭐ (향후 작성)
- [Model-Architecture.md](04-training-inference/Model-Architecture.md) - 모델 아키텍처

### 05-architecture/ (아키텍처)
- [System-Architecture.md](05-architecture/System-Architecture.md) - 시스템 아키텍처 (향후 작성)
- [Training-Inference-Separation-Design.md](05-architecture/Training-Inference-Separation-Design.md) - 학습-추론 분리 설계
- [Data-Source-Abstraction-Design.md](05-architecture/Data-Source-Abstraction-Design.md) - 데이터 소스 추상화

### 06-development/ (개발)
- [Contributing.md](06-development/Contributing.md) - 기여 가이드 (향후 작성)
- [Testing-Guide.md](06-development/Testing-Guide.md) - 테스팅 가이드 (향후 작성)
- [Code-Style.md](06-development/Code-Style.md) - 코드 스타일 (향후 작성)

### archive/ (아카이브)
- **implementation-history/** - Phase 1-4 구현 히스토리
- **refactoring/** - 리팩토링 계획 및 요약
- **legacy/** - 레거시 문서 (이전 버전)

---

## 🔍 주제별 빠른 찾기

### 데이터 입력
- 파일 기반: [Data-Source-Guide.md](03-data-management/Data-Source-Guide.md) ⭐
- 실시간 수집: [CFM-Data-Requirements.md](03-data-management/CFM-Data-Requirements.md)
- 데이터 형식: [Data-Format-Specification.md](03-data-management/Data-Format-Specification.md)

### 학습
- 빠른 시작: [Overview.md](04-training-inference/Overview.md) ⭐
- 상세 가이드: [Training-Guide.md](04-training-inference/Training-Guide.md)
- 모델 설명: [Model-Architecture.md](04-training-inference/Model-Architecture.md)

### 추론
- 추론 가이드: [Inference-Guide.md](04-training-inference/Inference-Guide.md) ⭐
- 전체 워크플로우: [Training-Inference-Workflow.md](02-user-guides/Training-Inference-Workflow.md)

### 운영
- 시스템 운영: [Operations-Guide.md](02-user-guides/Operations-Guide.md)
- 로그 분석: [Logging-Guide.md](02-user-guides/Logging-Guide.md)
- API 사용: [API.md](02-user-guides/API.md)

### 아키텍처
- 학습-추론 분리: [Training-Inference-Separation-Design.md](05-architecture/Training-Inference-Separation-Design.md)
- 데이터 소스 추상화: [Data-Source-Abstraction-Design.md](05-architecture/Data-Source-Abstraction-Design.md)

---

## 📝 문서 작성 현황

### ✅ 완료
- 01-getting-started/Quick-Start-Guide.md
- 02-user-guides/* (4개)
- 03-data-management/CFM-Data-Requirements.md
- 03-data-management/Data-Format-Specification.md
- 04-training-inference/Training-Guide.md
- 04-training-inference/Model-Architecture.md
- 05-architecture/* (2개)

### ⭐ 우선 작성 필요
- 03-data-management/Data-Source-Guide.md - 데이터 소스 통합 가이드
- 04-training-inference/Overview.md - 학습-추론 개요
- 04-training-inference/Inference-Guide.md - 추론 가이드

### 📅 향후 작성
- 05-architecture/System-Architecture.md
- 06-development/* (3개)

---

## 🔗 외부 링크

- [GitHub Repository](https://github.com/your-org/ocad) (TBD)
- [프로젝트 루트 README](../README.md)
- [CLAUDE.md](../CLAUDE.md) - Claude Code 가이드

---

## 📞 도움말

문서에서 찾을 수 없는 내용이 있다면:

1. [GitHub Issues](https://github.com/your-org/ocad/issues) (TBD)
2. 프로젝트 관리자에게 문의

---

**최종 업데이트**: 2025-10-27
**문서 버전**: 2.0.0 (재구조화 완료)
