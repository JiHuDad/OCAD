# OCAD 로그 시스템 가이드

## 📋 목차
1. [로그 시스템 개요](#1-로그-시스템-개요)
2. [로그 구조](#2-로그-구조)
3. [로그 설정 방법](#3-로그-설정-방법)
4. [로그 활용 시나리오](#4-로그-활용-시나리오)
5. [문제 해결을 위한 로그 분석](#5-문제-해결을-위한-로그-분석)

## 1. 로그 시스템 개요

OCAD는 **구조화된 다층 로그 시스템**을 제공하여 다양한 사용자의 요구에 맞는 정보를 제공합니다.

### 1.1 설계 원칙
- **분리된 관심사**: 디버그/요약/알람별 로그 분리
- **사용자 친화적**: 기술자와 운영자 모두를 위한 정보 제공
- **이식성**: 환경변수를 통한 유연한 로그 위치 설정
- **구조화**: 일관된 형식과 계층적 구조

### 1.2 주요 특징
- 📁 **계층적 구조**: 목적별 폴더 분리
- 🕐 **타임스탬프 기반**: 실행 시점별 로그 관리
- 🔧 **설정 가능**: 환경변수를 통한 커스터마이징
- 📖 **사람 친화적**: 기술적 배경 없이도 이해 가능한 분석

## 2. 로그 구조

### 2.1 전체 구조
```
logs/test_YYYYMMDD_HHMMSS/
├── debug/
│   └── detailed.log              # 🔍 상세 디버그 로그
├── summary/
│   └── summary.log               # 📋 주요 이벤트 로그
└── alerts/
    ├── alert_details.log         # 🚨 기술적 알람 정보
    └── human_readable_analysis.txt # 📖 사람 친화적 분석
```

### 2.2 각 로그 파일의 역할

#### 2.2.1 debug/detailed.log
**목적**: 시스템의 모든 상세 동작 기록

**포함 내용**:
- CUSUM 계산 과정
- 피처 추출 세부 과정
- 탐지 알고리즘별 점수 계산
- NETCONF 통신 로그
- 내부 상태 변화

**대상 사용자**: 개발자, 시스템 엔지니어

**예시**:
```log
[2025-09-30T02:03:30.718919Z] [info] Starting system orchestrator
[2025-09-30T02:03:30.719375Z] [info] System orchestrator started successfully
[2025-09-30T02:03:30.720492Z] [info] Endpoint registered [endpoint_id=sim-o-ru-001]
[2025-09-30T02:03:30.725051Z] [debug] UDP Echo CUSUM calculated [series_len=45, cusum=6.45]
[2025-09-30T02:03:30.725285Z] [debug] Rule-based score calculated [violations=2, score=0.85]
```

#### 2.2.2 summary/summary.log
**목적**: 시스템의 주요 이벤트만 기록

**포함 내용**:
- 시스템 시작/종료
- 엔드포인트 등록/해제
- 알람 생성/해제
- 중요한 설정 변경
- 오류 발생

**대상 사용자**: 운영자, 관리자

**예시**:
```log
[2025-09-30T02:03:30.718919Z] [info] OCAD System Started
[2025-09-30T02:03:30.720492Z] [info] Endpoint Registered [sim-o-ru-001]
[2025-09-30T02:04:45.123456Z] [warn] Alert Generated [sim-o-ru-001, WARNING, Network Delay]
[2025-09-30T02:05:30.654321Z] [info] Alert Resolved [sim-o-ru-001]
```

#### 2.2.3 alerts/alert_details.log
**목적**: 알람의 기술적 세부 정보 기록

**포함 내용**:
- 알람 ID 및 메타데이터
- 탐지 점수 세부 내역
- 증거(Evidence) 목록
- 기술적 임계값 정보

**대상 사용자**: 네트워크 엔지니어, 기술 지원팀

**예시**:
```log
알람 #1: sim-o-ru-001
심각도: WARNING
탐지 시간: 1696059471000ms
전체 설명: Anomaly detected with evidence: rule violations, changepoint spike
상세 원인:
  1. Rule violations: UDP Echo P99 24.7ms > 5.0ms (신뢰도: 90.0%)
  2. Sudden changes: UDP Echo CUSUM 6.45; LBM CUSUM 3.87 (신뢰도: 80.0%)
종합 점수: 0.416
```

#### 2.2.4 alerts/human_readable_analysis.txt
**목적**: 일반 운영자를 위한 사람 친화적 분석

**포함 내용**:
- 일반인이 이해할 수 있는 문제 설명
- 비즈니스 영향도 분석
- 구체적이고 실행 가능한 조치사항
- 지속 모니터링 가이드

**대상 사용자**: 운영팀, NOC 직원, 관리자

**특징**:
- 기술 용어 최소화
- 실행 가능한 조치사항
- O-RAN 특화 권장사항
- 단계별 문제 해결 가이드

## 3. 로그 설정 방법

### 3.1 기본 설정 (프로젝트 루트)
```bash
cd /path/to/OCAD
python3 scripts/quick_test.py
# 결과: ./logs/test_YYYYMMDD_HHMMSS/
```

### 3.2 환경변수를 통한 커스터마이징
```bash
# 임시 디렉토리 사용
export OCAD_LOG_DIR=/tmp/ocad_logs
python3 scripts/quick_test.py

# 시스템 로그 디렉토리 사용
export OCAD_LOG_DIR=/var/log/ocad
sudo mkdir -p /var/log/ocad
sudo chown $USER:$USER /var/log/ocad
python3 scripts/quick_test.py

# 사용자 홈 디렉토리 사용
export OCAD_LOG_DIR=$HOME/ocad_logs
python3 scripts/quick_test.py
```

### 3.3 운영 환경 설정 예시

#### 3.3.1 개발 환경
```bash
# .bashrc 또는 .profile에 추가
export OCAD_LOG_DIR=$HOME/dev/ocad_logs
```

#### 3.3.2 테스트 환경
```bash
# Jenkins/CI 스크립트에서
export OCAD_LOG_DIR=/var/log/ci/ocad/${BUILD_NUMBER}
```

#### 3.3.3 운영 환경
```bash
# systemd 서비스 파일에서
Environment=OCAD_LOG_DIR=/var/log/ocad
```

### 3.4 Docker 환경 설정
```bash
# Docker run 시
docker run -e OCAD_LOG_DIR=/app/logs \
           -v /host/logs:/app/logs \
           ocad:latest

# docker-compose.yml에서
environment:
  - OCAD_LOG_DIR=/app/logs
volumes:
  - ./logs:/app/logs
```

## 4. 로그 활용 시나리오

### 4.1 일상 운영 모니터링
**목표**: 시스템 상태 일반적 확인

**활용 방법**:
1. `summary/summary.log` 정기 확인
2. 새로운 알람 발생 시 `alerts/human_readable_analysis.txt` 확인
3. 문제 해결 후 결과 기록

**점검 주기**: 매 시간 또는 실시간 모니터링

### 4.2 알람 대응
**목표**: 발생한 알람의 원인 파악 및 조치

**활용 순서**:
1. **즉시 대응**: `human_readable_analysis.txt`에서 문제 요약 확인
2. **조치 실행**: 권장 조치사항 순서대로 실행
3. **기술 분석**: 필요 시 `alert_details.log`에서 세부 정보 확인
4. **추적 관찰**: 모니터링 포인트 지속 관찰

### 4.3 트러블슈팅
**목표**: 복잡한 문제의 근본 원인 분석

**활용 방법**:
1. `detailed.log`에서 문제 발생 시점 전후 로그 분석
2. CUSUM 계산 과정 및 탐지 점수 변화 추적
3. 시스템 구성 요소별 상태 변화 분석
4. 외부 요인과의 상관관계 분석

### 4.4 성능 분석
**목표**: 시스템 성능 및 탐지 효율성 분석

**활용 데이터**:
- 피처 추출 성능 로그
- 탐지 알고리즘별 실행 시간
- 메모리 사용량 변화
- 처리 지연 통계

### 4.5 운영 보고서 작성
**목표**: 정기적인 운영 현황 보고

**활용 내용**:
- 알람 발생 빈도 및 유형 분석
- 문제 해결 시간 통계
- 시스템 가용성 지표
- 개선 권장사항

## 5. 문제 해결을 위한 로그 분석

### 5.1 알람 미발생 문제
**증상**: 명백한 문제가 있는데 알람이 발생하지 않음

**분석 순서**:
1. `detailed.log`에서 피처 추출 확인
2. 탐지 알고리즘별 점수 계산 확인
3. 임계값 설정 확인
4. 데이터 수집 상태 확인

**주요 확인 포인트**:
```log
[debug] UDP Echo CUSUM calculated [series_len=45, cusum=6.45]
[debug] Rule-based score calculated [violations=2, score=0.85]
[debug] Composite score calculated [score=0.416]
[info] Evidence generated [count=2, min_required=1]
```

### 5.2 오탐 문제
**증상**: 정상 상황에서 불필요한 알람 발생

**분석 방법**:
1. `human_readable_analysis.txt`에서 탐지 근거 확인
2. `alert_details.log`에서 임계값과 실제 값 비교
3. `detailed.log`에서 데이터 품질 확인
4. 베이스라인 설정 적절성 검토

### 5.3 성능 저하 문제
**증상**: 시스템 응답 속도 저하

**분석 포인트**:
- 피처 추출 처리 시간
- 데이터 큐 대기 시간  
- 메모리 사용량 추이
- 외부 의존성(DB, Kafka) 응답 시간

### 5.4 로그 분석 도구 활용

#### 5.4.1 기본 명령어
```bash
# 특정 시간대 로그 확인
grep "2025-09-30T02:04" detailed.log

# 알람 관련 로그만 추출
grep -i "alert\|alarm" summary.log

# 오류 로그 확인
grep -i "error\|failed" detailed.log

# 특정 엔드포인트 로그 추적
grep "sim-o-ru-001" detailed.log
```

#### 5.4.2 고급 분석
```bash
# 시간대별 알람 발생 패턴 분석
awk '/Alert Generated/ {print $1}' summary.log | sort | uniq -c

# 탐지 점수 분포 분석
grep "Composite score" detailed.log | awk '{print $NF}' | sort -n

# 응답 시간 추이 분석
grep "UDP Echo.*ms" detailed.log | awk '{print $(NF-1)}' | sort -n
```

## 6. 로그 관리 모범 사례

### 6.1 보관 정책
- **단기 보관** (1주일): 모든 로그
- **중기 보관** (1개월): summary.log와 중요 alert 로그
- **장기 보관** (1년): 주요 알람의 human_readable_analysis.txt

### 6.2 백업 전략
```bash
# 중요 알람 로그 백업
tar -czf important_alerts_$(date +%Y%m%d).tar.gz \
    logs/*/alerts/human_readable_analysis.txt

# 주간 요약 로그 아카이브
find logs -name "summary.log" -mtime -7 | \
    xargs tar -czf weekly_summary_$(date +%Y%m%d).tar.gz
```

### 6.3 정기 정리
```bash
# 30일 이상 된 로그 삭제
find logs -type d -name "test_*" -mtime +30 -exec rm -rf {} \;

# 디스크 사용량 모니터링
du -sh logs/*/
```

### 6.4 모니터링 자동화
```bash
# 로그 디렉토리 크기 모니터링
#!/bin/bash
LOG_SIZE=$(du -s logs/ | cut -f1)
if [ $LOG_SIZE -gt 1000000 ]; then  # 1GB 이상
    echo "Warning: Log directory size exceeds 1GB"
fi

# 최근 알람 발생 확인
#!/bin/bash
RECENT_ALERTS=$(find logs -name "human_readable_analysis.txt" -mtime -1 | wc -l)
echo "Recent alerts in last 24 hours: $RECENT_ALERTS"
```

---

이 가이드를 통해 OCAD 로그 시스템을 효과적으로 활용하여 
시스템 운영 품질을 향상시킬 수 있습니다.
