# ORAN CFM-Lite AI Anomaly Detection System

ORAN 환경에서 축소된 CFM 기능을 활용한 하이브리드 이상탐지 시스템입니다.

## 주요 특징

- **Capability-Driven**: 장비가 지원하는 기능을 자동 인식하여 파이프라인 구성
- **하이브리드 탐지**: 룰 + 변화점(CUSUM/PELT) + 예측-잔차(TCN/LSTM) + 다변량(옵션)
- **조기 경보**: 끊기기 전 징조를 4분 이상 앞서 탐지 (목표)
- **최소 신호**: UDP-echo, eCPRI delay, LBM만으로도 효과적 탐지

## 시스템 구조

```
O-RU/O-DU → Capability Detector → Collectors → Feature Engine → Detectors → Alerts
```

## 설치 및 실행

### 빠른 시작

```bash
# 자동 설치 및 실행
./scripts/start.sh
```

### 수동 설치

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 설정
cp config/example.yaml config/local.yaml
# config/local.yaml 편집

# 서비스 실행 (API 서버)
python -m ocad.api.main

# 또는 백그라운드 시스템만 실행
python -m ocad.main
```

### CLI 사용

```bash
# 엔드포인트 추가
python -m ocad.cli add-endpoint 192.168.1.100 --role o-ru

# 시스템 상태 확인
python -m ocad.cli status

# 시뮬레이션 실행
python -m ocad.cli simulate --count 10 --duration 300

# 또는 스크립트 사용
./scripts/simulate.py
```

## 주요 컴포넌트

### 1. Capability Detection
- NETCONF/YANG을 통한 장비 기능 자동 인식
- 지원 기능에 따른 수집 파이프라인 자동 구성

### 2. Data Collection
- UDP Echo RTT 측정
- eCPRI One-way Delay 수집  
- LBM (Loopback) RTT 및 성공률
- CCM 최소 통계 (지원시)

### 3. Feature Engineering
- 백분위 (p95, p99) 계산
- EWMA, 기울기, CUSUM 누적량
- 예측-잔차 (TCN/LSTM)
- Run-length, 동시성 지표

### 4. Anomaly Detection
- **룰 기반**: 명확한 임계값 (빠른 탐지)
- **변화점**: CUSUM/PELT (급격한 변화)  
- **예측-잔차**: TCN/LSTM (드리프트 탐지)
- **다변량**: MSCRED/Isolation Forest (그룹 이상)

### 5. Alert Management
- 근거 3개 원칙 (드리프트/급등/동시성 중 2-3개)
- Hold-down, 중복 제거, 억제
- 스파크라인 및 capability 스냅샷

## 성능 목표

- MTTD 20-30% 단축 (룰 대비)
- 사전 경고 리드타임 p50 ≥ 4분
- 오경보율 ≤ 6%
- 운영자 승인율 ≥ 80%
- 전체 지연 ≤ 30초 (95th)

## 개발 스프린트

- Sprint 0: 프로젝트 초기화 및 기본 구조
- Sprint 1: Capability 감지 및 수집기
- Sprint 2: 피처링 및 탐지 엔진 v1
- Sprint 3: 운영 및 대시보드
- Sprint 4: 다변량 탐지 및 하드닝

## 라이선스

MIT License
