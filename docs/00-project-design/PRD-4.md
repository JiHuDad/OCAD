# ORAN CFM-Lite 기반 AI 이상탐지 — PRD
- 문서버전: v2.1 (ORAN 대응)
- 날짜: 2025-09-29
- 작성: AI 설계 도우미


## 1) 배경 & 스코프
O-RAN 환경(특히 프런트홀 O-RU/O-DU)은 IEEE 802.1Q CFM/Y.1731 기능이 축소되어 제공될 수 있다. 
일부 장비는 **LBM(Loopback)** 또는 **UDP-echo / eCPRI delay**만 제공하며, **LT/LMM/DMR/CCM** 등은 부재할 수 있다.
본 제품은 **capability-driven(장비 지원 기능 자동 인식)** 원칙으로, 가용 신호만으로 **조기 경보**를 제공하는 **하이브리드 이상탐지**다.

### In Scope
- NETCONF/YANG 기반 **capability 탐지** 및 **자동 파이프라인 구성**
- 수집: UDP-echo, eCPRI delay, LBM, (있으면) CCM 최소 통계
- 피처: p95/p99, 추세/기울기, CUSUM 기반 스파이크, 잔차 |y−ŷ|, run-length, 동시성 지표
- 탐지: **룰 + 변화점(CUSUM/PELT) + 예측-잔차(TCN/LSTM)** + (옵션) 다변량(MSCRED/IF)
- 알림: **근거 3개 원칙**(드리프트/급등/동시성 중 2~3개), hold-down, dedup
- 운영: 재학습(주간/월간), 임계 백분위 자동 보정, A/B 실험, KPI 대시보드

### Out of Scope (초기)
- 자동 리라우팅/제어(폐루프)
- GNN 기반 토폴로지 원인추론(파일럿 이후 고려)

## 2) 문제정의
룰(타임아웃/결함 플래그)은 **끊긴 후**에는 강력하지만, **천천히 악화**(지연 드리프트, 간헐 실패 버스트)와
**동시성 패턴**(여러 엔드포인트 동시 흔들림)을 놓치기 쉽다. ORAN에서는 입력 신호가 제한되어 이 문제가 더 크다.

## 3) 목표(OKR)
- **MTTD 20–30% 단축**(룰-only 대비)
- **사전 경고 리드타임 p50 ≥ 4분**(UDP/eCPRI/LBM 기반 잔차)
- **오경보율(FAR) ≤ 6%**, **운영자 승인율 ≥ 80%**
- **capability-coverage ≥ 95%**(장비가 제공하는 기능을 파이프라인에 반영)
- 수집→탐지→알림 **전체 지연 ≤ 30초** (95th)

## 4) 사용자 & 페르소나
- **NOC 엔지니어**: 실시간 알람, 근거 3개, 간단한 조치 가이드
- **NetOps/SRE**: 임계/가중치 튜닝, KPI 모니터링, 재학습 관리
- **서비스 오너**: SLA 리스크 조기 경보, 전/후 성과 리포트

## 5) 사용 시나리오
- **LBM-only RU**: LBM RTT(p95/p99), **실패 run-length** + 변화점/잔차
- **UDP-echo RU**: RTT 잔차 + CUSUM → 조기 경보
- **eCPRI delay RU**: one-way delay 잔차 + CUSUM
- **CCM 최소 통계 있음**: inter-arrival 지터, **연속 미수신 run-length**로 보강
- **LT/LMx 부재**: LLDP 변화율/이웃 통계로 경로 흔들림 추정

## 6) 데이터 사양(요약 스키마)
```yaml
endpoint_id: string
role: ["o-ru","o-du","transport"]
caps: {ccm_min: bool, lbm: bool, udp_echo: bool, ecpri_delay: bool}
ts_ms: int
metrics:
  udp_echo_rtt_ms: float?
  ecpri_ow_us: float?
  lbm_rtt_ms: float?
  lbm_success: bool?
  ccm_inter_arrival_ms: float?
  ccm_runlen_miss: int?
  port_crc_err: int?
  q_drop: int?
  lldp_changed: bool?
```

## 7) 기능 요구사항(Functional)
1. **Capability 감지**: NETCONF hello/capabilities 파싱 → 파이프라인 자동 구성  
2. **수집기**: UDP-echo/eCPRI/LBM/(CCM) 주기 수집, 타임스탬프 정렬/보정  
3. **피처링**: 백분위(p95/p99), EWMA/기울기, CUSUM 누적량, **예측-잔차**, run-length, 동시성 비율  
4. **탐지**: 룰 + CUSUM/PELT + TCN/LSTM(소형) + (옵션) 다변량(MSCRED/IF)  
5. **알림**: 근거 3개 + 스파크라인 + capability 스냅샷, hold-down/dedup/suppress  
6. **운영**: 튜닝 UI(민감도/가중치 프리셋), 라벨 수집(승인/반려), 주간 재학습/월간 베이스라인  
7. **KPI/리포트**: MTTD/리드타임/FAR/미탐/승인율/dedup%/알람볼륨/capability-coverage

## 8) 비기능 요구사항(Non-Functional)
- **성능**: 지연 ≤ 30s, 수천 엔드포인트 1분 윈도우 처리
- **가용성**: 메시지 재처리, 에이전트 재시도, 핵심 컴포넌트 이중화
- **관찰성**: Prometheus 지표, 구조화 로그, 트레이싱
- **보안**: NETCONF over TLS, 데이터 암호화, RBAC, 감사로그

## 9) 성능 평가 & A/B
- **오프라인**: 과거 2–4주 + 합성 장애(지연/버스트) 주입 → 룰 vs 하이브리드 비교
- **온라인**: MEG/섹터 50/50 A/B, 2주 관찰 → FAR/리드타임/승인율 비교
- **통계**: 리드타임(MWU/부트스트랩 CI), 비율(FAR) 2-proportion z-test

## 10) KPI 정의(계산식)
- `MTTD = mean(detect_ts - incident_start_ts)`  
- `LeadTime = incident_start_ts - detect_ts` (사전 탐지면 양수)  
- `FAR = FP / (TP + FP)` , `Miss = FN / (TP + FN)`  
- `Precision/Recall/F1`, **알람 볼륨/일**, **dedup%**, **승인율**  
- **capability-coverage%** = (capability 반영 엔드포인트 수 / 전체)×100

## 11) 릴리즈 플랜
- **PoC(2–4주)**: capability 감지 + UDP/LBM 수집 + CUSUM/TCN  
- **Pilot(4–8주)**: 다변량(옵션), dedup/hold-down, A/B  
- **GA**: 스케일/관찰성/런북 완비

## 12) 리스크 & 완화
- **기능 편차**: capability 기반 **fallback**(UDP/LBM/CCM 대체)  
- **동기/시각오차**: RTT 중심·백분위 기반, 스냅샷 동기 정보 기록  
- **튜닝 난이도/오탐**: 근거 3개+hold-down, 주간 재학습, 백분위 자동보정
