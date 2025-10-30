# Tasklist — ORAN CFM-Lite 이상탐지
- 문서버전: v2.1 — 2025-09-29


## 범례
- P: 우선순위(1=Highest) / E: 예상공수(일) / AC: 완료기준 / Own: 담당

## Sprint 0 — 부트스트랩 (P1)
- [ ] 리포/CI 초기화 **E2** **AC:** CI green **Own:** TBD
- [ ] NETCONF/YANG 클라이언트 베이스 **E2** **AC:** hello/capabilities 파싱 샘플
- [ ] Capability 스키마 정의(capabilities.yaml) **E1** **AC:** 리뷰 승인

## Sprint 1 — Capability 감지 & 수집 (P1)
- [ ] NETCONF hello 파서: lbm/udp_echo/ecpri/ccm_min 플래그 **E3** **AC:** 단위테스트 90%+
- [ ] 수집기: UDP-echo 폴러 **E3** **AC:** 샘플 RU로 1분 수집
- [ ] 수집기: eCPRI delay 리더 **E3** **AC:** 지연 타임시리즈 적재
- [ ] 수집기: LBM 트리거/응답 파서 **E3** **AC:** RTT/성공율/실패 run-length 산출
- [ ] (있다면) CCM 최소 통계 수집 **E2** **AC:** inter-arrival/미수신 run-length
- [ ] 시뮬레이터/리플레이(UDP/LBM/eCPRI) **E3** **AC:** 재현 스크립트 제공

## Sprint 2 — 피처/탐지 v1 (P1)
- [ ] 피처: p95/p99, slope, EWMA, CUSUM 누적량 **E3**
- [ ] 잔차: 소형 **TCN**(대안 LSTM) 훈련/추론 **E5**
- [ ] 변화점: **CUSUM/PELT** 적용 **E3**
- [ ] 스코어 합성 및 Severity 버킷 **E2**
- [ ] 알람 카드 v0(근거 2개) **E2**

## Sprint 3 — 운영/대시보드 (P1)
- [ ] dedup/hold-down/suppress **E3**
- [ ] 알람 카드 v1(근거 3개 + 스파크라인) **E2**
- [ ] KPI 보드: MTTD/LeadTime/FAR/승인율/dedup/capability-coverage **E3**
- [ ] 튜닝 UI(민감도/가중치 프리셋) **E3**

## Sprint 4 — 다변량 & 하드닝 (P2)
- [ ] MEG/섹터 그룹 **Isolation Forest** 시작 **E3**
- [ ] 효과시 **MSCRED**로 승격(경량 구성) **E5**
- [ ] 온라인 A/B 실험 파이프라인 **E3**
- [ ] 런북/장애 사례 카탈로그 **E2**

## 산출물 & Done
- PRD/Design/Explain(ORAN) / Dashboard JSON / capability 스키마 / 시뮬레이터 / 런북
