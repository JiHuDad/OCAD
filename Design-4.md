# Design — ORAN CFM-Lite 이상탐지
- 문서버전: v2.1 — 2025-09-29


## 1. 시스템 아키텍처

```mermaid
flowchart LR
  subgraph RU_DU[O-RU / O-DU]
    RU[장비]:::dev -->|NETCONF/YANG| MPlane[M-Plane]
  end
  MPlane --> Cap[Capability Detector]
  Cap -->|기능 매핑| Coll[Collectors]
  Coll --> Bus[(Message Bus)]
  Bus --> FE[Feature Engine\n(Sliding Window)]
  FE -->|변화점| CP[CUSUM/PELT]
  FE -->|예측-잔차| FR[TCN/LSTM (소형)]
  FE -->|옵션| MV[Group Multivariate\n(MSCRED/IF)]
  CP --> FX[Score Fusion]
  FR --> FX
  MV --> FX
  FX --> ST[(Alert Store)]
  FX --> OBS[Metrics/Logs]
  ST --> Noti[Notifier/ITSM]
  OBS --> Dash[Dashboard/KPI]
classDef dev fill:#eef,stroke:#99f;
```

## 2. Capability-Driven 파이프라인 흐름

```mermaid
sequenceDiagram
  participant RU as O-RU/O-DU
  participant CAP as Capability Detector
  participant COL as Collectors
  participant FE as Feature Engine
  participant DET as Detectors

  RU->>CAP: NETCONF hello (capabilities)
  CAP->>COL: { lbm:bool, udp_echo:bool, ecpri:bool, ccm_min:bool }
  COL->>FE: 활성 메트릭 스트림 publish
  FE->>DET: x = features(p95/p99, slope, CUSUM, residual, runlen)
  DET-->>FE: s = scores(rule, cp, residual, mv)
  FE->>FX: score fusion → severity
```

## 3. 데이터 모델(요약 ER)

```mermaid
erDiagram
  ENDPOINT ||--o{ SAMPLE : has
  ENDPOINT {
    string id
    string role
    bool lbm
    bool udp_echo
    bool ecpri_delay
    bool ccm_min
  }
  SAMPLE {
    int ts_ms
    float udp_echo_rtt
    float ecpri_ow_us
    float lbm_rtt
    bool  lbm_success
    float ccm_inter_ms
    int   ccm_runlen
    bool  lldp_changed
    int   port_crc_err
  }
```

## 4. 토픽/인터페이스(예시)
- `oran.caps` — endpoint_id, caps(json), ts
- `oran.metrics` — 정규화된 측정값 스트림
- `oran.features` — 윈도우 피처
- `oran.alerts` — severity, evidence(top3), caps_snapshot

## 5. 탐지 로직(의사코드)

```python
caps = get_caps(endpoint)
x = make_features(stream, caps)  # p95/p99, slope, EWMA, cusum, runlen...

s_rule = rule_score(x, caps)
s_cp   = cusum_score(x["spike_candidates"])
yhat   = tcn.predict(x["latency_window"])  # or lstm
s_res  = abs(x["latency_now"] - yhat)
s_mv   = mv_score(group_tensor) if caps.group_ok else 0.0

score = 0.35*s_rule + 0.25*s_cp + 0.30*s_res + 0.10*s_mv
sev   = bucketize(score, hysteresis=True, hold_down=120)
emit_alert(endpoint, sev, evidence=top3([s_cp,s_res,x.get("runlen"),x.get("delta_p99")]), caps=caps)
```

## 6. 배포 토폴로지
- 사이트별 경량 수집기(또는 중앙 수집) → 메시지버스(Kafka/NATS)
- 스트리밍 피처 엔진(1분 창/30초 중첩), 모델 레지스트리
- Grafana/ITSM 연동(링크 파라미터: endpoint_id, time_range)

## 7. SLO/운영
- 경보 지연 p95 ≤ 30s, 알람 중복 제거율 ≥ 30%
- 주간 재학습, 월간 베이스라인 재생성(계절성)
- 튜닝 프리셋: 보수/표준/공격적
