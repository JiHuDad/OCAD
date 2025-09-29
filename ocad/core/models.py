"""OCAD 시스템의 핵심 데이터 모델.
시스템 전반에서 사용되는 주요 데이터 구조와 타입을 정의합니다.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class EndpointRole(str, Enum):
    """ORAN 네트워크에서의 엔드포인트 역할."""
    
    O_RU = "o-ru"          # O-RAN Radio Unit
    O_DU = "o-du"          # O-RAN Distributed Unit  
    TRANSPORT = "transport" # Transport Network Element


class Severity(str, Enum):
    """알람 심각도 수준."""
    
    CRITICAL = "critical"  # 즉시 대응이 필요한 심각한 상황
    WARNING = "warning"    # 주의가 필요한 경고 상황
    INFO = "info"          # 정보성 알림


class Capabilities(BaseModel):
    """엔드포인트가 지원하는 기능들.
    
    O-RAN 표준에 따른 CFM-Lite 기능 지원 여부를 나타냅니다.
    """
    
    ccm_min: bool = False      # CCM 최소 통계 지원
    lbm: bool = False          # Loopback Message 지원
    udp_echo: bool = False     # UDP Echo (supervision) 지원
    ecpri_delay: bool = False  # eCPRI 지연 측정 지원
    lldp: bool = False         # LLDP 변화 모니터링 지원


class Endpoint(BaseModel):
    """네트워크 엔드포인트 정의.
    
    모니터링 대상이 되는 ORAN 장비의 기본 정보를 담습니다.
    """
    
    id: str = Field(..., description="고유 엔드포인트 식별자")
    role: EndpointRole = Field(..., description="엔드포인트 역할")
    host: str = Field(..., description="호스트명 또는 IP 주소")
    port: int = Field(default=830, description="NETCONF 포트")
    capabilities: Capabilities = Field(default_factory=Capabilities)
    last_seen: Optional[datetime] = None  # 마지막 접촉 시간
    active: bool = True  # 활성 상태 여부


class MetricSample(BaseModel):
    """엔드포인트에서 수집한 단일 메트릭 샘플.
    
    CFM-Lite 기능들로부터 수집된 성능 측정값들을 담습니다.
    """
    
    endpoint_id: str
    ts_ms: int = Field(..., description="밀리초 단위 타임스탬프")
    
    # CFM-Lite 메트릭들
    udp_echo_rtt_ms: Optional[float] = None    # UDP Echo 왕복 시간 (ms)
    ecpri_ow_us: Optional[float] = None        # eCPRI 일방향 지연 (μs)
    lbm_rtt_ms: Optional[float] = None         # LBM 왕복 시간 (ms)
    lbm_success: Optional[bool] = None         # LBM 성공 여부
    
    # CCM 최소 통계
    ccm_inter_arrival_ms: Optional[float] = None  # CCM 도착 간격 (ms)
    ccm_runlen_miss: Optional[int] = None          # CCM 연속 누락 횟수
    
    # 전송 계층 메트릭
    port_crc_err: Optional[int] = None         # 포트 CRC 오류 수
    q_drop: Optional[int] = None               # 큐 드롭 수
    lldp_changed: Optional[bool] = None        # LLDP 변화 여부


class FeatureVector(BaseModel):
    """이상 탐지를 위해 계산된 피처들.
    
    원시 메트릭 샘플들로부터 추출된 고급 통계 피처들을 담습니다.
    시계열 분석과 머신러닝 모델에서 사용됩니다.
    """
    
    endpoint_id: str
    ts_ms: int              # 피처 계산 시점
    window_size_ms: int     # 윈도우 크기 (밀리초)
    
    # 백분위 피처들 (성능 분포 특성)
    udp_echo_p95: Optional[float] = None     # UDP Echo 95퍼센타일
    udp_echo_p99: Optional[float] = None     # UDP Echo 99퍼센타일  
    ecpri_p95: Optional[float] = None        # eCPRI 95퍼센타일
    ecpri_p99: Optional[float] = None        # eCPRI 99퍼센타일
    lbm_rtt_p95: Optional[float] = None      # LBM RTT 95퍼센타일
    lbm_rtt_p99: Optional[float] = None      # LBM RTT 99퍼센타일
    
    # 추세 피처들 (시간에 따른 변화율)
    udp_echo_slope: Optional[float] = None   # UDP Echo 기울기
    ecpri_slope: Optional[float] = None      # eCPRI 기울기
    lbm_slope: Optional[float] = None        # LBM 기울기
    
    # EWMA 피처들 (지수가중이동평균)
    udp_echo_ewma: Optional[float] = None    # UDP Echo EWMA
    ecpri_ewma: Optional[float] = None       # eCPRI EWMA
    lbm_ewma: Optional[float] = None         # LBM EWMA
    
    # 변화점 탐지 (CUSUM 누적합)
    cusum_udp_echo: Optional[float] = None   # UDP Echo CUSUM
    cusum_ecpri: Optional[float] = None      # eCPRI CUSUM
    cusum_lbm: Optional[float] = None        # LBM CUSUM
    
    # 런길이 인코딩 (연속 실패/성공 횟수)
    lbm_fail_runlen: Optional[int] = None    # LBM 연속 실패 횟수
    ccm_miss_runlen: Optional[int] = None    # CCM 연속 누락 횟수
    
    # 예측 잔차 (예측값 vs 실제값 차이)
    udp_echo_residual: Optional[float] = None  # UDP Echo 잔차
    ecpri_residual: Optional[float] = None     # eCPRI 잔차
    lbm_residual: Optional[float] = None       # LBM 잔차


class DetectionScore(BaseModel):
    """다양한 알고리즘에서 나온 탐지 점수들.
    
    각 탐지 알고리즘의 개별 점수와 종합 점수를 담습니다.
    """
    
    endpoint_id: str
    ts_ms: int
    
    # 개별 알고리즘 점수들 (0-1 범위)
    rule_score: float = 0.0          # 룰 기반 탐지 점수
    changepoint_score: float = 0.0   # 변화점 탐지 점수
    residual_score: float = 0.0      # 잔차 기반 탐지 점수  
    multivariate_score: float = 0.0  # 다변량 탐지 점수
    
    # 종합 점수 (가중합)
    composite_score: float = 0.0
    
    # 증거 세부사항
    evidence: Dict[str, Any] = Field(default_factory=dict)


class AlertEvidence(BaseModel):
    """알람을 뒷받침하는 증거.
    
    알람 생성 근거가 되는 구체적인 증거 정보를 담습니다.
    """
    
    type: str       # 증거 유형: "drift"(드리프트), "spike"(급등), "concurrent"(동시성), "rule"(룰)
    value: float    # 증거 값 (메트릭 값, 점수 등)
    description: str # 증거 설명
    confidence: float = 1.0  # 신뢰도 (0-1)


class Alert(BaseModel):
    """이상 탐지 알람.
    
    시스템에서 감지된 이상 상황에 대한 완전한 정보를 담습니다.
    """
    
    id: str
    endpoint_id: str
    ts_ms: int
    severity: Severity
    
    # 알람 세부사항
    title: str           # 알람 제목
    description: str     # 알람 설명
    evidence: List[AlertEvidence] = Field(default_factory=list)  # 증거 목록
    
    # 컨텍스트 스냅샷 (알람 발생 시점의 상태)
    capabilities_snapshot: Capabilities                    # 기능 스냅샷
    feature_snapshot: Optional[FeatureVector] = None      # 피처 스냅샷
    score_snapshot: Optional[DetectionScore] = None       # 점수 스냅샷
    
    # 생명주기 관리
    acknowledged: bool = False   # 확인됨 여부
    resolved: bool = False       # 해결됨 여부
    suppressed: bool = False     # 억제됨 여부
    created_at: datetime = Field(default_factory=datetime.utcnow)  # 생성 시간
    updated_at: datetime = Field(default_factory=datetime.utcnow)  # 수정 시간


class KPIMetrics(BaseModel):
    """시스템 성능 평가를 위한 KPI 메트릭.
    
    OCAD 시스템의 탐지 성능과 운영 효율성을 측정하는 핵심 지표들을 담습니다.
    """
    
    period_start: datetime  # 측정 기간 시작
    period_end: datetime    # 측정 기간 종료
    
    # 탐지 성능 메트릭
    total_alerts: int = 0      # 총 알람 수
    true_positives: int = 0    # 정탐 (올바른 알람)
    false_positives: int = 0   # 오탐 (잘못된 알람)
    false_negatives: int = 0   # 미탐 (놓친 이상)
    
    # 시간 관련 메트릭
    mean_detection_time_sec: Optional[float] = None    # 평균 탐지 시간 (MTTD)
    mean_lead_time_sec: Optional[float] = None         # 평균 리드타임 (사전 경고)
    p95_processing_latency_sec: Optional[float] = None # 95퍼센타일 처리 지연
    
    # 운영 메트릭
    dedup_rate: float = 0.0           # 중복 제거율
    approval_rate: float = 0.0        # 운영자 승인율
    capability_coverage: float = 0.0  # 기능 커버리지
    
    # 파생 메트릭 (자동 계산)
    @property
    def false_alarm_rate(self) -> float:
        """오경보율 계산 (FAR)."""
        if self.false_positives + self.true_positives == 0:
            return 0.0
        return self.false_positives / (self.false_positives + self.true_positives)
    
    @property
    def precision(self) -> float:
        """정밀도 계산 (Precision)."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """재현율 계산 (Recall)."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """F1 점수 계산 (정밀도와 재현율의 조화평균)."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
