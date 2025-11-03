"""OCAD 시스템의 설정 관리.
시스템 전반의 설정을 중앙화하고 환경별 구성을 지원합니다.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """데이터베이스 설정.
    
    시계열 메트릭과 알람 이력을 저장하는 PostgreSQL 설정입니다.
    """
    
    url: str = "postgresql+asyncpg://ocad:ocad@localhost:5432/ocad"
    pool_size: int = 10      # 연결 풀 크기
    max_overflow: int = 20   # 최대 오버플로우 연결 수
    echo: bool = False       # SQL 쿼리 로깅 여부


class RedisConfig(BaseModel):
    """Redis 설정.
    
    실시간 데이터 캐싱과 세션 관리에 사용됩니다.
    """
    
    url: str = "redis://localhost:6379/0"
    max_connections: int = 10  # 최대 연결 수


class KafkaConfig(BaseModel):
    """Kafka 설정.
    
    대용량 메트릭 스트리밍과 이벤트 처리에 사용됩니다.
    """
    
    bootstrap_servers: List[str] = ["localhost:9092"]
    group_id: str = "ocad"                # 컨슈머 그룹 ID
    auto_offset_reset: str = "latest"     # 오프셋 리셋 정책


class NetconfConfig(BaseModel):
    """NETCONF 설정.
    
    ORAN 장비와의 연결 및 데이터 수집을 위한 설정입니다.
    """
    
    timeout: int = 30              # 연결 타임아웃 (초)
    port: int = 830               # 기본 NETCONF 포트
    username: str = "admin"       # 인증 사용자명
    password: str = "admin"       # 인증 비밀번호
    hostkey_verify: bool = False  # 호스트 키 검증 여부


class FeatureConfig(BaseModel):
    """피처 엔지니어링 설정.
    
    시계열 데이터에서 피처를 추출하는 방식을 제어합니다.
    """
    
    window_size_minutes: int = 1      # 슬라이딩 윈도우 크기 (분)
    overlap_seconds: int = 30         # 윈도우 중첩 시간 (초)
    percentiles: List[int] = [95, 99] # 계산할 백분위 목록
    ewma_alpha: float = 0.3           # EWMA 스무딩 계수
    cusum_threshold: float = 5.0      # CUSUM 변화점 임계값


class DetectionConfig(BaseModel):
    """이상 탐지 알고리즘 설정.
    
    각 탐지 방법의 가중치와 임계값을 정의합니다.
    """
    
    # 알고리즘별 가중치 (합이 1.0이 되도록)
    rule_weight: float = 0.35         # 룰 기반 탐지 가중치
    changepoint_weight: float = 0.25  # 변화점 탐지 가중치
    residual_weight: float = 0.30     # 잔차 기반 탐지 가중치
    multivariate_weight: float = 0.10 # 다변량 탐지 가중치
    
    # 탐지 임계값들
    rule_timeout_ms: float = 5000.0      # 룰 기반 타임아웃 임계값 (ms)
    rule_p99_threshold_ms: float = 100.0 # 룰 기반 p99 지연 임계값 (ms)
    rule_runlength_threshold: int = 3    # 룰 기반 연속 실패 임계값 (회)
    cusum_threshold: float = 5.0         # CUSUM 변화점 임계값
    residual_threshold: float = 6.5      # 잔차 이상 임계값 (정규화된 std 배수)
    
    # 알람 제어
    hold_down_seconds: int = 120      # Hold-down 시간 (초)
    dedup_window_seconds: int = 300   # 중복 제거 윈도우 (초)

    # 사전 훈련 모델 설정 (학습-추론 분리)
    use_pretrained_models: bool = True              # 사전 훈련 모델 사용 여부
    pretrained_model_dir: str = "ocad/models/tcn"   # 사전 훈련 모델 디렉토리
    inference_device: str = "cpu"                   # 추론 디바이스 ("cpu", "cuda", "mps")


class AlertConfig(BaseModel):
    """알람 관리 설정.
    
    알람 생성 조건과 심각도 분류를 정의합니다.
    """
    
    evidence_count: int = 3              # 최대 증거 개수
    min_evidence_for_alert: int = 2      # 알람 생성 최소 증거 개수
    severity_buckets: Dict[str, float] = {  # 심각도별 점수 임계값
        "critical": 0.8,  # 심각: 0.8 이상
        "warning": 0.6,   # 경고: 0.6 이상
        "info": 0.4       # 정보: 0.4 이상
    }


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    
    prometheus_port: int = 8000
    log_level: str = "INFO"
    enable_tracing: bool = True


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 1
    
    # Component configs
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    netconf: NetconfConfig = Field(default_factory=NetconfConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_config(config_path: Optional[Path] = None) -> Settings:
    """Load configuration from file and environment variables.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Loaded settings
    """
    if config_path and config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            return Settings(**config_data)
    
    return Settings()


# Global settings instance
settings = load_config()
