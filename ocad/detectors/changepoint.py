"""CUSUM 알고리즘을 사용한 변화점 탐지.
급격한 성능 변화를 실시간으로 감지하는 모듈
"""

from typing import Dict

from ..core.models import Capabilities, FeatureVector
from .base import BaseDetector


class ChangePointDetector(BaseDetector):
    """CUSUM 알고리즘을 사용한 변화점 탐지기.
    
    Cumulative Sum(CUSUM) 알고리즘을 사용하여 시계열 데이터에서
    급격한 성능 변화점을 실시간으로 감지합니다.
    룰 기반 탐지에서 놓칠 수 있는 점진적이지만 유의미한 변화를 포착합니다.
    """
    
    def can_detect(self, capabilities: Capabilities) -> bool:
        """CUSUM은 모든 시계열 데이터에서 작동할 수 있습니다.
        
        Args:
            capabilities: 엔드포인트 기능 정보
            
        Returns:
            사용 가능한 메트릭이 있으면 True
        """
        return any([
            capabilities.udp_echo,
            capabilities.ecpri_delay,
            capabilities.lbm,
        ])
    
    def detect(self, features: FeatureVector, capabilities: Capabilities) -> float:
        """CUSUM 값을 사용하여 변화점을 탐지합니다.
        
        Args:
            features: 분석할 피처 벡터
            capabilities: 엔드포인트 기능 정보
            
        Returns:
            0.0과 1.0 사이의 이상 점수
        """
        max_cusum = 0.0
        cusum_count = 0
        
        # 각 메트릭의 CUSUM 값 확인
        if capabilities.udp_echo and features.cusum_udp_echo is not None:
            max_cusum = max(max_cusum, features.cusum_udp_echo)
            cusum_count += 1
            
            if features.cusum_udp_echo > self.config.cusum_threshold:
                self.logger.debug(
                    "UDP echo CUSUM 임계값 초과",
                    endpoint_id=features.endpoint_id,
                    cusum=features.cusum_udp_echo,
                    threshold=self.config.cusum_threshold,
                )
        
        if capabilities.ecpri_delay and features.cusum_ecpri is not None:
            max_cusum = max(max_cusum, features.cusum_ecpri)
            cusum_count += 1
            
            if features.cusum_ecpri > self.config.cusum_threshold:
                self.logger.debug(
                    "eCPRI CUSUM 임계값 초과",
                    endpoint_id=features.endpoint_id,
                    cusum=features.cusum_ecpri,
                    threshold=self.config.cusum_threshold,
                )
        
        if capabilities.lbm and features.cusum_lbm is not None:
            max_cusum = max(max_cusum, features.cusum_lbm)
            cusum_count += 1
            
            if features.cusum_lbm > self.config.cusum_threshold:
                self.logger.debug(
                    "LBM CUSUM 임계값 초과",
                    endpoint_id=features.endpoint_id,
                    cusum=features.cusum_lbm,
                    threshold=self.config.cusum_threshold,
                )
        
        if cusum_count == 0:
            return 0.0
        
        # 임계값 기반으로 점수 정규화
        # 임계값에서 0.0, CUSUM이 증가하면 1.0에 접근
        score = min(1.0, max_cusum / (self.config.cusum_threshold * 2.0))
        
        return score
    
    def get_evidence(self, features: FeatureVector) -> Dict[str, float]:
        """변화점 탐지에 대한 증거 세부사항을 반환합니다.
        
        Args:
            features: 피처 벡터
            
        Returns:
            증거 세부사항이 담긴 딕셔너리
        """
        evidence = {}
        
        if features.cusum_udp_echo is not None:
            evidence["cusum_udp_echo"] = features.cusum_udp_echo
        
        if features.cusum_ecpri is not None:
            evidence["cusum_ecpri"] = features.cusum_ecpri
        
        if features.cusum_lbm is not None:
            evidence["cusum_lbm"] = features.cusum_lbm
        
        return evidence
