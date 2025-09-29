"""시계열 분석을 위한 피처 엔지니어링 엔진.
원시 메트릭 데이터를 머신러닝에 사용할 수 있는 고급 피처로 변환합니다.
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import structlog

from ..core.config import FeatureConfig
from ..core.logging import get_logger
from ..core.models import FeatureVector, MetricSample


logger = get_logger(__name__)


class SlidingWindow:
    """시계열 데이터를 위한 슬라이딩 윈도우.
    
    일정 시간 범위의 데이터를 유지하면서 오래된 데이터는 자동으로 제거합니다.
    메모리 효율적이고 실시간 스트리밍 처리에 적합합니다.
    """
    
    def __init__(self, window_size_ms: int, overlap_ms: int = 0):
        """슬라이딩 윈도우를 초기화합니다.
        
        Args:
            window_size_ms: 윈도우 크기 (밀리초)
            overlap_ms: 중첩 크기 (밀리초)
        """
        self.window_size_ms = window_size_ms
        self.overlap_ms = overlap_ms
        self.data: deque = deque()
        
    def add_sample(self, sample: MetricSample) -> None:
        """윈도우에 샘플을 추가합니다.
        
        Args:
            sample: 추가할 메트릭 샘플
        """
        self.data.append(sample)
        
        # 윈도우 범위를 벗어난 오래된 샘플들 제거
        current_time = sample.ts_ms
        while self.data and self.data[0].ts_ms < current_time - self.window_size_ms:
            self.data.popleft()
    
    def get_samples(self) -> List[MetricSample]:
        """현재 윈도우의 모든 샘플을 반환합니다.
        
        Returns:
            윈도우 내 샘플 리스트
        """
        return list(self.data)
    
    def is_ready(self) -> bool:
        """윈도우에 충분한 데이터가 있는지 확인합니다.
        
        Returns:
            피처 추출 준비가 되면 True
        """
        if len(self.data) < 2:
            return False
        
        # 윈도우의 최소 절반 이상 데이터가 있는지 확인
        time_span = self.data[-1].ts_ms - self.data[0].ts_ms
        return time_span >= self.window_size_ms // 2


class FeatureExtractor:
    """메트릭 샘플로부터 피처를 추출하는 클래스.
    
    시계열 데이터에서 통계적 특성, 추세, 변화점 등을 추출하여
    이상 탐지 알고리즘에서 사용할 수 있는 피처 벡터를 생성합니다.
    """
    
    def __init__(self, config: FeatureConfig):
        """피처 추출기를 초기화합니다.
        
        Args:
            config: 피처 설정
        """
        self.config = config
        self.logger = logger.bind(component="feature_extractor")
    
    def extract_features(self, window: SlidingWindow, endpoint_id: str) -> Optional[FeatureVector]:
        """슬라이딩 윈도우에서 피처를 추출합니다.
        
        Args:
            window: 샘플들의 슬라이딩 윈도우
            endpoint_id: 엔드포인트 식별자
            
        Returns:
            피처 벡터 또는 데이터 부족시 None
        """
        if not window.is_ready():
            return None
        
        samples = window.get_samples()
        if not samples:
            return None
        
        current_time = samples[-1].ts_ms
        
        # Extract time series for each metric
        udp_echo_series = self._extract_metric_series(samples, "udp_echo_rtt_ms")
        ecpri_series = self._extract_metric_series(samples, "ecpri_ow_us")
        lbm_rtt_series = self._extract_metric_series(samples, "lbm_rtt_ms")
        lbm_success_series = self._extract_boolean_series(samples, "lbm_success")
        ccm_miss_series = self._extract_metric_series(samples, "ccm_runlen_miss")
        
        # Create feature vector
        features = FeatureVector(
            endpoint_id=endpoint_id,
            ts_ms=current_time,
            window_size_ms=window.window_size_ms,
        )
        
        # UDP Echo features
        if udp_echo_series:
            features.udp_echo_p95 = self._calculate_percentile(udp_echo_series, 95)
            features.udp_echo_p99 = self._calculate_percentile(udp_echo_series, 99)
            features.udp_echo_slope = self._calculate_slope(udp_echo_series)
            features.udp_echo_ewma = self._calculate_ewma(udp_echo_series)
            features.cusum_udp_echo = self._calculate_cusum(udp_echo_series)
        
        # eCPRI features
        if ecpri_series:
            features.ecpri_p95 = self._calculate_percentile(ecpri_series, 95)
            features.ecpri_p99 = self._calculate_percentile(ecpri_series, 99)
            features.ecpri_slope = self._calculate_slope(ecpri_series)
            features.ecpri_ewma = self._calculate_ewma(ecpri_series)
            features.cusum_ecpri = self._calculate_cusum(ecpri_series)
        
        # LBM features
        if lbm_rtt_series:
            features.lbm_rtt_p95 = self._calculate_percentile(lbm_rtt_series, 95)
            features.lbm_rtt_p99 = self._calculate_percentile(lbm_rtt_series, 99)
            features.lbm_slope = self._calculate_slope(lbm_rtt_series)
            features.lbm_ewma = self._calculate_ewma(lbm_rtt_series)
            features.cusum_lbm = self._calculate_cusum(lbm_rtt_series)
        
        # Run-length features
        if lbm_success_series is not None:
            features.lbm_fail_runlen = self._calculate_failure_runlength(lbm_success_series)
        
        if ccm_miss_series:
            features.ccm_miss_runlen = int(max(ccm_miss_series)) if ccm_miss_series else 0
        
        return features
    
    def _extract_metric_series(self, samples: List[MetricSample], metric_name: str) -> List[float]:
        """Extract a metric time series from samples.
        
        Args:
            samples: List of metric samples
            metric_name: Name of the metric to extract
            
        Returns:
            List of metric values
        """
        values = []
        for sample in samples:
            value = getattr(sample, metric_name, None)
            if value is not None:
                values.append(float(value))
        
        return values
    
    def _extract_boolean_series(self, samples: List[MetricSample], metric_name: str) -> Optional[List[bool]]:
        """Extract a boolean metric series from samples.
        
        Args:
            samples: List of metric samples
            metric_name: Name of the boolean metric
            
        Returns:
            List of boolean values or None if no data
        """
        values = []
        for sample in samples:
            value = getattr(sample, metric_name, None)
            if value is not None:
                values.append(bool(value))
        
        return values if values else None
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> Optional[float]:
        """Calculate percentile of values.
        
        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value or None
        """
        if not values:
            return None
        
        try:
            return float(np.percentile(values, percentile))
        except Exception:
            return None
    
    def _calculate_slope(self, values: List[float]) -> Optional[float]:
        """Calculate linear trend slope.
        
        Args:
            values: List of values
            
        Returns:
            Slope value or None
        """
        if len(values) < 2:
            return None
        
        try:
            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, values)
            return float(slope)
        except Exception:
            return None
    
    def _calculate_ewma(self, values: List[float]) -> Optional[float]:
        """Calculate Exponentially Weighted Moving Average.
        
        Args:
            values: List of values
            
        Returns:
            EWMA value or None
        """
        if not values:
            return None
        
        try:
            alpha = self.config.ewma_alpha
            ewma = values[0]
            
            for value in values[1:]:
                ewma = alpha * value + (1 - alpha) * ewma
            
            return float(ewma)
        except Exception:
            return None
    
    def _calculate_cusum(self, values: List[float]) -> Optional[float]:
        """Calculate CUSUM (Cumulative Sum) for change detection.
        
        Args:
            values: List of values
            
        Returns:
            CUSUM value or None
        """
        if len(values) < 3:
            return None
        
        try:
            # Calculate mean and std of the series
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return 0.0
            
            # Normalize values
            normalized = [(v - mean_val) / std_val for v in values]
            
            # Calculate CUSUM
            threshold = self.config.cusum_threshold / 10.0  # Scale down for normalized data
            cusum_pos = 0.0
            cusum_neg = 0.0
            max_cusum = 0.0
            
            for value in normalized:
                cusum_pos = max(0, cusum_pos + value - threshold)
                cusum_neg = max(0, cusum_neg - value - threshold)
                max_cusum = max(max_cusum, cusum_pos, cusum_neg)
            
            return float(max_cusum)
        except Exception:
            return None
    
    def _calculate_failure_runlength(self, success_values: List[bool]) -> Optional[int]:
        """Calculate maximum failure run length.
        
        Args:
            success_values: List of success/failure boolean values
            
        Returns:
            Maximum consecutive failure count
        """
        if not success_values:
            return None
        
        max_failures = 0
        current_failures = 0
        
        for success in success_values:
            if not success:
                current_failures += 1
                max_failures = max(max_failures, current_failures)
            else:
                current_failures = 0
        
        return max_failures


class FeatureEngine:
    """Main feature engineering engine."""
    
    def __init__(self, config: FeatureConfig):
        """Initialize feature engine.
        
        Args:
            config: Feature configuration
        """
        self.config = config
        self.extractor = FeatureExtractor(config)
        self.windows: Dict[str, SlidingWindow] = {}
        self.logger = logger.bind(component="feature_engine")
        
        # Convert minutes to milliseconds
        window_size_ms = config.window_size_minutes * 60 * 1000
        overlap_ms = config.overlap_seconds * 1000
        
        self.window_size_ms = window_size_ms
        self.overlap_ms = overlap_ms
    
    def process_sample(self, sample: MetricSample) -> Optional[FeatureVector]:
        """Process a metric sample and extract features.
        
        Args:
            sample: Metric sample to process
            
        Returns:
            Feature vector if ready, None otherwise
        """
        endpoint_id = sample.endpoint_id
        
        # Get or create window for this endpoint
        if endpoint_id not in self.windows:
            self.windows[endpoint_id] = SlidingWindow(
                self.window_size_ms, 
                self.overlap_ms
            )
        
        window = self.windows[endpoint_id]
        window.add_sample(sample)
        
        # Extract features if window is ready
        if window.is_ready():
            features = self.extractor.extract_features(window, endpoint_id)
            
            if features:
                self.logger.debug(
                    "Features extracted",
                    endpoint_id=endpoint_id,
                    feature_count=len([v for v in features.dict().values() if v is not None]),
                )
            
            return features
        
        return None
    
    def get_window_stats(self) -> Dict[str, int]:
        """Get statistics about active windows.
        
        Returns:
            Dictionary with window statistics
        """
        stats = {}
        for endpoint_id, window in self.windows.items():
            stats[endpoint_id] = len(window.data)
        
        return stats
