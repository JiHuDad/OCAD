"""Unit tests for feature engine."""

import time

import pytest

from ocad.core.config import FeatureConfig
from ocad.core.models import MetricSample
from ocad.features.engine import FeatureEngine, FeatureExtractor, SlidingWindow


@pytest.fixture
def feature_config():
    """Create test feature configuration."""
    return FeatureConfig(
        window_size_minutes=1,
        overlap_seconds=30,
        percentiles=[95, 99],
        ewma_alpha=0.3,
        cusum_threshold=5.0,
    )


@pytest.fixture
def feature_extractor(feature_config):
    """Create feature extractor instance."""
    return FeatureExtractor(feature_config)


class TestSlidingWindow:
    """Test sliding window functionality."""
    
    def test_window_initialization(self):
        """Test window initialization."""
        window = SlidingWindow(window_size_ms=60000, overlap_ms=30000)
        
        assert window.window_size_ms == 60000
        assert window.overlap_ms == 30000
        assert len(window.data) == 0
        assert not window.is_ready()
    
    def test_add_sample(self):
        """Test adding samples to window."""
        window = SlidingWindow(window_size_ms=10000)
        
        current_time_ms = int(time.time() * 1000)
        
        # Add samples
        for i in range(5):
            sample = MetricSample(
                endpoint_id="test",
                ts_ms=current_time_ms + i * 1000,
                udp_echo_rtt_ms=10.0 + i,
            )
            window.add_sample(sample)
        
        assert len(window.data) == 5
        assert window.is_ready()
    
    def test_window_expiration(self):
        """Test that old samples are removed."""
        window = SlidingWindow(window_size_ms=5000)  # 5 second window
        
        current_time_ms = int(time.time() * 1000)
        
        # Add old samples
        for i in range(3):
            sample = MetricSample(
                endpoint_id="test",
                ts_ms=current_time_ms - 10000 + i * 1000,  # 10 seconds ago
                udp_echo_rtt_ms=10.0,
            )
            window.add_sample(sample)
        
        # Add recent sample
        recent_sample = MetricSample(
            endpoint_id="test",
            ts_ms=current_time_ms,
            udp_echo_rtt_ms=15.0,
        )
        window.add_sample(recent_sample)
        
        # Old samples should be removed
        assert len(window.data) == 1
        assert window.data[0] == recent_sample


class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    def test_extract_metric_series(self, feature_extractor):
        """Test metric series extraction."""
        samples = [
            MetricSample(endpoint_id="test", ts_ms=1000, udp_echo_rtt_ms=10.0),
            MetricSample(endpoint_id="test", ts_ms=2000, udp_echo_rtt_ms=15.0),
            MetricSample(endpoint_id="test", ts_ms=3000, udp_echo_rtt_ms=12.0),
        ]
        
        series = feature_extractor._extract_metric_series(samples, "udp_echo_rtt_ms")
        
        assert series == [10.0, 15.0, 12.0]
    
    def test_extract_boolean_series(self, feature_extractor):
        """Test boolean series extraction."""
        samples = [
            MetricSample(endpoint_id="test", ts_ms=1000, lbm_success=True),
            MetricSample(endpoint_id="test", ts_ms=2000, lbm_success=False),
            MetricSample(endpoint_id="test", ts_ms=3000, lbm_success=True),
        ]
        
        series = feature_extractor._extract_boolean_series(samples, "lbm_success")
        
        assert series == [True, False, True]
    
    def test_calculate_percentile(self, feature_extractor):
        """Test percentile calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        p50 = feature_extractor._calculate_percentile(values, 50)
        p95 = feature_extractor._calculate_percentile(values, 95)
        
        assert abs(p50 - 5.5) < 0.1  # Median
        assert abs(p95 - 9.55) < 0.1  # 95th percentile
    
    def test_calculate_slope(self, feature_extractor):
        """Test slope calculation."""
        # Increasing values
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        slope = feature_extractor._calculate_slope(increasing_values)
        assert slope > 0
        
        # Decreasing values
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        slope = feature_extractor._calculate_slope(decreasing_values)
        assert slope < 0
        
        # Flat values
        flat_values = [3.0, 3.0, 3.0, 3.0, 3.0]
        slope = feature_extractor._calculate_slope(flat_values)
        assert abs(slope) < 0.1
    
    def test_calculate_ewma(self, feature_extractor):
        """Test EWMA calculation."""
        values = [10.0, 12.0, 11.0, 13.0, 12.5]
        
        ewma = feature_extractor._calculate_ewma(values)
        
        assert ewma is not None
        assert 10.0 <= ewma <= 13.0  # Should be within range
    
    def test_calculate_cusum(self, feature_extractor):
        """Test CUSUM calculation."""
        # Stable values
        stable_values = [10.0] * 10
        cusum = feature_extractor._calculate_cusum(stable_values)
        assert cusum is not None
        assert cusum < 1.0  # Should be low for stable values
        
        # Values with change point
        changing_values = [10.0] * 5 + [20.0] * 5
        cusum = feature_extractor._calculate_cusum(changing_values)
        assert cusum is not None
        assert cusum > 1.0  # Should be high for changing values
    
    def test_calculate_failure_runlength(self, feature_extractor):
        """Test failure run-length calculation."""
        # No failures
        all_success = [True, True, True, True, True]
        runlen = feature_extractor._calculate_failure_runlength(all_success)
        assert runlen == 0
        
        # Some failures
        mixed_success = [True, False, False, False, True, False, True]
        runlen = feature_extractor._calculate_failure_runlength(mixed_success)
        assert runlen == 3  # Maximum consecutive failures
        
        # All failures
        all_failures = [False, False, False, False]
        runlen = feature_extractor._calculate_failure_runlength(all_failures)
        assert runlen == 4


class TestFeatureEngine:
    """Test feature engine functionality."""
    
    def test_process_sample_insufficient_data(self, feature_config):
        """Test processing with insufficient data."""
        engine = FeatureEngine(feature_config)
        
        sample = MetricSample(
            endpoint_id="test",
            ts_ms=int(time.time() * 1000),
            udp_echo_rtt_ms=10.0,
        )
        
        features = engine.process_sample(sample)
        assert features is None  # Not enough data yet
    
    def test_process_sample_with_sufficient_data(self, feature_config):
        """Test processing with sufficient data."""
        engine = FeatureEngine(feature_config)
        
        current_time_ms = int(time.time() * 1000)
        
        # Add multiple samples to build up window
        for i in range(20):
            sample = MetricSample(
                endpoint_id="test",
                ts_ms=current_time_ms + i * 3000,  # 3 second intervals
                udp_echo_rtt_ms=10.0 + i * 0.5,
            )
            features = engine.process_sample(sample)
        
        # Should have features after enough samples
        assert features is not None
        assert features.endpoint_id == "test"
        assert features.udp_echo_p95 is not None
        assert features.udp_echo_p99 is not None
    
    def test_get_window_stats(self, feature_config):
        """Test window statistics."""
        engine = FeatureEngine(feature_config)
        
        # Add samples for multiple endpoints
        current_time_ms = int(time.time() * 1000)
        
        for endpoint_id in ["ep1", "ep2"]:
            for i in range(5):
                sample = MetricSample(
                    endpoint_id=endpoint_id,
                    ts_ms=current_time_ms + i * 1000,
                    udp_echo_rtt_ms=10.0,
                )
                engine.process_sample(sample)
        
        stats = engine.get_window_stats()
        
        assert "ep1" in stats
        assert "ep2" in stats
        assert stats["ep1"] == 5
        assert stats["ep2"] == 5
