"""OCAD 시스템을 합성 데이터로 테스트하기 위한 시뮬레이터.
실제 ORAN 환경을 모사하여 시스템 검증과 성능 테스트를 수행합니다.
"""

import asyncio
import random
import time
from typing import List

import numpy as np
from rich.console import Console
from rich.progress import Progress

from ..core.config import Settings
from ..core.models import Capabilities, Endpoint, EndpointRole, MetricSample
from ..system.orchestrator import SystemOrchestrator


console = Console()


class SyntheticEndpoint:
    """실제와 유사한 동작을 하는 합성 ORAN 엔드포인트를 시뮬레이션합니다.
    
    실제 O-RU/O-DU/Transport 장비의 성능 특성을 모사하여:
    - 정상적인 베이스라인 성능
    - 랜덤 노이즈 및 변동
    - 의도적인 이상 상황 주입 (지연 스파이크, 드리프트, 패킷 손실)
    """
    
    def __init__(self, endpoint_id: str, role: EndpointRole):
        """합성 엔드포인트를 초기화합니다.
        
        Args:
            endpoint_id: 엔드포인트 식별자
            role: 엔드포인트 역할 (O-RU, O-DU, Transport)
        """
        self.endpoint_id = endpoint_id
        self.role = role
        
        # 역할에 따른 현실적인 기능 생성
        self.capabilities = self._generate_capabilities(role)
        
        # 베이스라인 성능 (정상 상태)
        self.baseline_udp_rtt = random.uniform(2.0, 8.0)      # 2-8ms
        self.baseline_ecpri_delay = random.uniform(50.0, 200.0)  # 50-200μs
        self.baseline_lbm_rtt = random.uniform(1.5, 6.0)      # 1.5-6ms
        
        # 노이즈 파라미터
        self.noise_factor = random.uniform(0.1, 0.3)
        
        # 이상 상황 주입 상태
        self.anomaly_start_time = None
        self.anomaly_type = None
        self.anomaly_severity = 1.0
        
        # 드리프트 상태
        self.drift_factor = 1.0
        self.drift_rate = 0.0
        
    def _generate_capabilities(self, role: EndpointRole) -> Capabilities:
        """Generate realistic capabilities based on role.
        
        Args:
            role: Endpoint role
            
        Returns:
            Generated capabilities
        """
        if role == EndpointRole.O_RU:
            return Capabilities(
                udp_echo=True,
                ecpri_delay=True,  # Enable for testing
                lbm=True,  # Enable for testing
                ccm_min=random.choice([True, False]),
                lldp=True,
            )
        elif role == EndpointRole.O_DU:
            return Capabilities(
                udp_echo=True,
                ecpri_delay=True,
                lbm=True,
                ccm_min=random.choice([True, False]),
                lldp=True,
            )
        else:  # Transport
            return Capabilities(
                udp_echo=random.choice([True, False]),
                ecpri_delay=False,
                lbm=True,
                ccm_min=True,
                lldp=True,
            )
    
    def inject_anomaly(self, anomaly_type: str, severity: float = 1.0, duration: float = 120.0):
        """Inject an anomaly into the endpoint.
        
        Args:
            anomaly_type: Type of anomaly ('latency_spike', 'latency_drift', 'packet_loss')
            severity: Anomaly severity (1.0 = normal, >1.0 = worse)
            duration: Anomaly duration in seconds
        """
        current_time = time.time()
        self.anomaly_start_time = current_time
        self.anomaly_type = anomaly_type
        self.anomaly_severity = severity
        
        if anomaly_type == "latency_drift":
            self.drift_rate = (severity - 1.0) / duration  # Gradual increase
        
        console.print(f"[yellow]Injected {anomaly_type} anomaly into {self.endpoint_id} (severity: {severity:.1f})[/yellow]")
    
    def generate_sample(self) -> MetricSample:
        """Generate a synthetic metric sample.
        
        Returns:
            Synthetic metric sample
        """
        current_time = time.time()
        
        # Update drift
        if self.anomaly_type == "latency_drift" and self.anomaly_start_time:
            time_since_anomaly = current_time - self.anomaly_start_time
            self.drift_factor = 1.0 + (self.drift_rate * time_since_anomaly)
        
        # Calculate current multipliers
        latency_multiplier = self.drift_factor
        
        if self.anomaly_type == "latency_spike" and self.anomaly_start_time:
            time_since_anomaly = current_time - self.anomaly_start_time
            if time_since_anomaly < 60:  # Spike lasts 1 minute
                latency_multiplier *= self.anomaly_severity
        
        # Generate metrics with noise
        sample = MetricSample(
            endpoint_id=self.endpoint_id,
            ts_ms=int(current_time * 1000),
        )
        
        # UDP Echo RTT
        if self.capabilities.udp_echo:
            noise = random.gauss(1.0, self.noise_factor)
            sample.udp_echo_rtt_ms = self.baseline_udp_rtt * latency_multiplier * noise
        
        # eCPRI delay
        if self.capabilities.ecpri_delay:
            noise = random.gauss(1.0, self.noise_factor)
            sample.ecpri_ow_us = self.baseline_ecpri_delay * latency_multiplier * noise
        
        # LBM RTT and success
        if self.capabilities.lbm:
            noise = random.gauss(1.0, self.noise_factor)
            sample.lbm_rtt_ms = self.baseline_lbm_rtt * latency_multiplier * noise
            
            # LBM success rate (lower with packet loss anomaly)
            success_rate = 0.99
            if self.anomaly_type == "packet_loss" and self.anomaly_start_time:
                success_rate = max(0.5, 0.99 - (self.anomaly_severity - 1.0) * 0.4)
            
            sample.lbm_success = random.random() < success_rate
        
        # CCM stats
        if self.capabilities.ccm_min:
            # Inter-arrival time with some jitter
            baseline_inter_arrival = 1000.0  # 1 second nominal
            jitter = random.gauss(0, 50)  # 50ms jitter
            sample.ccm_inter_arrival_ms = baseline_inter_arrival + jitter
            
            # Miss run-length (higher with packet loss)
            if self.anomaly_type == "packet_loss":
                sample.ccm_runlen_miss = random.poisson(self.anomaly_severity)
            else:
                sample.ccm_runlen_miss = 0
        
        return sample


class EndpointSimulator:
    """Simulates multiple endpoints for testing."""
    
    def __init__(self, config: Settings):
        """Initialize simulator.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.orchestrator = SystemOrchestrator(config)
        self.synthetic_endpoints: List[SyntheticEndpoint] = []
        
    async def run_simulation(self, endpoint_count: int, duration_seconds: int) -> None:
        """Run simulation with multiple endpoints.
        
        Args:
            endpoint_count: Number of endpoints to simulate
            duration_seconds: Simulation duration
        """
        console.print("Starting orchestrator...")
        await self.orchestrator.start()
        
        try:
            # Create synthetic endpoints
            await self._create_endpoints(endpoint_count)
            
            # Run simulation
            await self._run_data_generation(duration_seconds)
            
        finally:
            console.print("Stopping orchestrator...")
            await self.orchestrator.stop()
    
    async def _create_endpoints(self, count: int) -> None:
        """Create synthetic endpoints.
        
        Args:
            count: Number of endpoints to create
        """
        console.print(f"Creating {count} synthetic endpoints...")
        
        roles = [EndpointRole.O_RU, EndpointRole.O_DU, EndpointRole.TRANSPORT]
        
        with Progress() as progress:
            task = progress.add_task("Creating endpoints...", total=count)
            
            for i in range(count):
                role = random.choice(roles)
                endpoint_id = f"sim-{role.value}-{i:03d}"
                
                # Create synthetic endpoint
                synthetic_ep = SyntheticEndpoint(endpoint_id, role)
                self.synthetic_endpoints.append(synthetic_ep)
                
                # Create real endpoint for orchestrator
                endpoint = Endpoint(
                    id=endpoint_id,
                    host=f"192.168.1.{100 + i}",
                    port=830,
                    role=role,
                )
                
                # Register with orchestrator (skip capability detection)
                self.orchestrator.capability_registry.register_endpoint(
                    endpoint, synthetic_ep.capabilities
                )
                self.orchestrator.endpoints[endpoint_id] = endpoint
                
                progress.advance(task)
        
        console.print(f"[green]✓[/green] Created {count} synthetic endpoints")
    
    async def _run_data_generation(self, duration_seconds: int) -> None:
        """Run data generation simulation.
        
        Args:
            duration_seconds: Simulation duration
        """
        console.print(f"Running simulation for {duration_seconds} seconds...")
        
        start_time = time.time()
        sample_interval = 5.0  # Generate sample every 5 seconds
        anomaly_injection_interval = 60.0  # Inject anomaly every minute
        
        last_sample_time = start_time
        last_anomaly_time = start_time
        
        with Progress() as progress:
            task = progress.add_task("Simulation progress...", total=duration_seconds)
            
            while time.time() - start_time < duration_seconds:
                current_time = time.time()
                
                # Generate samples
                if current_time - last_sample_time >= sample_interval:
                    await self._generate_samples()
                    last_sample_time = current_time
                
                # Inject random anomalies
                if current_time - last_anomaly_time >= anomaly_injection_interval:
                    await self._inject_random_anomaly()
                    last_anomaly_time = current_time
                
                # Update progress
                elapsed = current_time - start_time
                progress.update(task, completed=elapsed)
                
                await asyncio.sleep(1.0)
        
        console.print("[green]✓[/green] Simulation completed")
    
    async def _generate_samples(self) -> None:
        """Generate samples from all synthetic endpoints."""
        for synthetic_ep in self.synthetic_endpoints:
            sample = synthetic_ep.generate_sample()
            
            # Process sample through orchestrator
            await self.orchestrator._process_sample(sample)
    
    async def _inject_random_anomaly(self) -> None:
        """Inject a random anomaly into a random endpoint."""
        if not self.synthetic_endpoints:
            return
        
        # Select random endpoint
        endpoint = random.choice(self.synthetic_endpoints)
        
        # Select random anomaly type
        anomaly_types = ["latency_spike", "latency_drift", "packet_loss"]
        anomaly_type = random.choice(anomaly_types)
        
        # Random severity
        severity = random.uniform(1.5, 3.0)
        
        # Random duration
        duration = random.uniform(30.0, 180.0)
        
        endpoint.inject_anomaly(anomaly_type, severity, duration)
