#!/usr/bin/env python3
"""
OCAD 대시보드 테스트 - API 서버와 함께 실시간 모니터링
웹 브라우저에서 실시간으로 시스템 동작을 확인할 수 있습니다
"""

import asyncio
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.config import Settings
from ocad.core.models import Endpoint, EndpointRole
from ocad.system.orchestrator import SystemOrchestrator
from ocad.utils.simulator import SyntheticEndpoint
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

console = Console()


class DashboardTester:
    """대시보드 기반 테스터"""
    
    def __init__(self):
        self.settings = Settings()
        self.orchestrator = SystemOrchestrator(self.settings)
        self.synthetic_endpoints = []
        self.running = False
    
    async def setup_endpoints(self):
        """엔드포인트 설정"""
        endpoints_config = [
            ("demo-ru-001", EndpointRole.O_RU, "192.168.100.10"),
            ("demo-du-001", EndpointRole.O_DU, "192.168.100.20"),
            ("demo-ru-002", EndpointRole.O_RU, "192.168.100.11"),
            ("demo-transport", EndpointRole.TRANSPORT, "192.168.100.1"),
        ]
        
        for ep_id, role, host in endpoints_config:
            synthetic_ep = SyntheticEndpoint(ep_id, role)
            self.synthetic_endpoints.append(synthetic_ep)
            
            endpoint = Endpoint(id=ep_id, host=host, port=830, role=role)
            self.orchestrator.capability_registry.register_endpoint(
                endpoint, synthetic_ep.capabilities
            )
            self.orchestrator.endpoints[ep_id] = endpoint
    
    def create_live_table(self):
        """실시간 테이블 생성"""
        table = Table(title="🔴 LIVE - OCAD 시스템 모니터링")
        table.add_column("메트릭", style="cyan", no_wrap=True)
        table.add_column("현재 값", style="green")
        table.add_column("상태", style="yellow")
        
        return table
    
    async def update_display_data(self, table):
        """디스플레이 데이터 업데이트"""
        try:
            stats = await self.orchestrator.get_statistics()
            alerts = await self.orchestrator.list_alerts(limit=5)
            
            table.rows.clear()
            
            # 기본 통계
            table.add_row("🔄 처리된 샘플", f"{stats['samples_processed']:,}", "정상" if stats['samples_processed'] > 0 else "대기")
            table.add_row("📊 추출된 피처", f"{stats['features_extracted']:,}", "정상" if stats['features_extracted'] > 0 else "대기")
            table.add_row("🚨 생성된 알람", f"{stats['alerts_generated']:,}", "⚠️ 주의" if stats['alerts_generated'] > 0 else "정상")
            table.add_row("🌐 활성 엔드포인트", f"{stats['endpoints']}", "연결됨")
            table.add_row("📈 기능 커버리지", f"{stats['capability_coverage']:.1f}%", "양호" if stats['capability_coverage'] > 80 else "개선필요")
            
            if stats.get("processing_latency_p95"):
                latency_status = "빠름" if stats['processing_latency_p95'] < 0.1 else "보통"
                table.add_row("⚡ 처리 지연 P95", f"{stats['processing_latency_p95']:.3f}s", latency_status)
            
            # 최근 알람
            if alerts:
                table.add_row("", "", "")  # 구분선
                table.add_row("🚨 최근 알람", "", "")
                for i, alert in enumerate(alerts[:3]):
                    severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert.severity, "⚪")
                    table.add_row(
                        f"  {severity_icon} {alert.endpoint_id}",
                        alert.severity.upper(),
                        "진행중" if not alert.resolved else "해결됨"
                    )
            
        except Exception as e:
            table.add_row("❌ 오류", str(e), "시스템 점검 필요")
    
    async def run_continuous_simulation(self):
        """지속적인 시뮬레이션 실행"""
        anomaly_injected = False
        start_time = time.time()
        
        while self.running:
            # 정기적으로 데이터 생성
            for synthetic_ep in self.synthetic_endpoints:
                sample = synthetic_ep.generate_sample()
                await self.orchestrator._process_sample(sample)
            
            # 60초 후 이상 상황 주입
            elapsed = time.time() - start_time
            if elapsed > 60 and not anomaly_injected:
                # 다양한 이상 상황 주입
                self.synthetic_endpoints[0].inject_anomaly("latency_spike", 2.8, 45)
                self.synthetic_endpoints[2].inject_anomaly("packet_loss", 1.6, 60)
                anomaly_injected = True
                console.print("\n🔥 [red]이상 상황 주입됨![/red] 대시보드에서 확인하세요")
            
            # 180초 후 추가 이상 상황
            if elapsed > 180 and anomaly_injected:
                self.synthetic_endpoints[1].inject_anomaly("latency_drift", 2.2, 120)
                console.print("\n📈 [yellow]점진적 성능 저하 시작![/yellow]")
                anomaly_injected = False  # 재설정
            
            await asyncio.sleep(3)  # 3초마다 데이터 생성
    
    async def run_dashboard_test(self):
        """대시보드 테스트 실행"""
        console.print(Panel.fit(
            "[bold green]OCAD 대시보드 테스트[/bold green]\n"
            "실시간 모니터링과 함께 시스템 동작을 확인합니다.\n"
            "[cyan]Ctrl+C로 중지[/cyan]",
            title="📊 Dashboard Test"
        ))
        
        # 시스템 시작
        await self.orchestrator.start()
        await self.setup_endpoints()
        
        console.print("\n✅ 시스템 시작 완료")
        console.print("🌐 API 서버: http://localhost:8080")
        console.print("📊 상태 확인: http://localhost:8080/stats")
        console.print("🚨 알람 목록: http://localhost:8080/alerts")
        
        # 실시간 테이블로 모니터링
        table = self.create_live_table()
        self.running = True
        
        try:
            with Live(table, refresh_per_second=2) as live:
                # 백그라운드에서 시뮬레이션 실행
                sim_task = asyncio.create_task(self.run_continuous_simulation())
                
                # 메인 루프에서 테이블 업데이트
                while self.running:
                    await self.update_display_data(table)
                    await asyncio.sleep(2)  # 2초마다 업데이트
                    
        except KeyboardInterrupt:
            console.print("\n\n👋 테스트 중지됨")
        
        finally:
            self.running = False
            await self.orchestrator.stop()
            console.print("🔚 시스템 종료 완료")


async def main():
    """메인 함수"""
    tester = DashboardTester()
    await tester.run_dashboard_test()


if __name__ == "__main__":
    asyncio.run(main())
