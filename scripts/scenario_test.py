#!/usr/bin/env python3
"""
OCAD 시나리오 테스트 - 다양한 이상 상황 시뮬레이션
실제 ORAN 환경에서 발생할 수 있는 다양한 시나리오를 재현
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.config import Settings
from ocad.core.models import Endpoint, EndpointRole
from ocad.system.orchestrator import SystemOrchestrator
from ocad.utils.simulator import SyntheticEndpoint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class ScenarioTester:
    """시나리오 기반 테스터"""
    
    def __init__(self):
        self.settings = Settings()
        self.orchestrator = SystemOrchestrator(self.settings)
        self.synthetic_endpoints = []
        self.scenario_results = []
    
    async def setup(self):
        """테스트 환경 설정"""
        await self.orchestrator.start()
        
        # 다양한 역할의 엔드포인트 생성
        endpoints_config = [
            ("cell-01-ru", EndpointRole.O_RU, "10.1.1.101"),
            ("cell-01-du", EndpointRole.O_DU, "10.1.1.102"), 
            ("cell-02-ru", EndpointRole.O_RU, "10.1.2.101"),
            ("cell-02-du", EndpointRole.O_DU, "10.1.2.102"),
            ("transport-sw1", EndpointRole.TRANSPORT, "10.1.0.1"),
        ]
        
        for ep_id, role, host in endpoints_config:
            synthetic_ep = SyntheticEndpoint(ep_id, role)
            self.synthetic_endpoints.append(synthetic_ep)
            
            endpoint = Endpoint(id=ep_id, host=host, port=830, role=role)
            self.orchestrator.capability_registry.register_endpoint(
                endpoint, synthetic_ep.capabilities
            )
            self.orchestrator.endpoints[ep_id] = endpoint
    
    async def run_scenario(self, name: str, description: str, duration: int, actions: list):
        """시나리오 실행"""
        console.print(f"\n[bold blue]🎬 시나리오: {name}[/bold blue]")
        console.print(f"[italic]{description}[/italic]")
        
        start_time = time.time()
        initial_alerts = len(await self.orchestrator.list_alerts())
        
        # 액션 스케줄 실행
        action_tasks = []
        for delay, action_func in actions:
            task = asyncio.create_task(self._schedule_action(delay, action_func))
            action_tasks.append(task)
        
        # 데이터 생성 루프
        data_task = asyncio.create_task(self._generate_data_loop(duration))
        
        # 모든 태스크 완료 대기
        await asyncio.gather(data_task, *action_tasks)
        
        # 결과 수집
        final_alerts = len(await self.orchestrator.list_alerts())
        alerts_generated = final_alerts - initial_alerts
        
        result = {
            "name": name,
            "duration": duration,
            "alerts_generated": alerts_generated,
            "success": alerts_generated > 0  # 이상이 감지되었는지 여부
        }
        
        self.scenario_results.append(result)
        
        if result["success"]:
            console.print(f"✅ 시나리오 완료: {alerts_generated}개 알람 생성")
        else:
            console.print("⚠️  시나리오 완료: 알람 미생성")
        
        return result
    
    async def _schedule_action(self, delay: int, action_func):
        """지연 후 액션 실행"""
        await asyncio.sleep(delay)
        await action_func()
    
    async def _generate_data_loop(self, duration: int):
        """데이터 생성 루프"""
        start_time = time.time()
        while time.time() - start_time < duration:
            for synthetic_ep in self.synthetic_endpoints:
                sample = synthetic_ep.generate_sample()
                await self.orchestrator._process_sample(sample)
            await asyncio.sleep(2)  # 2초마다 샘플 생성
    
    async def scenario_1_latency_spike(self):
        """시나리오 1: 급격한 지연 증가"""
        
        async def inject_spike():
            # O-RU에 갑작스런 지연 스파이크
            self.synthetic_endpoints[0].inject_anomaly("latency_spike", 3.0, 30)
            console.print("🔥 지연 스파이크 주입: 3배 증가")
        
        return await self.run_scenario(
            "급격한 지연 증가",
            "O-RU에서 갑작스런 지연 스파이크 발생 (네트워크 혼잡 시뮬레이션)",
            60,
            [(20, inject_spike)]
        )
    
    async def scenario_2_gradual_degradation(self):
        """시나리오 2: 점진적 성능 저하"""
        
        async def start_drift():
            # 점진적 성능 저하 시작
            self.synthetic_endpoints[1].inject_anomaly("latency_drift", 2.0, 90)
            console.print("📈 점진적 성능 저하 시작")
        
        return await self.run_scenario(
            "점진적 성능 저하", 
            "O-DU에서 서서히 진행되는 성능 저하 (하드웨어 열화 시뮬레이션)",
            120,
            [(30, start_drift)]
        )
    
    async def scenario_3_packet_loss(self):
        """시나리오 3: 패킷 손실"""
        
        async def inject_packet_loss():
            # 패킷 손실 시작
            self.synthetic_endpoints[2].inject_anomaly("packet_loss", 1.8, 45)
            console.print("📉 패킷 손실 시작")
        
        return await self.run_scenario(
            "패킷 손실",
            "O-RU에서 간헐적 패킷 손실 발생 (전송 품질 저하 시뮬레이션)",
            90,
            [(25, inject_packet_loss)]
        )
    
    async def scenario_4_concurrent_issues(self):
        """시나리오 4: 동시 다발적 문제"""
        
        async def inject_multiple():
            # 여러 엔드포인트에 동시 문제 발생
            self.synthetic_endpoints[0].inject_anomaly("latency_spike", 2.5, 40)
            self.synthetic_endpoints[1].inject_anomaly("packet_loss", 1.7, 35)
            self.synthetic_endpoints[3].inject_anomaly("latency_drift", 1.8, 60)
            console.print("💥 다중 엔드포인트 동시 이상 발생")
        
        return await self.run_scenario(
            "동시 다발적 문제",
            "여러 엔드포인트에서 동시에 다른 유형의 문제 발생",
            100,
            [(30, inject_multiple)]
        )
    
    async def run_all_scenarios(self):
        """모든 시나리오 실행"""
        console.print(Panel.fit(
            "[bold green]OCAD 시나리오 테스트[/bold green]\n"
            "실제 ORAN 환경의 다양한 장애 상황을 시뮬레이션합니다",
            title="🎭 Scenario Test"
        ))
        
        await self.setup()
        
        try:
            # 기본 동작 확인 (30초)
            console.print("\n[bold blue]0. 기본 동작 확인[/bold blue]")
            await self._generate_data_loop(30)
            console.print("✅ 정상 상태 데이터 수집 완료")
            
            # 각 시나리오 실행
            await self.scenario_1_latency_spike()
            await asyncio.sleep(10)  # 시나리오 간 간격
            
            await self.scenario_2_gradual_degradation()
            await asyncio.sleep(10)
            
            await self.scenario_3_packet_loss()
            await asyncio.sleep(10)
            
            await self.scenario_4_concurrent_issues()
            
            # 최종 결과 요약
            await self.show_final_results()
            
        finally:
            await self.orchestrator.stop()
    
    async def show_final_results(self):
        """최종 결과 표시"""
        console.print("\n[bold blue]📊 최종 시나리오 테스트 결과[/bold blue]")
        
        # 시나리오 결과 테이블
        table = Table(title="시나리오 테스트 결과")
        table.add_column("시나리오", style="cyan")
        table.add_column("소요시간", style="green")
        table.add_column("생성된 알람", style="yellow")
        table.add_column("탐지 성공", style="red")
        
        total_scenarios = len(self.scenario_results)
        successful_scenarios = 0
        
        for result in self.scenario_results:
            success_icon = "✅" if result["success"] else "❌"
            if result["success"]:
                successful_scenarios += 1
            
            table.add_row(
                result["name"],
                f"{result['duration']}초",
                str(result["alerts_generated"]),
                success_icon
            )
        
        console.print(table)
        
        # 전체 시스템 통계
        stats = await self.orchestrator.get_statistics()
        
        console.print(f"\n[bold green]시스템 성능 요약[/bold green]")
        console.print(f"• 처리된 샘플: {stats['samples_processed']:,}개")
        console.print(f"• 추출된 피처: {stats['features_extracted']:,}개")
        console.print(f"• 총 생성 알람: {stats['alerts_generated']:,}개")
        console.print(f"• 탐지 성공률: {successful_scenarios}/{total_scenarios} ({successful_scenarios/total_scenarios*100:.1f}%)")
        
        if stats.get("processing_latency_p95"):
            console.print(f"• 처리 지연 P95: {stats['processing_latency_p95']:.3f}초")
        
        # 결론
        if successful_scenarios >= total_scenarios * 0.75:
            console.print(f"\n🎉 [bold green]테스트 성공![/bold green] OCAD 시스템이 다양한 이상 상황을 효과적으로 탐지합니다.")
        else:
            console.print(f"\n⚠️  [bold yellow]부분 성공[/bold yellow] 일부 시나리오에서 탐지 성능 개선이 필요합니다.")


async def main():
    """메인 함수"""
    tester = ScenarioTester()
    await tester.run_all_scenarios()


if __name__ == "__main__":
    asyncio.run(main())
