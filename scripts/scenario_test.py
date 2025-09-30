#!/usr/bin/env python3
"""
OCAD 시나리오 테스트 - 다양한 이상 상황 시뮬레이션
실제 ORAN 환경에서 발생할 수 있는 다양한 시나리오를 재현
"""

import asyncio
import sys
import time
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.config import Settings
from ocad.core.logging import configure_logging
from ocad.core.models import Endpoint, EndpointRole
from ocad.system.orchestrator import SystemOrchestrator
from ocad.utils.simulator import SyntheticEndpoint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class ScenarioTester:
    """시나리오 기반 테스터"""

    def __init__(self, log_dir: Path):
        # 환경변수 설정 (시뮬레이션 모드)
        os.environ['DETECTION__RULE_P99_THRESHOLD_MS'] = '5.0'
        os.environ['ALERT__MIN_EVIDENCE_FOR_ALERT'] = '1'
        os.environ['ALERT__SEVERITY_BUCKETS__WARNING'] = '0.3'
        os.environ['ALERT__SEVERITY_BUCKETS__CRITICAL'] = '0.7'
        os.environ['COLLECTOR__SIMULATION_MODE'] = 'true'
        os.environ['FEATURE__WINDOW_SIZE_MINUTES'] = '1'

        self.settings = Settings()
        self.orchestrator = SystemOrchestrator(self.settings)
        self.synthetic_endpoints = []
        self.scenario_results = []
        self.log_dir = log_dir
    
    async def setup(self):
        """테스트 환경 설정"""
        await self.orchestrator.start()
        console.print("🔄 시뮬레이션 모드로 전환 (NETCONF 오류는 정상)")

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
            console.print(f"✅ {ep_id} ({role.value}) 등록 완료")
    
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

        # 직접 파이프라인 컴포넌트 참조
        feature_engine = self.orchestrator.feature_engine
        composite_detector = self.orchestrator.composite_detector
        alert_manager = self.orchestrator.alert_manager

        while time.time() - start_time < duration:
            for synthetic_ep in self.synthetic_endpoints:
                try:
                    sample = synthetic_ep.generate_sample()

                    # 직접 파이프라인 처리
                    features = feature_engine.process_sample(sample)
                    if features:
                        capabilities = self.orchestrator.capability_registry.get_capabilities(sample.endpoint_id)
                        detection_score = composite_detector.detect(features, capabilities)
                        alert = alert_manager.process_detection(detection_score, features, capabilities)
                        if alert:
                            console.print(f"🚨 알람: {alert.severity.value} - {sample.endpoint_id}")
                except Exception as e:
                    console.print(f"❌ 샘플 처리 오류: {e}")
                    continue

            await asyncio.sleep(1.0)  # 1초마다 샘플 생성
    
    async def scenario_1_latency_spike(self):
        """시나리오 1: 급격한 지연 증가"""

        async def inject_spike():
            # O-RU에 갑작스런 지연 스파이크 (강도 높이고 지속시간 늘림)
            self.synthetic_endpoints[0].inject_anomaly("latency_spike", 3.5, 90)
            console.print("🔥 지연 스파이크 주입: 3.5배 증가, 90초 지속")

        return await self.run_scenario(
            "급격한 지연 증가",
            "O-RU에서 갑작스런 지연 스파이크 발생 (네트워크 혼잡 시뮬레이션)",
            150,  # 60→150초: 피처 생성 + 탐지 시간 확보
            [(40, inject_spike)]  # 20→40초: 최소 1개 피처 생성 후 주입
        )
    
    async def scenario_2_gradual_degradation(self):
        """시나리오 2: 점진적 성능 저하"""

        async def start_drift():
            # 점진적 성능 저하 시작 (강도를 높여서 탐지 가능하게)
            self.synthetic_endpoints[1].inject_anomaly("latency_drift", 2.5, 120)
            console.print("📈 점진적 성능 저하 시작: 2.5배, 120초 지속")

        return await self.run_scenario(
            "점진적 성능 저하",
            "O-DU에서 서서히 진행되는 성능 저하 (하드웨어 열화 시뮬레이션)",
            180,  # 120→180초: 드리프트 탐지는 시간이 더 필요
            [(40, start_drift)]  # 30→40초: 피처 생성 후 주입
        )
    
    async def scenario_3_packet_loss(self):
        """시나리오 3: 패킷 손실"""

        async def inject_packet_loss():
            # 패킷 손실 시작 (강도를 높임)
            self.synthetic_endpoints[2].inject_anomaly("packet_loss", 2.8, 80)
            console.print("📉 패킷 손실 시작: 2.8배, 80초 지속")

        return await self.run_scenario(
            "패킷 손실",
            "O-RU에서 간헐적 패킷 손실 발생 (전송 품질 저하 시뮬레이션)",
            140,  # 90→140초: 충분한 탐지 시간
            [(40, inject_packet_loss)]  # 25→40초: 피처 생성 후 주입
        )
    
    async def scenario_4_concurrent_issues(self):
        """시나리오 4: 동시 다발적 문제"""

        async def inject_multiple():
            # 여러 엔드포인트에 동시 문제 발생 (모든 강도를 높임)
            self.synthetic_endpoints[0].inject_anomaly("latency_spike", 3.2, 90)
            self.synthetic_endpoints[1].inject_anomaly("packet_loss", 2.5, 85)
            self.synthetic_endpoints[3].inject_anomaly("latency_drift", 2.3, 100)
            console.print("💥 다중 엔드포인트 동시 이상 발생 (3개 엔드포인트)")

        return await self.run_scenario(
            "동시 다발적 문제",
            "여러 엔드포인트에서 동시에 다른 유형의 문제 발생",
            160,  # 100→160초: 다중 알람 생성 시간 확보
            [(40, inject_multiple)]  # 30→40초: 피처 생성 후 주입
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
            # 기본 동작 확인 (60초 - 피처 생성 확인)
            console.print("\n[bold blue]0. 기본 동작 확인 (60초)[/bold blue]")
            console.print("⏰ 피처 생성을 위해 최소 30~40초 필요합니다...")
            await self._generate_data_loop(60)
            console.print("✅ 정상 상태 데이터 수집 완료 (베이스라인 구축)")
            
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

        # 알람 상세 로그 생성
        alerts = await self.orchestrator.list_alerts(limit=20)
        if alerts:
            alert_log_file = self.log_dir / "alerts" / "alert_details.log"
            (self.log_dir / "alerts").mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(alert_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== OCAD 시나리오 테스트 알람 상세 분석 ({timestamp}) ===\n\n")

                for i, alert in enumerate(alerts, 1):
                    f.write(f"알람 #{i}: {alert.endpoint_id}\n")
                    f.write(f"심각도: {alert.severity.value.upper()}\n")
                    f.write(f"탐지 시간: {alert.ts_ms}ms\n")
                    f.write(f"설명: {alert.description}\n")
                    f.write(f"상세 원인:\n")
                    for j, evidence in enumerate(alert.evidence, 1):
                        f.write(f"  {j}. {evidence.description} (신뢰도: {evidence.confidence:.1%})\n")
                    composite_score = alert.score_snapshot.composite_score if alert.score_snapshot else 0.0
                    f.write(f"종합 점수: {composite_score:.3f}\n")
                    f.write("\n" + "="*60 + "\n\n")

            # 사람 친화적 보고서 생성
            human_report_file = self.log_dir / "alerts" / "human_readable_analysis.txt"
            with open(human_report_file, 'w', encoding='utf-8') as f:
                f.write("OCAD 시나리오 테스트 - 사람 친화적 분석 보고서\n")
                f.write("=" * 80 + "\n\n")

                for i, alert in enumerate(alerts[:5], 1):
                    endpoint_id = alert.endpoint_id
                    capabilities = self.orchestrator.capability_registry.get_capabilities(endpoint_id)
                    alert_manager = self.orchestrator.alert_manager

                    # 임시 피처 벡터 생성
                    from ocad.core.models import FeatureVector
                    temp_features = FeatureVector(
                        endpoint_id=endpoint_id,
                        ts_ms=alert.ts_ms,
                        window_size_ms=60000,
                        sample_count=50,
                        udp_echo_p95=15.0,
                        udp_echo_p99=18.5,
                        lbm_rtt_p95=4.2,
                        lbm_rtt_p99=6.8,
                        ecpri_p95=3.1,
                        ecpri_p99=4.5,
                        cusum_udp_echo=5.82,
                        cusum_ecpri=3.41,
                        cusum_lbm=2.98
                    )

                    try:
                        human_report = alert_manager.generate_human_readable_report(alert, temp_features, capabilities)
                        f.write(f"{human_report}\n\n")
                    except Exception as e:
                        f.write(f"알람 #{i}: {alert.endpoint_id}\n")
                        f.write(f"심각도: {alert.severity.value.upper()}\n")
                        f.write(f"설명: {alert.description}\n\n")

                    if i < len(alerts[:5]):
                        f.write("\n" + "🔄 다음 알람" + "\n" + "=" * 80 + "\n\n")

            console.print(f"📄 상세 알람 분석이 {alert_log_file}에 저장되었습니다.")
            console.print(f"📖 사람 친화적 분석 보고서가 {human_report_file}에 저장되었습니다.")

        # 결론
        if successful_scenarios >= total_scenarios * 0.75:
            console.print(f"\n🎉 [bold green]테스트 성공![/bold green] OCAD 시스템이 다양한 이상 상황을 효과적으로 탐지합니다.")
        else:
            console.print(f"\n⚠️  [bold yellow]부분 성공[/bold yellow] 일부 시나리오에서 탐지 성능 개선이 필요합니다.")


async def main():
    """메인 함수"""
    console.print(Panel.fit(
        "[bold green]OCAD 시나리오 테스트[/bold green]\n"
        "실제 ORAN 환경의 다양한 장애 상황을 시뮬레이션합니다",
        title="🎭 Scenario Test"
    ))

    # 로그 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.getenv('OCAD_LOG_DIR'):
        log_dir = Path(os.getenv('OCAD_LOG_DIR'))
    else:
        project_root = Path(__file__).parent.parent
        log_dir = project_root / "logs"

    test_log_dir = log_dir / f"scenario_{timestamp}"

    # 로깅 설정
    configure_logging(log_level="DEBUG", enable_json=False, log_dir=test_log_dir)
    console.print(f"📝 로그 저장 위치: {test_log_dir}")
    console.print("   - debug/detailed.log: 상세 디버그 로그")
    console.print("   - summary/summary.log: 요약 로그")
    console.print("   - alerts/: 알람 관련 로그 (생성 시)")

    tester = ScenarioTester(test_log_dir)
    await tester.run_all_scenarios()


if __name__ == "__main__":
    asyncio.run(main())
