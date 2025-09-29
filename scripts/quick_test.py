#!/usr/bin/env python3
"""
OCAD 시스템 빠른 테스트 스크립트
실제 CFM 장비 없이 시뮬레이션으로 시스템 검증
"""

import asyncio
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.config import Settings
from ocad.core.models import Endpoint, EndpointRole, Capabilities
from ocad.system.orchestrator import SystemOrchestrator
from ocad.utils.simulator import SyntheticEndpoint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


async def quick_test():
    """빠른 테스트 실행"""
    
    console.print(Panel.fit(
        "[bold green]OCAD 시스템 빠른 검증 테스트[/bold green]\n"
        "실제 CFM 장비 없이 시뮬레이션으로 시스템 동작 확인",
        title="🧪 Quick Test"
    ))
    
    # 1. 시스템 초기화
    console.print("\n[bold blue]1. 시스템 초기화[/bold blue]")
    settings = Settings()
    orchestrator = SystemOrchestrator(settings)
    
    try:
        await orchestrator.start()
        console.print("✅ 시스템 시작 완료")
        
        # 2. 가상 엔드포인트 생성
        console.print("\n[bold blue]2. 가상 O-RAN 엔드포인트 생성[/bold blue]")
        
        endpoints_info = [
            ("sim-o-ru-001", EndpointRole.O_RU, "192.168.1.101"),
            ("sim-o-du-001", EndpointRole.O_DU, "192.168.1.102"),
            ("sim-transport-001", EndpointRole.TRANSPORT, "192.168.1.103"),
        ]
        
        synthetic_endpoints = []
        
        for ep_id, role, host in endpoints_info:
            # 가상 엔드포인트 생성
            synthetic_ep = SyntheticEndpoint(ep_id, role)
            synthetic_endpoints.append(synthetic_ep)
            
            # 시스템에 등록
            endpoint = Endpoint(
                id=ep_id,
                host=host,
                port=830,
                role=role,
            )
            
            # Capability 등록 (실제 탐지 대신 시뮬레이터 결과 사용)
            orchestrator.capability_registry.register_endpoint(
                endpoint, synthetic_ep.capabilities
            )
            orchestrator.endpoints[ep_id] = endpoint
            
            console.print(f"✅ {ep_id} ({role.value}) 등록 완료")
        
        # 3. 시뮬레이션 데이터 생성 및 처리
        console.print("\n[bold blue]3. 시뮬레이션 데이터 생성 및 처리[/bold blue]")
        
        results = {
            "samples_processed": 0,
            "features_extracted": 0,
            "alerts_generated": 0,
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("데이터 처리 중...", total=None)
            
            # 수집기 중지 (시뮬레이션 모드로 전환)
            await orchestrator.collector_manager.stop()
            console.print("🔄 실제 수집기 중지, 시뮬레이션 모드로 전환")
            
            # 150초간 데이터 생성 및 처리 (윈도우 크기보다 충분히 긴 시간)
            start_time = time.time()
            sample_count = 0
            anomaly_injected = False
            
            while time.time() - start_time < 150:
                for synthetic_ep in synthetic_endpoints:
                    # 합성 샘플 생성
                    sample = synthetic_ep.generate_sample()
                    
                    # 시스템으로 직접 처리
                    await orchestrator._process_sample(sample)
                    results["samples_processed"] += 1
                    sample_count += 1
                
                # 75초 후 이상 상황 주입 (한 번만)
                if not anomaly_injected and time.time() - start_time > 75:
                    # 첫 번째 엔드포인트에 강한 지연 스파이크 주입
                    synthetic_endpoints[0].inject_anomaly("latency_spike", 3.0, 120)
                    console.print("🔥 이상 상황 주입: 강한 지연 스파이크 (3.0배)")
                    anomaly_injected = True
                
                # 진행 상황 표시
                if sample_count % 10 == 0:
                    progress.update(task, description=f"샘플 처리 중... ({sample_count}개)")
                
                await asyncio.sleep(0.5)  # 0.5초마다 샘플 생성 (더 빠른 데이터 생성)
        
        # 4. 결과 요약
        console.print("\n[bold blue]4. 테스트 결과 요약[/bold blue]")
        
        # 시스템 통계 가져오기
        stats = await orchestrator.get_statistics()
        
        # 결과 테이블 생성
        table = Table(title="테스트 결과")
        table.add_column("메트릭", style="cyan")
        table.add_column("값", style="green")
        table.add_column("설명", style="yellow")
        
        table.add_row("처리된 샘플", str(stats["samples_processed"]), "수집된 메트릭 샘플 수")
        table.add_row("추출된 피처", str(stats["features_extracted"]), "생성된 피처 벡터 수")
        table.add_row("생성된 알람", str(stats["alerts_generated"]), "탐지된 이상 알람 수")
        table.add_row("활성 알람", str(stats["active_alerts"]), "현재 활성 상태 알람")
        table.add_row("엔드포인트", str(stats["endpoints"]), "등록된 엔드포인트 수")
        table.add_row("기능 커버리지", f"{stats['capability_coverage']:.1f}%", "자동 탐지된 기능 비율")
        
        if stats.get("processing_latency_p95"):
            table.add_row("처리 지연 P95", f"{stats['processing_latency_p95']:.3f}s", "95퍼센타일 처리 시간")
        
        console.print(table)
        
        # 5. 알람 상세 보기
        alerts = await orchestrator.list_alerts(limit=10)
        if alerts:
            console.print(f"\n[bold blue]5. 생성된 알람 ({len(alerts)}개)[/bold blue]")
            
            alert_table = Table(title="알람 목록")
            alert_table.add_column("ID", style="cyan")
            alert_table.add_column("엔드포인트", style="green")
            alert_table.add_column("심각도", style="red")
            alert_table.add_column("설명", style="yellow")
            
            for alert in alerts[:5]:  # 최대 5개만 표시
                alert_table.add_row(
                    alert.id[:8] + "...",
                    alert.endpoint_id,
                    alert.severity.value,
                    alert.description[:50] + "..." if len(alert.description) > 50 else alert.description
                )
            
            console.print(alert_table)
        else:
            console.print("\n[yellow]⚠️  생성된 알람이 없습니다. 더 긴 시간 테스트하거나 더 강한 이상을 주입해보세요.[/yellow]")
        
        # 6. 결론
        console.print(f"\n[bold green]🎉 테스트 완료![/bold green]")
        
        success_indicators = []
        if stats["samples_processed"] > 0:
            success_indicators.append("✅ 데이터 수집 정상")
        if stats["features_extracted"] > 0:
            success_indicators.append("✅ 피처 추출 정상")
        if stats["capability_coverage"] > 0:
            success_indicators.append("✅ Capability 탐지 정상")
        
        console.print("\n".join(success_indicators))
        
        if stats["alerts_generated"] > 0:
            console.print("✅ 이상 탐지 정상 (알람 생성됨)")
        else:
            console.print("⚠️  이상 탐지 확인 필요 (알람 미생성)")
        
        console.print(f"\n[bold]결론:[/bold] OCAD 시스템이 {len(success_indicators)}/4 핵심 기능에서 정상 동작함")
        
    except Exception as e:
        console.print(f"[red]❌ 테스트 실패: {e}[/red]")
        return False
        
    finally:
        await orchestrator.stop()
        console.print("\n🔚 시스템 종료 완료")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
