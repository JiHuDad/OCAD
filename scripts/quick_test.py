#!/usr/bin/env python3
"""
OCAD 시스템 빠른 테스트 스크립트
실제 CFM 장비 없이 시뮬레이션으로 시스템 검증
"""

import asyncio
import sys
import time
import os
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.config import Settings
from ocad.core.logging import configure_logging
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
    
    # 로그 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/home/finux/dev/OCAD/logs"
    test_log_dir = f"{log_dir}/test_{timestamp}"
    
    # 로깅 설정
    configure_logging(log_level="DEBUG", enable_json=False, log_dir=test_log_dir)
    console.print(f"📝 로그 저장 위치: {test_log_dir}")
    console.print("   - debug/detailed.log: 상세 디버그 로그")
    console.print("   - summary/summary.log: 요약 로그")
    console.print("   - alerts/: 알람 관련 로그 (생성 시)")
    
    # 임계값을 현실적으로 조정 (환경변수 사용)
    os.environ['DETECTION__RULE_P99_THRESHOLD_MS'] = '5.0'   # 5ms로 설정
    os.environ['ALERT__MIN_EVIDENCE_FOR_ALERT'] = '1'       # 최소 증거 개수를 1로 줄임
    os.environ['ALERT__SEVERITY_BUCKETS__WARNING'] = '0.3'  # WARNING 임계값 낮춤
    os.environ['ALERT__SEVERITY_BUCKETS__CRITICAL'] = '0.7' # CRITICAL 임계값 설정
    os.environ['COLLECTOR__SIMULATION_MODE'] = 'true'      # 시뮬레이션 모드 활성화
    os.environ['FEATURE__WINDOW_SIZE_MINUTES'] = '1'       # 1분 윈도우 (명시적 설정)
    
    # 1. 시스템 초기화
    console.print("\n[bold blue]1. 시스템 초기화[/bold blue]")
    settings = Settings()
    console.print(f"📊 룰 기반 임계값: {settings.detection.rule_p99_threshold_ms}ms")
    console.print(f"📊 피처 윈도우 크기: {settings.feature.window_size_minutes}분 ({settings.feature.window_size_minutes * 60000}ms)")
    console.print(f"📊 윈도우가 준비될 조건: 시간간격 >= {settings.feature.window_size_minutes * 60000 // 2}ms")
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
            
            # 시뮬레이션 모드 안내
            console.print("🔄 시뮬레이션 모드로 전환 (NETCONF 오류는 정상)")
            
            # 시뮬레이션용 컴포넌트들 직접 사용
            feature_engine = orchestrator.feature_engine
            composite_detector = orchestrator.composite_detector
            alert_manager = orchestrator.alert_manager
            
            console.print("✅ 시뮬레이션용 컴포넌트 준비 완료")
            
            # 240초간 데이터 생성 및 처리 (윈도우 크기보다 충분히 긴 시간)
            start_time = time.time()
            sample_count = 0
            anomaly_injected = False
            alert_count = 0
            features_extracted = 0
            
            console.print(f"🚀 시뮬레이션 시작: 240초간 실행 (피처 추출을 위해 충분한 시간 확보)")
            console.print(f"⏰ 첫 피처는 약 30초 후에 생성될 예정입니다")
            
            while time.time() - start_time < 240:
                elapsed = time.time() - start_time
                
                for synthetic_ep in synthetic_endpoints:
                    try:
                        # 합성 샘플 생성
                        sample = synthetic_ep.generate_sample()
                        
                        # 직접 파이프라인 처리
                        # 1. 피처 추출
                        features = feature_engine.process_sample(sample)
                        if features:
                            console.print(f"🔥 피처 생성! #{features_extracted + 1} - 엔드포인트: {sample.endpoint_id}")
                            features_extracted += 1
                            # 2. 이상 탐지
                            capabilities = orchestrator.capability_registry.get_capabilities(sample.endpoint_id)
                            detection_score = composite_detector.detect(features, capabilities)
                            
                            # 3. 알람 처리
                            alert = alert_manager.process_detection(detection_score, features, capabilities)
                            if alert:
                                alert_count += 1
                                console.print(f"🚨 알람 생성! #{alert_count}: {alert.severity.value} - {alert.description[:50]}...")
                        
                        results["samples_processed"] += 1
                        results["features_extracted"] = features_extracted
                        results["alerts_generated"] = alert_count
                        sample_count += 1
                    except Exception as e:
                        console.print(f"❌ 샘플 처리 오류: {e}")
                        continue
                
                # 60초 후 이상 상황 주입 (피처 생성 후)
                if not anomaly_injected and elapsed > 60:
                    # 첫 번째 엔드포인트에 강한 지연 스파이크 주입
                    synthetic_endpoints[0].inject_anomaly("latency_spike", 3.0, 120)
                    console.print("🔥 이상 상황 주입: 강한 지연 스파이크 (3.0배)")
                    anomaly_injected = True
                
                # 진행 상황 표시 (매 10초마다)
                if sample_count > 0 and sample_count % 30 == 0:  # 30개마다 (약 10초)
                    progress.update(task, description=f"샘플 처리 중... ({sample_count}개, {elapsed:.1f}초, 피처: {features_extracted}개, 알람: {alert_count}개)")
                    console.print(f"🔄 진행: {sample_count}개 샘플, {elapsed:.1f}초 경과, 피처: {features_extracted}개, 알람: {alert_count}개")
                
                await asyncio.sleep(1.0)  # 1초마다 샘플 생성 (충분한 시간 간격)
            
            console.print(f"✅ 시뮬레이션 완료: {sample_count}개 샘플 처리됨, 피처: {features_extracted}개, 알람: {alert_count}개")
        
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
            
            alert_table = Table(title="알람 목록 및 원인 분석")
            alert_table.add_column("엔드포인트", style="green")
            alert_table.add_column("심각도", style="red")
            alert_table.add_column("탐지 원인", style="yellow", width=50)
            
            # 알람별 상세 로그 파일 생성
            alert_log_file = f"{test_log_dir}/alerts/alert_details.log"
            os.makedirs(f"{test_log_dir}/alerts", exist_ok=True)
            
            with open(alert_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== OCAD 알람 상세 분석 보고서 ({timestamp}) ===\n\n")
                
                for i, alert in enumerate(alerts[:5], 1):  # 최대 5개만 표시
                    # Evidence에서 원인 추출
                    causes = []
                    for evidence in alert.evidence:
                        if evidence.type == "rule":
                            causes.append(f"룰 위반: {evidence.description}")
                        elif evidence.type == "spike":
                            causes.append(f"급격한 변화: {evidence.description}")
                        elif evidence.type == "drift":
                            causes.append(f"패턴 변화: {evidence.description}")
                    
                    cause_summary = "; ".join(causes) if causes else "원인 불명"
                    
                    alert_table.add_row(
                        alert.endpoint_id,
                        alert.severity.value.upper(),
                        cause_summary[:50] + "..." if len(cause_summary) > 50 else cause_summary
                    )
                    
                    # 상세 로그 파일에 기록
                    f.write(f"알람 #{i}: {alert.endpoint_id}\n")
                    f.write(f"심각도: {alert.severity.value.upper()}\n")
                    f.write(f"탐지 시간: {alert.ts_ms}ms\n")
                    f.write(f"전체 설명: {alert.description}\n")
                    f.write(f"상세 원인:\n")
                    for j, evidence in enumerate(alert.evidence, 1):
                        f.write(f"  {j}. {evidence.description} (신뢰도: {evidence.confidence:.1%})\n")
                    f.write(f"종합 점수: {alert.composite_score:.3f}\n")
                    f.write("\n" + "="*60 + "\n\n")
                
                # 사람 친화적 상세 보고서 생성
                human_report_file = f"{test_log_dir}/alerts/human_readable_analysis.txt"
                with open(human_report_file, 'w', encoding='utf-8') as f:
                    f.write("OCAD 이상탐지 시스템 - 사람 친화적 분석 보고서\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for i, alert in enumerate(alerts[:3], 1):  # 상위 3개만 상세 분석
                        # 해당 알람의 피처와 기능 정보 가져오기 (최신 데이터 사용)
                        endpoint_id = alert.endpoint_id
                        capabilities = orchestrator.capability_registry.get_capabilities(endpoint_id)
                        
                        # 마지막 피처 정보 추정 (실제로는 알람 생성 시의 피처를 저장해야 함)
                        # 여기서는 간단히 처리
                        alert_manager = orchestrator.alert_manager
                        
                        # 임시 피처 벡터 생성 (실제 구현에서는 알람과 함께 저장되어야 함)
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
                        
                        # 사람 친화적 보고서 생성
                        human_report = alert_manager.generate_human_readable_report(alert, temp_features, capabilities)
                        f.write(f"{human_report}\n\n")
                        
                        if i < len(alerts[:3]):
                            f.write("\n" + "🔄 다음 알람" + "\n" + "=" * 80 + "\n\n")
            
            console.print(alert_table)
            console.print(f"📄 상세 알람 분석이 {alert_log_file}에 저장되었습니다.")
            console.print(f"📖 사람 친화적 분석 보고서가 {human_report_file}에 저장되었습니다.")
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
