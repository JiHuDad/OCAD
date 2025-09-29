"""OCAD 시스템의 메인 진입점.
시스템을 독립적으로 실행할 때 사용하는 백그라운드 서비스 모드입니다.
"""

import asyncio
import signal
import sys
from pathlib import Path

from .core.config import Settings, load_config
from .core.logging import configure_logging, get_logger
from .system.orchestrator import SystemOrchestrator


async def main():
    """메인 비동기 진입점."""
    # 설정 파일 로드
    config_path = Path("config/local.yaml")
    if config_path.exists():
        settings = load_config(config_path)
    else:
        settings = Settings()
    
    # 로깅 시스템 구성
    configure_logging(settings.monitoring.log_level)
    logger = get_logger(__name__)
    
    logger.info("ORAN CFM-Lite AI 이상탐지 시스템 시작")
    
    # 시스템 오케스트레이터 생성
    orchestrator = SystemOrchestrator(settings)
    
    # 우아한 종료를 위한 시그널 핸들러 설정
    def signal_handler(signum, frame):
        logger.info(f"시그널 {signum} 수신, 시스템 종료 중...")
        asyncio.create_task(orchestrator.stop())
    
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    try:
        # 시스템 시작
        await orchestrator.start()
        
        logger.info("OCAD 시스템이 성공적으로 시작되었습니다")
        logger.info("API 엔드포인트나 CLI 명령을 사용하여 시스템과 상호작용하세요")
        
        # 중단될 때까지 계속 실행
        while orchestrator.is_running:
            await asyncio.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("키보드 인터럽트 수신")
    
    except Exception as e:
        logger.error("시스템 오류", error=str(e))
        sys.exit(1)
    
    finally:
        logger.info("OCAD 시스템 종료 중")
        if orchestrator.is_running:
            await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
