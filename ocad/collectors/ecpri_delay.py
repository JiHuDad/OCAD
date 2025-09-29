"""eCPRI delay collector for O-RAN fronthaul.
O-RAN 프론트홀에서 eCPRI 지연 측정을 수집하는 모듈
"""

import time
from typing import Optional
import xml.etree.ElementTree as ET

from ncclient import manager
from ncclient.operations.errors import RPCError

from ..core.models import Capabilities, Endpoint, MetricSample
from .base import BaseCollector


class EcpriDelayCollector(BaseCollector):
    """eCPRI 일방향 지연 측정을 위한 수집기.
    
    O-RAN 표준에서 정의된 eCPRI(enhanced Common Public Radio Interface) 
    delay management 기능을 사용하여 프론트홀 지연을 측정합니다.
    """
    
    def can_collect(self, capabilities: Capabilities) -> bool:
        """eCPRI 지연 측정이 지원되는지 확인합니다.
        
        Args:
            capabilities: 엔드포인트 기능 정보
            
        Returns:
            eCPRI 지연 측정이 지원되면 True
        """
        return capabilities.ecpri_delay
    
    async def collect(self, endpoint: Endpoint, capabilities: Capabilities) -> Optional[MetricSample]:
        """eCPRI 지연 측정을 수집합니다.
        
        Args:
            endpoint: 측정할 엔드포인트
            capabilities: 엔드포인트 기능 정보
            
        Returns:
            eCPRI 지연이 포함된 메트릭 샘플
        """
        if not capabilities.ecpri_delay:
            return None
        
        try:
            # 장비에서 지연 측정값을 가져옵니다
            delay_us = await self._get_ecpri_delay(endpoint)
            
            if delay_us is None:
                return None
            
            # 측정 샘플 생성
            return MetricSample(
                endpoint_id=endpoint.id,
                ts_ms=int(time.time() * 1000),
                ecpri_ow_us=delay_us,  # 일방향 지연 (마이크로초)
            )
            
        except Exception as e:
            self.logger.error(
                "eCPRI 지연 수집 실패",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return None
    
    async def _get_ecpri_delay(self, endpoint: Endpoint) -> Optional[float]:
        """장비에서 eCPRI 일방향 지연을 가져옵니다.
        
        Args:
            endpoint: 조회할 엔드포인트
            
        Returns:
            마이크로초 단위의 지연 또는 None
        """
        try:
            # NETCONF 연결 생성
            with manager.connect(
                host=endpoint.host,
                port=endpoint.port,
                username=self.config.username,
                password=self.config.password,
                timeout=self.config.timeout,
                hostkey_verify=self.config.hostkey_verify,
            ) as mgr:
                
                # O-RAN delay management 상태 조회를 위한 필터
                filter_xml = """
                <filter>
                    <delay-management xmlns="urn:o-ran:delay-management:1.0">
                        <adaptive-delay-configuration>
                            <bandwidth-scs-delay-state/>
                        </adaptive-delay-configuration>
                    </delay-management>
                </filter>
                """
                
                # NETCONF GET 요청 실행
                result = mgr.get(filter=filter_xml)
                
                # 응답에서 지연 값 파싱
                delay_us = self._parse_delay_response(result.data)
                
                return delay_us
                
        except RPCError as e:
            self.logger.warning(
                "eCPRI 지연 조회 실패 (RPC 오류)",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return None
        except Exception as e:
            self.logger.error(
                "eCPRI 지연 조회 실패",
                endpoint_id=endpoint.id,
                error=str(e),
            )
            return None
    
    def _parse_delay_response(self, xml_data: str) -> Optional[float]:
        """XML 응답에서 eCPRI 지연 값을 파싱합니다.
        
        Args:
            xml_data: 장비에서 받은 XML 응답
            
        Returns:
            마이크로초 단위의 지연 또는 None
        """
        try:
            root = ET.fromstring(xml_data)
            
            # XML에서 지연 값 검색
            # 실제 YANG 모델 구현에 따라 달라질 수 있음
            # 벤더별로 네임스페이스가 다를 수 있음
            
            namespaces = {
                'dm': 'urn:o-ran:delay-management:1.0',
                'nc': 'urn:ietf:params:xml:ns:netconf:base:1.0',
            }
            
            # 가능한 지연 요소들의 XPath 경로
            delay_paths = [
                './/dm:adaptive-ru-profile/dm:delay-profile/dm:t2a-max-up',
                './/dm:bandwidth-scs-delay-state/dm:ru-delay-profile/dm:t2a-max-up',
                './/dm:t2a-max-up',  # T2a 상향링크 최대 지연
                './/dm:current-delay',  # 현재 지연
                './/dm:measured-delay',  # 측정된 지연
            ]
            
            # 각 경로에서 지연 값 검색
            for path in delay_paths:
                elements = root.findall(path, namespaces)
                if elements:
                    try:
                        # 필요시 나노초에서 마이크로초로 변환
                        delay_value = float(elements[0].text)
                        
                        # 다양한 단위 처리 (ns, us, ms)
                        if delay_value > 1000000:  # 나노초로 추정
                            return delay_value / 1000.0
                        elif delay_value > 1000:  # 마이크로초로 추정
                            return delay_value
                        else:  # 밀리초로 추정
                            return delay_value * 1000.0
                            
                    except (ValueError, AttributeError):
                        continue
            
            # 특정 지연이 없으면 지연 관련 요소에서 숫자 값 검색
            for elem in root.iter():
                if 'delay' in elem.tag.lower() and elem.text:
                    try:
                        delay_value = float(elem.text)
                        # 합리적인 범위라면 마이크로초로 가정
                        if 0.1 <= delay_value <= 10000:
                            return delay_value
                    except ValueError:
                        continue
            
            self.logger.warning(
                "응답에서 지연 값을 찾을 수 없음",
                xml_snippet=xml_data[:200] + "..." if len(xml_data) > 200 else xml_data,
            )
            return None
            
        except ET.ParseError as e:
            self.logger.error("지연 XML 파싱 실패", error=str(e))
            return None
        except Exception as e:
            self.logger.error("지연 파싱 중 예상치 못한 오류", error=str(e))
            return None
