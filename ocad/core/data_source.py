"""데이터 소스 추상화 인터페이스.

학습과 추론에서 사용할 데이터를 제공하는 인터페이스를 정의합니다.
파일 기반과 실시간 스트리밍을 모두 지원하도록 추상화되었습니다.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from ..core.schemas import MetricData
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataBatch:
    """데이터 배치."""

    metrics: List[MetricData]
    batch_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __len__(self):
        return len(self.metrics)


class DataSource(ABC):
    """데이터 소스 추상 인터페이스.

    학습과 추론에서 사용할 데이터를 제공하는 인터페이스입니다.
    파일 기반, 실시간 스트리밍 등 다양한 소스를 통일된 방식으로 사용할 수 있습니다.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[DataBatch]:
        """데이터 배치를 순회합니다.

        Yields:
            DataBatch: 데이터 배치
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """데이터 소스 메타데이터를 반환합니다.

        Returns:
            Dict: 메타데이터 (총 레코드 수, 시작/종료 시간 등)
        """
        pass

    @abstractmethod
    def close(self):
        """데이터 소스를 닫습니다.

        리소스 정리 (파일 핸들, 네트워크 연결 등)를 수행합니다.
        """
        pass


class FileDataSource(DataSource):
    """파일 기반 데이터 소스.

    CSV, Excel, Parquet 파일에서 데이터를 읽어옵니다.
    """

    def __init__(
        self,
        file_path: Path,
        batch_size: int = 100,
        loader_type: str = "auto"
    ):
        """초기화.

        Args:
            file_path: 파일 경로
            batch_size: 배치 크기
            loader_type: 로더 타입 ("auto", "csv", "excel", "parquet")
        """
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.loader_type = loader_type
        self._df: Optional[pd.DataFrame] = None
        self._current_idx = 0

        if not self.file_path.exists():
            raise FileNotFoundError(f"파일 없음: {file_path}")

        logger.info(
            "FileDataSource 초기화",
            file_path=str(file_path),
            batch_size=batch_size
        )

    def _load_file(self) -> pd.DataFrame:
        """파일을 로드합니다."""
        if self._df is not None:
            return self._df

        logger.info("파일 로드 중", file_path=str(self.file_path))

        # 파일 타입에 따라 로드
        suffix = self.file_path.suffix.lower()

        if suffix == ".csv" or self.loader_type == "csv":
            self._df = pd.read_csv(self.file_path)
        elif suffix in [".xlsx", ".xls"] or self.loader_type == "excel":
            self._df = pd.read_excel(self.file_path, engine="openpyxl")
        elif suffix in [".parquet", ".pq"] or self.loader_type == "parquet":
            self._df = pd.read_parquet(self.file_path, engine="pyarrow")
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

        logger.info(
            "파일 로드 완료",
            records=len(self._df),
            columns=list(self._df.columns)
        )

        return self._df

    def __iter__(self) -> Iterator[DataBatch]:
        """데이터 배치를 순회합니다."""
        df = self._load_file()
        self._current_idx = 0

        while self._current_idx < len(df):
            end_idx = min(self._current_idx + self.batch_size, len(df))
            batch_df = df.iloc[self._current_idx:end_idx]

            # DataFrame을 MetricData 리스트로 변환
            metrics = self._convert_to_metrics(batch_df)

            yield DataBatch(
                metrics=metrics,
                batch_id=f"batch_{self._current_idx}_{end_idx}",
                metadata={
                    "start_idx": self._current_idx,
                    "end_idx": end_idx,
                    "total": len(df)
                }
            )

            self._current_idx = end_idx

    def _convert_to_metrics(self, df: pd.DataFrame) -> List[MetricData]:
        """DataFrame을 MetricData 리스트로 변환.

        Args:
            df: DataFrame

        Returns:
            List[MetricData]: 메트릭 리스트
        """
        metrics = []

        for _, row in df.iterrows():
            try:
                # timestamp 처리
                timestamp = row.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = pd.Timestamp(timestamp).value // 10**6
                elif isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.value // 10**6
                else:
                    timestamp = int(timestamp)

                # MetricData 생성 (간소화된 버전)
                # 실제로는 각 메트릭별로 별도 MetricData 생성 필요
                # 여기서는 대표 메트릭만 사용
                metric = {
                    "timestamp": timestamp,
                    "endpoint_id": str(row.get("endpoint_id", "")),
                    "udp_echo_rtt": float(row.get("udp_echo_rtt_ms", 0)),
                    "ecpri_delay": float(row.get("ecpri_delay_us", 0)),
                    "lbm_rtt": float(row.get("lbm_rtt_ms", 0)),
                    "label": str(row.get("label", "unknown")),
                }

                metrics.append(metric)

            except Exception as e:
                logger.warning("메트릭 변환 실패", row_data=row.to_dict(), error=str(e))
                continue

        return metrics

    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터를 반환합니다."""
        df = self._load_file()

        metadata = {
            "source_type": "file",
            "file_path": str(self.file_path),
            "total_records": len(df),
            "batch_size": self.batch_size,
            "num_batches": (len(df) + self.batch_size - 1) // self.batch_size,
        }

        # 타임스탬프 범위
        if "timestamp" in df.columns:
            metadata["start_time"] = str(df["timestamp"].min())
            metadata["end_time"] = str(df["timestamp"].max())

        # 라벨 분포 (있으면)
        if "label" in df.columns:
            metadata["label_distribution"] = df["label"].value_counts().to_dict()

        return metadata

    def close(self):
        """데이터 소스를 닫습니다."""
        self._df = None
        logger.info("FileDataSource 종료", file_path=str(self.file_path))


class StreamingDataSource(DataSource):
    """실시간 스트리밍 데이터 소스.

    Kafka, WebSocket 등 실시간 데이터 스트림에서 데이터를 읽어옵니다.
    """

    def __init__(
        self,
        source_config: Dict[str, Any],
        batch_size: int = 100,
        timeout_seconds: float = 1.0
    ):
        """초기화.

        Args:
            source_config: 스트림 설정 (Kafka 토픽, WebSocket URL 등)
            batch_size: 배치 크기
            timeout_seconds: 배치 대기 타임아웃 (초)
        """
        self.source_config = source_config
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self._buffer: List[MetricData] = []
        self._total_processed = 0

        logger.info(
            "StreamingDataSource 초기화",
            config=source_config,
            batch_size=batch_size
        )

    def __iter__(self) -> Iterator[DataBatch]:
        """데이터 배치를 순회합니다.

        실시간 스트림에서 데이터를 읽어 배치 단위로 반환합니다.
        """
        # TODO: 실제 스트리밍 구현 (Kafka, WebSocket 등)
        # 현재는 인터페이스만 정의
        raise NotImplementedError(
            "StreamingDataSource는 아직 구현되지 않았습니다. "
            "Kafka/WebSocket 연동이 필요합니다."
        )

    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터를 반환합니다."""
        return {
            "source_type": "streaming",
            "config": self.source_config,
            "total_processed": self._total_processed,
            "batch_size": self.batch_size,
        }

    def close(self):
        """데이터 소스를 닫습니다."""
        logger.info("StreamingDataSource 종료")


class DataSourceFactory:
    """데이터 소스 팩토리.

    설정에 따라 적절한 데이터 소스를 생성합니다.
    """

    @staticmethod
    def create_from_file(
        file_path: Path,
        batch_size: int = 100
    ) -> FileDataSource:
        """파일 기반 데이터 소스를 생성합니다.

        Args:
            file_path: 파일 경로
            batch_size: 배치 크기

        Returns:
            FileDataSource: 파일 데이터 소스
        """
        return FileDataSource(file_path=file_path, batch_size=batch_size)

    @staticmethod
    def create_from_streaming(
        source_config: Dict[str, Any],
        batch_size: int = 100
    ) -> StreamingDataSource:
        """스트리밍 데이터 소스를 생성합니다.

        Args:
            source_config: 스트림 설정
            batch_size: 배치 크기

        Returns:
            StreamingDataSource: 스트리밍 데이터 소스
        """
        return StreamingDataSource(
            source_config=source_config,
            batch_size=batch_size
        )

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> DataSource:
        """설정에서 데이터 소스를 생성합니다.

        Args:
            config: 데이터 소스 설정
                - type: "file" 또는 "streaming"
                - file_path: 파일 경로 (type="file")
                - source_config: 스트림 설정 (type="streaming")
                - batch_size: 배치 크기

        Returns:
            DataSource: 데이터 소스

        Examples:
            >>> # 파일 기반
            >>> config = {
            ...     "type": "file",
            ...     "file_path": "data/training.csv",
            ...     "batch_size": 100
            ... }
            >>> source = DataSourceFactory.create_from_config(config)
            >>>
            >>> # 스트리밍
            >>> config = {
            ...     "type": "streaming",
            ...     "source_config": {"kafka_topic": "metrics"},
            ...     "batch_size": 100
            ... }
            >>> source = DataSourceFactory.create_from_config(config)
        """
        source_type = config.get("type", "file")
        batch_size = config.get("batch_size", 100)

        if source_type == "file":
            file_path = config.get("file_path")
            if not file_path:
                raise ValueError("file_path가 필요합니다")
            return DataSourceFactory.create_from_file(
                file_path=Path(file_path),
                batch_size=batch_size
            )

        elif source_type == "streaming":
            source_config = config.get("source_config")
            if not source_config:
                raise ValueError("source_config가 필요합니다")
            return DataSourceFactory.create_from_streaming(
                source_config=source_config,
                batch_size=batch_size
            )

        else:
            raise ValueError(f"지원하지 않는 소스 타입: {source_type}")
