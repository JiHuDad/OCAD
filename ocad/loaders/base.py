"""파일 로더 베이스 클래스."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd

from ..core.schemas import MetricData
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LoaderResult:
    """로더 결과 데이터."""

    success: bool
    total_records: int
    valid_records: int
    invalid_records: int
    metrics: List[MetricData]
    errors: List[Dict[str, Any]]

    @property
    def success_rate(self) -> float:
        """성공률 (0.0 ~ 1.0)."""
        if self.total_records == 0:
            return 0.0
        return self.valid_records / self.total_records


class BaseLoader(ABC):
    """파일 로더 베이스 클래스.

    모든 파일 로더는 이 클래스를 상속받아 구현합니다.
    """

    def __init__(self, strict_mode: bool = False):
        """초기화.

        Args:
            strict_mode: 엄격 모드. True면 하나라도 실패 시 전체 실패
        """
        self.strict_mode = strict_mode
        self.logger = logger

    @abstractmethod
    def load(self, file_path: Path) -> LoaderResult:
        """파일 로드.

        Args:
            file_path: 파일 경로

        Returns:
            LoaderResult: 로드 결과
        """
        pass

    def validate_file_exists(self, file_path: Path) -> None:
        """파일 존재 여부 확인.

        Args:
            file_path: 파일 경로

        Raises:
            FileNotFoundError: 파일이 없는 경우
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def validate_file_extension(self, file_path: Path, extensions: List[str]) -> None:
        """파일 확장자 확인.

        Args:
            file_path: 파일 경로
            extensions: 허용되는 확장자 리스트 (예: ['.csv', '.txt'])

        Raises:
            ValueError: 확장자가 허용되지 않는 경우
        """
        if file_path.suffix.lower() not in extensions:
            raise ValueError(
                f"Invalid file extension: {file_path.suffix}. "
                f"Expected one of: {', '.join(extensions)}"
            )

    def convert_row_to_metric(
        self, row: pd.Series, row_index: int
    ) -> tuple[Optional[MetricData], Optional[Dict[str, Any]]]:
        """DataFrame 행을 MetricData로 변환.

        Args:
            row: DataFrame 행
            row_index: 행 인덱스 (에러 리포팅용)

        Returns:
            (MetricData, error): 성공 시 (metric, None), 실패 시 (None, error_dict)
        """
        try:
            # timestamp 처리
            timestamp = row.get("timestamp")
            if pd.isna(timestamp):
                raise ValueError("timestamp is missing")

            # timestamp가 문자열이면 Unix timestamp로 변환
            if isinstance(timestamp, str):
                timestamp = pd.Timestamp(timestamp).value // 10**6  # nanoseconds to ms
            elif isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.value // 10**6
            else:
                timestamp = int(timestamp)

            # MetricData 생성
            metric = MetricData(
                endpoint_id=str(row.get("endpoint_id", "")),
                timestamp=timestamp,
                metric_type=str(row.get("metric_type", row.get("metric_name", ""))),
                value=float(row.get("value", 0.0)),
                unit=str(row.get("unit", "")),
                labels={
                    k: str(v)
                    for k, v in {
                        "site_name": row.get("site_name"),
                        "zone": row.get("zone"),
                    }.items()
                    if pd.notna(v)
                },
            )

            return metric, None

        except Exception as e:
            error = {
                "row_index": row_index,
                "error": str(e),
                "data": row.to_dict(),
            }
            return None, error

    def process_dataframe(self, df: pd.DataFrame) -> LoaderResult:
        """DataFrame을 MetricData 리스트로 변환.

        Args:
            df: 입력 DataFrame

        Returns:
            LoaderResult: 로드 결과
        """
        metrics = []
        errors = []

        for idx, row in df.iterrows():
            metric, error = self.convert_row_to_metric(row, idx)

            if metric:
                metrics.append(metric)
            elif error:
                errors.append(error)
                if self.strict_mode:
                    # 엄격 모드에서는 첫 에러에서 중단
                    break

        total = len(df)
        valid = len(metrics)
        invalid = len(errors)

        success = valid > 0 if not self.strict_mode else invalid == 0

        self.logger.info(
            "DataFrame 처리 완료",
            total=total,
            valid=valid,
            invalid=invalid,
            success_rate=f"{valid/total*100:.1f}%" if total > 0 else "N/A",
        )

        return LoaderResult(
            success=success,
            total_records=total,
            valid_records=valid,
            invalid_records=invalid,
            metrics=metrics,
            errors=errors,
        )
