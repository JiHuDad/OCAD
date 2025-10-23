"""Parquet 파일 로더."""

from pathlib import Path
import pandas as pd

from .base import BaseLoader, LoaderResult
from ..core.logging import get_logger

logger = get_logger(__name__)


class ParquetLoader(BaseLoader):
    """Parquet 파일 로더.

    고성능 컬럼 기반 스토리지 포맷인 Parquet을 지원합니다.
    대용량 데이터 처리에 유리합니다.
    """

    def __init__(self, strict_mode: bool = False, format_type: str = "auto"):
        """초기화.

        Args:
            strict_mode: 엄격 모드
            format_type: 파일 형식 ("auto", "wide", "long")
        """
        super().__init__(strict_mode)
        self.format_type = format_type

    def load(self, file_path: Path) -> LoaderResult:
        """Parquet 파일 로드.

        Args:
            file_path: Parquet 파일 경로

        Returns:
            LoaderResult: 로드 결과
        """
        file_path = Path(file_path)

        # 파일 검증
        self.validate_file_exists(file_path)
        self.validate_file_extension(file_path, [".parquet", ".pq"])

        logger.info("Parquet 파일 로드 시작", file_path=str(file_path))

        try:
            # Parquet 읽기
            df = pd.read_parquet(file_path, engine="pyarrow")

            logger.info(
                "Parquet 파일 읽기 완료",
                rows=len(df),
                columns=list(df.columns),
                size_mb=file_path.stat().st_size / 1024 / 1024,
            )

            # 형식 자동 감지
            if self.format_type == "auto":
                detected_format = self._detect_format(df)
                logger.info("파일 형식 자동 감지", format=detected_format)
            else:
                detected_format = self.format_type

            # 형식에 따라 변환
            if detected_format == "long":
                df = self._convert_long_to_wide(df)
                logger.info("Long Format → Wide Format 변환 완료")

            # Wide Format을 MetricData로 변환
            df_normalized = self._normalize_wide_format(df)

            # 처리
            result = self.process_dataframe(df_normalized)

            logger.info(
                "Parquet 파일 로드 완료",
                file_path=str(file_path),
                success=result.success,
                valid=result.valid_records,
                invalid=result.invalid_records,
            )

            return result

        except Exception as e:
            logger.error(
                "Parquet 파일 로드 실패", file_path=str(file_path), error=str(e)
            )
            return LoaderResult(
                success=False,
                total_records=0,
                valid_records=0,
                invalid_records=0,
                metrics=[],
                errors=[{"error": str(e), "file_path": str(file_path)}],
            )

    def _detect_format(self, df: pd.DataFrame) -> str:
        """파일 형식 자동 감지."""
        columns = set(df.columns)

        if "metric_name" in columns and "value" in columns:
            return "long"

        metric_columns = {
            "udp_echo_rtt_ms",
            "ecpri_delay_us",
            "lbm_rtt_ms",
            "ccm_interval_ms",
        }
        if any(col in columns for col in metric_columns):
            return "wide"

        logger.warning("파일 형식 자동 감지 실패, Wide Format으로 가정")
        return "wide"

    def _convert_long_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Long Format을 Wide Format으로 변환."""
        df_wide = df.pivot_table(
            index=["timestamp", "endpoint_id"],
            columns="metric_name",
            values="value",
            aggfunc="first",
        ).reset_index()

        df_wide.columns.name = None

        for col in ["site_name", "zone"]:
            if col in df.columns:
                metadata = df.groupby(["timestamp", "endpoint_id"])[col].first()
                df_wide = df_wide.merge(
                    metadata.reset_index(), on=["timestamp", "endpoint_id"], how="left"
                )

        return df_wide

    def _normalize_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wide Format을 정규화."""
        records = []

        metric_mapping = {
            "udp_echo_rtt_ms": ("udp_echo_rtt", "ms"),
            "udp_echo_rtt": ("udp_echo_rtt", "ms"),
            "ecpri_delay_us": ("ecpri_delay", "us"),
            "ecpri_delay": ("ecpri_delay", "us"),
            "lbm_rtt_ms": ("lbm_rtt", "ms"),
            "lbm_rtt": ("lbm_rtt", "ms"),
            "lbm_success": ("lbm_success", "bool"),
            "ccm_interval_ms": ("ccm_interval", "ms"),
            "ccm_interval": ("ccm_interval", "ms"),
            "ccm_miss_count": ("ccm_miss_count", "count"),
        }

        for _, row in df.iterrows():
            base_data = {
                "timestamp": row.get("timestamp"),
                "endpoint_id": row.get("endpoint_id"),
                "site_name": row.get("site_name"),
                "zone": row.get("zone"),
            }

            for col_name, (metric_type, unit) in metric_mapping.items():
                if col_name in df.columns and pd.notna(row.get(col_name)):
                    record = base_data.copy()
                    record["metric_type"] = metric_type
                    record["value"] = row[col_name]
                    record["unit"] = unit
                    records.append(record)

        return pd.DataFrame(records)
