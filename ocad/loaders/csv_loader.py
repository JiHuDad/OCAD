"""CSV 파일 로더."""

from pathlib import Path
import pandas as pd

from .base import BaseLoader, LoaderResult
from ..core.logging import get_logger

logger = get_logger(__name__)


class CSVLoader(BaseLoader):
    """CSV 파일 로더.

    Wide Format과 Long Format 모두 지원합니다.

    Wide Format 예시:
        timestamp,endpoint_id,site_name,udp_echo_rtt_ms,ecpri_delay_us,...

    Long Format 예시:
        timestamp,endpoint_id,metric_name,value,unit,...
    """

    def __init__(
        self,
        strict_mode: bool = False,
        format_type: str = "auto",
        encoding: str = "utf-8",
    ):
        """초기화.

        Args:
            strict_mode: 엄격 모드
            format_type: 파일 형식 ("auto", "wide", "long")
            encoding: 파일 인코딩
        """
        super().__init__(strict_mode)
        self.format_type = format_type
        self.encoding = encoding

    def load(self, file_path: Path) -> LoaderResult:
        """CSV 파일 로드.

        Args:
            file_path: CSV 파일 경로

        Returns:
            LoaderResult: 로드 결과
        """
        file_path = Path(file_path)

        # 파일 검증
        self.validate_file_exists(file_path)
        self.validate_file_extension(file_path, [".csv", ".txt"])

        logger.info("CSV 파일 로드 시작", file_path=str(file_path))

        try:
            # CSV 읽기
            df = pd.read_csv(file_path, encoding=self.encoding)

            logger.info(
                "CSV 파일 읽기 완료",
                rows=len(df),
                columns=list(df.columns),
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
                "CSV 파일 로드 완료",
                file_path=str(file_path),
                success=result.success,
                valid=result.valid_records,
                invalid=result.invalid_records,
            )

            return result

        except Exception as e:
            logger.error("CSV 파일 로드 실패", file_path=str(file_path), error=str(e))
            return LoaderResult(
                success=False,
                total_records=0,
                valid_records=0,
                invalid_records=0,
                metrics=[],
                errors=[{"error": str(e), "file_path": str(file_path)}],
            )

    def _detect_format(self, df: pd.DataFrame) -> str:
        """파일 형식 자동 감지 (Wide vs Long).

        Args:
            df: DataFrame

        Returns:
            "wide" 또는 "long"
        """
        columns = set(df.columns)

        # Long Format 특징: metric_name, value 열이 있음
        if "metric_name" in columns and "value" in columns:
            return "long"

        # Wide Format 특징: udp_echo_rtt_ms, ecpri_delay_us 등의 메트릭 열이 있음
        metric_columns = {
            "udp_echo_rtt_ms",
            "ecpri_delay_us",
            "lbm_rtt_ms",
            "ccm_interval_ms",
        }
        if any(col in columns for col in metric_columns):
            return "wide"

        # 기본값: Wide
        logger.warning("파일 형식 자동 감지 실패, Wide Format으로 가정")
        return "wide"

    def _convert_long_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Long Format을 Wide Format으로 변환.

        Args:
            df: Long Format DataFrame

        Returns:
            Wide Format DataFrame
        """
        # metric_name별로 pivot
        df_wide = df.pivot_table(
            index=["timestamp", "endpoint_id"],
            columns="metric_name",
            values="value",
            aggfunc="first",  # 중복 시 첫 번째 값 사용
        ).reset_index()

        # 열 이름 정리 (multi-index 제거)
        df_wide.columns.name = None

        # 메타데이터 컬럼 복원 (site_name, zone 등)
        if "site_name" in df.columns:
            site_names = df.groupby(["timestamp", "endpoint_id"])["site_name"].first()
            df_wide = df_wide.merge(
                site_names.reset_index(), on=["timestamp", "endpoint_id"], how="left"
            )

        if "zone" in df.columns:
            zones = df.groupby(["timestamp", "endpoint_id"])["zone"].first()
            df_wide = df_wide.merge(
                zones.reset_index(), on=["timestamp", "endpoint_id"], how="left"
            )

        return df_wide

    def _normalize_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wide Format을 정규화하여 Long Format으로 변환.

        각 메트릭을 별도의 행으로 분리합니다.

        Args:
            df: Wide Format DataFrame

        Returns:
            정규화된 DataFrame (metric_type, value, unit 포함)
        """
        records = []

        # 메트릭 매핑 (컬럼명 → metric_type, unit)
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

            # 각 메트릭에 대해 별도의 레코드 생성
            for col_name, (metric_type, unit) in metric_mapping.items():
                if col_name in df.columns and pd.notna(row.get(col_name)):
                    record = base_data.copy()
                    record["metric_type"] = metric_type
                    record["value"] = row[col_name]
                    record["unit"] = unit
                    records.append(record)

        return pd.DataFrame(records)
