"""파일 형식 변환기.

CSV ↔ Parquet, Wide ↔ Long 형식 변환을 지원합니다.
"""

from pathlib import Path
import pandas as pd
from typing import Optional

from ..core.logging import get_logger

logger = get_logger(__name__)


class FormatConverter:
    """파일 형식 변환기."""

    @staticmethod
    def csv_to_parquet(
        csv_path: Path,
        parquet_path: Optional[Path] = None,
        compression: str = "snappy",
    ) -> Path:
        """CSV를 Parquet으로 변환.

        Args:
            csv_path: CSV 파일 경로
            parquet_path: 출력 Parquet 파일 경로 (None이면 자동 생성)
            compression: 압축 방식 ("snappy", "gzip", "brotli", "zstd", "none")

        Returns:
            생성된 Parquet 파일 경로
        """
        csv_path = Path(csv_path)

        if parquet_path is None:
            parquet_path = csv_path.with_suffix(".parquet")

        logger.info(
            "CSV → Parquet 변환 시작",
            csv_path=str(csv_path),
            parquet_path=str(parquet_path),
            compression=compression,
        )

        # CSV 읽기
        df = pd.read_csv(csv_path)

        # Parquet 저장
        df.to_parquet(parquet_path, engine="pyarrow", compression=compression)

        original_size = csv_path.stat().st_size
        converted_size = parquet_path.stat().st_size
        compression_ratio = (1 - converted_size / original_size) * 100

        logger.info(
            "CSV → Parquet 변환 완료",
            original_size_mb=f"{original_size / 1024 / 1024:.2f}",
            converted_size_mb=f"{converted_size / 1024 / 1024:.2f}",
            compression_ratio=f"{compression_ratio:.1f}%",
        )

        return parquet_path

    @staticmethod
    def parquet_to_csv(
        parquet_path: Path, csv_path: Optional[Path] = None, encoding: str = "utf-8"
    ) -> Path:
        """Parquet을 CSV로 변환.

        Args:
            parquet_path: Parquet 파일 경로
            csv_path: 출력 CSV 파일 경로 (None이면 자동 생성)
            encoding: CSV 인코딩

        Returns:
            생성된 CSV 파일 경로
        """
        parquet_path = Path(parquet_path)

        if csv_path is None:
            csv_path = parquet_path.with_suffix(".csv")

        logger.info(
            "Parquet → CSV 변환 시작",
            parquet_path=str(parquet_path),
            csv_path=str(csv_path),
        )

        # Parquet 읽기
        df = pd.read_parquet(parquet_path, engine="pyarrow")

        # CSV 저장
        df.to_csv(csv_path, index=False, encoding=encoding)

        logger.info("Parquet → CSV 변환 완료", rows=len(df))

        return csv_path

    @staticmethod
    def wide_to_long(
        input_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """Wide Format을 Long Format으로 변환.

        Args:
            input_path: 입력 파일 경로 (CSV 또는 Parquet)
            output_path: 출력 파일 경로 (None이면 자동 생성)

        Returns:
            생성된 파일 경로
        """
        input_path = Path(input_path)

        # 파일 읽기
        if input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        logger.info("Wide → Long 변환 시작", input_path=str(input_path))

        # Wide to Long 변환
        id_vars = ["timestamp", "endpoint_id"]
        optional_vars = ["site_name", "zone"]

        # 존재하는 optional_vars만 추가
        for var in optional_vars:
            if var in df.columns:
                id_vars.append(var)

        # 메트릭 컬럼 찾기
        metric_columns = [
            col
            for col in df.columns
            if col not in id_vars
            and not col.startswith("_")
            and col not in ["notes", "status"]
        ]

        # Melt (Wide → Long)
        df_long = df.melt(
            id_vars=id_vars, value_vars=metric_columns, var_name="metric_name", value_name="value"
        )

        # NaN 제거
        df_long = df_long.dropna(subset=["value"])

        # Unit 추출 (컬럼명에서)
        def extract_unit(metric_name):
            if "_ms" in metric_name:
                return "ms"
            elif "_us" in metric_name:
                return "us"
            elif "count" in metric_name:
                return "count"
            elif "success" in metric_name:
                return "bool"
            else:
                return ""

        df_long["unit"] = df_long["metric_name"].apply(extract_unit)

        # metric_name 정리 (suffix 제거)
        df_long["metric_name"] = (
            df_long["metric_name"]
            .str.replace("_ms$", "", regex=True)
            .str.replace("_us$", "", regex=True)
        )

        # 출력 경로 결정
        if output_path is None:
            stem = input_path.stem
            if not stem.endswith("_long"):
                stem += "_long"
            output_path = input_path.parent / f"{stem}{input_path.suffix}"

        # 파일 저장
        if output_path.suffix.lower() == ".csv":
            df_long.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in [".parquet", ".pq"]:
            df_long.to_parquet(output_path, engine="pyarrow")

        logger.info(
            "Wide → Long 변환 완료",
            output_path=str(output_path),
            original_rows=len(df),
            converted_rows=len(df_long),
        )

        return output_path

    @staticmethod
    def long_to_wide(
        input_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """Long Format을 Wide Format으로 변환.

        Args:
            input_path: 입력 파일 경로 (CSV 또는 Parquet)
            output_path: 출력 파일 경로 (None이면 자동 생성)

        Returns:
            생성된 파일 경로
        """
        input_path = Path(input_path)

        # 파일 읽기
        if input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        logger.info("Long → Wide 변환 시작", input_path=str(input_path))

        # Long to Wide 변환 (pivot)
        df_wide = df.pivot_table(
            index=["timestamp", "endpoint_id"],
            columns="metric_name",
            values="value",
            aggfunc="first",  # 중복 시 첫 번째 값
        ).reset_index()

        # 컬럼명 정리
        df_wide.columns.name = None

        # 메타데이터 복원
        for col in ["site_name", "zone"]:
            if col in df.columns:
                metadata = df.groupby(["timestamp", "endpoint_id"])[col].first()
                df_wide = df_wide.merge(
                    metadata.reset_index(), on=["timestamp", "endpoint_id"], how="left"
                )

        # Unit 정보로 컬럼명 정리 (metric_name + unit suffix)
        if "unit" in df.columns:
            # 각 metric의 unit 매핑 생성
            unit_map = (
                df.groupby("metric_name")["unit"].first().to_dict()
            )

            # 컬럼명 변경
            rename_dict = {}
            for metric_name, unit in unit_map.items():
                if unit == "ms":
                    rename_dict[metric_name] = f"{metric_name}_ms"
                elif unit == "us":
                    rename_dict[metric_name] = f"{metric_name}_us"

            df_wide = df_wide.rename(columns=rename_dict)

        # 출력 경로 결정
        if output_path is None:
            stem = input_path.stem
            if stem.endswith("_long"):
                stem = stem[:-5]  # "_long" 제거
            stem += "_wide"
            output_path = input_path.parent / f"{stem}{input_path.suffix}"

        # 파일 저장
        if output_path.suffix.lower() == ".csv":
            df_wide.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in [".parquet", ".pq"]:
            df_wide.to_parquet(output_path, engine="pyarrow")

        logger.info(
            "Long → Wide 변환 완료",
            output_path=str(output_path),
            original_rows=len(df),
            converted_rows=len(df_wide),
        )

        return output_path
