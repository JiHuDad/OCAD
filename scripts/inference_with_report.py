#!/usr/bin/env python3
"""추론 실행 + 보고서 생성 통합 스크립트.

하나의 스크립트로 추론 실행과 보고서 생성을 모두 수행합니다.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocad.core.data_source import DataSourceFactory
from ocad.core.logging import get_logger

logger = get_logger(__name__)


class InferenceRunner:
    """추론 실행 클래스."""

    def __init__(self, model_path: Path, config: dict):
        """초기화.

        Args:
            model_path: 학습된 모델 경로
            config: 추론 설정
        """
        self.model_path = model_path
        self.config = config
        self.threshold = config.get("threshold", 0.5)
        self.rule_threshold = config.get("rule_threshold", 10.0)
        logger.info("InferenceRunner 초기화 완료", config=config)

    def run(self, data_source, output_path: Path = None):
        """추론 실행.

        Args:
            data_source: 데이터 소스 (DataSource 인터페이스)
            output_path: 결과 출력 경로 (선택)

        Returns:
            DataFrame: 추론 결과
        """
        print(f"\n{'=' * 70}")
        print(f"추론 실행")
        print(f"{'=' * 70}")

        # 데이터 소스 메타데이터 출력
        metadata = data_source.get_metadata()
        print(f"\n데이터 소스 정보:")
        for key, value in metadata.items():
            if key == "label_distribution":
                print(f"  {key}:")
                for label, count in value.items():
                    print(f"    {label}: {count}개")
            else:
                print(f"  {key}: {value}")

        # 데이터 수집 및 추론
        all_results = []
        print(f"\n추론 실행 중...")

        batch_count = 0
        total_processed = 0

        for batch in data_source:
            batch_count += 1
            # batch.metrics가 실제 MetricData 리스트
            total_processed += len(batch.metrics)

            # 각 레코드에 대해 추론 수행
            for metric in batch.metrics:
                # metric은 이미 dict 형태 (FileDataSource의 경우)
                result = self._detect_anomaly(metric)
                all_results.append(result)

            # 진행 상황 출력 (100개 배치마다)
            if batch_count % 5 == 0:
                print(f"  배치 {batch_count}: {total_processed}개 처리됨")

        print(f"\n✅ 총 {total_processed}개 레코드 처리 완료")

        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(all_results)

        # 결과 저장
        if output_path:
            results_df.to_csv(output_path, index=False)
            file_size_kb = output_path.stat().st_size / 1024
            print(f"\n✅ 결과 저장: {output_path}")
            print(f"   파일 크기: {file_size_kb:.2f} KB")

        # 결과 분석 출력
        self._print_analysis(results_df)

        return results_df

    def _detect_anomaly(self, record: dict) -> dict:
        """단일 레코드에 대해 이상 탐지 수행 (룰 기반).

        Args:
            record: 데이터 레코드

        Returns:
            dict: 추론 결과
        """
        # 메트릭 추출 (FileDataSource는 _ms, _us 없이 반환)
        udp_rtt = record.get("udp_echo_rtt", record.get("udp_echo_rtt_ms", 0))
        ecpri_delay = record.get("ecpri_delay", record.get("ecpri_delay_us", 0))
        lbm_rtt = record.get("lbm_rtt", record.get("lbm_rtt_ms", 0))

        # 룰 기반 탐지
        rule_based_score = 1.0 if udp_rtt > self.rule_threshold else 0.0
        ecpri_score = 1.0 if ecpri_delay > 200 else 0.0
        lbm_score = 1.0 if lbm_rtt > self.rule_threshold else 0.0

        # 앙상블 점수 (평균)
        composite_score = (rule_based_score + ecpri_score + lbm_score) / 3.0

        # 예측 라벨
        predicted_label = "anomaly" if composite_score >= self.threshold else "normal"

        return {
            "timestamp": record.get("timestamp"),
            "endpoint_id": record.get("endpoint_id"),
            "udp_echo_rtt": udp_rtt,
            "ecpri_delay": ecpri_delay,
            "lbm_rtt": lbm_rtt,
            "label": record.get("label", "unknown"),
            "rule_based_score": rule_based_score,
            "ecpri_score": ecpri_score,
            "lbm_score": lbm_score,
            "composite_score": composite_score,
            "predicted_label": predicted_label,
        }

    def _print_analysis(self, results_df: pd.DataFrame):
        """결과 분석 출력."""
        print(f"\n{'=' * 70}")
        print(f"결과 분석")
        print(f"{'=' * 70}")

        # 예측 분포
        print(f"\n예측 분포:")
        pred_counts = results_df["predicted_label"].value_counts()
        for label, count in pred_counts.items():
            percentage = count / len(results_df) * 100
            print(f"  {label}: {count}개 ({percentage:.1f}%)")

        # 정확도 계산 (label 컬럼이 있는 경우)
        if "label" in results_df.columns and results_df["label"].notna().any():
            correct = (results_df["predicted_label"] == results_df["label"]).sum()
            accuracy = correct / len(results_df) * 100
            print(f"\n정확도: {accuracy:.2f}%")

            # Confusion Matrix
            print(f"\nConfusion Matrix:")
            cm = pd.crosstab(
                results_df["label"],
                results_df["predicted_label"],
                rownames=["실제"],
                colnames=["예측"],
            )
            print(cm)

        # 탐지기별 평균 점수
        print(f"\n탐지기별 평균 점수:")
        for col in ["rule_based", "ecpri", "lbm", "composite"]:
            score_col = f"{col}_score"
            if score_col in results_df.columns:
                mean_score = results_df[score_col].mean()
                print(f"  {col:15s}: {mean_score:.3f}")


def generate_report(input_file: Path, output_file: Path = None):
    """추론 결과 보고서 생성.

    Args:
        input_file: 추론 결과 CSV 파일
        output_file: 보고서 출력 파일 (선택, 기본값: reports/inference_report_<timestamp>.md)
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    print(f"\n{'=' * 70}")
    print(f"OCAD 추론 결과 보고서 생성")
    print(f"{'=' * 70}")
    print(f"입력: {input_file}")

    # 출력 파일명 자동 생성 (날짜 시간 포함)
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("reports") / f"inference_report_{timestamp}.md"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"출력: {output_file}")

    # 데이터 로드
    df = pd.read_csv(input_file)

    # 보고서 생성
    lines = []

    # 헤더
    lines.append("# OCAD 추론 결과 보고서")
    lines.append("")
    lines.append(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**입력 파일**: {input_file}")
    lines.append(f"**총 레코드**: {len(df)}개")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 전체 요약
    lines.append("## 📊 전체 요약")
    lines.append("")

    # 시간 범위
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()
    duration = end_time - start_time

    lines.append(f"- **분석 기간**: {start_time} ~ {end_time}")
    lines.append(f"- **분석 시간**: {duration}")
    lines.append("")

    # 엔드포인트 수
    endpoints = df["endpoint_id"].value_counts()
    lines.append(f"- **엔드포인트 수**: {len(endpoints)}개")
    for endpoint, count in endpoints.items():
        lines.append(f"  - `{endpoint}`: {count}개 레코드")
    lines.append("")

    # 실제 라벨 분포
    if "label" in df.columns and df["label"].notna().any():
        lines.append("### 실제 라벨 분포")
        lines.append("")
        label_counts = df["label"].value_counts()
        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            lines.append(f"- **{label}**: {count}개 ({percentage:.1f}%)")
        lines.append("")

    # 예측 라벨 분포
    lines.append("### 예측 라벨 분포")
    lines.append("")
    pred_counts = df["predicted_label"].value_counts()
    for label, count in pred_counts.items():
        percentage = count / len(df) * 100
        lines.append(f"- **{label}**: {count}개 ({percentage:.1f}%)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 성능 지표
    if "label" in df.columns and df["label"].notna().any():
        lines.append("## 🎯 성능 지표")
        lines.append("")

        # 정확도
        correct = (df["predicted_label"] == df["label"]).sum()
        accuracy = correct / len(df) * 100
        lines.append(f"### 정확도: **{accuracy:.2f}%**")
        lines.append("")

        # Confusion Matrix
        lines.append("### Confusion Matrix")
        lines.append("")
        lines.append("```")
        cm = pd.crosstab(
            df["label"],
            df["predicted_label"],
            rownames=["실제"],
            colnames=["예측"],
        )
        lines.append(str(cm))
        lines.append("```")
        lines.append("")

        # Precision, Recall, F1
        try:
            # 이진 분류 지표 계산
            y_true = (df["label"] == "anomaly").astype(int)
            y_pred = (df["predicted_label"] == "anomaly").astype(int)

            if y_pred.sum() > 0:  # anomaly 예측이 있는 경우
                precision = precision_score(y_true, y_pred, zero_division=0) * 100
                recall = recall_score(y_true, y_pred, zero_division=0) * 100
                f1 = f1_score(y_true, y_pred, zero_division=0) * 100

                lines.append("### 세부 성능 지표")
                lines.append("")
                lines.append(f"- **Precision (정밀도)**: {precision:.2f}%")
                lines.append("  - 이상으로 예측한 것 중 실제 이상 비율")
                lines.append(f"- **Recall (재현율)**: {recall:.2f}%")
                lines.append("  - 실제 이상 중 탐지한 비율")
                lines.append(f"- **F1 Score**: {f1:.2f}%")
                lines.append("  - Precision과 Recall의 조화 평균")
                lines.append("")

            # TP, FP, TN, FN
            cm_array = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm_array.ravel() if cm_array.size == 4 else (0, 0, 0, 0)

            lines.append("### 분류 결과 상세")
            lines.append("")
            lines.append(f"- **True Positive (TP)**: {tp}개 - ✅ 이상을 이상으로 정확히 탐지")
            lines.append(f"- **False Positive (FP)**: {fp}개 - ⚠️ 정상을 이상으로 오탐")
            lines.append(f"- **True Negative (TN)**: {tn}개 - ✅ 정상을 정상으로 정확히 분류")
            lines.append(f"- **False Negative (FN)**: {fn}개 - ❌ 이상을 정상으로 미탐")
            lines.append("")

        except Exception as e:
            lines.append(f"⚠️ 성능 지표 계산 실패: {e}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # 탐지기별 분석
    lines.append("## 🔍 탐지기별 분석")
    lines.append("")

    for detector_name, score_col in [
        ("Rule Based", "rule_based_score"),
        ("Ecpri", "ecpri_score"),
        ("Lbm", "lbm_score"),
        ("Composite", "composite_score"),
    ]:
        if score_col in df.columns:
            scores = df[score_col]
            mean_score = scores.mean()
            max_score = scores.max()
            active_count = (scores > 0).sum()
            active_pct = active_count / len(df) * 100

            lines.append(f"### {detector_name}")
            lines.append("")
            lines.append(f"- **평균 점수**: {mean_score:.3f}")
            lines.append(f"- **최대 점수**: {max_score:.3f}")
            lines.append(f"- **활성화 횟수**: {active_count}회 ({active_pct:.1f}%)")

            if active_count > 0:
                active_mean = scores[scores > 0].mean()
                lines.append(f"- **활성화 시 평균 점수**: {active_mean:.3f}")

            lines.append("")

    lines.append("---")
    lines.append("")

    # 메트릭 통계
    lines.append("## 📈 메트릭 통계")
    lines.append("")

    for metric_name, col_name in [
        ("UDP ECHO RTT", "udp_echo_rtt"),
        ("ECPRI DELAY", "ecpri_delay"),
        ("LBM RTT", "lbm_rtt"),
    ]:
        if col_name in df.columns:
            values = df[col_name]
            lines.append(f"### {metric_name}")
            lines.append("")
            lines.append(f"- **평균**: {values.mean():.2f}")
            lines.append(f"- **표준편차**: {values.std():.2f}")
            lines.append(f"- **최소값**: {values.min():.2f}")
            lines.append(f"- **최대값**: {values.max():.2f}")
            lines.append(f"- **중앙값**: {values.median():.2f}")
            lines.append(f"- **P95**: {values.quantile(0.95):.2f}")
            lines.append(f"- **P99**: {values.quantile(0.99):.2f}")
            lines.append("")

    lines.append("---")
    lines.append("")

    # False Negative 분석
    if "label" in df.columns and df["label"].notna().any():
        fn_df = df[(df["label"] == "anomaly") & (df["predicted_label"] == "normal")]

        if len(fn_df) > 0:
            lines.append("## ❌ False Negative 분석 (미탐지)")
            lines.append("")
            lines.append(f"**총 {len(fn_df)}건의 이상이 탐지되지 않았습니다.**")
            lines.append("")

            # 미탐지 메트릭 범위
            lines.append("### 미탐지 메트릭 범위")
            lines.append("")
            for metric_name, col_name in [
                ("udp_echo_rtt", "udp_echo_rtt"),
                ("ecpri_delay", "ecpri_delay"),
                ("lbm_rtt", "lbm_rtt"),
            ]:
                if col_name in fn_df.columns:
                    values = fn_df[col_name]
                    lines.append(
                        f"- **{metric_name}**: {values.min():.2f} ~ {values.max():.2f} (평균: {values.mean():.2f})"
                    )
            lines.append("")

            # 미탐지 사례 샘플
            lines.append("### 미탐지 사례 (처음 10개)")
            lines.append("")
            lines.append("| 시간 | UDP RTT | eCPRI Delay | LBM RTT | Composite Score |")
            lines.append("|------|---------|-------------|---------|-----------------|")

            for _, row in fn_df.head(10).iterrows():
                time_str = row["timestamp"].strftime("%H:%M:%S") if isinstance(row["timestamp"], pd.Timestamp) else str(row["timestamp"])
                lines.append(
                    f"| {time_str} | {row['udp_echo_rtt']:.2f} | "
                    f"{row['ecpri_delay']:.2f} | {row['lbm_rtt']:.2f} | "
                    f"{row['composite_score']:.3f} |"
                )

            lines.append("")
            lines.append("---")
            lines.append("")

    # Top 이상 케이스
    top_anomalies = df.nlargest(10, "composite_score")

    lines.append("## ⚠️ 탐지된 이상 케이스 (상위 10개)")
    lines.append("")
    lines.append("가장 높은 이상 점수를 기록한 케이스들:")
    lines.append("")
    lines.append("| 순위 | 시간 | UDP RTT | eCPRI Delay | LBM RTT | Composite Score | 실제 라벨 |")
    lines.append("|------|------|---------|-------------|---------|-----------------|-----------|")

    for idx, (_, row) in enumerate(top_anomalies.iterrows(), 1):
        time_str = row["timestamp"].strftime("%H:%M:%S") if isinstance(row["timestamp"], pd.Timestamp) else str(row["timestamp"])
        label = row.get("label", "unknown")
        label_emoji = "✅" if label == "anomaly" else "❌" if label == "normal" else "❓"
        lines.append(
            f"| {idx} | {time_str} | {row['udp_echo_rtt']:.2f} | "
            f"{row['ecpri_delay']:.2f} | {row['lbm_rtt']:.2f} | "
            f"{row['composite_score']:.3f} | {label} {label_emoji} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # 시간대별 분석
    lines.append("## 📅 시간대별 분석")
    lines.append("")
    lines.append("### 5분 간격 이상 탐지 빈도")
    lines.append("")

    df["time_bin"] = df["timestamp"].dt.floor("5min")
    time_groups = df.groupby("time_bin")

    for time_bin, group in time_groups:
        anomaly_count = (group["predicted_label"] == "anomaly").sum()
        avg_score = group["composite_score"].mean()
        time_str = time_bin.strftime("%H:%M")

        # 간단한 시각화 (█ 개수)
        bar_length = min(int(anomaly_count / 2), 15)
        bar = "█" * bar_length if bar_length > 0 else "█"

        lines.append(f"`{time_str}` {bar} {anomaly_count}건 (평균 점수: {avg_score:.3f})")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 권장 사항
    if "label" in df.columns and df["label"].notna().any():
        y_true = (df["label"] == "anomaly").astype(int)
        y_pred = (df["predicted_label"] == "anomaly").astype(int)

        fp_rate = (y_pred - y_true).clip(lower=0).sum() / len(df) * 100
        fn_rate = (y_true - y_pred).clip(lower=0).sum() / y_true.sum() * 100 if y_true.sum() > 0 else 0

        lines.append("## 💡 권장 사항")
        lines.append("")

        if fn_rate > 20:
            lines.append("### ⚠️ 높은 False Negative율")
            lines.append("")
            lines.append(f"- 현재 **{fn_rate:.1f}%**의 이상이 탐지되지 않고 있습니다.")
            lines.append("- **권장 조치**:")
            lines.append("  1. 탐지 임계값 낮추기 (`--threshold 0.3`)")
            lines.append("  2. 룰 기반 임계값 조정 (`--rule-threshold 8.0`)")
            lines.append("  3. 변화점 탐지기 추가 고려")
        elif fp_rate > 5:
            lines.append("### ⚠️ 높은 False Positive율")
            lines.append("")
            lines.append(f"- 현재 **{fp_rate:.1f}%**의 정상 데이터가 오탐되고 있습니다.")
            lines.append("- **권장 조치**:")
            lines.append("  1. 탐지 임계값 높이기 (`--threshold 0.7`)")
            lines.append("  2. 룰 기반 임계값 완화 (`--rule-threshold 12.0`)")
        else:
            lines.append("### ✅ 양호한 탐지 성능")
            lines.append("")
            lines.append("현재 탐지 성능이 우수합니다:")
            lines.append(f"- False Negative율: {fn_rate:.1f}% (목표: ≤ 20%)")
            lines.append(f"- False Positive율: {fp_rate:.1f}% (목표: ≤ 5%)")

        lines.append("")
        lines.append("---")
        lines.append("")

    # 부록
    lines.append("## 📝 부록")
    lines.append("")
    lines.append("### 파일 정보")
    lines.append("")
    lines.append(f"- **보고서 파일**: {output_file}")
    lines.append(f"- **데이터 파일**: {input_file}")
    lines.append(f"- **생성 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 파일 저장
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    file_size_kb = output_file.stat().st_size / 1024
    print(f"\n✅ 보고서 생성 완료: {output_file}")
    print(f"   파일 크기: {file_size_kb:.2f} KB")
    print(f"\n보고서 확인:")
    print(f"  cat {output_file}")
    print(f"  code {output_file}")
    print(f"{'=' * 70}")


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="추론 실행 + 보고서 생성 (통합)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 데이터 소스
    parser.add_argument(
        "--data-source",
        type=str,
        required=True,
        help="데이터 소스 경로 (CSV/Excel/Parquet 파일)",
    )

    # 모델 경로
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ocad/models/tcn"),
        help="학습된 모델 경로",
    )

    # 추론 설정
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="이상 탐지 임계값 (0.0 ~ 1.0)",
    )

    parser.add_argument(
        "--rule-threshold",
        type=float,
        default=10.0,
        help="룰 기반 탐지 임계값 (ms)",
    )

    # 출력 설정
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="추론 결과 CSV 파일 (기본값: data/inference_results_<timestamp>.csv)",
    )

    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="보고서 마크다운 파일 (기본값: reports/inference_report_<timestamp>.md)",
    )

    # 데이터 배치 크기
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="데이터 소스 배치 크기",
    )

    args = parser.parse_args()

    # 시작 시간
    start_time = datetime.now()
    print(f"{'=' * 70}")
    print(f"OCAD 추론 + 보고서 생성 (통합)")
    print(f"{'=' * 70}")
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n설정:")
    print(f"  데이터 소스: {args.data_source}")
    print(f"  모델 경로: {args.model_path}")
    print(f"  임계값: {args.threshold}")
    print(f"  배치 크기: {args.batch_size}")
    print(f"{'=' * 70}")

    # 출력 파일명 자동 생성 (날짜 시간 포함)
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    if args.output_csv is None:
        args.output_csv = Path("data") / f"inference_results_{timestamp}.csv"

    if args.output_report is None:
        args.output_report = Path("reports") / f"inference_report_{timestamp}.md"

    # 출력 디렉토리 생성
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    # 데이터 소스 생성
    data_source = DataSourceFactory.create_from_file(
        args.data_source,
        batch_size=args.batch_size,
    )
    logger.info("데이터 소스 생성 완료")
    print(f"\n✅ 데이터 소스 생성 완료")

    # 추론 실행
    config = {
        "threshold": args.threshold,
        "rule_threshold": args.rule_threshold,
    }
    runner = InferenceRunner(args.model_path, config)

    try:
        results_df = runner.run(data_source, output_path=args.output_csv)
    finally:
        # 데이터 소스 종료
        data_source.close()

    # 보고서 생성
    print(f"\n{'=' * 70}")
    generate_report(args.output_csv, args.output_report)

    # 종료 시간
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 70}")
    print(f"✅ 모든 작업 완료!")
    print(f"{'=' * 70}")
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {duration}")
    print(f"\n생성된 파일:")
    print(f"  - 추론 결과: {args.output_csv}")
    print(f"  - 보고서: {args.output_report}")
    print(f"\n다음 단계:")
    print(f"  - 결과 확인: head -20 {args.output_csv}")
    print(f"  - 보고서 확인: cat {args.output_report}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
