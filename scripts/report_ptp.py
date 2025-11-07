#!/usr/bin/env python3
"""Generate performance report for PTP anomaly detection.

This script analyzes inference predictions and generates a comprehensive
performance report in Markdown and HTML formats.

Usage:
    # Generate report from predictions
    python scripts/report_ptp.py --predictions results/ptp/predictions.csv

    # Custom output path
    python scripts/report_ptp.py --predictions results/ptp/predictions.csv --output results/ptp/report.md
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load predictions from CSV.

    Args:
        predictions_path: Path to predictions CSV

    Returns:
        DataFrame with predictions
    """
    print(f"Loading predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path)
    print(f"Loaded {len(df):,} predictions")
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate performance metrics.

    Args:
        df: DataFrame with predictions

    Returns:
        Dictionary of metrics
    """
    # Filter valid predictions (with predicted values)
    valid_df = df[df["predicted_value"].notna()].copy()

    if len(valid_df) == 0:
        return {"error": "No valid predictions found"}

    # Overall metrics
    metrics = {
        "total_samples": len(df),
        "valid_predictions": len(valid_df),
        "skipped_samples": len(df) - len(valid_df),
    }

    # Prediction accuracy
    metrics["mean_prediction_error"] = valid_df["prediction_error"].mean()
    metrics["std_prediction_error"] = valid_df["prediction_error"].std()
    metrics["median_prediction_error"] = valid_df["prediction_error"].median()
    metrics["p95_prediction_error"] = valid_df["prediction_error"].quantile(0.95)
    metrics["p99_prediction_error"] = valid_df["prediction_error"].quantile(0.99)
    metrics["max_prediction_error"] = valid_df["prediction_error"].max()

    # Anomaly detection metrics
    if "is_anomaly_actual" in valid_df.columns:
        tp = ((valid_df["is_anomaly_predicted"] == True) & (valid_df["is_anomaly_actual"] == True)).sum()
        tn = ((valid_df["is_anomaly_predicted"] == False) & (valid_df["is_anomaly_actual"] == False)).sum()
        fp = ((valid_df["is_anomaly_predicted"] == True) & (valid_df["is_anomaly_actual"] == False)).sum()
        fn = ((valid_df["is_anomaly_predicted"] == False) & (valid_df["is_anomaly_actual"] == True)).sum()

        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

        # Derived metrics
        if tp + fp > 0:
            metrics["precision"] = tp / (tp + fp)
        else:
            metrics["precision"] = 0.0

        if tp + fn > 0:
            metrics["recall"] = tp / (tp + fn)
        else:
            metrics["recall"] = 0.0

        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        else:
            metrics["f1_score"] = 0.0

        if tp + tn + fp + fn > 0:
            metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
        else:
            metrics["accuracy"] = 0.0

        # Per-anomaly-type metrics
        if "anomaly_type_actual" in valid_df.columns:
            anomaly_types = valid_df[valid_df["is_anomaly_actual"] == True]["anomaly_type_actual"].unique()
            anomaly_types = [at for at in anomaly_types if pd.notna(at)]

            metrics["anomaly_type_metrics"] = {}
            for anomaly_type in anomaly_types:
                anomaly_subset = valid_df[valid_df["anomaly_type_actual"] == anomaly_type]
                tp_type = ((anomaly_subset["is_anomaly_predicted"] == True) & (anomaly_subset["is_anomaly_actual"] == True)).sum()
                fn_type = ((anomaly_subset["is_anomaly_predicted"] == False) & (anomaly_subset["is_anomaly_actual"] == True)).sum()

                if tp_type + fn_type > 0:
                    recall_type = tp_type / (tp_type + fn_type)
                else:
                    recall_type = 0.0

                metrics["anomaly_type_metrics"][anomaly_type] = {
                    "samples": len(anomaly_subset),
                    "detected": int(tp_type),
                    "missed": int(fn_type),
                    "recall": recall_type,
                }

    return metrics


def generate_markdown_report(metrics: dict, output_path: Path) -> None:
    """Generate Markdown report.

    Args:
        metrics: Metrics dictionary
        output_path: Output path for report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# PTP 프로토콜 이상 탐지 성능 리포트\n\n")
        f.write(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## 요약\n\n")
        if "error" in metrics:
            f.write(f"**오류**: {metrics['error']}\n\n")
            return

        f.write(f"TCN(Temporal Convolutional Network) 탐지기를 사용한 PTP(Precision Time Protocol) 시간 동기화 이상 탐지 결과입니다.\n\n")

        if "accuracy" in metrics:
            f.write(f"- **정확도(Accuracy)**: {metrics['accuracy'] * 100:.2f}%\n")
            f.write(f"- **정밀도(Precision)**: {metrics['precision'] * 100:.2f}%\n")
            f.write(f"- **재현율(Recall)**: {metrics['recall'] * 100:.2f}%\n")
            f.write(f"- **F1-Score**: {metrics['f1_score'] * 100:.2f}%\n\n")

        f.write("---\n\n")

        # Dataset Info
        f.write("## 데이터셋\n\n")
        f.write(f"- **전체 샘플 수**: {metrics['total_samples']:,}개\n")
        f.write(f"- **유효 예측 수**: {metrics['valid_predictions']:,}개\n")
        f.write(f"- **건너뛴 샘플**: {metrics['skipped_samples']:,}개 (히스토리 부족)\n\n")

        if "true_positives" in metrics:
            total_anomalies = metrics["true_positives"] + metrics["false_negatives"]
            total_normal = metrics["true_negatives"] + metrics["false_positives"]
            f.write(f"- **실제 정상 샘플**: {total_normal:,}개 ({total_normal / metrics['valid_predictions'] * 100:.1f}%)\n")
            f.write(f"- **실제 비정상 샘플**: {total_anomalies:,}개 ({total_anomalies / metrics['valid_predictions'] * 100:.1f}%)\n")

        f.write("\n---\n\n")

        # Performance Metrics
        f.write("## 성능 지표\n\n")

        f.write("### 전체 성능\n\n")
        f.write("| 지표 | 값 |\n")
        f.write("|------|----|\n")
        if "accuracy" in metrics:
            f.write(f"| 정확도 (Accuracy) | {metrics['accuracy'] * 100:.2f}% |\n")
            f.write(f"| 정밀도 (Precision) | {metrics['precision'] * 100:.2f}% |\n")
            f.write(f"| 재현율 (Recall) | {metrics['recall'] * 100:.2f}% |\n")
            f.write(f"| F1-Score | {metrics['f1_score'] * 100:.2f}% |\n")
            f.write(f"| 참 긍정 (True Positive) | {metrics['true_positives']:,} |\n")
            f.write(f"| 참 부정 (True Negative) | {metrics['true_negatives']:,} |\n")
            f.write(f"| 거짓 긍정 (False Positive) | {metrics['false_positives']:,} |\n")
            f.write(f"| 거짓 부정 (False Negative) | {metrics['false_negatives']:,} |\n")

        f.write("\n---\n\n")

        # Anomaly Type Performance
        if "anomaly_type_metrics" in metrics and metrics["anomaly_type_metrics"]:
            f.write("## 이상 유형별 탐지 성능\n\n")
            f.write("| 이상 유형 | 샘플 수 | 탐지 성공 | 탐지 실패 | 재현율 |\n")
            f.write("|----------|---------|----------|----------|--------|\n")

            anomaly_type_names = {
                "clock_drift": "클럭 드리프트",
                "master_change": "마스터 변경",
                "path_delay_increase": "경로 지연 증가",
                "sync_failure": "동기화 실패",
            }

            for anomaly_type, type_metrics in metrics["anomaly_type_metrics"].items():
                korean_name = anomaly_type_names.get(anomaly_type, anomaly_type)
                f.write(f"| {korean_name} | {type_metrics['samples']:,} | {type_metrics['detected']:,} | {type_metrics['missed']:,} | {type_metrics['recall'] * 100:.2f}% |\n")

            f.write("\n### 이상 유형 설명\n\n")
            f.write("- **클럭 드리프트**: 클럭이 높은 드리프트율을 보이며 오프셋이 계속 증가\n")
            f.write("- **마스터 변경**: PTP 마스터 클럭이 변경되어 포트 상태 전환 및 오프셋 스파이크 발생\n")
            f.write("- **경로 지연 증가**: 네트워크 지연이 증가하여 경로 지연이 증가\n")
            f.write("- **동기화 실패**: PTP 동기화가 실패하여 오프셋이 계속 증가\n\n")

            f.write("---\n\n")

        # Prediction Accuracy
        f.write("## 나노초 정밀도 분석\n\n")
        f.write("TCN 모델의 PTP 오프셋 예측 정확도를 나노초 단위로 평가했습니다.\n\n")
        f.write("| 통계량 | 예측 오차 (정규화) |\n")
        f.write("|--------|--------------------|\n")
        f.write(f"| 평균 | {metrics['mean_prediction_error']:.6f} |\n")
        f.write(f"| 표준편차 | {metrics['std_prediction_error']:.6f} |\n")
        f.write(f"| 중앙값 | {metrics['median_prediction_error']:.6f} |\n")
        f.write(f"| 95th percentile | {metrics['p95_prediction_error']:.6f} |\n")
        f.write(f"| 99th percentile | {metrics['p99_prediction_error']:.6f} |\n")
        f.write(f"| 최댓값 | {metrics['max_prediction_error']:.6f} |\n\n")

        f.write("**참고**: 예측 오차는 정규화된 값입니다. 실제 나노초 단위 오차는 모델의 스케일러에 따라 달라집니다.\n\n")

        f.write("---\n\n")

        # TCN Architecture
        f.write("## TCN 아키텍처 특성\n\n")
        f.write("Temporal Convolutional Network(TCN)은 시계열 예측에 최적화된 딥러닝 모델입니다.\n\n")
        f.write("### 주요 특징\n\n")
        f.write("- **Dilated Causal Convolution**: 과거 데이터만 참조하여 미래 정보 누출 방지\n")
        f.write("- **Receptive Field**: 긴 시퀀스 패턴 캡처 가능\n")
        f.write("  - Dilation rates: 1, 2, 4, 8 (4 layers)\n")
        f.write("  - Receptive field: 약 45 timesteps (kernel_size=3 기준)\n")
        f.write("- **병렬 학습**: RNN과 달리 병렬 처리 가능하여 학습 속도 빠름\n")
        f.write("- **잔차 연결(Residual Connection)**: 깊은 네트워크에서도 안정적인 학습\n\n")

        f.write("### PTP 이상 탐지 적합성\n\n")
        f.write("- **나노초 정밀도**: PTP 오프셋의 미세한 변화 캡처\n")
        f.write("- **장기 패턴**: 클럭 드리프트 같은 장기 추세 탐지\n")
        f.write("- **빠른 응답**: 실시간 모니터링에 적합한 추론 속도\n\n")

        f.write("---\n\n")

        # Conclusions
        f.write("## 결론\n\n")
        if "f1_score" in metrics:
            if metrics["f1_score"] >= 0.9:
                f.write(f"TCN 모델은 PTP 이상 탐지에서 **우수한 성능**을 보였습니다 (F1-Score: {metrics['f1_score'] * 100:.2f}%).\n\n")
            elif metrics["f1_score"] >= 0.7:
                f.write(f"TCN 모델은 PTP 이상 탐지에서 **양호한 성능**을 보였습니다 (F1-Score: {metrics['f1_score'] * 100:.2f}%).\n\n")
            else:
                f.write(f"TCN 모델은 PTP 이상 탐지에서 **개선이 필요한 성능**을 보였습니다 (F1-Score: {metrics['f1_score'] * 100:.2f}%).\n\n")

        f.write("## 예측 vs 실제 비교 분석\n\n")

        # Calculate distribution
        # PTP uses is_anomaly_actual/is_anomaly_predicted column names
        if "is_anomaly_actual" in df.columns and "is_anomaly_predicted" in df.columns:
            y_true = df["is_anomaly_actual"].astype(int)
            y_pred = df["is_anomaly_predicted"].astype(int)
        else:
            # Fallback to old column names if present
            y_true = df.get("ground_truth", pd.Series([0] * len(df))).astype(int)
            y_pred = df.get("predicted", pd.Series([0] * len(df))).astype(int)

        total = len(df)
        true_anomalies = y_true.sum()
        pred_anomalies = y_pred.sum()
        true_normal = total - true_anomalies
        pred_normal = total - pred_anomalies

        tp = metrics.get("true_positives", 0)
        tn = metrics.get("true_negatives", 0)
        fp = metrics.get("false_positives", 0)
        fn = metrics.get("false_negatives", 0)
        matches = tp + tn

        f.write("| 구분 | 정상 | 이상 | 합계 |\n")
        f.write("|------|------|------|------|\n")
        f.write(f"| **실제 (Ground Truth)** | {true_normal}개 ({true_normal/total*100:.1f}%) | {true_anomalies}개 ({true_anomalies/total*100:.1f}%) | {total}개 |\n")
        f.write(f"| **예측 (Predicted)** | {pred_normal}개 ({pred_normal/total*100:.1f}%) | {pred_anomalies}개 ({pred_anomalies/total*100:.1f}%) | {total}개 |\n")
        f.write(f"| **일치 여부** | TN: {tn}개 | TP: {tp}개 | 일치: {matches}개 ({matches/total*100:.1f}%) |\n\n")

        # Error analysis
        if fp > 0 or fn > 0:
            f.write("### 오류 분석\n\n")
            if fp > 0:
                f.write(f"- **False Positive (오탐)**: {fp}건 ({fp/total*100:.1f}%) - 정상을 이상으로 오판\n")
            if fn > 0:
                f.write(f"- **False Negative (미탐)**: {fn}건 ({fn/total*100:.1f}%) - 이상을 정상으로 오판\n")
            f.write("\n")

        f.write("---\n\n")
        f.write(f"**리포트 생성**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Markdown report saved: {output_path}")


def generate_html_report(markdown_path: Path, html_path: Path) -> None:
    """Convert Markdown report to HTML.

    Args:
        markdown_path: Path to Markdown report
        html_path: Path to save HTML report
    """
    try:
        import markdown
    except ImportError:
        print("Warning: markdown package not installed. Skipping HTML generation.")
        print("  Install with: pip install markdown")
        return

    with open(markdown_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    html_content = markdown.markdown(md_content, extensions=["tables"])

    # Add CSS styling
    html_with_style = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PTP 프로토콜 이상 탐지 성능 리포트</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        hr {{
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 30px 0;
        }}
        ul, ol {{
            padding-left: 30px;
        }}
        li {{
            margin: 5px 0;
        }}
        strong {{
            color: #2c3e50;
        }}
        code {{
            background-color: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_with_style)

    print(f"HTML report saved: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance report for PTP anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions CSV file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ptp/report.md"),
        help="Output path for Markdown report (default: results/ptp/report.md)",
    )

    args = parser.parse_args()

    # Validate
    if not args.predictions.exists():
        parser.error(f"Predictions file does not exist: {args.predictions}")

    print("=" * 70)
    print("PTP Performance Report Generation")
    print("=" * 70)

    # Load predictions
    df = load_predictions(args.predictions)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(df)

    # Generate Markdown report
    print("\nGenerating Markdown report...")
    generate_markdown_report(metrics, args.output)

    # Generate HTML report
    html_path = args.output.with_suffix(".html")
    print("\nGenerating HTML report...")
    generate_html_report(args.output, html_path)

    print("\n" + "=" * 70)
    print("Report Generation Complete!")
    print("=" * 70)
    print(f"\nGenerated reports:")
    print(f"  - Markdown: {args.output}")
    print(f"  - HTML: {html_path}")


if __name__ == "__main__":
    main()
