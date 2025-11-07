#!/usr/bin/env python3
"""Generate CFM anomaly detection performance report.

This script generates a comprehensive, easy-to-understand report
analyzing CFM anomaly detection performance using TCN models.

Usage:
    # Generate report from predictions
    python scripts/report_cfm.py \
        --predictions results/cfm/predictions_*.csv \
        --metrics results/cfm/metrics_*.csv \
        --output results/cfm/report.md
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
import glob


CFM_METRICS = [
    "udp_echo_rtt_ms",
    "ecpri_delay_us",
    "lbm_rtt_ms",
]

METRIC_NAMES_KR = {
    "udp_echo_rtt_ms": "UDP Echo RTT",
    "ecpri_delay_us": "eCPRI 지연",
    "lbm_rtt_ms": "LBM RTT",
}

ANOMALY_TYPES_KR = {
    "latency_spike": "지연 급증",
    "packet_loss": "패킷 손실",
    "network_instability": "네트워크 불안정",
    "sustained_degradation": "지속적 성능 저하",
    "normal": "정상",
}


def analyze_per_metric_performance(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze performance per CFM metric.

    Args:
        df: Predictions dataframe

    Returns:
        Dictionary of metric-level performance
    """
    results = {}

    # Filter to records with predictions
    df_eval = df[df["ensemble_score"] > 0].copy()

    for metric in CFM_METRICS:
        if f"{metric}_pred_anomaly" not in df_eval.columns:
            continue

        y_true = df_eval["is_anomaly"].values
        y_pred = df_eval[f"{metric}_pred_anomaly"].values

        tp = ((y_true == True) & (y_pred == True)).sum()
        fp = ((y_true == False) & (y_pred == True)).sum()
        tn = ((y_true == False) & (y_pred == False)).sum()
        fn = ((y_true == True) & (y_pred == False)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[metric] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    return results


def analyze_per_anomaly_type(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze performance per anomaly type.

    Args:
        df: Predictions dataframe

    Returns:
        Dictionary of anomaly-type-level performance
    """
    results = {}

    # Filter to records with predictions
    df_eval = df[df["ensemble_score"] > 0].copy()

    # Get unique anomaly types
    anomaly_types = df_eval["anomaly_type"].unique()

    for anom_type in anomaly_types:
        if anom_type == "normal":
            continue

        # Filter to this anomaly type
        df_type = df_eval[df_eval["anomaly_type"] == anom_type].copy()

        if len(df_type) == 0:
            continue

        y_true = df_type["is_anomaly"].values
        y_pred = df_type["ensemble_anomaly"].values

        tp = ((y_true == True) & (y_pred == True)).sum()
        fp = ((y_true == False) & (y_pred == True)).sum()
        fn = ((y_true == True) & (y_pred == False)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        results[anom_type] = {
            "total": len(df_type),
            "detected": int(tp),
            "missed": int(fn),
            "recall": recall,
        }

    return results


def generate_markdown_report(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate markdown report.

    Args:
        predictions_df: Predictions dataframe
        metrics_df: Metrics dataframe
        output_path: Output file path
    """
    # Analyze data
    per_metric_perf = analyze_per_metric_performance(predictions_df)
    per_anomaly_perf = analyze_per_anomaly_type(predictions_df)

    # Get overall metrics
    overall_rows = metrics_df[metrics_df["dataset"] == "combined"]
    overall = overall_rows.iloc[0].to_dict() if len(overall_rows) > 0 else None

    # Start building report
    lines = []
    lines.append("# CFM 프로토콜 이상 탐지 성능 리포트")
    lines.append("")
    lines.append(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Executive Summary
    lines.append("## 요약")
    lines.append("")
    if overall is not None:
        lines.append(f"TCN(Temporal Convolutional Network) 탐지기를 사용한 CFM 지연 이상 탐지 결과, "
                    f"**정확도 {overall['accuracy']*100:.1f}%**, **F1-score {overall['f1_score']*100:.1f}%**를 달성했습니다.")
    else:
        lines.append("TCN(Temporal Convolutional Network) 탐지기를 사용한 CFM 지연 이상 탐지를 수행했습니다.")
    lines.append("")

    # Dataset info
    lines.append("## 데이터셋")
    lines.append("")

    n_total = len(predictions_df)
    n_anomaly = predictions_df["is_anomaly"].sum()
    n_normal = n_total - n_anomaly

    lines.append(f"- **전체 샘플**: {n_total:,}개")
    lines.append(f"- **정상 샘플**: {n_normal:,}개 ({n_normal/n_total*100:.1f}%)")
    lines.append(f"- **비정상 샘플**: {n_anomaly:,}개 ({n_anomaly/n_total*100:.1f}%)")
    lines.append("")

    # Anomaly type distribution
    if n_anomaly > 0:
        lines.append("### 이상 유형 분포")
        lines.append("")
        anomaly_dist = predictions_df[predictions_df["is_anomaly"]]["anomaly_type"].value_counts()
        for anom_type, count in anomaly_dist.items():
            if anom_type == "normal":
                continue
            type_kr = ANOMALY_TYPES_KR.get(anom_type, anom_type)
            lines.append(f"- **{type_kr}**: {count:,}개 ({count/n_anomaly*100:.1f}%)")
        lines.append("")

    # Overall performance
    lines.append("## 전체 성능 지표")
    lines.append("")

    if overall is not None:
        lines.append("| 지표 | 값 |")
        lines.append("|------|-----|")
        lines.append(f"| 정확도 (Accuracy) | {overall['accuracy']*100:.2f}% |")
        lines.append(f"| 정밀도 (Precision) | {overall['precision']*100:.2f}% |")
        lines.append(f"| 재현율 (Recall) | {overall['recall']*100:.2f}% |")
        lines.append(f"| F1-Score | {overall['f1_score']*100:.2f}% |")
        lines.append("")
        lines.append("### 혼동 행렬 (Confusion Matrix)")
        lines.append("")
        lines.append("| | 예측: 정상 | 예측: 이상 |")
        lines.append("|---------|------------|------------|")
        lines.append(f"| **실제: 정상** | {overall['true_negatives']:,} (TN) | {overall['false_positives']:,} (FP) |")
        lines.append(f"| **실제: 이상** | {overall['false_negatives']:,} (FN) | {overall['true_positives']:,} (TP) |")
        lines.append("")

    # Per-metric performance
    lines.append("## 메트릭별 탐지 성능")
    lines.append("")
    lines.append("각 CFM 메트릭별로 독립적으로 학습된 TCN 모델의 성능입니다.")
    lines.append("")

    lines.append("| 메트릭 | 정밀도 (Precision) | 재현율 (Recall) | F1-Score |")
    lines.append("|--------|-------------------|----------------|----------|")

    for metric, perf in per_metric_perf.items():
        metric_name = METRIC_NAMES_KR.get(metric, metric)
        lines.append(f"| {metric_name} | {perf['precision']*100:.2f}% | {perf['recall']*100:.2f}% | {perf['f1_score']*100:.2f}% |")

    lines.append("")

    # Per-anomaly-type performance
    if per_anomaly_perf:
        lines.append("## 이상 유형별 탐지 성능")
        lines.append("")
        lines.append("각 이상 유형별로 얼마나 잘 탐지했는지 분석한 결과입니다.")
        lines.append("")

        lines.append("| 이상 유형 | 전체 | 탐지 | 미탐지 | 재현율 |")
        lines.append("|----------|------|------|--------|--------|")

        for anom_type, perf in per_anomaly_perf.items():
            type_kr = ANOMALY_TYPES_KR.get(anom_type, anom_type)
            lines.append(f"| {type_kr} | {perf['total']:,} | {perf['detected']:,} | {perf['missed']:,} | {perf['recall']*100:.2f}% |")

        lines.append("")

    # Interpretation
    lines.append("## 결과 해석")
    lines.append("")
    lines.append("### CFM 프로토콜 특화 분석")
    lines.append("")

    if overall is not None and overall['f1_score'] > 0.8:
        lines.append("✅ **우수한 성능**: TCN 모델이 CFM 메트릭의 시계열 패턴을 효과적으로 학습하여 "
                    "높은 이상 탐지 성능을 보였습니다.")
    elif overall is not None and overall['f1_score'] > 0.6:
        lines.append("⚠️ **양호한 성능**: TCN 모델이 대부분의 CFM 이상을 탐지했지만, "
                    "일부 개선의 여지가 있습니다.")
    else:
        lines.append("❌ **개선 필요**: TCN 모델의 성능이 기대에 미치지 못합니다. "
                    "하이퍼파라미터 조정이나 더 많은 학습 데이터가 필요할 수 있습니다.")

    lines.append("")
    lines.append("### 메트릭별 분석")
    lines.append("")

    # Find best and worst metrics
    if per_metric_perf:
        metrics_sorted = sorted(per_metric_perf.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        best_metric = metrics_sorted[0]
        worst_metric = metrics_sorted[-1]

        best_name = METRIC_NAMES_KR.get(best_metric[0], best_metric[0])
        worst_name = METRIC_NAMES_KR.get(worst_metric[0], worst_metric[0])

        lines.append(f"- **최고 성능 메트릭**: {best_name} (F1-Score: {best_metric[1]['f1_score']*100:.2f}%)")
        lines.append(f"- **최저 성능 메트릭**: {worst_name} (F1-Score: {worst_metric[1]['f1_score']*100:.2f}%)")
        lines.append("")

    # Anomaly type analysis
    if per_anomaly_perf:
        lines.append("### 이상 유형별 분석")
        lines.append("")

        anom_sorted = sorted(per_anomaly_perf.items(), key=lambda x: x[1]['recall'], reverse=True)
        best_anom = anom_sorted[0]
        worst_anom = anom_sorted[-1]

        best_anom_name = ANOMALY_TYPES_KR.get(best_anom[0], best_anom[0])
        worst_anom_name = ANOMALY_TYPES_KR.get(worst_anom[0], worst_anom[0])

        lines.append(f"- **가장 잘 탐지된 이상**: {best_anom_name} (재현율: {best_anom[1]['recall']*100:.2f}%)")
        lines.append(f"- **가장 어려운 이상**: {worst_anom_name} (재현율: {worst_anom[1]['recall']*100:.2f}%)")
        lines.append("")

    # Prediction vs Actual Comparison
    lines.append("## 예측 vs 실제 비교 분석")
    lines.append("")

    if overall is not None:
        total = overall['total']
        true_anomalies = overall['true_positives'] + overall['false_negatives']
        pred_anomalies = overall['true_positives'] + overall['false_positives']
        true_normal = total - true_anomalies
        pred_normal = total - pred_anomalies

        tp = overall['true_positives']
        tn = overall['true_negatives']
        fp = overall['false_positives']
        fn = overall['false_negatives']
        matches = tp + tn

        lines.append("| 구분 | 정상 | 이상 | 합계 |")
        lines.append("|------|------|------|------|")
        lines.append(f"| **실제 (Ground Truth)** | {true_normal}개 ({true_normal/total*100:.1f}%) | {true_anomalies}개 ({true_anomalies/total*100:.1f}%) | {total}개 |")
        lines.append(f"| **예측 (Predicted)** | {pred_normal}개 ({pred_normal/total*100:.1f}%) | {pred_anomalies}개 ({pred_anomalies/total*100:.1f}%) | {total}개 |")
        lines.append(f"| **일치 여부** | TN: {tn}개 | TP: {tp}개 | 일치: {matches}개 ({matches/total*100:.1f}%) |")
        lines.append("")

        # Error analysis
        if fp > 0 or fn > 0:
            lines.append("### 오류 분석")
            lines.append("")
            if fp > 0:
                lines.append(f"- **False Positive (오탐)**: {fp}건 ({fp/total*100:.1f}%) - 정상을 이상으로 오판")
            if fn > 0:
                lines.append(f"- **False Negative (미탐)**: {fn}건 ({fn/total*100:.1f}%) - 이상을 정상으로 오판")
            lines.append("")

    lines.append("")
    lines.append("## 기술적 세부사항")
    lines.append("")
    lines.append("### 사용된 모델")
    lines.append("")
    lines.append("- **모델 아키텍처**: TCN (Temporal Convolutional Network)")
    lines.append("- **시퀀스 길이**: 20 timesteps")
    lines.append("- **탐지 방식**: 시계열 예측 오차 기반 이상 탐지")
    lines.append("")

    lines.append("### CFM 메트릭")
    lines.append("")
    lines.append("- **UDP Echo RTT** (ms): UDP 에코 왕복 시간")
    lines.append("- **eCPRI 지연** (us): eCPRI 단방향 지연")
    lines.append("- **LBM RTT** (ms): Loopback Message 왕복 시간")
    lines.append("")

    # Write to file
    report_text = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"✅ 리포트 생성 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CFM anomaly detection performance report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions CSV file (supports wildcards)",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics CSV file (supports wildcards)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cfm/report.md"),
        help="Output report path (default: results/cfm/report.md)",
    )

    args = parser.parse_args()

    print("="*70)
    print("CFM 성능 리포트 생성")
    print("="*70)

    # Find files
    predictions_files = glob.glob(args.predictions)
    metrics_files = glob.glob(args.metrics)

    if not predictions_files:
        print(f"❌ No prediction files found: {args.predictions}")
        return 1

    if not metrics_files:
        print(f"❌ No metrics files found: {args.metrics}")
        return 1

    print(f"\nPredictions file: {predictions_files[0]}")
    print(f"Metrics file: {metrics_files[0]}")

    # Load data
    predictions_df = pd.read_csv(predictions_files[0])
    metrics_df = pd.read_csv(metrics_files[0])

    print(f"\nLoaded {len(predictions_df):,} predictions")
    print(f"Loaded {len(metrics_df)} metric rows")

    # Generate report
    print(f"\n리포트 생성 중...")
    generate_markdown_report(predictions_df, metrics_df, args.output)

    print(f"\n{'='*70}")
    print("✅ 리포트 생성 완료!")
    print(f"{'='*70}")
    print(f"\n리포트 파일: {args.output}")
    print(f"\n리포트를 확인하세요:")
    print(f"  cat {args.output}")
    print(f"  또는 마크다운 뷰어로 열기")

    return 0


if __name__ == "__main__":
    sys.exit(main())
