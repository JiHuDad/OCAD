#!/usr/bin/env python3
"""Generate comprehensive BGP anomaly detection report.

This script analyzes inference results and generates a detailed report
with performance metrics, anomaly type breakdown, and recommendations.

Usage:
    python scripts/report_bgp.py --predictions results/bgp/predictions.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions from CSV."""
    print(f"Loading predictions from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records")
    return df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    y_true = df["ground_truth"].astype(int).values
    y_pred = df["predicted_anomaly"].astype(int).values

    # Confusion matrix
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "specificity": specificity,
    }


def analyze_anomaly_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Analyze detection performance by anomaly type."""
    # Classify anomaly types based on BGP metrics
    anomaly_types = {}

    # 1. Route flapping: high flap count
    flapping = df[df["route_flap_count"] > 5].copy()
    if len(flapping) > 0:
        y_true = flapping["ground_truth"].astype(int).values
        y_pred = flapping["predicted_anomaly"].astype(int).values

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        precision = tp / (tp + ((y_true == 0) & (y_pred == 1)).sum()) if (tp + ((y_true == 0) & (y_pred == 1)).sum()) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        anomaly_types["Route Flapping"] = {
            "count": len(flapping),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    # 2. Prefix hijacking: sudden prefix count spike
    hijacking = df[df["prefix_count"] > 150].copy()
    if len(hijacking) > 0:
        y_true = hijacking["ground_truth"].astype(int).values
        y_pred = hijacking["predicted_anomaly"].astype(int).values

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        precision = tp / (tp + ((y_true == 0) & (y_pred == 1)).sum()) if (tp + ((y_true == 0) & (y_pred == 1)).sum()) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        anomaly_types["Prefix Hijacking"] = {
            "count": len(hijacking),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    # 3. AS-path poisoning: long AS-path
    poisoning = df[df["as_path_length"] > 9].copy()
    if len(poisoning) > 0:
        y_true = poisoning["ground_truth"].astype(int).values
        y_pred = poisoning["predicted_anomaly"].astype(int).values

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        precision = tp / (tp + ((y_true == 0) & (y_pred == 1)).sum()) if (tp + ((y_true == 0) & (y_pred == 1)).sum()) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        anomaly_types["AS-path Poisoning"] = {
            "count": len(poisoning),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    # 4. Session instability: IDLE state
    instability = df[df["session_state"] == 0].copy()
    if len(instability) > 0:
        y_true = instability["ground_truth"].astype(int).values
        y_pred = instability["predicted_anomaly"].astype(int).values

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        precision = tp / (tp + ((y_true == 0) & (y_pred == 1)).sum()) if (tp + ((y_true == 0) & (y_pred == 1)).sum()) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        anomaly_types["Session Instability"] = {
            "count": len(instability),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    return anomaly_types


def generate_report(df: pd.DataFrame, metrics: Dict[str, Any], anomaly_types: Dict[str, Dict[str, Any]]) -> str:
    """Generate comprehensive report in Markdown format."""
    report = []

    # Header
    report.append("# BGP 프로토콜 이상 탐지 성능 리포트")
    report.append("")
    report.append(f"**생성 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**탐지 모델**: GNN (Graph Neural Network)")
    report.append("")

    # Summary
    report.append("## 요약")
    report.append("")
    report.append(
        f"GNN 탐지기를 사용한 BGP AS-path 이상 탐지 결과, "
        f"정확도 **{metrics['accuracy']*100:.2f}%**, F1-score **{metrics['f1_score']*100:.2f}%**를 달성했습니다."
    )
    report.append("")

    # Dataset stats
    n_total = len(df)
    n_anomalies = df["ground_truth"].sum()
    n_normal = n_total - n_anomalies

    report.append("## 데이터셋")
    report.append("")
    report.append(f"- **전체 레코드**: {n_total:,}개")
    report.append(f"- **정상 데이터**: {n_normal:,}개 ({n_normal/n_total*100:.1f}%)")
    report.append(f"- **비정상 데이터**: {n_anomalies:,}개 ({n_anomalies/n_total*100:.1f}%)")
    report.append("")

    # Performance metrics
    cm = metrics["confusion_matrix"]
    report.append("## 성능 지표")
    report.append("")
    report.append("### Confusion Matrix")
    report.append("")
    report.append("| 실제 \\ 예측 | 정상 | 비정상 |")
    report.append("|------------|------|--------|")
    report.append(f"| **정상** | {cm['tn']} | {cm['fp']} |")
    report.append(f"| **비정상** | {cm['fn']} | {cm['tp']} |")
    report.append("")

    report.append("### 주요 지표")
    report.append("")
    report.append("| 지표 | 값 | 설명 |")
    report.append("|------|----|----|")
    report.append(f"| **Accuracy** | {metrics['accuracy']*100:.2f}% | 전체 예측 중 정확한 예측의 비율 |")
    report.append(f"| **Precision** | {metrics['precision']*100:.2f}% | 비정상으로 예측한 것 중 실제 비정상의 비율 |")
    report.append(f"| **Recall** | {metrics['recall']*100:.2f}% | 실제 비정상 중 탐지된 비율 |")
    report.append(f"| **F1-score** | {metrics['f1_score']*100:.2f}% | Precision과 Recall의 조화평균 |")
    report.append(f"| **Specificity** | {metrics['specificity']*100:.2f}% | 실제 정상 중 정상으로 예측된 비율 |")
    report.append("")

    # Anomaly type breakdown
    if anomaly_types:
        report.append("## 이상 유형별 탐지 성능")
        report.append("")
        report.append("BGP 프로토콜의 다양한 이상 유형에 대한 탐지 성능을 분석했습니다.")
        report.append("")
        report.append("| 이상 유형 | 건수 | Precision | Recall | F1-score |")
        report.append("|----------|------|-----------|--------|----------|")

        for anom_type, stats in anomaly_types.items():
            report.append(
                f"| **{anom_type}** | {stats['count']} | "
                f"{stats['precision']*100:.1f}% | {stats['recall']*100:.1f}% | {stats['f1_score']*100:.1f}% |"
            )
        report.append("")

        # Type descriptions
        report.append("### 이상 유형 설명")
        report.append("")
        report.append("- **Route Flapping**: 라우트가 빠르게 변경되는 현상 (route_flap_count > 5)")
        report.append("- **Prefix Hijacking**: 비정상적으로 많은 prefix를 광고 (prefix_count > 150)")
        report.append("- **AS-path Poisoning**: 비정상적으로 긴 AS 경로 (as_path_length > 9)")
        report.append("- **Session Instability**: BGP 세션 불안정 (session_state = IDLE)")
        report.append("")

    # Results interpretation
    report.append("## 결과 해석")
    report.append("")

    # Accuracy
    if metrics["accuracy"] >= 0.9:
        report.append("### ✅ 우수한 탐지 정확도")
        report.append(f"전체 정확도가 **{metrics['accuracy']*100:.2f}%**로 우수합니다. ")
    elif metrics["accuracy"] >= 0.8:
        report.append("### ⚠️ 양호한 탐지 정확도")
        report.append(f"전체 정확도가 **{metrics['accuracy']*100:.2f}%**로 양호합니다. ")
    else:
        report.append("### ❌ 개선 필요한 탐지 정확도")
        report.append(f"전체 정확도가 **{metrics['accuracy']*100:.2f}%**로 개선이 필요합니다. ")

    # False positives
    if cm["fp"] > 0:
        fp_rate = cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) > 0 else 0.0
        report.append(f"정상 트래픽을 비정상으로 오탐한 비율(False Positive Rate)이 **{fp_rate*100:.2f}%**입니다. ")

    # False negatives
    if cm["fn"] > 0:
        fn_rate = cm["fn"] / (cm["fn"] + cm["tp"]) if (cm["fn"] + cm["tp"]) > 0 else 0.0
        report.append(f"비정상 트래픽을 놓친 비율(False Negative Rate)이 **{fn_rate*100:.2f}%**입니다. ")

    report.append("")

    # BGP-specific insights
    report.append("### BGP 특화 분석")
    report.append("")

    # AS-path analysis
    if "as_path_length" in df.columns:
        avg_path_normal = df[df["ground_truth"] == False]["as_path_length"].mean()
        avg_path_anomaly = df[df["ground_truth"] == True]["as_path_length"].mean()

        report.append(f"- **AS-path 길이**: 정상 {avg_path_normal:.1f}, 비정상 {avg_path_anomaly:.1f}")
        report.append(f"  - 비정상 트래픽에서 AS-path가 {avg_path_anomaly - avg_path_normal:.1f} 더 긴 경향")

    # Update rate analysis
    if "update_delta" in df.columns:
        avg_update_normal = df[df["ground_truth"] == False]["update_delta"].mean()
        avg_update_anomaly = df[df["ground_truth"] == True]["update_delta"].mean()

        report.append(f"- **UPDATE 메시지 빈도**: 정상 {avg_update_normal:.1f}/주기, 비정상 {avg_update_anomaly:.1f}/주기")
        if avg_update_anomaly > avg_update_normal * 2:
            report.append(f"  - 비정상 트래픽에서 UPDATE 메시지가 크게 증가")

    report.append("")

    # Prediction vs Actual Comparison
    report.append("## 예측 vs 실제 비교 분석")
    report.append("")

    total = len(df)
    true_anomalies = df["ground_truth"].sum()
    pred_anomalies = df["predicted_anomaly"].sum()
    true_normal = total - true_anomalies
    pred_normal = total - pred_anomalies

    cm = metrics["confusion_matrix"]
    matches = cm["tp"] + cm["tn"]

    report.append("| 구분 | 정상 | 이상 | 합계 |")
    report.append("|------|------|------|------|")
    report.append(f"| **실제 (Ground Truth)** | {true_normal}개 ({true_normal/total*100:.1f}%) | {true_anomalies}개 ({true_anomalies/total*100:.1f}%) | {total}개 |")
    report.append(f"| **예측 (Predicted)** | {pred_normal}개 ({pred_normal/total*100:.1f}%) | {pred_anomalies}개 ({pred_anomalies/total*100:.1f}%) | {total}개 |")
    report.append(f"| **일치 여부** | TN: {cm['tn']}개 | TP: {cm['tp']}개 | 일치: {matches}개 ({matches/total*100:.1f}%) |")
    report.append("")

    # Error analysis
    if cm["fp"] > 0 or cm["fn"] > 0:
        report.append("### 오류 분석")
        report.append("")
        if cm["fp"] > 0:
            report.append(f"- **False Positive (오탐)**: {cm['fp']}건 ({cm['fp']/total*100:.1f}%) - 정상을 이상으로 오판")
        if cm["fn"] > 0:
            report.append(f"- **False Negative (미탐)**: {cm['fn']}건 ({cm['fn']/total*100:.1f}%) - 이상을 정상으로 오판")
        report.append("")

    # Footer
    report.append("---")
    report.append("")
    report.append("*이 리포트는 자동으로 생성되었습니다.*")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Generate BGP anomaly detection report",
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
        default=Path("results/bgp/report.md"),
        help="Output report file (default: results/bgp/report.md)",
    )

    args = parser.parse_args()

    # Validate
    if not args.predictions.exists():
        print(f"ERROR: Predictions file not found: {args.predictions}")
        sys.exit(1)

    # Load predictions
    try:
        df = load_predictions(args.predictions)
    except Exception as e:
        print(f"ERROR: Failed to load predictions: {e}")
        sys.exit(1)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(df)

    # Analyze anomaly types
    print("Analyzing anomaly types...")
    anomaly_types = analyze_anomaly_types(df)

    # Generate report
    print("Generating report...")
    report = generate_report(df, metrics, anomaly_types)

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {args.output}")

    # Print summary
    print("\n" + "="*60)
    print("REPORT SUMMARY")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-score:  {metrics['f1_score']*100:.2f}%")

    if anomaly_types:
        print(f"\nAnomal types analyzed: {len(anomaly_types)}")
        for anom_type, stats in anomaly_types.items():
            print(f"  {anom_type}: F1={stats['f1_score']*100:.1f}%")


if __name__ == "__main__":
    main()
