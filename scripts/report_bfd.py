#!/usr/bin/env python3
"""Generate detailed performance report for BFD anomaly detection.

This script analyzes inference results and generates comprehensive
reports in both Korean (Markdown) and visualization formats.

Usage:
    # Generate report from predictions
    python scripts/report_bfd.py --predictions results/bfd/predictions.csv

    # Custom output location
    python scripts/report_bfd.py --predictions results/bfd/predictions.csv \
        --output results/bfd/report.md
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load predictions from CSV.

    Args:
        predictions_path: Path to predictions CSV

    Returns:
        Predictions DataFrame
    """
    print(f"Loading predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path)
    print(f"Loaded {len(df):,} predictions")
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive performance metrics.

    Args:
        df: Predictions DataFrame

    Returns:
        Dictionary of metrics and statistics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
    )

    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values
    y_score = df["anomaly_score"].values

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Calculate ROC-AUC if we have both classes
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_score)
        else:
            roc_auc = None
    except:
        roc_auc = None

    # Detailed statistics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": len(df),
        "n_true_anomalies": int(y_true.sum()),
        "n_pred_anomalies": int(y_pred.sum()),
    }


def generate_markdown_report(df: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    """Generate Korean Markdown report.

    Args:
        df: Predictions DataFrame
        metrics: Performance metrics
        output_path: Output file path
    """
    print(f"\nGenerating Markdown report...")

    # Interpretation helpers
    def interpret_metric(value: float, metric_type: str) -> str:
        """Interpret metric value in Korean."""
        if metric_type == "accuracy":
            if value >= 0.95:
                return "매우 우수한"
            elif value >= 0.90:
                return "우수한"
            elif value >= 0.80:
                return "양호한"
            elif value >= 0.70:
                return "보통의"
            else:
                return "개선이 필요한"
        elif metric_type == "precision":
            if value >= 0.95:
                return "오탐지가 거의 없는 매우 신뢰성 높은"
            elif value >= 0.85:
                return "신뢰성 높은"
            elif value >= 0.70:
                return "적절한"
            else:
                return "오탐지가 많은"
        elif metric_type == "recall":
            if value >= 0.95:
                return "이상을 거의 모두 탐지하는"
            elif value >= 0.85:
                return "대부분의 이상을 탐지하는"
            elif value >= 0.70:
                return "많은 이상을 탐지하는"
            else:
                return "이상 탐지율이 낮은"
        return ""

    # Get example predictions
    true_positives = df[(df["true_label"] == True) & (df["predicted_label"] == True)].head(3)
    true_negatives = df[(df["true_label"] == False) & (df["predicted_label"] == False)].head(3)
    false_positives = df[(df["true_label"] == False) & (df["predicted_label"] == True)].head(3)
    false_negatives = df[(df["true_label"] == True) & (df["predicted_label"] == False)].head(3)

    # Generate report
    report = []
    report.append("# BFD 프로토콜 이상 탐지 성능 리포트\n")
    report.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n\n")

    # Executive Summary
    report.append("## 요약\n\n")
    acc_interp = interpret_metric(metrics["accuracy"], "accuracy")
    report.append(
        f"BFD 세션 상태 이상 탐지 모델을 검증한 결과, "
        f"**{acc_interp} 정확도 {metrics['accuracy']*100:.1f}%**를 달성했습니다. "
    )

    if metrics["f1_score"] > 0:
        report.append(f"F1-score는 **{metrics['f1_score']*100:.1f}%**로, ")
        if metrics["f1_score"] >= 0.85:
            report.append("이상 탐지와 정상 판별의 균형이 매우 우수합니다.\n\n")
        elif metrics["f1_score"] >= 0.70:
            report.append("이상 탐지와 정상 판별의 균형이 양호합니다.\n\n")
        else:
            report.append("이상 탐지와 정상 판별 간 균형을 개선할 여지가 있습니다.\n\n")
    else:
        report.append("\n\n")

    # Dataset
    report.append("## 데이터셋\n\n")
    report.append(f"- **전체 샘플 수**: {metrics['total']:,}개\n")
    report.append(f"- **실제 정상**: {metrics['total'] - metrics['n_true_anomalies']:,}개 "
                  f"({(metrics['total'] - metrics['n_true_anomalies'])/metrics['total']*100:.1f}%)\n")
    report.append(f"- **실제 이상**: {metrics['n_true_anomalies']:,}개 "
                  f"({metrics['n_true_anomalies']/metrics['total']*100:.1f}%)\n\n")

    # Performance Metrics
    report.append("## 성능 지표\n\n")
    report.append("| 지표 | 값 | 해석 |\n")
    report.append("|------|-----|------|\n")

    report.append(f"| **Accuracy (정확도)** | {metrics['accuracy']*100:.2f}% | "
                  f"전체 예측 중 올바른 예측의 비율. {acc_interp} 수준입니다. |\n")

    prec_interp = interpret_metric(metrics["precision"], "precision")
    report.append(f"| **Precision (정밀도)** | {metrics['precision']*100:.2f}% | "
                  f"이상이라고 예측한 것 중 실제 이상의 비율. {prec_interp} 수준입니다. |\n")

    rec_interp = interpret_metric(metrics["recall"], "recall")
    report.append(f"| **Recall (재현율)** | {metrics['recall']*100:.2f}% | "
                  f"실제 이상 중 탐지한 비율. {rec_interp} 수준입니다. |\n")

    report.append(f"| **F1-score** | {metrics['f1_score']*100:.2f}% | "
                  f"Precision과 Recall의 조화평균. 두 지표의 균형을 나타냅니다. |\n")

    if metrics["roc_auc"] is not None:
        report.append(f"| **ROC-AUC** | {metrics['roc_auc']:.4f} | "
                      f"이상/정상 구분 능력. 1.0에 가까울수록 우수합니다. |\n")

    report.append("\n")

    # Confusion Matrix
    report.append("## 혼동 행렬 (Confusion Matrix)\n\n")
    report.append("```\n")
    report.append("                예측: 정상     예측: 이상\n")
    report.append(f"실제 정상:      {metrics['tn']:6d}        {metrics['fp']:6d}  (FP: 오탐)\n")
    report.append(f"실제 이상:      {metrics['fn']:6d}        {metrics['tp']:6d}  (FN: 미탐)\n")
    report.append("```\n\n")

    report.append("- **TN (True Negative)**: 정상을 정상으로 올바르게 판별한 수\n")
    report.append("- **FP (False Positive)**: 정상을 이상으로 잘못 판별한 수 (오탐)\n")
    report.append("- **FN (False Negative)**: 이상을 정상으로 잘못 판별한 수 (미탐)\n")
    report.append("- **TP (True Positive)**: 이상을 이상으로 올바르게 판별한 수\n\n")

    # Interpretation
    report.append("## 결과 해석\n\n")
    report.append("### 강점\n\n")

    strengths = []
    if metrics["accuracy"] >= 0.90:
        strengths.append(f"- 전체 정확도가 {metrics['accuracy']*100:.1f}%로 매우 높음")
    if metrics["precision"] >= 0.85:
        strengths.append(f"- 정밀도가 {metrics['precision']*100:.1f}%로 오탐지가 적음")
    if metrics["recall"] >= 0.85:
        strengths.append(f"- 재현율이 {metrics['recall']*100:.1f}%로 이상 탐지율이 높음")
    if metrics["fp"] == 0:
        strengths.append("- 오탐지(FP)가 없어 신뢰성이 매우 높음")
    if metrics["fn"] == 0:
        strengths.append("- 미탐지(FN)가 없어 모든 이상을 탐지함")

    if strengths:
        report.append("\n".join(strengths) + "\n\n")
    else:
        report.append("- 기본적인 탐지 기능을 수행하고 있음\n\n")

    report.append("### 약점 및 개선 방향\n\n")

    weaknesses = []
    if metrics["precision"] < 0.70:
        weaknesses.append(f"- 정밀도가 {metrics['precision']*100:.1f}%로 낮아 오탐지가 많음")
        weaknesses.append("  - **개선방안**: 임계값(threshold) 조정, 더 많은 학습 데이터 확보")
    if metrics["recall"] < 0.70:
        weaknesses.append(f"- 재현율이 {metrics['recall']*100:.1f}%로 낮아 놓치는 이상이 많음")
        weaknesses.append("  - **개선방안**: 모델 복잡도 증가, 이상 데이터 증강")
    if metrics["fp"] > metrics["total"] * 0.1:
        weaknesses.append(f"- 오탐지(FP)가 {metrics['fp']:,}건으로 많음 ({metrics['fp']/metrics['total']*100:.1f}%)")
        weaknesses.append("  - **개선방안**: 임계값 상향 조정, 정상 데이터 학습 강화")
    if metrics["fn"] > metrics["total"] * 0.1:
        weaknesses.append(f"- 미탐지(FN)가 {metrics['fn']:,}건으로 많음")
        weaknesses.append("  - **개선방안**: 다양한 이상 패턴 학습, 앙상블 모델 고려")

    if weaknesses:
        report.append("\n".join(weaknesses) + "\n\n")
    else:
        report.append("- 현재 모델의 성능이 우수하여 특별한 개선이 필요하지 않음\n\n")

    # Examples
    report.append("## 예측 샘플\n\n")

    if len(true_positives) > 0:
        report.append("### True Positive (이상을 올바르게 탐지한 경우)\n\n")
        report.append("| 시간 | 소스 | 값 | 이상 점수 |\n")
        report.append("|------|------|-----|----------|\n")
        for _, row in true_positives.iterrows():
            report.append(f"| {row['timestamp']} | {row['source_id']} | {row['value']:.2f} | {row['anomaly_score']:.3f} |\n")
        report.append("\n")

    if len(true_negatives) > 0:
        report.append("### True Negative (정상을 올바르게 판별한 경우)\n\n")
        report.append("| 시간 | 소스 | 값 | 이상 점수 |\n")
        report.append("|------|------|-----|----------|\n")
        for _, row in true_negatives.head(3).iterrows():
            report.append(f"| {row['timestamp']} | {row['source_id']} | {row['value']:.2f} | {row['anomaly_score']:.3f} |\n")
        report.append("\n")

    if len(false_positives) > 0:
        report.append("### False Positive (오탐: 정상을 이상으로 잘못 판별)\n\n")
        report.append("| 시간 | 소스 | 값 | 이상 점수 |\n")
        report.append("|------|------|-----|----------|\n")
        for _, row in false_positives.iterrows():
            report.append(f"| {row['timestamp']} | {row['source_id']} | {row['value']:.2f} | {row['anomaly_score']:.3f} |\n")
        report.append("\n")

    if len(false_negatives) > 0:
        report.append("### False Negative (미탐: 이상을 정상으로 잘못 판별)\n\n")
        report.append("| 시간 | 소스 | 값 | 이상 점수 |\n")
        report.append("|------|------|-----|----------|\n")
        for _, row in false_negatives.iterrows():
            report.append(f"| {row['timestamp']} | {row['source_id']} | {row['value']:.2f} | {row['anomaly_score']:.3f} |\n")
        report.append("\n")

    # Recommendations
    report.append("## 권장사항\n\n")

    if metrics["accuracy"] >= 0.90 and metrics["f1_score"] >= 0.85:
        report.append("### ✅ 프로덕션 배포 가능\n\n")
        report.append("현재 모델의 성능이 우수하여 실제 환경에 배포할 수 있는 수준입니다.\n\n")
        report.append("**다음 단계**:\n")
        report.append("1. 실제 BFD 세션 데이터로 추가 검증\n")
        report.append("2. 프로덕션 환경 성능 모니터링 체계 구축\n")
        report.append("3. 정기적인 재학습 주기 설정 (예: 월 1회)\n\n")
    elif metrics["accuracy"] >= 0.80:
        report.append("### ⚠️  개선 후 배포 권장\n\n")
        report.append("기본적인 성능은 달성했으나, 추가 개선이 필요합니다.\n\n")
        report.append("**개선 방안**:\n")
        report.append("1. 하이퍼파라미터 튜닝 (learning rate, hidden size, epochs)\n")
        report.append("2. 더 많은 학습 데이터 수집 (현재보다 2-3배)\n")
        report.append("3. 데이터 증강 기법 적용\n\n")
    else:
        report.append("### ❌ 추가 개발 필요\n\n")
        report.append("현재 성능으로는 프로덕션 배포가 어렵습니다.\n\n")
        report.append("**필수 개선 작업**:\n")
        report.append("1. 모델 아키텍처 재검토 (LSTM → Transformer, Ensemble 등)\n")
        report.append("2. 학습 데이터 품질 점검 및 정제\n")
        report.append("3. 피처 엔지니어링 강화\n")
        report.append("4. 전문가와 함께 이상 패턴 재정의\n\n")

    # Footer
    report.append("---\n\n")
    report.append("**문서 정보**\n\n")
    report.append(f"- 생성 도구: OCAD BFD Anomaly Detection Report Generator\n")
    report.append(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"- 버전: 1.0.0\n")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(report)

    print(f"✅ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BFD anomaly detection performance report",
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
        default=Path("results/bfd/report.md"),
        help="Output report path (default: results/bfd/report.md)",
    )

    args = parser.parse_args()

    # Load predictions
    df = load_predictions(args.predictions)

    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics = calculate_metrics(df)

    # Generate report
    generate_markdown_report(df, metrics, args.output)

    print(f"\n✅ Report generation completed!")
    print(f"\nGenerated files:")
    print(f"  - {args.output}")
    print(f"\nView the report:")
    print(f"  cat {args.output}")


if __name__ == "__main__":
    main()
