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


def analyze_training_data(train_data_path: Path = None) -> dict:
    """Analyze training data quality if available.

    Args:
        train_data_path: Path to training data directory

    Returns:
        Dictionary of training data statistics, or None if not available
    """
    if train_data_path is None or not train_data_path.exists():
        return None

    try:
        import glob
        parquet_files = glob.glob(str(train_data_path / "*.parquet"))
        if not parquet_files:
            return None

        # Load all training data
        dfs = [pd.read_parquet(f) for f in parquet_files]
        train_df = pd.concat(dfs, ignore_index=True)

        # Analyze
        total = len(train_df)
        normal = len(train_df[train_df["is_anomaly"] == False])
        anomaly = len(train_df[train_df["is_anomaly"] == True])

        # State distribution
        state_dist = train_df["local_state"].value_counts().to_dict()

        return {
            "total": total,
            "normal": normal,
            "anomaly": anomaly,
            "anomaly_rate": anomaly / total if total > 0 else 0,
            "state_distribution": state_dist,
            "has_issues": (anomaly == 0) or (total < 500),
        }
    except Exception as e:
        print(f"Warning: Could not analyze training data: {e}")
        return None


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


def generate_markdown_report(
    df: pd.DataFrame, metrics: dict, train_stats: dict, output_path: Path
) -> None:
    """Generate Korean Markdown report.

    Args:
        df: Predictions DataFrame
        metrics: Performance metrics
        train_stats: Training data statistics (optional)
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

    # Get example predictions (more samples for better analysis)
    true_positives = df[(df["true_label"] == True) & (df["predicted_label"] == True)].head(10)
    true_negatives = df[(df["true_label"] == False) & (df["predicted_label"] == False)].head(10)
    false_positives = df[(df["true_label"] == False) & (df["predicted_label"] == True)].head(10)
    false_negatives = df[(df["true_label"] == True) & (df["predicted_label"] == False)].head(10)

    # Analyze error patterns
    fp_values = df[df["true_label"] == False & (df["predicted_label"] == True)]["value"] if len(false_positives) > 0 else pd.Series()
    fn_values = df[df["true_label"] == True & (df["predicted_label"] == False)]["value"] if len(false_negatives) > 0 else pd.Series()

    # Extract model state from evidence field
    def extract_model_info(df):
        """Extract threshold and likelihood from evidence field."""
        try:
            import json
            import ast

            thresholds = []
            likelihoods = []

            for evidence_str in df["evidence"].dropna():
                try:
                    # Try to parse as dict
                    if isinstance(evidence_str, str):
                        evidence = ast.literal_eval(evidence_str)
                    else:
                        evidence = evidence_str

                    if isinstance(evidence, dict):
                        if "anomaly_threshold" in evidence:
                            thresholds.append(evidence["anomaly_threshold"])
                        if "log_likelihood" in evidence:
                            likelihoods.append(evidence["log_likelihood"])
                except:
                    continue

            return {
                "threshold": np.mean(thresholds) if thresholds else None,
                "likelihood_mean": np.mean(likelihoods) if likelihoods else None,
                "likelihood_std": np.std(likelihoods) if likelihoods else None,
                "likelihood_min": np.min(likelihoods) if likelihoods else None,
                "likelihood_max": np.max(likelihoods) if likelihoods else None,
            }
        except Exception as e:
            print(f"Warning: Could not extract model info: {e}")
            return {}

    model_info = extract_model_info(df)

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

    # Prediction vs Actual Comparison
    report.append("## 예측 vs 실제 비교 분석\n\n")

    # Overall distribution comparison
    report.append("### 전체 분포 비교\n\n")
    report.append("| 구분 | 정상 | 이상 | 합계 |\n")
    report.append("|------|------|------|------|\n")
    report.append(f"| **실제 (True Label)** | {metrics['total'] - metrics['n_true_anomalies']:,}개 "
                  f"({(metrics['total'] - metrics['n_true_anomalies'])/metrics['total']*100:.1f}%) | "
                  f"{metrics['n_true_anomalies']:,}개 "
                  f"({metrics['n_true_anomalies']/metrics['total']*100:.1f}%) | "
                  f"{metrics['total']:,}개 |\n")
    report.append(f"| **예측 (Predicted)** | {metrics['total'] - metrics['n_pred_anomalies']:,}개 "
                  f"({(metrics['total'] - metrics['n_pred_anomalies'])/metrics['total']*100:.1f}%) | "
                  f"{metrics['n_pred_anomalies']:,}개 "
                  f"({metrics['n_pred_anomalies']/metrics['total']*100:.1f}%) | "
                  f"{metrics['total']:,}개 |\n")
    report.append(f"| **일치 여부** | TN: {metrics['tn']:,}개 | TP: {metrics['tp']:,}개 | "
                  f"일치: {metrics['tn'] + metrics['tp']:,}개 "
                  f"({(metrics['tn'] + metrics['tp'])/metrics['total']*100:.1f}%) |\n")
    report.append("\n")

    # Error pattern analysis
    if len(false_positives) > 0 or len(false_negatives) > 0:
        report.append("### 오류 패턴 분석\n\n")

        if len(false_positives) > 0:
            fp_value_counts = fp_values.value_counts().head(5)
            report.append(f"**False Positive (오탐) 패턴**: 정상 데이터를 이상으로 잘못 판별한 {len(fp_values):,}건 분석\n\n")
            report.append("| BFD 상태 값 | 발생 횟수 | 비율 |\n")
            report.append("|------------|----------|------|\n")
            state_names = {0: "ADMIN_DOWN", 1: "DOWN", 2: "INIT", 3: "UP"}
            for value, count in fp_value_counts.items():
                state_name = state_names.get(int(value), f"Unknown({value})")
                report.append(f"| {state_name} ({int(value)}) | {count:,}건 | {count/len(fp_values)*100:.1f}% |\n")
            report.append("\n")

        if len(false_negatives) > 0:
            fn_value_counts = fn_values.value_counts().head(5)
            report.append(f"**False Negative (미탐) 패턴**: 이상 데이터를 정상으로 잘못 판별한 {len(fn_values):,}건 분석\n\n")
            report.append("| BFD 상태 값 | 발생 횟수 | 비율 |\n")
            report.append("|------------|----------|------|\n")
            state_names = {0: "ADMIN_DOWN", 1: "DOWN", 2: "INIT", 3: "UP"}
            for value, count in fn_value_counts.items():
                state_name = state_names.get(int(value), f"Unknown({value})")
                report.append(f"| {state_name} ({int(value)}) | {count:,}건 | {count/len(fn_values)*100:.1f}% |\n")
            report.append("\n")

    # Model state analysis
    if model_info:
        report.append("## 모델 상태 분석\n\n")

        if model_info.get("threshold") is not None:
            report.append(f"**이상 탐지 임계값 (Threshold)**: `{model_info['threshold']:.6e}`\n\n")

        if model_info.get("likelihood_mean") is not None:
            report.append("**Log-Likelihood 분포**:\n\n")
            report.append("| 통계량 | 값 |\n")
            report.append("|--------|----|\n")
            report.append(f"| 평균 | {model_info['likelihood_mean']:.2e} |\n")
            if model_info.get("likelihood_std") is not None:
                report.append(f"| 표준편차 | {model_info['likelihood_std']:.2e} |\n")
            if model_info.get("likelihood_min") is not None:
                report.append(f"| 최소값 | {model_info['likelihood_min']:.2e} |\n")
            if model_info.get("likelihood_max") is not None:
                report.append(f"| 최대값 | {model_info['likelihood_max']:.2e} |\n")
            report.append("\n")

        # Threshold sensitivity analysis
        if model_info.get("threshold") is not None and model_info.get("likelihood_mean") is not None:
            threshold = model_info["threshold"]
            likelihood_mean = model_info["likelihood_mean"]

            if abs(threshold) < 1e-6:
                report.append("⚠️ **임계값이 거의 0에 가까움** - 모델이 매우 민감하게 반응할 수 있습니다.\n\n")
            elif abs(likelihood_mean) > abs(threshold) * 1000:
                report.append("⚠️ **Likelihood 값이 임계값보다 훨씬 큼** - 대부분 샘플이 이상으로 판정될 수 있습니다.\n\n")

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

    # Training data quality analysis (if available)
    if train_stats is not None:
        report.append("## 학습 데이터 품질 분석\n\n")

        report.append("| 항목 | 값 |\n")
        report.append("|------|----|\n")
        report.append(f"| **전체 샘플 수** | {train_stats['total']:,}개 |\n")
        report.append(f"| **정상 샘플** | {train_stats['normal']:,}개 ({train_stats['normal']/train_stats['total']*100:.1f}%) |\n")
        report.append(f"| **이상 샘플** | {train_stats['anomaly']:,}개 ({train_stats['anomaly_rate']*100:.1f}%) |\n")
        report.append("\n")

        if train_stats.get("state_distribution"):
            report.append("**상태 분포**:\n\n")
            state_names = {0: "ADMIN_DOWN", 1: "DOWN", 2: "INIT", 3: "UP"}
            for state_value, count in sorted(train_stats["state_distribution"].items()):
                state_name = state_names.get(state_value, f"Unknown({state_value})")
                report.append(f"- {state_name} ({state_value}): {count:,}개\n")
            report.append("\n")

        # Quality warnings
        if train_stats.get("has_issues"):
            report.append("### ⚠️ 데이터 품질 경고\n\n")
            issues = []
            if train_stats['anomaly'] == 0:
                issues.append("- ❌ **이상 샘플이 없음**: 모델이 이상 패턴을 학습할 수 없습니다.")
            if train_stats['total'] < 500:
                issues.append(f"- ⚠️ **학습 데이터 부족**: 현재 {train_stats['total']:,}개, 권장 1,000개 이상")
            if train_stats['anomaly_rate'] < 0.1 and train_stats['anomaly'] > 0:
                issues.append(f"- ⚠️ **이상 샘플 비율 낮음**: 현재 {train_stats['anomaly_rate']*100:.1f}%, 권장 15-30%")

            if issues:
                report.append("\n".join(issues) + "\n\n")

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

    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Path to training data directory (optional, for quality analysis)",
    )

    args = parser.parse_args()

    # Load predictions
    df = load_predictions(args.predictions)

    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics = calculate_metrics(df)

    # Analyze training data (if provided)
    train_stats = None
    if args.train_data:
        print(f"\nAnalyzing training data from: {args.train_data}")
        train_stats = analyze_training_data(args.train_data)
        if train_stats:
            print(f"  Training samples: {train_stats['total']:,} "
                  f"(Normal: {train_stats['normal']:,}, Anomaly: {train_stats['anomaly']:,})")
        else:
            print("  Warning: Could not analyze training data")

    # Generate report
    generate_markdown_report(df, metrics, train_stats, args.output)

    print(f"\n✅ Report generation completed!")
    print(f"\nGenerated files:")
    print(f"  - {args.output}")
    print(f"\nView the report:")
    print(f"  cat {args.output}")


if __name__ == "__main__":
    main()
