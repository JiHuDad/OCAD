#!/usr/bin/env python3
"""Validate CFM training data quality.

This script validates that CFM training data follows the expected patterns:
1. Normal data should have metrics within expected ranges
2. Normal data should have low CCM miss rates
3. Anomaly data should show clear metric degradation

Usage:
    python scripts/validate_cfm_training_data.py --data data/cfm/train
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
from typing import Dict, List, Any


def load_data(data_path: Path) -> pd.DataFrame:
    """Load CFM data from parquet file(s)."""
    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        files = sorted(data_path.glob("*.parquet"))
    else:
        raise ValueError(f"Path not found: {data_path}")

    if not files:
        raise ValueError(f"No parquet files found in {data_path}")

    print(f"Loading {len(files)} file(s)...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} records\n")
    return df


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate CFM data quality."""
    issues = []
    warnings = []

    # Rule 1: Check anomaly_type consistency
    normal_data = df[df["is_anomaly"] == False]
    anomaly_data = df[df["is_anomaly"] == True]

    if len(normal_data) > 0 and "anomaly_type" in normal_data.columns:
        non_normal_type = normal_data[normal_data["anomaly_type"] != "normal"]
        if len(non_normal_type) > 0:
            issues.append({
                "severity": "CRITICAL",
                "rule": "Normal data must have anomaly_type='normal'",
                "found": f"{len(non_normal_type)} records with anomaly_type != 'normal'",
                "types": non_normal_type["anomaly_type"].value_counts().to_dict(),
            })

    # Rule 2: Check UDP Echo RTT ranges (normal should be 3-15ms)
    if len(normal_data) > 0:
        high_rtt = normal_data[normal_data["udp_echo_rtt_ms"] > 15]
        if len(high_rtt) > 0:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal UDP Echo RTT should be 3-15ms",
                "found": f"{len(high_rtt)} records with RTT > 15ms",
                "max": f"{normal_data['udp_echo_rtt_ms'].max():.2f}ms",
            })

    # Rule 3: Check eCPRI delay ranges (normal should be 50-250us)
    if len(normal_data) > 0:
        high_delay = normal_data[normal_data["ecpri_delay_us"] > 250]
        if len(high_delay) > 0:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal eCPRI delay should be 50-250us",
                "found": f"{len(high_delay)} records with delay > 250us",
                "max": f"{normal_data['ecpri_delay_us'].max():.2f}us",
            })

    # Rule 4: Check LBM RTT ranges (normal should be 8-18ms)
    if len(normal_data) > 0:
        high_lbm = normal_data[normal_data["lbm_rtt_ms"] > 18]
        if len(high_lbm) > 0:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal LBM RTT should be 8-18ms",
                "found": f"{len(high_lbm)} records with RTT > 18ms",
                "max": f"{normal_data['lbm_rtt_ms'].max():.2f}ms",
            })

    # Rule 5: Check CCM miss count reasonableness
    if len(normal_data) > 0:
        # Normal data should have very few CCM misses
        mean_miss_rate = normal_data["ccm_miss_count"].diff().fillna(0).mean()
        if mean_miss_rate > 0.05:  # More than 5% miss rate
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal data should have low CCM miss rate",
                "found": f"Mean miss rate: {mean_miss_rate:.3f} per sample",
            })

    # Rule 6: Check for negative values
    negative_udp = df[df["udp_echo_rtt_ms"] < 0]
    negative_ecpri = df[df["ecpri_delay_us"] < 0]
    negative_lbm = df[df["lbm_rtt_ms"] < 0]
    negative_ccm = df[df["ccm_miss_count"] < 0]

    if len(negative_udp) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "udp_echo_rtt_ms must be non-negative",
            "found": f"{len(negative_udp)} records",
        })

    if len(negative_ecpri) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "ecpri_delay_us must be non-negative",
            "found": f"{len(negative_ecpri)} records",
        })

    if len(negative_lbm) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "lbm_rtt_ms must be non-negative",
            "found": f"{len(negative_lbm)} records",
        })

    if len(negative_ccm) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "ccm_miss_count must be non-negative",
            "found": f"{len(negative_ccm)} records",
        })

    # Rule 7: Check anomaly data has elevated metrics
    if len(anomaly_data) > 0 and len(normal_data) > 0:
        normal_mean_rtt = normal_data["udp_echo_rtt_ms"].mean()
        anomaly_mean_rtt = anomaly_data["udp_echo_rtt_ms"].mean()

        if anomaly_mean_rtt <= normal_mean_rtt * 1.5:
            warnings.append({
                "severity": "WARNING",
                "rule": "Anomaly data should have elevated RTT (>1.5x normal)",
                "found": f"Anomaly mean: {anomaly_mean_rtt:.2f}ms vs Normal mean: {normal_mean_rtt:.2f}ms",
            })

    return {
        "total_records": len(df),
        "normal_records": len(normal_data),
        "anomaly_records": len(anomaly_data),
        "unique_endpoints": df["source_id"].nunique(),
        "normal_metrics": {
            "udp_rtt_range": f"[{normal_data['udp_echo_rtt_ms'].min():.2f}, {normal_data['udp_echo_rtt_ms'].max():.2f}] ms" if len(normal_data) > 0 else "N/A",
            "ecpri_delay_range": f"[{normal_data['ecpri_delay_us'].min():.2f}, {normal_data['ecpri_delay_us'].max():.2f}] us" if len(normal_data) > 0 else "N/A",
            "lbm_rtt_range": f"[{normal_data['lbm_rtt_ms'].min():.2f}, {normal_data['lbm_rtt_ms'].max():.2f}] ms" if len(normal_data) > 0 else "N/A",
            "ccm_miss_range": f"[{normal_data['ccm_miss_count'].min()}, {normal_data['ccm_miss_count'].max()}]" if len(normal_data) > 0 else "N/A",
        },
        "issues": issues,
        "warnings": warnings,
        "status": "FAIL" if len(issues) > 0 else ("WARNING" if len(warnings) > 0 else "PASS"),
    }


def print_report(report: Dict[str, Any]) -> None:
    """Print validation report."""
    print("=" * 80)
    print("CFM TRAINING DATA VALIDATION REPORT")
    print("=" * 80)
    print()

    # Summary
    print("📊 Dataset Summary")
    print("-" * 80)
    print(f"Total records:        {report['total_records']:,}")
    print(f"Normal records:       {report['normal_records']:,} ({report['normal_records']/report['total_records']*100:.1f}%)")
    print(f"Anomaly records:      {report['anomaly_records']:,} ({report['anomaly_records']/report['total_records']*100:.1f}%)")
    print(f"Unique endpoints:     {report['unique_endpoints']}")
    print()

    if report["normal_metrics"]:
        print("📏 Metric Ranges (Normal Data)")
        print("-" * 80)
        for metric, range_val in report["normal_metrics"].items():
            print(f"  {metric:20s}: {range_val}")
        print()

    # Issues
    if report["issues"]:
        print("❌ CRITICAL ISSUES")
        print("=" * 80)
        for i, issue in enumerate(report["issues"], 1):
            print(f"\n{i}. {issue['rule']}")
            print(f"   Severity: {issue['severity']}")
            print(f"   Found: {issue['found']}")
        print()

    # Warnings
    if report["warnings"]:
        print("⚠️  WARNINGS")
        print("=" * 80)
        for i, warning in enumerate(report["warnings"], 1):
            print(f"\n{i}. {warning['rule']}")
            print(f"   Found: {warning['found']}")
        print()

    # Status
    print("=" * 80)
    status_emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}
    print(f"{status_emoji[report['status']]} VALIDATION STATUS: {report['status']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Validate CFM training data quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data file or directory",
    )

    args = parser.parse_args()

    df = load_data(args.data)
    report = validate_data(df)
    print_report(report)

    sys.exit(1 if report["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
