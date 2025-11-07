#!/usr/bin/env python3
"""Validate PTP training data quality.

This script validates that PTP training data follows the expected patterns:
1. Normal data should only have SLAVE states
2. Normal data metrics should be within expected ranges
3. Anomaly data should have abnormal states or metrics

Usage:
    python scripts/validate_ptp_training_data.py --data data/ptp/train
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
from typing import Dict, List, Any


class PTPPortState:
    INITIALIZING = 1
    FAULTY = 2
    DISABLED = 3
    LISTENING = 4
    PRE_MASTER = 5
    MASTER = 6
    PASSIVE = 7
    UNCALIBRATED = 8
    SLAVE = 9


def load_data(data_path: Path) -> pd.DataFrame:
    """Load PTP data from parquet file(s)."""
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
    """Validate PTP data quality."""
    issues = []
    warnings = []

    # Rule 1: Normal data should only have SLAVE states
    normal_data = df[df["is_anomaly"] == False]
    non_slave = normal_data[normal_data["port_state"] != PTPPortState.SLAVE]

    if len(non_slave) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "Normal data must only have SLAVE states (port_state=9)",
            "found": f"{len(non_slave)} records with non-SLAVE state",
            "percentage": f"{len(non_slave) / len(normal_data) * 100:.2f}%",
            "states": non_slave["port_state_name"].value_counts().to_dict(),
        })

    # Rule 2: Check offset_from_master_ns range (should be within ±1000ns for normal)
    if len(normal_data) > 0:
        large_offset = normal_data[abs(normal_data["offset_from_master_ns"]) > 1000]
        if len(large_offset) > 0:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal offset_from_master_ns should be within ±1000ns",
                "found": f"{len(large_offset)} records with |offset| > 1000ns",
                "range": f"[{normal_data['offset_from_master_ns'].min():.2f}, {normal_data['offset_from_master_ns'].max():.2f}]",
            })

    # Rule 3: Check mean_path_delay_ns range (should be 1000-20000ns for normal)
    if len(normal_data) > 0:
        path_delay_range = (normal_data["mean_path_delay_ns"].min(), normal_data["mean_path_delay_ns"].max())
        if path_delay_range[1] > 20000:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal mean_path_delay_ns should be < 20000ns",
                "found": f"Max: {path_delay_range[1]:.2f}ns",
            })

    # Rule 4: Check clock_drift_ppb range (should be within ±50ppb for normal)
    if len(normal_data) > 0:
        large_drift = normal_data[abs(normal_data["clock_drift_ppb"]) > 50]
        if len(large_drift) > 0:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal clock_drift_ppb should be within ±50ppb",
                "found": f"{len(large_drift)} records with |drift| > 50ppb",
                "range": f"[{normal_data['clock_drift_ppb'].min():.2f}, {normal_data['clock_drift_ppb'].max():.2f}]",
            })

    # Rule 5: Check steps_removed (should be 1 for direct slaves)
    if len(normal_data) > 0:
        steps_range = (normal_data["steps_removed"].min(), normal_data["steps_removed"].max())
        if steps_range[1] > 3:
            warnings.append({
                "severity": "WARNING",
                "rule": "steps_removed should typically be 1-3",
                "found": f"Max: {steps_range[1]}",
            })

    return {
        "total_records": len(df),
        "normal_records": len(normal_data),
        "anomaly_records": len(df) - len(normal_data),
        "unique_slaves": df["source_id"].nunique(),
        "state_distribution": df["port_state_name"].value_counts().to_dict(),
        "normal_state_distribution": normal_data["port_state_name"].value_counts().to_dict() if len(normal_data) > 0 else {},
        "normal_metrics": {
            "offset_range": f"[{normal_data['offset_from_master_ns'].min():.2f}, {normal_data['offset_from_master_ns'].max():.2f}]" if len(normal_data) > 0 else "N/A",
            "path_delay_range": f"[{normal_data['mean_path_delay_ns'].min():.2f}, {normal_data['mean_path_delay_ns'].max():.2f}]" if len(normal_data) > 0 else "N/A",
            "drift_range": f"[{normal_data['clock_drift_ppb'].min():.2f}, {normal_data['clock_drift_ppb'].max():.2f}]" if len(normal_data) > 0 else "N/A",
        },
        "issues": issues,
        "warnings": warnings,
        "status": "FAIL" if len(issues) > 0 else ("WARNING" if len(warnings) > 0 else "PASS"),
    }


def print_report(report: Dict[str, Any]) -> None:
    """Print validation report."""
    print("=" * 80)
    print("PTP TRAINING DATA VALIDATION REPORT")
    print("=" * 80)
    print()

    # Summary
    print("📊 Dataset Summary")
    print("-" * 80)
    print(f"Total records:        {report['total_records']:,}")
    print(f"Normal records:       {report['normal_records']:,} ({report['normal_records']/report['total_records']*100:.1f}%)")
    print(f"Anomaly records:      {report['anomaly_records']:,} ({report['anomaly_records']/report['total_records']*100:.1f}%)")
    print(f"Unique slaves:        {report['unique_slaves']}")
    print()

    # State distribution
    print("📈 State Distribution (Overall)")
    print("-" * 80)
    for state, count in report["state_distribution"].items():
        print(f"  {state:15s}: {count:6,} ({count/report['total_records']*100:.1f}%)")
    print()

    if report["normal_state_distribution"]:
        print("✅ State Distribution (Normal Data)")
        print("-" * 80)
        for state, count in report["normal_state_distribution"].items():
            print(f"  {state:15s}: {count:6,} ({count/report['normal_records']*100:.1f}%)")
        print()

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
        description="Validate PTP training data quality",
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
