#!/usr/bin/env python3
"""Validate BFD training data quality.

This script validates that BFD training data follows the expected patterns:
1. Normal data (is_anomaly=False) should only have UP states
2. Anomaly data (is_anomaly=True) should have DOWN states or flapping
3. Metric ranges should be consistent
4. flap_count should be monotonically increasing per session

Usage:
    python scripts/validate_bfd_training_data.py --data data/bfd/train
    python scripts/validate_bfd_training_data.py --data data/bfd/train/bfd_data_*.parquet
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any


# BFD State definitions
class BFDState:
    ADMIN_DOWN = 0
    DOWN = 1
    INIT = 2
    UP = 3

    NAMES = {
        0: "ADMIN_DOWN",
        1: "DOWN",
        2: "INIT",
        3: "UP",
    }


def load_data(data_path: Path) -> pd.DataFrame:
    """Load BFD data from parquet file(s)."""
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
    """Validate BFD data quality.

    Returns:
        Validation report with issues found
    """
    issues = []
    warnings = []

    # Rule 1: Normal data should only have UP states
    normal_data = df[df["is_anomaly"] == False]
    non_up_in_normal = normal_data[normal_data["local_state"] != BFDState.UP]

    if len(non_up_in_normal) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "Normal data (is_anomaly=False) must only have UP states (local_state=3)",
            "found": f"{len(non_up_in_normal)} records with local_state != UP",
            "percentage": f"{len(non_up_in_normal) / len(normal_data) * 100:.2f}%",
            "sample_states": non_up_in_normal["local_state_name"].value_counts().to_dict(),
            "samples": non_up_in_normal[["timestamp", "source_id", "local_state_name", "is_anomaly", "flap_count"]].head(5).to_dict("records"),
        })

    # Rule 2: Anomaly data should have DOWN states or flapping
    anomaly_data = df[df["is_anomaly"] == True]
    if len(anomaly_data) > 0:
        up_no_flap_in_anomaly = anomaly_data[
            (anomaly_data["local_state"] == BFDState.UP) & (anomaly_data["flap_count"] == 0)
        ]

        if len(up_no_flap_in_anomaly) > 0:
            warnings.append({
                "severity": "WARNING",
                "rule": "Anomaly data should have DOWN states or flapping",
                "found": f"{len(up_no_flap_in_anomaly)} records with UP state and no flapping",
                "percentage": f"{len(up_no_flap_in_anomaly) / len(anomaly_data) * 100:.2f}%",
            })

    # Rule 3: Check flap_count consistency (should only increase by 0 or 1)
    for source_id in df["source_id"].unique():
        session_data = df[df["source_id"] == source_id].sort_values("timestamp")
        flap_diffs = session_data["flap_count"].diff().fillna(0)
        invalid_flaps = (flap_diffs < 0) | (flap_diffs > 1)

        if invalid_flaps.sum() > 0:
            warnings.append({
                "severity": "WARNING",
                "rule": f"flap_count for {source_id} should only increase by 0 or 1",
                "found": f"{invalid_flaps.sum()} invalid transitions",
            })

    # Rule 4: Check detection_time_ms ranges
    normal_detection_times = normal_data["detection_time_ms"]
    if len(normal_detection_times) > 0:
        if normal_detection_times.max() > 100:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal data should have detection_time_ms < 100ms",
                "found": f"Max detection time: {normal_detection_times.max():.2f}ms",
            })

    # Rule 5: Check for duplicate timestamps
    duplicates = df.groupby(["timestamp", "source_id"]).size()
    duplicate_count = (duplicates > 1).sum()
    if duplicate_count > 0:
        warnings.append({
            "severity": "WARNING",
            "rule": "No duplicate (timestamp, source_id) pairs",
            "found": f"{duplicate_count} duplicate pairs",
        })

    return {
        "total_records": len(df),
        "normal_records": len(normal_data),
        "anomaly_records": len(anomaly_data),
        "unique_sessions": df["source_id"].nunique(),
        "state_distribution": df["local_state_name"].value_counts().to_dict(),
        "normal_state_distribution": normal_data["local_state_name"].value_counts().to_dict() if len(normal_data) > 0 else {},
        "anomaly_state_distribution": anomaly_data["local_state_name"].value_counts().to_dict() if len(anomaly_data) > 0 else {},
        "issues": issues,
        "warnings": warnings,
        "status": "FAIL" if len(issues) > 0 else ("WARNING" if len(warnings) > 0 else "PASS"),
    }


def print_report(report: Dict[str, Any]) -> None:
    """Print validation report."""
    print("=" * 80)
    print("BFD TRAINING DATA VALIDATION REPORT")
    print("=" * 80)
    print()

    # Summary
    print("📊 Dataset Summary")
    print("-" * 80)
    print(f"Total records:        {report['total_records']:,}")
    print(f"Normal records:       {report['normal_records']:,} ({report['normal_records']/report['total_records']*100:.1f}%)")
    print(f"Anomaly records:      {report['anomaly_records']:,} ({report['anomaly_records']/report['total_records']*100:.1f}%)")
    print(f"Unique sessions:      {report['unique_sessions']}")
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

    # Issues
    if report["issues"]:
        print("❌ CRITICAL ISSUES")
        print("=" * 80)
        for i, issue in enumerate(report["issues"], 1):
            print(f"\n{i}. {issue['rule']}")
            print(f"   Severity: {issue['severity']}")
            print(f"   Found: {issue['found']}")
            if "percentage" in issue:
                print(f"   Percentage: {issue['percentage']}")
            if "sample_states" in issue:
                print(f"   State breakdown:")
                for state, count in issue["sample_states"].items():
                    print(f"     - {state}: {count}")
            if "samples" in issue:
                print(f"\n   Sample records:")
                for sample in issue["samples"]:
                    print(f"     {sample}")
        print()

    # Warnings
    if report["warnings"]:
        print("⚠️  WARNINGS")
        print("=" * 80)
        for i, warning in enumerate(report["warnings"], 1):
            print(f"\n{i}. {warning['rule']}")
            print(f"   Severity: {warning['severity']}")
            print(f"   Found: {warning['found']}")
        print()

    # Status
    print("=" * 80)
    status_emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}
    print(f"{status_emoji[report['status']]} VALIDATION STATUS: {report['status']}")
    print("=" * 80)

    if report["status"] == "PASS":
        print("\nAll validation checks passed! Data quality is good.")
    elif report["status"] == "WARNING":
        print("\nValidation passed with warnings. Review warnings above.")
    else:
        print("\n❌ Validation FAILED! Critical issues found. Data cannot be used for training.")
        print("Please fix the data generation logic and regenerate the dataset.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate BFD training data quality",
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

    # Load data
    df = load_data(args.data)

    # Validate
    report = validate_data(df)

    # Print report
    print_report(report)

    # Exit with appropriate code
    if report["status"] == "FAIL":
        sys.exit(1)
    elif report["status"] == "WARNING":
        sys.exit(0)  # Warnings don't fail
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
