#!/usr/bin/env python3
"""Validate BGP training data quality.

This script validates that BGP training data follows the expected patterns:
1. Normal data should only have ESTABLISHED states
2. Anomaly data should have non-ESTABLISHED states or anomalous metrics
3. Metric ranges should be reasonable
4. No negative counters

Usage:
    python scripts/validate_bgp_training_data.py --data data/bgp/train
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
from typing import Dict, List, Any


class BGPState:
    IDLE = 0
    CONNECT = 1
    ACTIVE = 2
    OPEN_SENT = 3
    OPEN_CONFIRM = 4
    ESTABLISHED = 5


def load_data(data_path: Path) -> pd.DataFrame:
    """Load BGP data from parquet file(s)."""
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
    """Validate BGP data quality."""
    issues = []
    warnings = []

    # Rule 1: Normal data should only have ESTABLISHED states
    normal_data = df[df["is_anomaly"] == False]
    non_established = normal_data[normal_data["session_state"] != BGPState.ESTABLISHED]

    if len(non_established) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "Normal data must only have ESTABLISHED states (session_state=5)",
            "found": f"{len(non_established)} records with non-ESTABLISHED state",
            "percentage": f"{len(non_established) / len(normal_data) * 100:.2f}%",
            "states": non_established["session_state_name"].value_counts().to_dict(),
        })

    # Rule 2: Check for negative counters
    negative_update = df[df["update_count"] < 0]
    negative_withdraw = df[df["withdraw_count"] < 0]
    negative_prefix = df[df["prefix_count"] < 0]

    if len(negative_update) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "update_count must be non-negative",
            "found": f"{len(negative_update)} records with negative update_count",
        })

    if len(negative_withdraw) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "withdraw_count must be non-negative",
            "found": f"{len(negative_withdraw)} records with negative withdraw_count",
        })

    if len(negative_prefix) > 0:
        issues.append({
            "severity": "CRITICAL",
            "rule": "prefix_count must be non-negative",
            "found": f"{len(negative_prefix)} records with negative prefix_count",
        })

    # Rule 3: Check AS path length reasonableness
    long_as_path = df[df["as_path_length"] > 20]
    if len(long_as_path) > 0:
        warnings.append({
            "severity": "WARNING",
            "rule": "AS path length should typically be < 20",
            "found": f"{len(long_as_path)} records with as_path_length > 20",
            "max_length": int(df["as_path_length"].max()),
        })

    # Rule 4: Check prefix count ranges
    if len(normal_data) > 0:
        normal_prefix_range = (normal_data["prefix_count"].min(), normal_data["prefix_count"].max())
        if normal_prefix_range[1] > 1000:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal prefix count seems high",
                "found": f"Range: {normal_prefix_range}",
            })

    # Rule 5: Check update/withdraw rates
    if "update_delta" in df.columns and "withdraw_delta" in df.columns:
        high_update = normal_data[normal_data["update_delta"] > 20]
        high_withdraw = normal_data[normal_data["withdraw_delta"] > 20]

        if len(high_update) > 0:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal data should have low update rates",
                "found": f"{len(high_update)} records with update_delta > 20",
            })

        if len(high_withdraw) > 0:
            warnings.append({
                "severity": "WARNING",
                "rule": "Normal data should have low withdraw rates",
                "found": f"{len(high_withdraw)} records with withdraw_delta > 20",
            })

    return {
        "total_records": len(df),
        "normal_records": len(normal_data),
        "anomaly_records": len(df) - len(normal_data),
        "unique_peers": df["source_id"].nunique(),
        "state_distribution": df["session_state_name"].value_counts().to_dict(),
        "normal_state_distribution": normal_data["session_state_name"].value_counts().to_dict() if len(normal_data) > 0 else {},
        "issues": issues,
        "warnings": warnings,
        "status": "FAIL" if len(issues) > 0 else ("WARNING" if len(warnings) > 0 else "PASS"),
    }


def print_report(report: Dict[str, Any]) -> None:
    """Print validation report."""
    print("=" * 80)
    print("BGP TRAINING DATA VALIDATION REPORT")
    print("=" * 80)
    print()

    # Summary
    print("📊 Dataset Summary")
    print("-" * 80)
    print(f"Total records:        {report['total_records']:,}")
    print(f"Normal records:       {report['normal_records']:,} ({report['normal_records']/report['total_records']*100:.1f}%)")
    print(f"Anomaly records:      {report['anomaly_records']:,} ({report['anomaly_records']/report['total_records']*100:.1f}%)")
    print(f"Unique peers:         {report['unique_peers']}")
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
        description="Validate BGP training data quality",
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
