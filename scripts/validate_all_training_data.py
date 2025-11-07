#!/usr/bin/env python3
"""Validate training data quality for all protocols.

This script runs validation checks on all protocol training data and provides
a comprehensive quality report.

Usage:
    # Validate all protocols
    python scripts/validate_all_training_data.py

    # Validate specific protocols
    python scripts/validate_all_training_data.py --protocols bfd,ptp

    # Specify custom data directory
    python scripts/validate_all_training_data.py --data-dir data
"""

import argparse
from pathlib import Path
import sys
import subprocess
from typing import Dict, List


def run_validation(script: Path, data_path: Path) -> Dict:
    """Run a validation script and capture results."""
    if not script.exists():
        return {"status": "SKIP", "reason": "Script not found"}

    if not data_path.exists():
        return {"status": "SKIP", "reason": "Data not found"}

    try:
        result = subprocess.run(
            [sys.executable, str(script), "--data", str(data_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        return {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "reason": "Timeout (>60s)"}
    except Exception as e:
        return {"status": "FAIL", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Validate training data quality for all protocols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory (default: data)",
    )

    parser.add_argument(
        "--protocols",
        type=str,
        default="bfd,bgp,ptp,cfm",
        help="Comma-separated list of protocols to validate (default: all)",
    )

    args = parser.parse_args()

    protocols = [p.strip() for p in args.protocols.split(",")]
    scripts_dir = Path(__file__).parent

    print("=" * 80)
    print("ALL PROTOCOLS TRAINING DATA VALIDATION")
    print("=" * 80)
    print()
    print(f"Data directory: {args.data_dir}")
    print(f"Protocols: {', '.join(protocols)}")
    print()

    results = {}
    all_passed = True

    for protocol in protocols:
        print(f"\n{'=' * 80}")
        print(f"  {protocol.upper()} Protocol Validation")
        print(f"{'=' * 80}\n")

        script = scripts_dir / f"validate_{protocol}_training_data.py"
        data_path = args.data_dir / protocol / "train"

        result = run_validation(script, data_path)
        results[protocol] = result

        if result["status"] == "SKIP":
            print(f"⏭️  SKIPPED: {result.get('reason', 'Unknown')}")
        elif result["status"] == "PASS":
            print(result.get("stdout", ""))
            print(f"\n✅ {protocol.upper()} validation PASSED")
        else:
            print(result.get("stdout", ""))
            if result.get("stderr"):
                print(f"\nErrors:\n{result['stderr']}")
            print(f"\n❌ {protocol.upper()} validation FAILED")
            all_passed = False

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()

    for protocol, result in results.items():
        status_emoji = {
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⏭️",
        }
        emoji = status_emoji.get(result["status"], "❓")
        print(f"{emoji} {protocol.upper():6s}: {result['status']}")

    print()
    print("=" * 80)

    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("=" * 80)
        print("\nPlease fix the data generation issues and regenerate the datasets.")
        sys.exit(1)


if __name__ == "__main__":
    main()
