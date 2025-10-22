#!/usr/bin/env python3
"""훈련 데이터 확인 스크립트.

생성된 Parquet 파일의 내용을 확인하고 샘플 데이터를 출력합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np


def view_data_summary(data_path: Path):
    """데이터 요약 정보를 출력합니다."""
    print("=" * 80)
    print(f"데이터 파일: {data_path}")
    print("=" * 80)

    # Parquet 파일 로드
    df = pd.read_parquet(data_path)

    print(f"\n총 시퀀스 수: {len(df):,}")
    print(f"컬럼 수: {len(df.columns)}")
    print(f"파일 크기: {data_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 컬럼 정보
    print("\n컬럼 정보:")
    print(df.dtypes)

    # 메트릭 타입 분포
    if 'metric_type' in df.columns:
        print("\n메트릭 타입 분포:")
        metric_counts = df['metric_type'].value_counts()
        for metric, count in metric_counts.items():
            percentage = count / len(df) * 100
            print(f"  {metric:15s}: {count:6,}개 ({percentage:5.2f}%)")

    # 이상 레이블 분포
    if 'is_anomaly' in df.columns:
        print("\n이상 레이블 분포:")
        anomaly_counts = df['is_anomaly'].value_counts()
        normal_count = anomaly_counts.get(False, 0)
        anomaly_count = anomaly_counts.get(True, 0)
        total = len(df)
        print(f"  정상:   {normal_count:6,}개 ({normal_count/total*100:5.2f}%)")
        print(f"  이상:   {anomaly_count:6,}개 ({anomaly_count/total*100:5.2f}%)")

    # 이상 타입 분포
    if 'anomaly_type' in df.columns:
        print("\n이상 타입 분포:")
        anomaly_type_counts = df['anomaly_type'].value_counts()
        for anom_type, count in anomaly_type_counts.items():
            if anom_type != 'none':
                print(f"  {anom_type:10s}: {count:6,}개")

    # 통계 정보
    if 'sequence' in df.columns:
        print("\n시퀀스 통계:")
        # 시퀀스는 리스트이므로 길이를 확인
        seq_lengths = df['sequence'].apply(len)
        print(f"  시퀀스 길이: {seq_lengths.iloc[0]} timesteps")

        # 첫 번째 시퀀스의 값 범위
        first_seq = df['sequence'].iloc[0]
        print(f"  값 범위 (첫 번째 시퀀스): [{min(first_seq):.2f}, {max(first_seq):.2f}]")

    if 'target' in df.columns:
        print("\n타겟 통계:")
        print(f"  평균: {df['target'].mean():.4f}")
        print(f"  표준편차: {df['target'].std():.4f}")
        print(f"  최소: {df['target'].min():.4f}")
        print(f"  최대: {df['target'].max():.4f}")

    return df


def show_samples(df: pd.DataFrame, num_samples: int = 3):
    """샘플 데이터를 출력합니다."""
    print("\n" + "=" * 80)
    print(f"샘플 데이터 (처음 {num_samples}개)")
    print("=" * 80)

    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        print(f"\n[샘플 #{i+1}]")
        print(f"  메트릭 타입: {row.get('metric_type', 'N/A')}")
        print(f"  엔드포인트: {row.get('endpoint_id', 'N/A')}")
        print(f"  타임스탬프: {row.get('ts_ms', 'N/A')}")
        print(f"  이상 여부: {row.get('is_anomaly', 'N/A')}")
        print(f"  이상 타입: {row.get('anomaly_type', 'N/A')}")

        if 'sequence' in row:
            seq = row['sequence']
            if len(seq) > 0:
                print(f"  시퀀스 (길이 {len(seq)}): {seq[:5]}... (처음 5개)")

        if 'target' in row:
            print(f"  타겟 값: {row['target']:.4f}")


def compare_splits(train_path: Path, val_path: Path, test_path: Path):
    """Train/Val/Test 분할 비교."""
    print("\n" + "=" * 80)
    print("Train/Val/Test 분할 비교")
    print("=" * 80)

    splits = {
        'Train': train_path,
        'Val': val_path,
        'Test': test_path,
    }

    stats = {}
    for split_name, path in splits.items():
        if path.exists():
            df = pd.read_parquet(path)
            stats[split_name] = {
                'count': len(df),
                'size_mb': path.stat().st_size / 1024 / 1024,
            }

            if 'is_anomaly' in df.columns:
                stats[split_name]['anomaly_rate'] = df['is_anomaly'].sum() / len(df)

    # 표 형식으로 출력
    print(f"\n{'분할':<10} {'샘플 수':>12} {'비율':>8} {'파일 크기':>12} {'이상 비율':>10}")
    print("-" * 65)

    total_count = sum(s['count'] for s in stats.values())

    for split_name, split_stats in stats.items():
        count = split_stats['count']
        ratio = count / total_count * 100 if total_count > 0 else 0
        size = split_stats['size_mb']
        anomaly_rate = split_stats.get('anomaly_rate', 0) * 100

        print(f"{split_name:<10} {count:>12,} {ratio:>7.2f}% {size:>10.2f} MB {anomaly_rate:>9.2f}%")

    print("-" * 65)
    print(f"{'Total':<10} {total_count:>12,} {100.0:>7.2f}%")


def export_to_csv(df: pd.DataFrame, output_path: Path, num_samples: int = 100):
    """샘플 데이터를 CSV로 내보냅니다."""
    # 시퀀스를 펼쳐서 CSV로 저장
    sample_df = df.head(num_samples).copy()

    # 시퀀스를 개별 컬럼으로 변환
    if 'sequence' in sample_df.columns:
        seq_length = len(sample_df['sequence'].iloc[0])
        for i in range(seq_length):
            sample_df[f'seq_{i}'] = sample_df['sequence'].apply(lambda x: x[i] if i < len(x) else None)
        sample_df = sample_df.drop('sequence', axis=1)

    sample_df.to_csv(output_path, index=False)
    print(f"\n✓ {num_samples}개 샘플을 CSV로 저장했습니다: {output_path}")


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(description="훈련 데이터 확인")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='ocad/data/training',
        help='데이터 디렉토리 (default: ocad/data/training)'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test', 'all'],
        default='train',
        help='확인할 분할 (default: train)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=3,
        help='표시할 샘플 수 (default: 3)'
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='샘플을 CSV로 내보내기'
    )
    parser.add_argument(
        '--csv-samples',
        type=int,
        default=100,
        help='CSV로 내보낼 샘플 수 (default: 100)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"❌ 데이터 디렉토리가 없습니다: {data_dir}")
        print(f"\n힌트: 먼저 훈련 데이터를 생성하세요:")
        print(f"  python scripts/generate_training_data.py --output-dir {data_dir}")
        return 1

    # 파일 경로
    train_path = data_dir / "timeseries_train.parquet"
    val_path = data_dir / "timeseries_val.parquet"
    test_path = data_dir / "timeseries_test.parquet"

    # 분할별 확인
    if args.split == 'all':
        # 전체 비교
        if all(p.exists() for p in [train_path, val_path, test_path]):
            compare_splits(train_path, val_path, test_path)
        else:
            print("❌ 일부 분할 파일이 없습니다")
            return 1
    else:
        # 특정 분할 상세 확인
        split_files = {
            'train': train_path,
            'val': val_path,
            'test': test_path,
        }

        file_path = split_files[args.split]
        if not file_path.exists():
            print(f"❌ 파일이 없습니다: {file_path}")
            return 1

        # 데이터 요약
        df = view_data_summary(file_path)

        # 샘플 출력
        if args.samples > 0:
            show_samples(df, args.samples)

        # CSV 내보내기
        if args.export_csv:
            csv_path = data_dir / f"timeseries_{args.split}_sample.csv"
            export_to_csv(df, csv_path, args.csv_samples)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
