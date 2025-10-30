#!/usr/bin/env python3
"""
추론 결과 상세 보고서 생성

Usage:
    python scripts/generate_inference_report.py \\
        --input data/inference_results.csv \\
        --output reports/inference_report.md
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def generate_report(input_file: Path, output_file: Path):
    """추론 결과 보고서 생성"""

    # 데이터 로드
    df = pd.read_csv(input_file)

    # 타임스탬프 변환
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 보고서 작성
    report_lines = []

    # 헤더
    report_lines.append("# OCAD 추론 결과 보고서")
    report_lines.append("")
    report_lines.append(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**입력 파일**: {input_file}")
    report_lines.append(f"**총 레코드**: {len(df):,}개")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # 1. 전체 요약
    report_lines.append("## 📊 전체 요약")
    report_lines.append("")

    # 시간 범위
    start_time = df['timestamp_dt'].min()
    end_time = df['timestamp_dt'].max()
    duration = end_time - start_time
    report_lines.append(f"- **분석 기간**: {start_time} ~ {end_time}")
    report_lines.append(f"- **분석 시간**: {duration}")
    report_lines.append("")

    # 엔드포인트
    endpoints = df['endpoint_id'].unique()
    report_lines.append(f"- **엔드포인트 수**: {len(endpoints)}개")
    for endpoint in endpoints:
        count = len(df[df['endpoint_id'] == endpoint])
        report_lines.append(f"  - `{endpoint}`: {count:,}개 레코드")
    report_lines.append("")

    # 라벨 분포
    if 'label' in df.columns:
        label_dist = df['label'].value_counts()
        report_lines.append("### 실제 라벨 분포")
        report_lines.append("")
        for label, count in label_dist.items():
            pct = count / len(df) * 100
            report_lines.append(f"- **{label}**: {count:,}개 ({pct:.1f}%)")
        report_lines.append("")

    # 예측 분포
    pred_dist = df['predicted_label'].value_counts()
    report_lines.append("### 예측 라벨 분포")
    report_lines.append("")
    for label, count in pred_dist.items():
        pct = count / len(df) * 100
        report_lines.append(f"- **{label}**: {count:,}개 ({pct:.1f}%)")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # 2. 성능 지표
    if 'label' in df.columns:
        report_lines.append("## 🎯 성능 지표")
        report_lines.append("")

        # 정확도
        accuracy = (df['label'] == df['predicted_label']).mean() * 100
        report_lines.append(f"### 정확도: **{accuracy:.2f}%**")
        report_lines.append("")

        # Confusion Matrix
        cm = confusion_matrix(df['label'], df['predicted_label'])
        report_lines.append("### Confusion Matrix")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("예측       anomaly  normal")
        report_lines.append("실제")

        labels = sorted(df['label'].unique())
        for i, actual_label in enumerate(labels):
            row = f"{actual_label:10s}"
            for j in range(len(labels)):
                row += f" {cm[i][j]:7d}"
            report_lines.append(row)
        report_lines.append("```")
        report_lines.append("")

        # 세부 지표
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            df['label'], df['predicted_label'], average='binary', pos_label='anomaly'
        )

        report_lines.append("### 세부 성능 지표")
        report_lines.append("")
        report_lines.append(f"- **Precision (정밀도)**: {precision:.2%}")
        report_lines.append(f"  - 이상으로 예측한 것 중 실제 이상 비율")
        report_lines.append(f"- **Recall (재현율)**: {recall:.2%}")
        report_lines.append(f"  - 실제 이상 중 탐지한 비율")
        report_lines.append(f"- **F1 Score**: {f1:.2%}")
        report_lines.append(f"  - Precision과 Recall의 조화 평균")
        report_lines.append("")

        # True/False Positive/Negative
        tp = ((df['label'] == 'anomaly') & (df['predicted_label'] == 'anomaly')).sum()
        fp = ((df['label'] == 'normal') & (df['predicted_label'] == 'anomaly')).sum()
        tn = ((df['label'] == 'normal') & (df['predicted_label'] == 'normal')).sum()
        fn = ((df['label'] == 'anomaly') & (df['predicted_label'] == 'normal')).sum()

        report_lines.append("### 분류 결과 상세")
        report_lines.append("")
        report_lines.append(f"- **True Positive (TP)**: {tp:,}개 - ✅ 이상을 이상으로 정확히 탐지")
        report_lines.append(f"- **False Positive (FP)**: {fp:,}개 - ⚠️ 정상을 이상으로 오탐")
        report_lines.append(f"- **True Negative (TN)**: {tn:,}개 - ✅ 정상을 정상으로 정확히 분류")
        report_lines.append(f"- **False Negative (FN)**: {fn:,}개 - ❌ 이상을 정상으로 미탐")
        report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

    # 3. 탐지기별 분석
    report_lines.append("## 🔍 탐지기별 분석")
    report_lines.append("")

    detector_cols = [col for col in df.columns if col.endswith('_score')]

    for col in detector_cols:
        detector_name = col.replace('_score', '')
        mean_score = df[col].mean()
        max_score = df[col].max()

        # 점수 > 0인 경우
        active = df[df[col] > 0]
        active_count = len(active)
        active_pct = active_count / len(df) * 100

        report_lines.append(f"### {detector_name.replace('_', ' ').title()}")
        report_lines.append("")
        report_lines.append(f"- **평균 점수**: {mean_score:.3f}")
        report_lines.append(f"- **최대 점수**: {max_score:.3f}")
        report_lines.append(f"- **활성화 횟수**: {active_count:,}회 ({active_pct:.1f}%)")

        if len(active) > 0:
            report_lines.append(f"- **활성화 시 평균 점수**: {active[col].mean():.3f}")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # 4. 메트릭 통계
    report_lines.append("## 📈 메트릭 통계")
    report_lines.append("")

    metric_cols = ['udp_echo_rtt', 'ecpri_delay', 'lbm_rtt']

    for col in metric_cols:
        if col not in df.columns:
            continue

        col_data = df[col]

        report_lines.append(f"### {col.replace('_', ' ').upper()}")
        report_lines.append("")
        report_lines.append(f"- **평균**: {col_data.mean():.2f}")
        report_lines.append(f"- **표준편차**: {col_data.std():.2f}")
        report_lines.append(f"- **최소값**: {col_data.min():.2f}")
        report_lines.append(f"- **최대값**: {col_data.max():.2f}")
        report_lines.append(f"- **중앙값**: {col_data.median():.2f}")
        report_lines.append(f"- **P95**: {col_data.quantile(0.95):.2f}")
        report_lines.append(f"- **P99**: {col_data.quantile(0.99):.2f}")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # 5. False Negative 분석 (이상을 정상으로 오판)
    if 'label' in df.columns:
        false_negatives = df[(df['label'] == 'anomaly') & (df['predicted_label'] == 'normal')]

        if len(false_negatives) > 0:
            report_lines.append("## ❌ False Negative 분석 (미탐지)")
            report_lines.append("")
            report_lines.append(f"**총 {len(false_negatives):,}건의 이상이 탐지되지 않았습니다.**")
            report_lines.append("")

            report_lines.append("### 미탐지 메트릭 범위")
            report_lines.append("")
            for col in metric_cols:
                if col in false_negatives.columns:
                    fn_data = false_negatives[col]
                    report_lines.append(f"- **{col}**: {fn_data.min():.2f} ~ {fn_data.max():.2f} (평균: {fn_data.mean():.2f})")
            report_lines.append("")

            report_lines.append("### 미탐지 사례 (처음 10개)")
            report_lines.append("")
            report_lines.append("| 시간 | UDP RTT | eCPRI Delay | LBM RTT | Composite Score |")
            report_lines.append("|------|---------|-------------|---------|-----------------|")

            for _, row in false_negatives.head(10).iterrows():
                time_str = row['timestamp_dt'].strftime('%H:%M:%S')
                report_lines.append(
                    f"| {time_str} | "
                    f"{row.get('udp_echo_rtt', 0):.2f} | "
                    f"{row.get('ecpri_delay', 0):.2f} | "
                    f"{row.get('lbm_rtt', 0):.2f} | "
                    f"{row.get('composite_score', 0):.3f} |"
                )

            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

    # 6. 탐지된 이상 케이스 (Top 10)
    anomalies = df[df['predicted_label'] == 'anomaly'].sort_values('composite_score', ascending=False)

    if len(anomalies) > 0:
        report_lines.append("## ⚠️ 탐지된 이상 케이스 (상위 10개)")
        report_lines.append("")
        report_lines.append("가장 높은 이상 점수를 기록한 케이스들:")
        report_lines.append("")

        report_lines.append("| 순위 | 시간 | UDP RTT | eCPRI Delay | LBM RTT | Composite Score | 실제 라벨 |")
        report_lines.append("|------|------|---------|-------------|---------|-----------------|-----------|")

        for idx, (_, row) in enumerate(anomalies.head(10).iterrows(), 1):
            time_str = row['timestamp_dt'].strftime('%H:%M:%S')
            actual_label = row.get('label', '-')
            match = "✅" if actual_label == 'anomaly' else "❌"

            report_lines.append(
                f"| {idx} | {time_str} | "
                f"{row.get('udp_echo_rtt', 0):.2f} | "
                f"{row.get('ecpri_delay', 0):.2f} | "
                f"{row.get('lbm_rtt', 0):.2f} | "
                f"{row.get('composite_score', 0):.3f} | "
                f"{actual_label} {match} |"
            )

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    # 7. 시간대별 분석
    if 'label' in df.columns:
        report_lines.append("## 📅 시간대별 분석")
        report_lines.append("")

        # 5분 간격으로 그룹화
        df['time_window'] = df['timestamp_dt'].dt.floor('5min')
        time_analysis = df.groupby('time_window').agg({
            'predicted_label': lambda x: (x == 'anomaly').sum(),
            'composite_score': 'mean'
        }).reset_index()

        report_lines.append("### 5분 간격 이상 탐지 빈도")
        report_lines.append("")

        for _, row in time_analysis.head(20).iterrows():
            time_str = row['time_window'].strftime('%H:%M')
            anomaly_count = int(row['predicted_label'])
            avg_score = row['composite_score']

            bar = "█" * max(1, int(anomaly_count / 2))
            report_lines.append(f"`{time_str}` {bar} {anomaly_count}건 (평균 점수: {avg_score:.3f})")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    # 8. 권장 사항
    report_lines.append("## 💡 권장 사항")
    report_lines.append("")

    if 'label' in df.columns:
        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        if fn_rate > 0.2:
            report_lines.append("### ⚠️ 높은 False Negative율")
            report_lines.append("")
            report_lines.append(f"- 현재 **{fn_rate:.1%}**의 이상이 탐지되지 않고 있습니다.")
            report_lines.append("- **권장 조치**:")
            report_lines.append("  1. 탐지 임계값 낮추기 (`--threshold 0.3`)")
            report_lines.append("  2. 룰 기반 임계값 조정 (`--rule-threshold 8.0`)")
            report_lines.append("  3. 변화점 탐지기 추가 고려")
            report_lines.append("")

        if fp_rate > 0.05:
            report_lines.append("### ⚠️ False Positive 주의")
            report_lines.append("")
            report_lines.append(f"- 현재 **{fp_rate:.1%}**의 오탐이 발생하고 있습니다.")
            report_lines.append("- **권장 조치**:")
            report_lines.append("  1. 탐지 임계값 높이기 (`--threshold 0.7`)")
            report_lines.append("  2. 여러 탐지기의 합의 요구")
            report_lines.append("")

        if fn_rate <= 0.2 and fp_rate <= 0.05:
            report_lines.append("### ✅ 양호한 탐지 성능")
            report_lines.append("")
            report_lines.append("현재 탐지 성능이 우수합니다:")
            report_lines.append(f"- False Negative율: {fn_rate:.1%} (목표: ≤ 20%)")
            report_lines.append(f"- False Positive율: {fp_rate:.1%} (목표: ≤ 5%)")
            report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # 푸터
    report_lines.append("## 📝 부록")
    report_lines.append("")
    report_lines.append("### 파일 정보")
    report_lines.append("")
    report_lines.append(f"- **보고서 파일**: {output_file}")
    report_lines.append(f"- **데이터 파일**: {input_file}")
    report_lines.append(f"- **생성 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 파일 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(report_lines), encoding='utf-8')

    print(f"✅ 보고서 생성 완료: {output_file}")
    print(f"   파일 크기: {output_file.stat().st_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description="추론 결과 상세 보고서 생성")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/inference_results.csv"),
        help="추론 결과 CSV 파일 (기본값: data/inference_results.csv)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="보고서 출력 파일 (기본값: reports/inference_report_YYYYMMDD_HHMMSS.md)"
    )

    args = parser.parse_args()

    # 기본 출력 파일명 생성
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"reports/inference_report_{timestamp}.md")

    print("="*70)
    print("OCAD 추론 결과 보고서 생성")
    print("="*70)
    print(f"입력: {args.input}")
    print(f"출력: {args.output}")
    print("")

    if not args.input.exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        return 1

    generate_report(args.input, args.output)

    print("")
    print("보고서 확인:")
    print(f"  cat {args.output}")
    print(f"  code {args.output}")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
