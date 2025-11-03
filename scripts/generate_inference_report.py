#!/usr/bin/env python3
"""추론 결과 리포트 생성 스크립트.

추론 결과를 분석하고 시각화된 리포트를 생성합니다.
- 시계열 그래프로 이상 구간 표시
- 이상 데이터 통계 및 설명
- Markdown 형식의 리포트 생성
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class InferenceReportGenerator:
    """추론 결과 리포트 생성기."""

    def __init__(self, inference_result_path: Path, original_data_path: Path):
        """초기화.

        Args:
            inference_result_path: 추론 결과 CSV 파일 경로
            original_data_path: 원본 데이터 CSV 파일 경로
        """
        self.inference_result_path = inference_result_path
        self.original_data_path = original_data_path

        # 데이터 로드
        self.results_df = pd.read_csv(inference_result_path)
        self.original_df = pd.read_csv(original_data_path)

        # timestamp를 datetime으로 변환
        self.results_df['timestamp'] = pd.to_datetime(self.results_df['timestamp'])
        self.original_df['timestamp'] = pd.to_datetime(self.original_df['timestamp'])

        # 두 데이터 병합
        self.merged_df = pd.merge(
            self.original_df,
            self.results_df[['timestamp', 'endpoint_id', 'residual_score',
                             'multivariate_score', 'final_score', 'is_anomaly']],
            on=['timestamp', 'endpoint_id'],
            how='left'
        )

    def generate_summary(self) -> dict:
        """전체 요약 통계 생성."""
        total_samples = len(self.results_df)
        anomaly_count = self.results_df['is_anomaly'].sum()
        anomaly_rate = (anomaly_count / total_samples) * 100

        # 점수 통계
        residual_mean = self.results_df['residual_score'].mean()
        multivariate_mean = self.results_df['multivariate_score'].mean()
        final_mean = self.results_df['final_score'].mean()

        return {
            'total_samples': total_samples,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'normal_count': total_samples - anomaly_count,
            'residual_mean': residual_mean,
            'multivariate_mean': multivariate_mean,
            'final_mean': final_mean,
        }

    def find_anomaly_periods(self) -> list:
        """이상 구간 찾기."""
        anomaly_df = self.merged_df[self.merged_df['is_anomaly'] == 1].copy()

        if len(anomaly_df) == 0:
            return []

        # 연속된 이상 구간 그룹화
        anomaly_df['group'] = (anomaly_df['timestamp'].diff() > pd.Timedelta(minutes=2)).cumsum()

        periods = []
        for group_id, group in anomaly_df.groupby('group'):
            period = {
                'start': group['timestamp'].min(),
                'end': group['timestamp'].max(),
                'duration_minutes': (group['timestamp'].max() - group['timestamp'].min()).total_seconds() / 60,
                'count': len(group),
                'max_score': group['final_score'].max(),
                'avg_score': group['final_score'].mean(),
            }
            periods.append(period)

        return periods

    def analyze_anomaly_causes(self) -> dict:
        """이상 원인 분석."""
        anomaly_df = self.merged_df[self.merged_df['is_anomaly'] == 1]

        if len(anomaly_df) == 0:
            return {}

        # 메트릭별 평균값 비교
        normal_df = self.merged_df[self.merged_df['is_anomaly'] == 0]

        analysis = {}
        metrics = ['udp_echo_rtt_ms', 'ecpri_delay_us', 'lbm_rtt_ms', 'ccm_miss_count']

        for metric in metrics:
            if metric in anomaly_df.columns and metric in normal_df.columns:
                normal_mean = normal_df[metric].mean()
                anomaly_mean = anomaly_df[metric].mean()
                normal_std = normal_df[metric].std()

                # 변화율 계산
                if normal_mean > 0:
                    change_pct = ((anomaly_mean - normal_mean) / normal_mean) * 100
                else:
                    change_pct = 0

                # 표준편차 배수
                if normal_std > 0:
                    sigma_diff = (anomaly_mean - normal_mean) / normal_std
                else:
                    sigma_diff = 0

                analysis[metric] = {
                    'normal_mean': normal_mean,
                    'anomaly_mean': anomaly_mean,
                    'change_pct': change_pct,
                    'sigma_diff': sigma_diff,
                    'is_significant': abs(change_pct) > 20 or abs(sigma_diff) > 2,
                }

        return analysis

    def generate_markdown_report(self, output_path: Path):
        """Markdown 리포트 생성."""
        summary = self.generate_summary()
        periods = self.find_anomaly_periods()
        causes = self.analyze_anomaly_causes()

        report_lines = []

        # 헤더
        report_lines.append("# OCAD 추론 결과 리포트")
        report_lines.append("")
        report_lines.append(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**원본 데이터**: {self.original_data_path.name}")
        report_lines.append(f"**추론 결과**: {self.inference_result_path.name}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

        # 전체 요약
        report_lines.append("## 📊 전체 요약")
        report_lines.append("")
        report_lines.append(f"- **총 샘플 수**: {summary['total_samples']:,}개")
        report_lines.append(f"- **정상 데이터**: {summary['normal_count']:,}개 ({100 - summary['anomaly_rate']:.1f}%)")
        report_lines.append(f"- **이상 데이터**: {summary['anomaly_count']:,}개 ({summary['anomaly_rate']:.1f}%)")
        report_lines.append("")
        report_lines.append("### 탐지 점수 평균")
        report_lines.append("")
        report_lines.append(f"- **Residual Detector**: {summary['residual_mean']:.4f}")
        report_lines.append(f"- **Multivariate Detector**: {summary['multivariate_mean']:.4f}")
        report_lines.append(f"- **Final Score**: {summary['final_mean']:.4f}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

        # 이상 구간 분석
        if periods:
            report_lines.append("## ⚠️ 이상 구간 분석")
            report_lines.append("")
            report_lines.append(f"**총 {len(periods)}개의 이상 구간이 탐지되었습니다.**")
            report_lines.append("")

            for i, period in enumerate(periods, 1):
                report_lines.append(f"### 이상 구간 #{i}")
                report_lines.append("")
                report_lines.append(f"- **시작 시간**: {period['start'].strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"- **종료 시간**: {period['end'].strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"- **지속 시간**: {period['duration_minutes']:.1f}분")
                report_lines.append(f"- **이상 샘플 수**: {period['count']}개")
                report_lines.append(f"- **최대 이상 점수**: {period['max_score']:.4f}")
                report_lines.append(f"- **평균 이상 점수**: {period['avg_score']:.4f}")
                report_lines.append("")

            report_lines.append("---")
            report_lines.append("")
        else:
            report_lines.append("## ✅ 이상 구간 없음")
            report_lines.append("")
            report_lines.append("모든 데이터가 정상 범위 내에 있습니다.")
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        # 이상 원인 분석
        if causes:
            report_lines.append("## 🔍 이상 원인 분석")
            report_lines.append("")
            report_lines.append("정상 데이터와 이상 데이터의 메트릭 비교:")
            report_lines.append("")

            metric_names = {
                'udp_echo_rtt_ms': 'UDP Echo RTT',
                'ecpri_delay_us': 'eCPRI Delay',
                'lbm_rtt_ms': 'LBM RTT',
                'ccm_miss_count': 'CCM Miss Count',
            }

            for metric, data in causes.items():
                if data['is_significant']:
                    report_lines.append(f"### ⚠️ {metric_names.get(metric, metric)}")
                    report_lines.append("")
                    report_lines.append(f"- **정상 시 평균**: {data['normal_mean']:.2f}")
                    report_lines.append(f"- **이상 시 평균**: {data['anomaly_mean']:.2f}")
                    report_lines.append(f"- **변화율**: {data['change_pct']:+.1f}%")
                    report_lines.append(f"- **표준편차 배수**: {data['sigma_diff']:+.2f}σ")
                    report_lines.append("")

                    # 설명 추가
                    if data['change_pct'] > 50:
                        report_lines.append(f"**💡 분석**: {metric_names.get(metric, metric)}가 정상 대비 **{data['change_pct']:.0f}% 이상 증가**했습니다. 네트워크 지연 또는 성능 저하가 발생했을 가능성이 높습니다.")
                    elif data['change_pct'] > 20:
                        report_lines.append(f"**💡 분석**: {metric_names.get(metric, metric)}가 정상 대비 **{data['change_pct']:.0f}% 증가**했습니다. 성능 저하 징후가 관찰됩니다.")
                    elif data['change_pct'] < -20:
                        report_lines.append(f"**💡 분석**: {metric_names.get(metric, metric)}가 정상 대비 **{abs(data['change_pct']):.0f}% 감소**했습니다.")

                    if abs(data['sigma_diff']) > 3:
                        report_lines.append(f"**⚠️ 경고**: 정상 범위에서 **{abs(data['sigma_diff']):.1f} 표준편차** 벗어났습니다. 매우 이례적인 패턴입니다.")

                    report_lines.append("")

            report_lines.append("---")
            report_lines.append("")

        # 권장 사항
        report_lines.append("## 💡 권장 사항")
        report_lines.append("")

        if summary['anomaly_rate'] > 50:
            report_lines.append("### 🔴 높은 이상 탐지율 (50% 이상)")
            report_lines.append("")
            report_lines.append("1. **즉시 조치 필요**: 네트워크 또는 장비에 심각한 문제가 있을 수 있습니다.")
            report_lines.append("2. **원인 파악**: 이상 구간의 시작 시간과 시스템 로그를 대조하여 원인을 파악하세요.")
            report_lines.append("3. **장비 점검**: 해당 시간대에 장비 재시작, 설정 변경 등이 있었는지 확인하세요.")
        elif summary['anomaly_rate'] > 20:
            report_lines.append("### 🟡 중간 이상 탐지율 (20-50%)")
            report_lines.append("")
            report_lines.append("1. **모니터링 강화**: 이상 구간이 계속 증가하는지 관찰하세요.")
            report_lines.append("2. **성능 분석**: 네트워크 트래픽 증가, 간섭 등 외부 요인을 확인하세요.")
            report_lines.append("3. **예방 조치**: 필요시 장비 설정 최적화를 고려하세요.")
        elif summary['anomaly_rate'] > 5:
            report_lines.append("### 🟢 낮은 이상 탐지율 (5-20%)")
            report_lines.append("")
            report_lines.append("1. **정상 범위**: 일시적인 이상 패턴으로 보입니다.")
            report_lines.append("2. **계속 모니터링**: 이상 구간이 반복되는지 확인하세요.")
            report_lines.append("3. **패턴 분석**: 특정 시간대에 이상이 집중되는지 확인하세요.")
        else:
            report_lines.append("### ✅ 정상 상태 (5% 미만)")
            report_lines.append("")
            report_lines.append("1. **양호한 상태**: 시스템이 정상적으로 동작하고 있습니다.")
            report_lines.append("2. **일상적 모니터링**: 정기적인 점검을 계속하세요.")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

        # 데이터 샘플 (상세 설명 포함)
        report_lines.append("## 📋 이상 데이터 샘플 (상위 10개)")
        report_lines.append("")

        anomaly_samples = self.merged_df[self.merged_df['is_anomaly'] == 1].head(10)

        if len(anomaly_samples) > 0:
            # 정상 데이터 기준값 계산
            normal_df = self.merged_df[self.merged_df['is_anomaly'] == 0]
            normal_means = {}
            normal_stds = {}

            for metric in ['udp_echo_rtt_ms', 'ecpri_delay_us', 'lbm_rtt_ms', 'ccm_miss_count']:
                if metric in normal_df.columns:
                    normal_means[metric] = normal_df[metric].mean()
                    normal_stds[metric] = normal_df[metric].std()

            report_lines.append("각 샘플이 왜 이상으로 판단되었는지 상세히 설명합니다:")
            report_lines.append("")

            for idx, (_, row) in enumerate(anomaly_samples.iterrows(), 1):
                report_lines.append(f"### 🔴 이상 샘플 #{idx}")
                report_lines.append("")
                report_lines.append(f"**시간**: {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"**최종 이상 점수**: {row['final_score']:.4f}")
                report_lines.append("")

                # 메트릭별 상세 분석
                report_lines.append("**메트릭 분석**:")
                report_lines.append("")

                problems = []

                # UDP Echo RTT
                if 'udp_echo_rtt_ms' in row.index and 'udp_echo_rtt_ms' in normal_means:
                    value = row['udp_echo_rtt_ms']
                    normal = normal_means['udp_echo_rtt_ms']
                    std = normal_stds['udp_echo_rtt_ms']
                    diff_pct = ((value - normal) / normal * 100) if normal > 0 else 0
                    sigma = ((value - normal) / std) if std > 0 else 0

                    status = "🔴" if abs(diff_pct) > 50 or abs(sigma) > 3 else "🟡" if abs(diff_pct) > 20 or abs(sigma) > 2 else "🟢"
                    report_lines.append(f"- {status} **UDP Echo RTT**: {value:.2f} ms")
                    report_lines.append(f"  - 정상 평균: {normal:.2f} ms")
                    report_lines.append(f"  - 차이: {diff_pct:+.1f}% ({sigma:+.2f}σ)")

                    if abs(diff_pct) > 50:
                        problems.append(f"UDP Echo RTT가 정상 대비 {abs(diff_pct):.0f}% {'증가' if diff_pct > 0 else '감소'}")
                    elif abs(diff_pct) > 20:
                        problems.append(f"UDP Echo RTT가 약간 {'높음' if diff_pct > 0 else '낮음'}")
                    report_lines.append("")

                # eCPRI Delay
                if 'ecpri_delay_us' in row.index and 'ecpri_delay_us' in normal_means:
                    value = row['ecpri_delay_us']
                    normal = normal_means['ecpri_delay_us']
                    std = normal_stds['ecpri_delay_us']
                    diff_pct = ((value - normal) / normal * 100) if normal > 0 else 0
                    sigma = ((value - normal) / std) if std > 0 else 0

                    status = "🔴" if abs(diff_pct) > 50 or abs(sigma) > 3 else "🟡" if abs(diff_pct) > 20 or abs(sigma) > 2 else "🟢"
                    report_lines.append(f"- {status} **eCPRI Delay**: {value:.2f} μs")
                    report_lines.append(f"  - 정상 평균: {normal:.2f} μs")
                    report_lines.append(f"  - 차이: {diff_pct:+.1f}% ({sigma:+.2f}σ)")

                    if abs(diff_pct) > 50:
                        problems.append(f"eCPRI 지연이 정상 대비 {abs(diff_pct):.0f}% {'증가' if diff_pct > 0 else '감소'}")
                    elif abs(diff_pct) > 20:
                        problems.append(f"eCPRI 지연이 약간 {'높음' if diff_pct > 0 else '낮음'}")
                    report_lines.append("")

                # LBM RTT
                if 'lbm_rtt_ms' in row.index and 'lbm_rtt_ms' in normal_means:
                    value = row['lbm_rtt_ms']
                    normal = normal_means['lbm_rtt_ms']
                    std = normal_stds['lbm_rtt_ms']
                    diff_pct = ((value - normal) / normal * 100) if normal > 0 else 0
                    sigma = ((value - normal) / std) if std > 0 else 0

                    status = "🔴" if abs(diff_pct) > 50 or abs(sigma) > 3 else "🟡" if abs(diff_pct) > 20 or abs(sigma) > 2 else "🟢"
                    report_lines.append(f"- {status} **LBM RTT**: {value:.2f} ms")
                    report_lines.append(f"  - 정상 평균: {normal:.2f} ms")
                    report_lines.append(f"  - 차이: {diff_pct:+.1f}% ({sigma:+.2f}σ)")

                    if abs(diff_pct) > 50:
                        problems.append(f"LBM RTT가 정상 대비 {abs(diff_pct):.0f}% {'증가' if diff_pct > 0 else '감소'}")
                    elif abs(diff_pct) > 20:
                        problems.append(f"LBM RTT가 약간 {'높음' if diff_pct > 0 else '낮음'}")
                    report_lines.append("")

                # CCM Miss Count
                if 'ccm_miss_count' in row.index and 'ccm_miss_count' in normal_means:
                    value = int(row['ccm_miss_count'])
                    normal = normal_means['ccm_miss_count']

                    status = "🔴" if value > 5 else "🟡" if value > 0 else "🟢"
                    report_lines.append(f"- {status} **CCM Miss Count**: {value}회")
                    report_lines.append(f"  - 정상 평균: {normal:.1f}회")

                    if value > 5:
                        problems.append(f"패킷 손실이 심각함 ({value}회)")
                    elif value > 0:
                        problems.append(f"패킷 손실 발생 ({value}회)")
                    report_lines.append("")

                # 종합 판단
                if problems:
                    report_lines.append("**💡 종합 판단**:")
                    report_lines.append("")
                    for problem in problems:
                        report_lines.append(f"- {problem}")
                else:
                    report_lines.append("**💡 종합 판단**: 모든 메트릭이 정상 범위이지만, 다변량 패턴 분석에서 이상으로 탐지되었습니다.")

                report_lines.append("")
                report_lines.append("---")
                report_lines.append("")
        else:
            report_lines.append("이상 데이터가 없습니다.")
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        # 푸터
        report_lines.append("## 📌 참고사항")
        report_lines.append("")
        report_lines.append("- **이상 점수 (Final Score)**: 0.0 (정상) ~ 1.0 (이상)")
        report_lines.append("- **이상 기준**: Final Score > 0.5")
        report_lines.append("- **Residual Detector**: 시계열 예측-잔차 기반 탐지")
        report_lines.append("- **Multivariate Detector**: 다변량 이상 탐지 (Isolation Forest)")
        report_lines.append("")

        # 파일 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n✅ 리포트 생성 완료: {output_path}")
        return output_path


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="추론 결과 리포트 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--inference-result",
        type=Path,
        required=True,
        help="추론 결과 CSV 파일 (inference_simple.py 출력)",
    )
    parser.add_argument(
        "--original-data",
        type=Path,
        required=True,
        help="원본 데이터 CSV 파일",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 리포트 경로 (기본: 자동 생성)",
    )

    args = parser.parse_args()

    # 출력 경로 자동 생성
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"reports/inference_report_{timestamp}.md")

    print("=" * 70)
    print("📊 OCAD 추론 결과 리포트 생성기")
    print("=" * 70)
    print(f"\n추론 결과: {args.inference_result}")
    print(f"원본 데이터: {args.original_data}")
    print(f"출력 경로: {args.output}")

    # 리포트 생성
    generator = InferenceReportGenerator(args.inference_result, args.original_data)
    output_path = generator.generate_markdown_report(args.output)

    print("\n" + "=" * 70)
    print("✅ 리포트 생성 완료!")
    print("=" * 70)
    print(f"\n리포트를 확인하세요: {output_path.absolute()}")
    print(f"\n명령어: cat {output_path}")


if __name__ == "__main__":
    main()
