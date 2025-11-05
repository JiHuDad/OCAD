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

    def _create_metric_chart(self, value: float, normal: float, range_min: float, range_max: float) -> str:
        """메트릭 값을 시각적으로 표현하는 간단한 차트 생성.

        Args:
            value: 현재 값
            normal: 정상 평균값
            range_min: 정상 범위 최소값
            range_max: 정상 범위 최대값

        Returns:
            str: ASCII 차트 문자열
        """
        # 차트 길이 (총 50칸)
        chart_width = 50

        # 범위 확장 (여유 20%)
        margin = (range_max - range_min) * 0.2
        chart_min = max(0, range_min - margin)
        chart_max = range_max + margin

        # 위치 계산
        def calc_pos(val):
            if chart_max == chart_min:
                return chart_width // 2
            pos = int((val - chart_min) / (chart_max - chart_min) * chart_width)
            return max(0, min(chart_width - 1, pos))

        normal_pos = calc_pos(normal)
        value_pos = calc_pos(value)
        range_start = calc_pos(range_min)
        range_end = calc_pos(range_max)

        # 차트 생성
        chart = ['·'] * chart_width

        # 정상 범위 표시 (━)
        for i in range(range_start, range_end + 1):
            chart[i] = '━'

        # 정상 평균 위치 (│)
        chart[normal_pos] = '│'

        # 현재 값 위치
        if value < range_min:
            chart[value_pos] = '▼'  # 정상 범위 아래
        elif value > range_max:
            chart[value_pos] = '▲'  # 정상 범위 위
        else:
            chart[value_pos] = '●'  # 정상 범위 내

        return ''.join(chart)

    def analyze_multivariate_patterns(self) -> dict:
        """다변량 패턴 분석 (개별 메트릭이 정상이지만 조합이 이상인 경우)."""
        anomaly_df = self.merged_df[self.merged_df['is_anomaly'] == 1]

        if len(anomaly_df) == 0:
            return {}

        normal_df = self.merged_df[self.merged_df['is_anomaly'] == 0]

        # Multivariate Detector가 주도한 이상 탐지 찾기
        # (Multivariate Score가 높고, 개별 메트릭은 정상 범위인 케이스)
        multivariate_driven = anomaly_df[
            (anomaly_df['multivariate_score'] > 0.3) &
            (anomaly_df['residual_score'] < 0.3)
        ]

        if len(multivariate_driven) == 0:
            return {}

        # 메트릭 간 상관관계 분석
        metrics = ['udp_echo_rtt_ms', 'ecpri_delay_us', 'lbm_rtt_ms']
        available_metrics = [m for m in metrics if m in normal_df.columns]

        if len(available_metrics) < 2:
            return {}

        # 정상 데이터의 상관관계
        normal_corr = normal_df[available_metrics].corr()

        # 이상 데이터의 상관관계
        anomaly_corr = multivariate_driven[available_metrics].corr()

        # 상관관계 변화 계산
        corr_changes = []
        for i in range(len(available_metrics)):
            for j in range(i + 1, len(available_metrics)):
                metric1 = available_metrics[i]
                metric2 = available_metrics[j]
                normal_val = normal_corr.loc[metric1, metric2]
                anomaly_val = anomaly_corr.loc[metric1, metric2]
                change = abs(anomaly_val - normal_val)

                if change > 0.3:  # 상관관계가 크게 변한 경우
                    corr_changes.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'normal_corr': normal_val,
                        'anomaly_corr': anomaly_val,
                        'change': change,
                    })

        return {
            'multivariate_driven_count': len(multivariate_driven),
            'correlation_changes': corr_changes,
            'available_metrics': available_metrics,
        }

    def generate_markdown_report(self, output_path: Path):
        """Markdown 리포트 생성."""
        summary = self.generate_summary()
        periods = self.find_anomaly_periods()
        causes = self.analyze_anomaly_causes()
        multivariate = self.analyze_multivariate_patterns()

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

        # 다변량 패턴 분석
        if multivariate and multivariate.get('multivariate_driven_count', 0) > 0:
            report_lines.append("## 🧩 다변량 패턴 이상 탐지")
            report_lines.append("")
            report_lines.append(f"**{multivariate['multivariate_driven_count']}개의 샘플**이 개별 메트릭은 정상 범위이지만, **메트릭 간 상관관계(조합 패턴)가 비정상**으로 탐지되었습니다.")
            report_lines.append("")
            report_lines.append("### 💡 다변량 이상 탐지란?")
            report_lines.append("")
            report_lines.append("- **개별 메트릭 분석**: 각 메트릭을 독립적으로 평가 (예: RTT < 10ms, 손실 < 1%)")
            report_lines.append("- **다변량 패턴 분석**: 여러 메트릭의 **조합 패턴**을 평가")
            report_lines.append("- **핵심**: 개별 값은 정상이지만, **메트릭 간의 상관관계가 학습 데이터와 다른 경우** 이상으로 탐지")
            report_lines.append("")
            report_lines.append("### 🔍 탐지된 패턴 변화")
            report_lines.append("")

            if multivariate.get('correlation_changes'):
                metric_names = {
                    'udp_echo_rtt_ms': 'UDP Echo RTT',
                    'ecpri_delay_us': 'eCPRI Delay',
                    'lbm_rtt_ms': 'LBM RTT',
                }

                report_lines.append("정상 데이터와 이상 데이터의 **메트릭 간 상관관계 변화**:")
                report_lines.append("")

                for change in multivariate['correlation_changes']:
                    m1_name = metric_names.get(change['metric1'], change['metric1'])
                    m2_name = metric_names.get(change['metric2'], change['metric2'])

                    report_lines.append(f"#### {m1_name} ↔ {m2_name}")
                    report_lines.append("")
                    report_lines.append(f"- **정상 시 상관계수**: {change['normal_corr']:.3f}")
                    report_lines.append(f"- **이상 시 상관계수**: {change['anomaly_corr']:.3f}")
                    report_lines.append(f"- **변화량**: {change['change']:.3f}")
                    report_lines.append("")

                    # 해석 추가
                    if abs(change['normal_corr']) > 0.5 and abs(change['anomaly_corr']) < 0.2:
                        report_lines.append(f"**💡 해석**: 정상 시에는 {m1_name}과 {m2_name}이 **{'정' if change['normal_corr'] > 0 else '부'}의 상관관계**를 보였으나, 이상 구간에서는 **상관관계가 약화**되었습니다. 이는 두 메트릭이 독립적으로 변동하는 비정상 패턴입니다.")
                    elif abs(change['normal_corr']) < 0.2 and abs(change['anomaly_corr']) > 0.5:
                        report_lines.append(f"**💡 해석**: 정상 시에는 {m1_name}과 {m2_name}이 **독립적**이었으나, 이상 구간에서는 **강한 {'정' if change['anomaly_corr'] > 0 else '부'}의 상관관계**가 나타났습니다. 이는 학습 데이터에서 보지 못한 비정상 패턴입니다.")
                    else:
                        report_lines.append(f"**💡 해석**: {m1_name}과 {m2_name}의 상관관계가 정상 데이터와 크게 달라졌습니다. 이는 네트워크 경로 변경, 장비 동작 모드 변화 등의 신호일 수 있습니다.")

                    report_lines.append("")

            else:
                report_lines.append("메트릭 간 상관관계 변화는 미미하지만, **고차원 공간에서의 패턴**이 정상 학습 데이터와 다른 것으로 탐지되었습니다.")
                report_lines.append("")
                report_lines.append("이는 Isolation Forest가 2개 이상의 메트릭을 조합했을 때 학습 데이터와 다른 **드문 패턴(isolated)**을 발견했음을 의미합니다.")
                report_lines.append("")

            report_lines.append("### ⚠️ 왜 중요한가?")
            report_lines.append("")
            report_lines.append("다변량 패턴 이상 탐지는 **개별 임계값 기반 탐지로는 놓칠 수 있는 미묘한 장애 신호**를 조기 발견할 수 있습니다:")
            report_lines.append("")
            report_lines.append("- **하드웨어 부분 고장**: 개별 메트릭은 임계값 이하지만, 메트릭 조합이 비정상")
            report_lines.append("- **네트워크 경로 변경**: RTT, 손실률은 정상이지만, 그들의 관계가 평소와 다름")
            report_lines.append("- **간헐적 문제**: 평균적으로 정상이지만, 메트릭 간 동기화 패턴이 변함")
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

        # 최종 점수 기준으로 정렬하여 상위 10개 추출
        anomaly_samples = self.merged_df[self.merged_df['is_anomaly'] == 1].nlargest(10, 'final_score')

        if len(anomaly_samples) > 0:
            # 정상 데이터 기준값 계산
            normal_df = self.merged_df[self.merged_df['is_anomaly'] == 0]
            normal_means = {}
            normal_stds = {}
            normal_ranges = {}  # 정상 범위 (mean ± 2*std)

            for metric in ['udp_echo_rtt_ms', 'ecpri_delay_us', 'lbm_rtt_ms', 'ccm_miss_count']:
                if metric in normal_df.columns:
                    normal_means[metric] = normal_df[metric].mean()
                    normal_stds[metric] = normal_df[metric].std()
                    # 정상 범위: mean ± 2*std (약 95% 신뢰구간)
                    normal_ranges[metric] = {
                        'min': normal_means[metric] - 2 * normal_stds[metric],
                        'max': normal_means[metric] + 2 * normal_stds[metric],
                    }

            report_lines.append("각 샘플이 **왜 이상으로 판단되었는지** 구체적으로 설명합니다:")
            report_lines.append("")

            for idx, (_, row) in enumerate(anomaly_samples.iterrows(), 1):
                report_lines.append(f"### 🔴 이상 샘플 #{idx}")
                report_lines.append("")
                report_lines.append(f"**시간**: {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"**최종 이상 점수**: {row['final_score']:.4f}")
                report_lines.append("")

                # 탐지기별 점수 분석
                residual_score = row['residual_score'] if 'residual_score' in row.index else 0
                multivariate_score = row['multivariate_score'] if 'multivariate_score' in row.index else 0

                report_lines.append("**탐지기 분석**:")
                report_lines.append("")
                report_lines.append(f"- **Residual Score (TCN)**: {residual_score:.4f} {'🔴' if residual_score > 0.7 else '🟡' if residual_score > 0.3 else '🟢'}")
                report_lines.append(f"- **Multivariate Score (Isolation Forest)**: {multivariate_score:.4f} {'🔴' if multivariate_score > 0.7 else '🟡' if multivariate_score > 0.3 else '🟢'}")
                report_lines.append("")

                # 주도 탐지기 파악
                if residual_score > multivariate_score and residual_score > 0.3:
                    report_lines.append("**🎯 주도 탐지기**: TCN (Residual Detector)")
                    report_lines.append("→ 시계열 패턴이 학습 데이터와 다릅니다.")
                elif multivariate_score > residual_score and multivariate_score > 0.3:
                    report_lines.append("**🎯 주도 탐지기**: Isolation Forest (Multivariate Detector)")
                    report_lines.append("→ 메트릭 간 조합 패턴이 비정상입니다.")
                else:
                    report_lines.append("**🎯 주도 탐지기**: 복합 탐지 (여러 탐지기가 함께 탐지)")
                report_lines.append("")

                # 메트릭별 상세 분석
                report_lines.append("**메트릭 분석**:")
                report_lines.append("")

                problems = []
                metric_details = []

                # UDP Echo RTT
                if 'udp_echo_rtt_ms' in row.index and 'udp_echo_rtt_ms' in normal_means:
                    value = row['udp_echo_rtt_ms']
                    normal = normal_means['udp_echo_rtt_ms']
                    std = normal_stds['udp_echo_rtt_ms']
                    range_min = normal_ranges['udp_echo_rtt_ms']['min']
                    range_max = normal_ranges['udp_echo_rtt_ms']['max']

                    diff_pct = ((value - normal) / normal * 100) if normal > 0 else 0
                    sigma = ((value - normal) / std) if std > 0 else 0

                    # 정상 범위 벗어남 판단
                    out_of_range = value < range_min or value > range_max

                    status = "🔴" if abs(diff_pct) > 50 or abs(sigma) > 3 else "🟡" if abs(diff_pct) > 20 or abs(sigma) > 2 else "🟢"
                    report_lines.append(f"- {status} **UDP Echo RTT**: {value:.2f} ms")
                    report_lines.append(f"  - 정상 평균: {normal:.2f} ms (범위: {max(0, range_min):.2f} ~ {range_max:.2f} ms)")
                    report_lines.append(f"  - 차이: {diff_pct:+.1f}% ({sigma:+.2f}σ)")

                    # 시각적 표현 (간단한 차트)
                    chart = self._create_metric_chart(value, normal, range_min, range_max)
                    report_lines.append(f"  - 시각화: {chart}")

                    if out_of_range:
                        if value > range_max:
                            problems.append(f"UDP Echo RTT가 정상 범위를 크게 초과 ({value:.1f}ms > {range_max:.1f}ms)")
                            metric_details.append("**네트워크 지연 증가**")
                        else:
                            problems.append(f"UDP Echo RTT가 비정상적으로 낮음 ({value:.1f}ms < {range_min:.1f}ms)")
                            metric_details.append("**비정상적으로 낮은 RTT (캐싱/우회?)**")
                    elif abs(diff_pct) > 20:
                        problems.append(f"UDP Echo RTT가 약간 {'높음' if diff_pct > 0 else '낮음'} (정상 대비 {abs(diff_pct):.0f}%)")
                    report_lines.append("")

                # eCPRI Delay
                if 'ecpri_delay_us' in row.index and 'ecpri_delay_us' in normal_means:
                    value = row['ecpri_delay_us']
                    normal = normal_means['ecpri_delay_us']
                    std = normal_stds['ecpri_delay_us']
                    range_min = normal_ranges['ecpri_delay_us']['min']
                    range_max = normal_ranges['ecpri_delay_us']['max']

                    diff_pct = ((value - normal) / normal * 100) if normal > 0 else 0
                    sigma = ((value - normal) / std) if std > 0 else 0

                    out_of_range = value < range_min or value > range_max

                    status = "🔴" if abs(diff_pct) > 50 or abs(sigma) > 3 else "🟡" if abs(diff_pct) > 20 or abs(sigma) > 2 else "🟢"
                    report_lines.append(f"- {status} **eCPRI Delay**: {value:.2f} μs")
                    report_lines.append(f"  - 정상 평균: {normal:.2f} μs (범위: {max(0, range_min):.2f} ~ {range_max:.2f} μs)")
                    report_lines.append(f"  - 차이: {diff_pct:+.1f}% ({sigma:+.2f}σ)")

                    chart = self._create_metric_chart(value, normal, range_min, range_max)
                    report_lines.append(f"  - 시각화: {chart}")

                    if out_of_range:
                        if value > range_max:
                            problems.append(f"eCPRI 지연이 정상 범위를 초과 ({value:.1f}μs > {range_max:.1f}μs)")
                            metric_details.append("**프론트홀 지연 증가**")
                        else:
                            problems.append(f"eCPRI 지연이 비정상적으로 낮음 ({value:.1f}μs < {range_min:.1f}μs)")
                    elif abs(diff_pct) > 20:
                        problems.append(f"eCPRI 지연이 약간 {'높음' if diff_pct > 0 else '낮음'} (정상 대비 {abs(diff_pct):.0f}%)")
                    report_lines.append("")

                # LBM RTT
                if 'lbm_rtt_ms' in row.index and 'lbm_rtt_ms' in normal_means:
                    value = row['lbm_rtt_ms']
                    normal = normal_means['lbm_rtt_ms']
                    std = normal_stds['lbm_rtt_ms']
                    range_min = normal_ranges['lbm_rtt_ms']['min']
                    range_max = normal_ranges['lbm_rtt_ms']['max']

                    diff_pct = ((value - normal) / normal * 100) if normal > 0 else 0
                    sigma = ((value - normal) / std) if std > 0 else 0

                    out_of_range = value < range_min or value > range_max

                    status = "🔴" if abs(diff_pct) > 50 or abs(sigma) > 3 else "🟡" if abs(diff_pct) > 20 or abs(sigma) > 2 else "🟢"
                    report_lines.append(f"- {status} **LBM RTT**: {value:.2f} ms")
                    report_lines.append(f"  - 정상 평균: {normal:.2f} ms (범위: {max(0, range_min):.2f} ~ {range_max:.2f} ms)")
                    report_lines.append(f"  - 차이: {diff_pct:+.1f}% ({sigma:+.2f}σ)")

                    chart = self._create_metric_chart(value, normal, range_min, range_max)
                    report_lines.append(f"  - 시각화: {chart}")

                    if out_of_range:
                        if value > range_max:
                            problems.append(f"LBM RTT가 정상 범위를 초과 ({value:.1f}ms > {range_max:.1f}ms)")
                            metric_details.append("**이더넷 링크 지연 증가**")
                        else:
                            problems.append(f"LBM RTT가 비정상적으로 낮음 ({value:.1f}ms < {range_min:.1f}ms)")
                    elif abs(diff_pct) > 20:
                        problems.append(f"LBM RTT가 약간 {'높음' if diff_pct > 0 else '낮음'} (정상 대비 {abs(diff_pct):.0f}%)")
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
                        metric_details.append("**심각한 패킷 손실**")
                    elif value > 0:
                        problems.append(f"패킷 손실 발생 ({value}회)")
                        metric_details.append("**간헐적 패킷 손실**")
                    report_lines.append("")

                # 종합 판단 (개선된 버전)
                report_lines.append("**💡 종합 판단 - 왜 이상으로 탐지되었나?**:")
                report_lines.append("")

                if problems:
                    report_lines.append("**메트릭 기반 이상 탐지**:")
                    report_lines.append("")
                    for problem in problems:
                        report_lines.append(f"  - {problem}")
                    report_lines.append("")

                    if metric_details:
                        report_lines.append("**진단**:")
                        for detail in metric_details:
                            report_lines.append(f"  - {detail}")
                        report_lines.append("")

                # 탐지기별 구체적 설명
                if residual_score > 0.5:
                    report_lines.append("**TCN (Residual Detector) 탐지 이유**:")
                    report_lines.append("")
                    report_lines.append(f"  - TCN 모델이 학습한 시계열 패턴과 **현재 패턴이 {residual_score:.1%} 차이**")
                    report_lines.append("  - 메트릭 간 상관관계가 학습 데이터와 다름")
                    report_lines.append("  - 예: 정상 시 'UDP RTT ↑ → eCPRI Delay ↑'인데, 현재는 역관계 또는 무상관")
                    report_lines.append("")

                if multivariate_score > 0.3:
                    report_lines.append("**Isolation Forest (Multivariate Detector) 탐지 이유**:")
                    report_lines.append("")
                    report_lines.append(f"  - 메트릭 조합 패턴이 학습 데이터에서 **{multivariate_score:.1%} isolated (드문 패턴)**")
                    report_lines.append("  - 개별 메트릭은 정상이어도, 조합이 비정상")
                    report_lines.append("  - 예: UDP RTT=5ms, eCPRI=150μs 각각은 정상이지만, 이 조합은 학습 데이터에 없음")
                    report_lines.append("")

                # 결론
                if not problems and (residual_score < 0.3 and multivariate_score < 0.3):
                    report_lines.append("**⚠️ 경미한 이상**: 메트릭 값은 정상 범위이지만, 미묘한 패턴 변화가 감지됨")
                    report_lines.append("")
                elif residual_score > 0.7 or multivariate_score > 0.7:
                    report_lines.append("**🚨 심각한 이상**: 즉시 조치가 필요합니다!")
                    report_lines.append("")
                else:
                    report_lines.append("**⚠️ 중간 수준 이상**: 모니터링 강화 권장")
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
