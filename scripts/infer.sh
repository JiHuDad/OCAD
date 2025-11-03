#!/bin/bash
# OCAD 추론 스크립트
# TCN + Isolation Forest 추론 + 리포트 생성

set -e  # 에러 발생 시 중단

# 기본값 설정
INPUT_DATA=""
OUTPUT_FILE=""
REPORT_FILE=""
MODEL_DIR="ocad/models"
GENERATE_REPORT=true

# 사용법 출력
usage() {
    cat << EOF
사용법: $0 [OPTIONS]

OCAD 모델로 추론을 수행하고 리포트를 생성합니다.

옵션:
    -i, --input FILE            입력 데이터 CSV 파일 (필수)
    -o, --output FILE           추론 결과 CSV 파일 (기본값: 자동 생성)
    -r, --report FILE           리포트 마크다운 파일 (기본값: 자동 생성)
    -m, --model-dir DIR         모델 디렉토리 (기본값: ocad/models)
    --no-report                 리포트 생성 안 함
    -h, --help                  도움말 표시

예제:
    # 기본 추론 + 리포트 생성
    $0 --input data/datasets/03_validation_drift_anomaly.csv

    # 출력 파일 명시
    $0 --input my_data.csv --output results/result.csv --report results/report.md

    # 리포트 없이 추론만
    $0 --input my_data.csv --no-report

    # 커스텀 모델 디렉토리
    $0 --input my_data.csv --model-dir /path/to/models
EOF
    exit 1
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DATA="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -r|--report)
            REPORT_FILE="$2"
            shift 2
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --no-report)
            GENERATE_REPORT=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "❌ 알 수 없는 옵션: $1"
            usage
            ;;
    esac
done

# 필수 인자 확인
if [ -z "$INPUT_DATA" ]; then
    echo "❌ 입력 데이터 파일이 필요합니다."
    usage
fi

if [ ! -f "$INPUT_DATA" ]; then
    echo "❌ 입력 데이터 파일을 찾을 수 없습니다: $INPUT_DATA"
    exit 1
fi

# 스크립트 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 프로젝트 루트로 이동
cd "$PROJECT_ROOT"

# 가상환경 활성화
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ 가상환경을 찾을 수 없습니다."
    exit 1
fi

# 출력 파일명 자동 생성
if [ -z "$OUTPUT_FILE" ]; then
    INPUT_BASENAME=$(basename "$INPUT_DATA" .csv)
    OUTPUT_FILE="data/results/${INPUT_BASENAME}_result.csv"
fi

if [ -z "$REPORT_FILE" ] && [ "$GENERATE_REPORT" = true ]; then
    INPUT_BASENAME=$(basename "$INPUT_DATA" .csv)
    REPORT_FILE="reports/${INPUT_BASENAME}_report.md"
fi

# 디렉토리 생성
mkdir -p "$(dirname "$OUTPUT_FILE")"
if [ "$GENERATE_REPORT" = true ]; then
    mkdir -p "$(dirname "$REPORT_FILE")"
fi

echo "========================================================================"
echo "🔮 OCAD 추론 실행"
echo "========================================================================"
echo "입력 데이터: $INPUT_DATA"
echo "모델 디렉토리: $MODEL_DIR"
echo "출력 파일: $OUTPUT_FILE"
if [ "$GENERATE_REPORT" = true ]; then
    echo "리포트 파일: $REPORT_FILE"
else
    echo "리포트: 생성 안 함"
fi
echo "========================================================================"
echo ""

# Step 1: 추론 실행
echo "========================================================================"
echo "🚀 Step 1: 추론 실행 중..."
echo "========================================================================"
echo ""

python scripts/inference_simple.py \
    --input "$INPUT_DATA" \
    --output "$OUTPUT_FILE" \
    --model-dir "$MODEL_DIR"

if [ $? -ne 0 ]; then
    echo "❌ 추론 실패"
    exit 1
fi

echo ""
echo "✅ 추론 완료"
echo ""

# Step 2: 리포트 생성 (요청된 경우)
if [ "$GENERATE_REPORT" = true ]; then
    echo "========================================================================"
    echo "📊 Step 2: 리포트 생성 중..."
    echo "========================================================================"
    echo ""

    python scripts/generate_inference_report.py \
        --inference-result "$OUTPUT_FILE" \
        --original-data "$INPUT_DATA" \
        --output "$REPORT_FILE"

    if [ $? -ne 0 ]; then
        echo "❌ 리포트 생성 실패"
        exit 1
    fi

    echo ""
    echo "✅ 리포트 생성 완료"
    echo ""
fi

# 완료 메시지
echo "========================================================================"
echo "✅ 추론 완료!"
echo "========================================================================"
echo ""
echo "생성된 파일:"
echo ""
echo "📊 추론 결과:"
ls -lh "$OUTPUT_FILE"
echo ""

if [ "$GENERATE_REPORT" = true ] && [ -f "$REPORT_FILE" ]; then
    echo "📄 리포트:"
    ls -lh "$REPORT_FILE"
    echo ""
    echo "리포트 미리보기:"
    echo "========================================================================"
    head -n 30 "$REPORT_FILE"
    echo "..."
    echo "========================================================================"
    echo ""
    echo "전체 리포트 확인: cat $REPORT_FILE"
fi

echo ""
echo "추론 결과 요약:"
echo "========================================================================"
python << EOF
import pandas as pd

df = pd.read_csv('$OUTPUT_FILE')
total = len(df)
anomalies = df['is_anomaly'].sum()
anomaly_rate = (anomalies / total * 100) if total > 0 else 0

print(f"총 샘플: {total:,}개")
print(f"정상: {total - anomalies:,}개 ({(100 - anomaly_rate):.1f}%)")
print(f"이상: {anomalies:,}개 ({anomaly_rate:.1f}%)")
print(f"")
print(f"평균 점수:")
print(f"  Residual: {df['residual_score'].mean():.4f}")
print(f"  Multivariate: {df['multivariate_score'].mean():.4f}")
print(f"  Final: {df['final_score'].mean():.4f}")
EOF
echo "========================================================================"
echo ""
