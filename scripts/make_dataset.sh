#!/bin/bash
# OCAD 데이터셋 생성 스크립트
# CSV 생성 + Parquet 변환까지 원스톱으로 수행

set -e  # 에러 발생 시 중단

# 기본값 설정
OUTPUT_DIR="data/datasets"
TYPE="normal"
TRAINING_HOURS=24
VALIDATION_HOURS=12
ANOMALY_HOURS=6
FORMATS="csv parquet"

# 사용법 출력
usage() {
    cat << EOF
사용법: $0 [OPTIONS]

OCAD 학습/검증용 데이터셋을 생성합니다.

옵션:
    -o, --output-dir DIR        출력 디렉토리 (기본값: data/datasets)
    -t, --type TYPE             데이터 타입: all, normal, drift, spike, packet_loss (기본값: all)
    -th, --training-hours N     학습 데이터 시간 (기본값: 24)
    -vh, --validation-hours N   검증 데이터 시간 (기본값: 12)
    -ah, --anomaly-hours N      이상 데이터 시간 (기본값: 6)
    -f, --formats FORMATS       출력 포맷: csv, parquet 또는 둘 다 (기본값: "csv parquet")
    -h, --help                  도움말 표시

예제:
    # 모든 데이터셋 생성 (학습용 정상 + 검증용 정상/이상)
    $0

    # 정상 데이터만 생성
    $0 --type normal

    # 커스텀 디렉토리에 drift 이상 데이터 생성
    $0 --output-dir data/my_data --type drift --anomaly-hours 12

    # CSV만 생성 (Parquet 변환 없이)
    $0 --formats csv
EOF
    exit 1
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--type)
            TYPE="$2"
            shift 2
            ;;
        -th|--training-hours)
            TRAINING_HOURS="$2"
            shift 2
            ;;
        -vh|--validation-hours)
            VALIDATION_HOURS="$2"
            shift 2
            ;;
        -ah|--anomaly-hours)
            ANOMALY_HOURS="$2"
            shift 2
            ;;
        -f|--formats)
            FORMATS="$2"
            shift 2
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

# 스크립트 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 프로젝트 루트로 이동
cd "$PROJECT_ROOT"

# 가상환경 활성화
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ 가상환경을 찾을 수 없습니다. 먼저 'python -m venv .venv'로 생성하세요."
    exit 1
fi

echo "========================================================================"
echo "📊 OCAD 데이터셋 생성"
echo "========================================================================"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "데이터 타입: $TYPE"
echo "학습 시간: ${TRAINING_HOURS}시간"
echo "검증 시간: ${VALIDATION_HOURS}시간"
echo "이상 시간: ${ANOMALY_HOURS}시간"
echo "출력 포맷: $FORMATS"
echo "========================================================================"
echo ""

# Python 명령어 확인 (python3 우선)
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "❌ Python을 찾을 수 없습니다."
        exit 1
    fi
fi

# Step 1: CSV 데이터 생성
echo "📁 Step 1: CSV 데이터 생성 중..."
$PYTHON_CMD scripts/generate_datasets.py \
    --output-dir "$OUTPUT_DIR" \
    --training-hours "$TRAINING_HOURS" \
    --validation-hours "$VALIDATION_HOURS" \
    --anomaly-hours "$ANOMALY_HOURS" \
    --formats csv

if [ $? -ne 0 ]; then
    echo "❌ CSV 생성 실패"
    exit 1
fi

echo ""
echo "✅ CSV 생성 완료"
echo ""

# Step 2: Parquet 변환 (요청된 경우)
if [[ "$FORMATS" == *"parquet"* ]]; then
    echo "📦 Step 2: Parquet 변환 중..."

    PROCESSED_DIR="data/processed"
    mkdir -p "$PROCESSED_DIR"

    # TYPE에 따라 변환할 파일 결정
    case $TYPE in
        all)
            FILES=("01_training_normal.csv" "02_validation_normal.csv" "03_validation_drift_anomaly.csv" "04_validation_spike_anomaly.csv" "05_validation_packet_loss_anomaly.csv")
            ;;
        normal)
            FILES=("01_training_normal.csv" "02_validation_normal.csv")
            ;;
        drift)
            FILES=("03_validation_drift_anomaly.csv")
            ;;
        spike)
            FILES=("04_validation_spike_anomaly.csv")
            ;;
        packet_loss)
            FILES=("05_validation_packet_loss_anomaly.csv")
            ;;
        *)
            echo "⚠️  알 수 없는 타입: $TYPE. 모든 파일을 변환합니다."
            FILES=("01_training_normal.csv" "02_validation_normal.csv" "03_validation_drift_anomaly.csv" "04_validation_spike_anomaly.csv" "05_validation_packet_loss_anomaly.csv")
            ;;
    esac

    # 각 메트릭별로 Parquet 변환
    for metric in udp_echo ecpri lbm; do
        echo ""
        echo "  📊 $metric 변환 중..."

        # 학습용 데이터가 있으면 변환
        if [ -f "$OUTPUT_DIR/01_training_normal.csv" ]; then
            $PYTHON_CMD scripts/prepare_timeseries_data_v2.py \
                --input "$OUTPUT_DIR/01_training_normal.csv" \
                --output-dir "$PROCESSED_DIR" \
                --metric-type "$metric" \
                --sequence-length 10

            if [ $? -ne 0 ]; then
                echo "❌ $metric Parquet 변환 실패"
                exit 1
            fi
        fi
    done

    # Multivariate 데이터 준비 (Isolation Forest용)
    if [ -f "$OUTPUT_DIR/01_training_normal.csv" ]; then
        echo ""
        echo "  📊 Multivariate 데이터 생성 중..."
        $PYTHON_CMD scripts/prepare_multivariate_data.py \
            --train-data "$OUTPUT_DIR/01_training_normal.csv" \
            --val-data "$OUTPUT_DIR/02_validation_normal.csv" \
            --test-data "$OUTPUT_DIR/02_validation_normal.csv" \
            --output-dir "$PROCESSED_DIR"

        if [ $? -ne 0 ]; then
            echo "❌ Multivariate 데이터 생성 실패"
            exit 1
        fi
    fi

    echo ""
    echo "✅ Parquet 변환 완료"
fi

echo ""
echo "========================================================================"
echo "✅ 데이터셋 생성 완료!"
echo "========================================================================"
echo ""
echo "생성된 파일:"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null || true
if [[ "$FORMATS" == *"parquet"* ]]; then
    echo ""
    echo "변환된 Parquet 파일:"
    ls -lh "$PROCESSED_DIR"/*.parquet 2>/dev/null || true
fi
echo ""
echo "다음 단계:"
echo "  학습: ./scripts/train.sh --train-data $OUTPUT_DIR/01_training_normal.csv"
echo "  추론: ./scripts/infer.sh --input $OUTPUT_DIR/03_validation_drift_anomaly.csv"
echo ""
