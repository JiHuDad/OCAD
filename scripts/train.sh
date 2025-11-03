#!/bin/bash
# OCAD 모델 학습 스크립트
# TCN (3개) + Isolation Forest 모두 학습

set -e  # 에러 발생 시 중단

# 기본값 설정
TRAIN_DATA=""
VAL_DATA=""
TEST_DATA=""
EPOCHS=10
BATCH_SIZE=32
VERSION="2.0.2"
MODEL_DIR="ocad/models"
PROCESSED_DIR="data/processed"

# 사용법 출력
usage() {
    cat << EOF
사용법: $0 [OPTIONS]

OCAD 모델을 학습합니다 (TCN 3개 + Isolation Forest).

옵션:
    -t, --train-data FILE       학습 데이터 CSV 파일 (필수)
    -v, --val-data FILE         검증 데이터 CSV 파일 (기본값: 자동 탐색)
    -ts, --test-data FILE       테스트 데이터 CSV 파일 (기본값: 자동 탐색)
    -e, --epochs N              학습 에포크 수 (기본값: 10)
    -b, --batch-size N          배치 크기 (기본값: 32)
    -V, --version VERSION       모델 버전 (기본값: 2.0.2)
    -m, --model-dir DIR         모델 저장 디렉토리 (기본값: ocad/models)
    -h, --help                  도움말 표시

예제:
    # 기본 학습 (모든 모델)
    $0 --train-data data/datasets/01_training_normal.csv

    # 커스텀 에포크와 버전
    $0 --train-data data/datasets/01_training_normal.csv --epochs 20 --version 3.0.0

    # 검증/테스트 데이터 명시
    $0 --train-data train.csv --val-data val.csv --test-data test.csv
EOF
    exit 1
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        -v|--val-data)
            VAL_DATA="$2"
            shift 2
            ;;
        -ts|--test-data)
            TEST_DATA="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -V|--version)
            VERSION="$2"
            shift 2
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
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

# 필수 인자 확인
if [ -z "$TRAIN_DATA" ]; then
    echo "❌ 학습 데이터 파일이 필요합니다."
    usage
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 학습 데이터 파일을 찾을 수 없습니다: $TRAIN_DATA"
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

# Val/Test 데이터 자동 탐색
TRAIN_DIR=$(dirname "$TRAIN_DATA")
if [ -z "$VAL_DATA" ]; then
    VAL_DATA="$TRAIN_DIR/02_validation_normal.csv"
    if [ ! -f "$VAL_DATA" ]; then
        VAL_DATA="$TRAIN_DATA"  # 없으면 train 데이터 사용
    fi
fi

if [ -z "$TEST_DATA" ]; then
    TEST_DATA="$VAL_DATA"  # Val과 동일하게
fi

echo "========================================================================"
echo "🎓 OCAD 모델 학습"
echo "========================================================================"
echo "학습 데이터: $TRAIN_DATA"
echo "검증 데이터: $VAL_DATA"
echo "테스트 데이터: $TEST_DATA"
echo "에포크: $EPOCHS"
echo "배치 크기: $BATCH_SIZE"
echo "버전: $VERSION"
echo "모델 디렉토리: $MODEL_DIR"
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

# 디렉토리 생성
mkdir -p "$PROCESSED_DIR"
mkdir -p "$MODEL_DIR/tcn"
mkdir -p "$MODEL_DIR/isolation_forest"
mkdir -p "$MODEL_DIR/metadata/performance_reports"

# Step 1: 데이터 전처리
echo "========================================================================"
echo "📦 Step 1: 데이터 전처리"
echo "========================================================================"
echo ""

# TCN용 시계열 데이터 준비 (3개 메트릭)
for metric in udp_echo ecpri lbm; do
    echo "📊 $metric 시계열 데이터 준비 중..."
    $PYTHON_CMD scripts/prepare_timeseries_data_v2.py \
        --input "$TRAIN_DATA" \
        --output-dir "$PROCESSED_DIR" \
        --metric-type "$metric" \
        --window-size 10

    if [ $? -ne 0 ]; then
        echo "❌ $metric 데이터 준비 실패"
        exit 1
    fi
    echo ""
done

# Isolation Forest용 다변량 데이터 준비
echo "📊 Multivariate 데이터 준비 중..."
$PYTHON_CMD scripts/prepare_multivariate_data.py \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --test-data "$TEST_DATA" \
    --output-dir "$PROCESSED_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Multivariate 데이터 준비 실패"
    exit 1
fi

echo ""
echo "✅ 데이터 전처리 완료"
echo ""

# Step 2: TCN 모델 학습 (3개)
echo "========================================================================"
echo "🧠 Step 2: TCN 모델 학습 (3개)"
echo "========================================================================"
echo ""

for metric in udp_echo ecpri lbm; do
    echo "----------------------------------------"
    echo "🎯 $metric TCN 학습 중..."
    echo "----------------------------------------"

    $PYTHON_CMD scripts/train_tcn_model.py \
        --metric-type "$metric" \
        --train-data "$PROCESSED_DIR/timeseries_${metric}_train.parquet" \
        --val-data "$PROCESSED_DIR/timeseries_${metric}_val.parquet" \
        --test-data "$PROCESSED_DIR/timeseries_${metric}_test.parquet" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --version "$VERSION" \
        --output-dir "$MODEL_DIR/tcn"

    if [ $? -ne 0 ]; then
        echo "❌ $metric TCN 학습 실패"
        exit 1
    fi

    echo ""
    echo "✅ $metric TCN 학습 완료"
    echo ""
done

# Step 3: Isolation Forest 학습
echo "========================================================================"
echo "🌲 Step 3: Isolation Forest 학습"
echo "========================================================================"
echo ""

$PYTHON_CMD scripts/train_isolation_forest.py \
    --train-data "$PROCESSED_DIR/multivariate_train.parquet" \
    --val-data "$PROCESSED_DIR/multivariate_val.parquet" \
    --test-data "$PROCESSED_DIR/multivariate_test.parquet" \
    --output-dir "$MODEL_DIR/isolation_forest" \
    --version "$VERSION"

if [ $? -ne 0 ]; then
    echo "❌ Isolation Forest 학습 실패"
    exit 1
fi

echo ""
echo "✅ Isolation Forest 학습 완료"
echo ""

# 완료 메시지
echo "========================================================================"
echo "✅ 모든 모델 학습 완료!"
echo "========================================================================"
echo ""
echo "생성된 모델:"
echo ""
echo "📦 TCN 모델 (3개):"
ls -lh "$MODEL_DIR/tcn/"*_v${VERSION}.pth 2>/dev/null || echo "  (파일 없음)"
echo ""
echo "📦 TCN Scaler (3개):"
ls -lh "$MODEL_DIR/tcn/"*_v${VERSION}_scaler.pkl 2>/dev/null || echo "  (파일 없음)"
echo ""
echo "🌲 Isolation Forest:"
ls -lh "$MODEL_DIR/isolation_forest/"*_${VERSION}.pkl 2>/dev/null || echo "  (파일 없음)"
echo ""
echo "다음 단계:"
echo "  추론: ./scripts/infer.sh --input data/datasets/03_validation_drift_anomaly.csv"
echo ""
