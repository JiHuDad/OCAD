#!/bin/bash
# OCAD TCN 모델 전체 학습 스크립트
#
# 사용법:
#   ./scripts/train_all_models.sh              # 모든 단계 실행
#   ./scripts/train_all_models.sh --data-only  # 데이터 생성만
#   ./scripts/train_all_models.sh --train-only # 학습만 (데이터 있어야 함)

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리로 이동
cd "$(dirname "$0")/.."

# 가상환경 활성화
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}OCAD TCN 모델 학습 시작${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ 가상환경 활성화 완료${NC}"
else
    echo -e "${RED}✗ 가상환경을 찾을 수 없습니다: .venv/bin/activate${NC}"
    exit 1
fi

# 설정 변수
DATA_DIR="ocad/data/training"
MODEL_DIR="ocad/models/tcn"
EPOCHS=50
BATCH_SIZE=32
HIDDEN_SIZE=32
LEARNING_RATE=0.001

# 훈련 데이터 생성 설정
NUM_ENDPOINTS=10
DURATION_HOURS=6
ANOMALY_RATE=0.15

# 플래그 파싱
GENERATE_DATA=true
TRAIN_MODELS=true

if [ "$1" = "--data-only" ]; then
    TRAIN_MODELS=false
elif [ "$1" = "--train-only" ]; then
    GENERATE_DATA=false
fi

# 1. 훈련 데이터 생성
if [ "$GENERATE_DATA" = true ]; then
    echo ""
    echo -e "${BLUE}==================================================${NC}"
    echo -e "${BLUE}[1/4] 훈련 데이터 생성${NC}"
    echo -e "${BLUE}==================================================${NC}"
    echo ""
    echo "설정:"
    echo "  - 엔드포인트 수: ${NUM_ENDPOINTS}"
    echo "  - 수집 기간: ${DURATION_HOURS}시간"
    echo "  - 이상 비율: ${ANOMALY_RATE}"
    echo "  - 출력 디렉토리: ${DATA_DIR}"
    echo ""

    python scripts/generate_training_data.py \
        --dataset-type timeseries \
        --endpoints ${NUM_ENDPOINTS} \
        --duration-hours ${DURATION_HOURS} \
        --anomaly-rate ${ANOMALY_RATE} \
        --output-dir ${DATA_DIR}

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ 훈련 데이터 생성 완료${NC}"
        echo ""
        echo "생성된 파일:"
        ls -lh ${DATA_DIR}/*.parquet
        echo ""

        # 데이터 요약 출력
        echo "데이터 요약:"
        python -c "
import pandas as pd
import sys
try:
    df = pd.read_parquet('${DATA_DIR}/timeseries_train.parquet')
    print(f'  총 시퀀스: {len(df):,}개')
    print(f'  메트릭 분포:')
    for metric, count in df['metric_type'].value_counts().items():
        print(f'    - {metric}: {count:,}개')
except Exception as e:
    print(f'  데이터 요약 실패: {e}', file=sys.stderr)
"
    else
        echo -e "${RED}✗ 훈련 데이터 생성 실패${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${YELLOW}⊘ 데이터 생성 단계 건너뛰기${NC}"

    # 데이터 존재 확인
    if [ ! -f "${DATA_DIR}/timeseries_train.parquet" ]; then
        echo -e "${RED}✗ 훈련 데이터가 없습니다: ${DATA_DIR}/timeseries_train.parquet${NC}"
        echo -e "${YELLOW}힌트: --data-only 옵션으로 데이터를 먼저 생성하세요${NC}"
        exit 1
    fi
fi

# 2. 모델 학습 (선택된 경우)
if [ "$TRAIN_MODELS" = false ]; then
    echo ""
    echo -e "${YELLOW}⊘ 모델 학습 단계 건너뛰기${NC}"
    echo ""
    echo -e "${GREEN}모든 작업 완료!${NC}"
    exit 0
fi

# 학습 설정 출력
echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}모델 학습 설정${NC}"
echo -e "${BLUE}==================================================${NC}"
echo "  - 에포크: ${EPOCHS}"
echo "  - 배치 크기: ${BATCH_SIZE}"
echo "  - 히든 크기: ${HIDDEN_SIZE}"
echo "  - 학습률: ${LEARNING_RATE}"
echo "  - 출력 디렉토리: ${MODEL_DIR}"
echo ""

# 2.1 UDP Echo 모델 학습
echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}[2/4] UDP Echo 모델 학습${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

timeout 300 python scripts/train_tcn_model.py \
    --metric-type udp_echo \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --hidden-size ${HIDDEN_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --output-dir ${MODEL_DIR}

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ UDP Echo 모델 학습 완료${NC}"
    ls -lh ${MODEL_DIR}/udp_echo_v*.pth 2>/dev/null | tail -1
else
    echo -e "${RED}✗ UDP Echo 모델 학습 실패${NC}"
    exit 1
fi

# 2.2 eCPRI 모델 학습
echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}[3/4] eCPRI 모델 학습${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

timeout 300 python scripts/train_tcn_model.py \
    --metric-type ecpri \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --hidden-size ${HIDDEN_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --output-dir ${MODEL_DIR}

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ eCPRI 모델 학습 완료${NC}"
    ls -lh ${MODEL_DIR}/ecpri_v*.pth 2>/dev/null | tail -1
else
    echo -e "${RED}✗ eCPRI 모델 학습 실패${NC}"
    exit 1
fi

# 2.3 LBM 모델 학습
echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}[4/4] LBM 모델 학습${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

timeout 300 python scripts/train_tcn_model.py \
    --metric-type lbm \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --hidden-size ${HIDDEN_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --output-dir ${MODEL_DIR}

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ LBM 모델 학습 완료${NC}"
    ls -lh ${MODEL_DIR}/lbm_v*.pth 2>/dev/null | tail -1
else
    echo -e "${RED}✗ LBM 모델 학습 실패${NC}"
    exit 1
fi

# 3. 최종 요약
echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}모든 모델 학습 완료!${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""
echo "생성된 모델:"
ls -lh ${MODEL_DIR}/*_v*.pth 2>/dev/null
echo ""

echo "모델 메타데이터:"
ls -lh ${MODEL_DIR}/*_v*.json 2>/dev/null
echo ""

echo "성능 리포트:"
ls -lh ocad/models/metadata/performance_reports/*_report.json 2>/dev/null
echo ""

echo -e "${GREEN}다음 단계:${NC}"
echo "  1. 추론 성능 테스트:"
echo "     python scripts/test_inference_performance.py --model-dir ${MODEL_DIR}"
echo ""
echo "  2. 시스템 통합 테스트:"
echo "     python scripts/test_system_integration.py"
echo ""
echo "  3. 데이터 확인:"
echo "     python scripts/view_training_data.py"
echo ""
