#!/bin/bash
#
# Run inference for OCAD protocol anomaly detection
#
# Usage:
#   ./scripts/infer.sh --protocol bfd --model models/bfd/hmm_v1.0.0 --data data/bfd/val
#   ./scripts/infer.sh --protocol bgp --model models/bgp/gnn_v1.0.0 --data data/bgp/val
#   ./scripts/infer.sh --protocol ptp --model models/ptp/tcn_v1.0.0 --data data/ptp/val
#   ./scripts/infer.sh --protocol cfm --model models/cfm/isoforest_v1.0.0 --data data/cfm/val
#
# Options:
#   --protocol       Protocol to infer (bfd|bgp|ptp|cfm)
#   --model          Model directory (auto-detects .pkl/.pth files)
#   --data           Inference data directory
#   --help           Show this help message
#
# Output:
#   Creates results/{protocol}/infer_YYYYMMDD_HHMMSS/ directory with:
#     - predictions.csv
#     - report.md
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Function to find model file in directory
find_model_file() {
    local model_dir=$1
    
    # Try .pkl first (HMM, Isolation Forest)
    local pkl_file=$(find "$model_dir" -maxdepth 1 -name "*.pkl" | head -1)
    if [ -n "$pkl_file" ]; then
        echo "$pkl_file"
        return 0
    fi
    
    # Try .pth (LSTM, GNN, TCN)
    local pth_file=$(find "$model_dir" -maxdepth 1 -name "*.pth" | head -1)
    if [ -n "$pth_file" ]; then
        echo "$pth_file"
        return 0
    fi
    
    return 1
}

# Parse arguments
PROTOCOL=""
MODEL_DIR=""
DATA_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --protocol)
            PROTOCOL="$2"
            shift 2
            ;;
        --model)
            MODEL_DIR="$2"
            shift 2
            ;;
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            echo ""
            show_help
            ;;
    esac
done

# Validate required arguments
if [ -z "$PROTOCOL" ]; then
    log_error "Missing required argument: --protocol"
    echo ""
    show_help
fi

if [ -z "$MODEL_DIR" ]; then
    log_error "Missing required argument: --model"
    echo ""
    show_help
fi

if [ -z "$DATA_DIR" ]; then
    log_error "Missing required argument: --data"
    echo ""
    show_help
fi

# Validate protocol
case $PROTOCOL in
    bfd|bgp|ptp|cfm)
        ;;
    *)
        log_error "Invalid protocol: $PROTOCOL (must be bfd|bgp|ptp|cfm)"
        exit 1
        ;;
esac

# Validate model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    log_error "Model directory not found: $MODEL_DIR"
    exit 1
fi

# Find model file
MODEL_FILE=$(find_model_file "$MODEL_DIR")
if [ -z "$MODEL_FILE" ]; then
    log_error "No model file (.pkl or .pth) found in: $MODEL_DIR"
    exit 1
fi

log_info "Found model file: $MODEL_FILE"

# Validate data directory exists
if [ ! -d "$DATA_DIR" ]; then
    log_error "Data directory not found: $DATA_DIR"
    exit 1
fi

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment if exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    log_info "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/$PROTOCOL/infer_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

log_info "Output directory: $OUTPUT_DIR"

# Inference logic per protocol
echo ""
log_info "Starting inference for protocol: $PROTOCOL"
echo ""

PREDICTIONS_FILE="$OUTPUT_DIR/predictions.csv"

case $PROTOCOL in
    bfd)
        # Detect model type from file extension
        if [[ "$MODEL_FILE" == *.pkl ]]; then
            DETECTOR="hmm"
        elif [[ "$MODEL_FILE" == *.pth ]]; then
            DETECTOR="lstm"
        else
            log_error "Unknown model type for BFD"
            exit 1
        fi

        log_info "Detected BFD detector type: $DETECTOR"
        python "$SCRIPT_DIR/infer_bfd.py" \
            --model "$MODEL_FILE" \
            --detector "$DETECTOR" \
            --data "$DATA_DIR" \
            --output "$PREDICTIONS_FILE"
        ;;

    bgp)
        log_info "Running BGP GNN inference..."
        python "$SCRIPT_DIR/infer_bgp.py" \
            --model "$MODEL_FILE" \
            --data "$DATA_DIR" \
            --output "$PREDICTIONS_FILE"
        ;;

    ptp)
        log_info "Running PTP TCN inference..."
        python "$SCRIPT_DIR/infer_ptp.py" \
            --model "$MODEL_FILE" \
            --data "$DATA_DIR" \
            --output "$PREDICTIONS_FILE"
        ;;

    cfm)
        log_info "Running CFM Isolation Forest inference..."
        python "$SCRIPT_DIR/infer_cfm_isoforest.py" \
            --model "$MODEL_FILE" \
            --data "$DATA_DIR" \
            --output "$PREDICTIONS_FILE"
        ;;
esac

# Check if inference was successful
if [ $? -eq 0 ] && [ -f "$PREDICTIONS_FILE" ]; then
    echo ""
    log_success "Inference completed successfully!"
    log_success "Predictions saved to: $PREDICTIONS_FILE"

    # Show predictions info
    echo ""
    log_info "Predictions file size:"
    ls -lh "$PREDICTIONS_FILE"

    # Generate report
    echo ""
    log_info "Generating performance report..."
    REPORT_FILE="$OUTPUT_DIR/report.md"

    case $PROTOCOL in
        bfd)
            python "$SCRIPT_DIR/report_bfd.py" \
                --predictions "$PREDICTIONS_FILE" \
                --output "$REPORT_FILE"
            ;;
        bgp)
            python "$SCRIPT_DIR/report_bgp.py" \
                --predictions "$PREDICTIONS_FILE" \
                --output "$REPORT_FILE"
            ;;
        ptp)
            python "$SCRIPT_DIR/report_ptp.py" \
                --predictions "$PREDICTIONS_FILE" \
                --output "$REPORT_FILE"
            ;;
        cfm)
            python "$SCRIPT_DIR/report_cfm.py" \
                --predictions "$PREDICTIONS_FILE" \
                --output "$REPORT_FILE"
            ;;
    esac

    if [ $? -eq 0 ] && [ -f "$REPORT_FILE" ]; then
        log_success "Report saved to: $REPORT_FILE"
        
        echo ""
        log_info "Generated files:"
        echo "  📊 Predictions: $PREDICTIONS_FILE"
        echo "  📄 Report:      $REPORT_FILE"
    else
        log_warning "Report generation failed (predictions are still available)"
    fi

else
    echo ""
    log_error "Inference failed!"
    exit 1
fi

echo ""
log_success "All done! 🎉"
echo ""
echo "View report:"
echo "  cat $REPORT_FILE"
