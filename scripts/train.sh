#!/bin/bash
#
# Train ML models for OCAD protocol anomaly detection
#
# Usage:
#   ./scripts/train.sh --protocol bfd --data data/bfd/train --output models/bfd/hmm_v1.0.0
#   ./scripts/train.sh --protocol bgp --data data/bgp/train --output models/bgp/gnn_v1.0.0
#   ./scripts/train.sh --protocol ptp --data data/ptp/train --output models/ptp/tcn_v1.0.0
#   ./scripts/train.sh --protocol cfm --data data/cfm/train --output models/cfm/isoforest_v1.0.0
#
# Options:
#   --protocol       Protocol to train (bfd|bgp|ptp|cfm)
#   --data           Training data directory
#   --output         Output model directory (will create if not exists)
#   --model-type     Model type (optional, for protocols with multiple models)
#                    BFD: hmm (default), lstm
#   --help           Show this help message
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

# Parse arguments
PROTOCOL=""
DATA_DIR=""
OUTPUT_DIR=""
MODEL_TYPE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --protocol)
            PROTOCOL="$2"
            shift 2
            ;;
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
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

if [ -z "$DATA_DIR" ]; then
    log_error "Missing required argument: --data"
    echo ""
    show_help
fi

if [ -z "$OUTPUT_DIR" ]; then
    log_error "Missing required argument: --output"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"
log_info "Output directory: $OUTPUT_DIR"

# Training logic per protocol
echo ""
log_info "Starting training for protocol: $PROTOCOL"
echo ""

case $PROTOCOL in
    bfd)
        # Determine model type
        if [ -z "$MODEL_TYPE" ]; then
            MODEL_TYPE="hmm"
            log_warning "No model type specified, using default: hmm"
        fi

        case $MODEL_TYPE in
            hmm)
                MODEL_FILE="$OUTPUT_DIR/model.pkl"
                log_info "Training BFD HMM model..."
                python "$SCRIPT_DIR/train_bfd_hmm.py" \
                    --data "$DATA_DIR" \
                    --output "$MODEL_FILE"
                ;;
            lstm)
                MODEL_FILE="$OUTPUT_DIR/model.pth"
                log_info "Training BFD LSTM model..."
                python "$SCRIPT_DIR/train_bfd_lstm.py" \
                    --data "$DATA_DIR" \
                    --output "$MODEL_FILE"
                ;;
            *)
                log_error "Invalid BFD model type: $MODEL_TYPE (must be hmm|lstm)"
                exit 1
                ;;
        esac
        ;;

    bgp)
        MODEL_FILE="$OUTPUT_DIR/model.pth"
        log_info "Training BGP GNN model..."
        python "$SCRIPT_DIR/train_bgp_gnn.py" \
            --data "$DATA_DIR" \
            --output "$MODEL_FILE"
        ;;

    ptp)
        MODEL_FILE="$OUTPUT_DIR/model.pth"
        log_info "Training PTP TCN model..."
        python "$SCRIPT_DIR/train_ptp_tcn.py" \
            --data "$DATA_DIR" \
            --output "$MODEL_FILE"
        ;;

    cfm)
        # CFM creates multiple model files (one per metric)
        MODEL_FILE="$OUTPUT_DIR"  # Directory, not single file
        log_info "Training CFM Isolation Forest model..."

        # Find training data file (parquet or csv)
        if [ -f "$DATA_DIR" ]; then
            # DATA_DIR is a file
            TRAIN_DATA_FILE="$DATA_DIR"
        elif [ -d "$DATA_DIR" ]; then
            # DATA_DIR is a directory, find first parquet file
            TRAIN_DATA_FILE=$(find "$DATA_DIR" -maxdepth 1 -name "*.parquet" | head -1)
            if [ -z "$TRAIN_DATA_FILE" ]; then
                # No parquet, try csv
                TRAIN_DATA_FILE=$(find "$DATA_DIR" -maxdepth 1 -name "*.csv" | head -1)
            fi
            if [ -z "$TRAIN_DATA_FILE" ]; then
                log_error "No training data file (.parquet or .csv) found in $DATA_DIR"
                exit 1
            fi
        else
            log_error "Training data not found: $DATA_DIR"
            exit 1
        fi

        log_info "Using training data: $TRAIN_DATA_FILE"
        python "$SCRIPT_DIR/train_cfm_isoforest.py" \
            --train-data "$TRAIN_DATA_FILE" \
            --output-dir "$OUTPUT_DIR"
        ;;
esac

# Check if training was successful
if [ $? -eq 0 ]; then
    # For CFM, check if any .pkl files were created
    if [ "$PROTOCOL" = "cfm" ]; then
        MODEL_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.pkl" | wc -l)
        if [ "$MODEL_COUNT" -gt 0 ]; then
            echo ""
            log_success "Training completed successfully!"
            log_success "Models saved to: $OUTPUT_DIR"
            echo ""
            log_info "Model files:"
            ls -lh "$OUTPUT_DIR"/*.pkl
        else
            echo ""
            log_error "Training failed - no model files created!"
            exit 1
        fi
    # For other protocols, check single model file
    elif [ -f "$MODEL_FILE" ]; then
        echo ""
        log_success "Training completed successfully!"
        log_success "Model saved to: $MODEL_FILE"

        # Show model info
        echo ""
        log_info "Model information:"
        ls -lh "$MODEL_FILE"
    else
        echo ""
        log_error "Training failed - model file not created!"
        exit 1
    fi

    # Create metadata file
    METADATA_FILE="$OUTPUT_DIR/metadata.json"
    cat > "$METADATA_FILE" <<EOF
{
  "protocol": "$PROTOCOL",
  "model_type": "${MODEL_TYPE:-${PROTOCOL}}",
  "model_file": "$(basename "$MODEL_FILE")",
  "training_data": "$DATA_DIR",
  "trained_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "version": "1.0.0"
}
EOF
    log_info "Metadata saved to: $METADATA_FILE"

else
    echo ""
    log_error "Training failed!"
    exit 1
fi

echo ""
log_success "All done! 🎉"
