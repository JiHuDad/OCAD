#!/usr/bin/env bash
# =============================================================================
# OCAD ML Pipeline: Complete Sample Data Generation Script
# =============================================================================
#
# This script generates training and validation datasets for all 4 protocols:
# - BFD (Bidirectional Forwarding Detection)
# - BGP (Border Gateway Protocol)
# - PTP (Precision Time Protocol)
# - CFM (Connectivity Fault Management)
#
# Usage:
#   # Quick test (5 minutes, 3 sessions/peers/slaves/endpoints)
#   ./scripts/generate_all_ml_data.sh --quick
#
#   # Medium dataset (1 hour training, 10 sessions)
#   ./scripts/generate_all_ml_data.sh --medium
#
#   # Large dataset (4 hours training, 20 sessions)
#   ./scripts/generate_all_ml_data.sh --large
#
#   # Custom configuration
#   ./scripts/generate_all_ml_data.sh --train-hours 2 --val-hours 0.5 --count 15
#
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Color output
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# =============================================================================
# Default Configuration
# =============================================================================
MODE="medium"  # quick, medium, large, custom
TRAIN_HOURS=1.0
VAL_HOURS=0.3
COUNT=10  # Number of sessions/peers/slaves/endpoints
COLLECTION_INTERVAL_BFD=5
COLLECTION_INTERVAL_BGP=10
COLLECTION_INTERVAL_PTP=5
COLLECTION_INTERVAL_CFM=10
OUTPUT_DIR="data"
SEED=""  # Empty means no seed (random)
PROTOCOLS="all"  # all, bfd, bgp, ptp, cfm (comma-separated)

# =============================================================================
# Parse Command Line Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            TRAIN_HOURS=0.083  # 5 minutes
            VAL_HOURS=0.033    # 2 minutes
            COUNT=3
            shift
            ;;
        --medium)
            MODE="medium"
            TRAIN_HOURS=1.0
            VAL_HOURS=0.3
            COUNT=10
            shift
            ;;
        --large)
            MODE="large"
            TRAIN_HOURS=4.0
            VAL_HOURS=1.0
            COUNT=20
            shift
            ;;
        --train-hours)
            MODE="custom"
            TRAIN_HOURS="$2"
            shift 2
            ;;
        --val-hours)
            MODE="custom"
            VAL_HOURS="$2"
            shift 2
            ;;
        --count)
            MODE="custom"
            COUNT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --protocols)
            PROTOCOLS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Presets:"
            echo "  --quick         Quick test (5min train, 3 count)"
            echo "  --medium        Medium dataset (1h train, 10 count) [default]"
            echo "  --large         Large dataset (4h train, 20 count)"
            echo ""
            echo "Custom Options:"
            echo "  --train-hours N      Training data duration in hours (default: 1.0)"
            echo "  --val-hours N        Validation data duration in hours (default: 0.3)"
            echo "  --count N            Number of sessions/peers/slaves/endpoints (default: 10)"
            echo "  --output DIR         Output directory (default: data)"
            echo "  --seed N             Random seed for reproducibility"
            echo "  --protocols LIST     Comma-separated list: all,bfd,bgp,ptp,cfm (default: all)"
            echo ""
            echo "Examples:"
            echo "  $0 --quick"
            echo "  $0 --large --seed 42"
            echo "  $0 --train-hours 2 --count 15"
            echo "  $0 --protocols bfd,cfm"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Display Configuration
# =============================================================================
echo ""
echo "================================================================================"
echo "  OCAD ML Pipeline: Sample Data Generation"
echo "================================================================================"
echo ""
info "Configuration:"
echo "  Mode:                ${MODE}"
echo "  Training duration:   ${TRAIN_HOURS}h"
echo "  Validation duration: ${VAL_HOURS}h"
echo "  Count:               ${COUNT} sessions/peers/slaves/endpoints"
echo "  Output directory:    ${OUTPUT_DIR}"
echo "  Protocols:           ${PROTOCOLS}"
if [ -n "$SEED" ]; then
    echo "  Random seed:         ${SEED}"
else
    echo "  Random seed:         (random)"
fi
echo ""

# =============================================================================
# Helper function to check if protocol should be generated
# =============================================================================
should_generate_protocol() {
    local protocol=$1
    if [ "$PROTOCOLS" = "all" ]; then
        return 0
    elif echo "$PROTOCOLS" | grep -q "$protocol"; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Generate BFD Data
# =============================================================================
if should_generate_protocol "bfd"; then
    echo "================================================================================"
    echo "  1/4: BFD (Bidirectional Forwarding Detection)"
    echo "================================================================================"
    echo ""

    BFD_CMD="python3 scripts/generate_bfd_ml_data.py \
        --sessions ${COUNT} \
        --train-hours ${TRAIN_HOURS} \
        --val-hours ${VAL_HOURS} \
        --collection-interval ${COLLECTION_INTERVAL_BFD} \
        --output ${OUTPUT_DIR}/bfd"

    if [ -n "$SEED" ]; then
        BFD_CMD="$BFD_CMD --seed $SEED"
    fi

    info "Executing: $BFD_CMD"
    eval $BFD_CMD

    if [ $? -eq 0 ]; then
        success "BFD data generation completed!"
    else
        error "BFD data generation failed!"
        exit 1
    fi
    echo ""
else
    warning "Skipping BFD (not in --protocols list)"
    echo ""
fi

# =============================================================================
# Generate BGP Data
# =============================================================================
if should_generate_protocol "bgp"; then
    echo "================================================================================"
    echo "  2/4: BGP (Border Gateway Protocol)"
    echo "================================================================================"
    echo ""

    BGP_CMD="python3 scripts/generate_bgp_ml_data.py \
        --peers ${COUNT} \
        --train-hours ${TRAIN_HOURS} \
        --val-hours ${VAL_HOURS} \
        --collection-interval ${COLLECTION_INTERVAL_BGP} \
        --output ${OUTPUT_DIR}/bgp"

    if [ -n "$SEED" ]; then
        BGP_CMD="$BGP_CMD --seed $SEED"
    fi

    info "Executing: $BGP_CMD"
    eval $BGP_CMD

    if [ $? -eq 0 ]; then
        success "BGP data generation completed!"
    else
        error "BGP data generation failed!"
        exit 1
    fi
    echo ""
else
    warning "Skipping BGP (not in --protocols list)"
    echo ""
fi

# =============================================================================
# Generate PTP Data
# =============================================================================
if should_generate_protocol "ptp"; then
    echo "================================================================================"
    echo "  3/4: PTP (Precision Time Protocol)"
    echo "================================================================================"
    echo ""

    PTP_CMD="python3 scripts/generate_ptp_ml_data.py \
        --slaves ${COUNT} \
        --train-hours ${TRAIN_HOURS} \
        --val-hours ${VAL_HOURS} \
        --collection-interval ${COLLECTION_INTERVAL_PTP} \
        --output ${OUTPUT_DIR}/ptp"

    if [ -n "$SEED" ]; then
        PTP_CMD="$PTP_CMD --seed $SEED"
    fi

    info "Executing: $PTP_CMD"
    eval $PTP_CMD

    if [ $? -eq 0 ]; then
        success "PTP data generation completed!"
    else
        error "PTP data generation failed!"
        exit 1
    fi
    echo ""
else
    warning "Skipping PTP (not in --protocols list)"
    echo ""
fi

# =============================================================================
# Generate CFM Data
# =============================================================================
if should_generate_protocol "cfm"; then
    echo "================================================================================"
    echo "  4/4: CFM (Connectivity Fault Management)"
    echo "================================================================================"
    echo ""

    CFM_CMD="python3 scripts/generate_cfm_ml_data.py \
        --endpoints ${COUNT} \
        --train-hours ${TRAIN_HOURS} \
        --val-hours ${VAL_HOURS} \
        --collection-interval ${COLLECTION_INTERVAL_CFM} \
        --output ${OUTPUT_DIR}/cfm"

    if [ -n "$SEED" ]; then
        CFM_CMD="$CFM_CMD --seed $SEED"
    fi

    info "Executing: $CFM_CMD"
    eval $CFM_CMD

    if [ $? -eq 0 ]; then
        success "CFM data generation completed!"
    else
        error "CFM data generation failed!"
        exit 1
    fi
    echo ""
else
    warning "Skipping CFM (not in --protocols list)"
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo "================================================================================"
echo "  ✅ DATA GENERATION COMPLETE!"
echo "================================================================================"
echo ""
info "Generated datasets:"
if should_generate_protocol "bfd"; then
    echo "  📁 BFD:  ${OUTPUT_DIR}/bfd/{train,val_normal,val_anomaly}/"
fi
if should_generate_protocol "bgp"; then
    echo "  📁 BGP:  ${OUTPUT_DIR}/bgp/{train,val_normal,val_anomaly}/"
fi
if should_generate_protocol "ptp"; then
    echo "  📁 PTP:  ${OUTPUT_DIR}/ptp/{train,val_normal,val_anomaly}/"
fi
if should_generate_protocol "cfm"; then
    echo "  📁 CFM:  ${OUTPUT_DIR}/cfm/{train,val_normal,val_anomaly}/"
fi
echo ""

info "Next steps:"
echo "  1. Train models:"
if should_generate_protocol "bfd"; then
    echo "     python3 scripts/train_bfd_hmm.py --data ${OUTPUT_DIR}/bfd/train"
fi
if should_generate_protocol "bgp"; then
    echo "     python3 scripts/train_bgp_gnn.py --data ${OUTPUT_DIR}/bgp/train"
fi
if should_generate_protocol "ptp"; then
    echo "     python3 scripts/train_ptp_tcn.py --data ${OUTPUT_DIR}/ptp/train"
fi
if should_generate_protocol "cfm"; then
    echo "     python3 scripts/train_cfm_isoforest.py --train-data ${OUTPUT_DIR}/cfm/train/*.parquet"
fi
echo ""
echo "  2. Run inference:"
if should_generate_protocol "bfd"; then
    echo "     python3 scripts/infer_bfd.py --model models/bfd/hmm_v1.0.0.pkl --detector hmm --data ${OUTPUT_DIR}/bfd/val_normal ${OUTPUT_DIR}/bfd/val_anomaly"
fi
if should_generate_protocol "cfm"; then
    echo "     python3 scripts/infer_cfm_isoforest.py --val-normal ${OUTPUT_DIR}/cfm/val_normal/*.parquet --val-anomaly ${OUTPUT_DIR}/cfm/val_anomaly/*.parquet"
fi
echo ""

success "All done! 🎉"
echo ""
