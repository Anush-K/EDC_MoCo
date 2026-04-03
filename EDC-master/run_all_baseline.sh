#!/usr/bin/env bash
# =============================================================================
# run_all_baseline.sh
#
# Baseline EDC — ImageNet-pretrained ResNet50 encoder + ResNet50 decoder.
# No SSL. This is the "EDC Baseline" condition for the paper comparison table.
#
# Datasets: APTOS, LungCT, BUSI
#
# USAGE:
#   cd /home/cs24d0008/EDC_SSL/EDC_Moco/EDC-master
#   chmod +x run_all_baseline.sh
#   tmux new -s baseline
#   ./run_all_baseline.sh
#   Detach: Ctrl+B D   |   Re-attach: tmux attach -t baseline
# =============================================================================

set -e

VENV_DIR="/home/cs24d0008/EDC_SSL/moco_env"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Hyperparameters (keep identical across all conditions for fair comparison)
GPU=0
SEED=42
AMP=True

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${CODE_DIR}/logs"
cd "$CODE_DIR"
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="/home/cs24d0008/EDC_SSL/EDC_Moco/EDC-master:$PYTHONPATH"
echo "Python : $(which python)"
echo "Version: $(python --version)"
echo "============================================================"
echo " EDC BASELINE (ImageNet encoder, no SSL)"
echo " GPU  : $GPU  |  SEED : $SEED  |  AMP : $AMP"
echo "============================================================"

# ── Helper ────────────────────────────────────────────────────────────────────
run_dataset () {
    local RUNNER=$1
    local SAVE_NAME=$2
    local LOG="${CODE_DIR}/logs/${SAVE_NAME}.log"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Starting : $SAVE_NAME"
    echo "  Log      : $LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python "$RUNNER" \
        --gpu             "$GPU"       \
        --save_name       "$SAVE_NAME" \
        --seed            "$SEED"      \
        --amp             "$AMP"       \
        --use_tensorboard \
        2>&1 | tee "$LOG"
    echo "  Finished : $SAVE_NAME"
}

# ── Run each dataset ──────────────────────────────────────────────────────────
run_dataset "runners_edc/edc_aptos.py"   "edc_baseline_aptos"
run_dataset "runners_edc/edc_lungct.py"  "edc_baseline_lungct"
run_dataset "runners_edc/edc_busi.py"    "edc_baseline_busi"

echo ""
echo "============================================================"
echo " BASELINE: ALL DATASETS COMPLETE"
echo " Logs   → logs/edc_baseline_*.log"
echo " Models → saved_models/edc_baseline_*/"
echo "============================================================"