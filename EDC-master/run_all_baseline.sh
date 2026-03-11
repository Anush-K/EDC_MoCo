#!/usr/bin/env bash
# =============================================================================
# run_all_baseline.sh
#
# Baseline EDC — ImageNet-pretrained ResNet50 encoder + ResNet50 decoder.
# No SSL. This is the "EDC Baseline" condition for your paper comparison table.
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#  cd /path/to/EDC-master
#  chmod +x run_all_baseline.sh
#  tmux new -s baseline
#  ./run_all_baseline.sh
#  Detach: Ctrl+B then D   |   Re-attach: tmux attach -t baseline
# =============================================================================

set -e

# ─── EDIT THIS LINE ──────────────────────────────────────────────────────────
VENV_DIR="/home/cs24d0008/EDC_SSL/moco_env"
# ─────────────────────────────────────────────────────────────────────────────

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── HYPER-PARAMETERS ────────────────────────────────────────────────────────
# Keep identical to SSL runs for a fair comparison
GPU=0
# BATCH=32
# EVAL_BATCH=64
# NUM_TRAIN_ITER=1000       # increase for full runs (e.g. 5000)
# NUM_EVAL_ITER=250
# LR=5e-4
# LR_ENCODER=1e-5           # lower lr keeps SSL features stable early on
# WEIGHT_DECAY=1e-4
SEED=42
AMP=True

# ─── SETUP ───────────────────────────────────────────────────────────────────
mkdir -p "${CODE_DIR}/logs"
cd "$CODE_DIR"
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="/home/cs24d0008/EDC_SSL/EDC_Moco/EDC-master:$PYTHONPATH"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo "============================================================"
echo " EDC BASELINE (ImageNet encoder, no SSL) - ALL 5 DATASETS"echo " GPU          : $GPU"
echo " GPU          : $GPU"
echo " AMP          : $AMP"
echo "=============================================="
# ─── HELPER ──────────────────────────────────────────────────────────────────
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
        --gpu             "$GPU"            \
        --save_name       "$SAVE_NAME"      \
        --seed            "$SEED"           \
        --amp             "$AMP"            \
        --use_tensorboard \
        2>&1 | tee "$LOG"

    echo "  Finished : $SAVE_NAME"
}

# ─── RUN EACH DATASET ────────────────────────────────────────────────────────
run_dataset "runners_edc/edc_aptos.py"    "edc_baseline_aptos"
run_dataset "runners_edc/edc_br35h.py"    "edc_baseline_br35h"
run_dataset "runners_edc/edc_isic2018.py" "edc_baseline_isic2018"
run_dataset "runners_edc/edc_oct2017.py"  "edc_baseline_oct2017"
run_dataset "runners_edc/edc_lungct.py"   "edc_baseline_lungct"

echo ""
echo "============================================================"
echo " BASELINE: ALL DATASETS COMPLETE"
echo " Logs    → logs/edc_baseline_*.log"
echo " Models  → saved_models/edc_baseline_*/"
echo "============================================================"
