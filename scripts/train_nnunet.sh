#!/bin/bash
# nnU-Net Training Script with Ablation Support
# Usage:
#   bash scripts/train_nnunet.sh baseline          # A. No augmentation
#   bash scripts/train_nnunet.sh hardware           # B. Hardware only
#   bash scripts/train_nnunet.sh fracture_trad      # C. Traditional CV fracture
#   bash scripts/train_nnunet.sh fracture_phys      # D. Physics-based fracture
#   bash scripts/train_nnunet.sh full_trad           # E. Hardware + Traditional fracture
#   bash scripts/train_nnunet.sh full_phys           # F. Hardware + Physics fracture (MAIN)
#
# Resume:   bash scripts/train_nnunet.sh <experiment> (auto-detects checkpoint)
# Optional: bash scripts/train_nnunet.sh <experiment> <fold> <num_gpus>

set -e

# ─── Parse arguments ─────────────────────────────────────────────────
EXPERIMENT="${1:-baseline}"
FOLD="${2:-0}"
NUM_GPUS="${3:-2}"
DATASET_ID=500

# Map experiment name → nnU-Net trainer class
case "$EXPERIMENT" in
    baseline)
        TRAINER="nnUNetTrainer"
        DESC="A. Baseline (no augmentation)"
        MASTER_PORT=29500
        ;;
    hardware)
        TRAINER="nnUNetTrainer_HardwareOnly"
        DESC="B. Hardware Only"
        MASTER_PORT=29501
        ;;
    fracture_trad)
        TRAINER="nnUNetTrainer_FractureTrad"
        DESC="C. Fracture Only (Traditional CV)"
        MASTER_PORT=29502
        ;;
    fracture_phys)
        TRAINER="nnUNetTrainer_FracturePhys"
        DESC="D. Fracture Only (Physics-Based)"
        MASTER_PORT=29503
        ;;
    full_trad)
        TRAINER="nnUNetTrainer_FullTrad"
        DESC="E. Full (Hardware + Traditional Fracture)"
        MASTER_PORT=29504
        ;;
    full_phys)
        TRAINER="nnUNetTrainer_FullPhys"
        DESC="F. Full (Hardware + Physics Fracture) [MAIN]"
        MASTER_PORT=29505
        ;;
    *)
        echo "ERROR: Unknown experiment '$EXPERIMENT'"
        echo ""
        echo "Available experiments:"
        echo "  baseline       - A. No augmentation"
        echo "  hardware       - B. Hardware only"
        echo "  fracture_trad  - C. Traditional CV fracture only"
        echo "  fracture_phys  - D. Physics-based fracture only"
        echo "  full_trad      - E. Hardware + Traditional fracture"
        echo "  full_phys      - F. Hardware + Physics fracture (MAIN)"
        echo ""
        echo "Usage: bash scripts/train_nnunet.sh <experiment> [fold] [num_gpus]"
        exit 1
        ;;
esac

# ─── Environment setup ───────────────────────────────────────────────
export nnUNet_raw="/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/nnunet/nnUNet_preprocessed"
export nnUNet_results="/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/nnunet/nnUNet_results"

# ─── Redirect all caches to scrubbed (avoid home disk quota) ─────────
CACHE_BASE="/gscratch/scrubbed/june0604/.cache_training"
mkdir -p "$CACHE_BASE"

export TMPDIR="$CACHE_BASE/tmp"
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="$CACHE_BASE"
export MPLCONFIGDIR="$CACHE_BASE/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export TORCH_HOME="$CACHE_BASE/torch"
mkdir -p "$TORCH_HOME"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
mkdir -p "$TRITON_CACHE_DIR"

# Performance tuning
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Disable torch.compile (avoid compilation overhead)
export nnUNet_compile="False"

# Disable AMP (PyTorch 2.3.1 compatibility)
export nnUNet_disable_amp="True"

# Better error messages
export CUDA_LAUNCH_BLOCKING=1

# GPU selection
export CUDA_VISIBLE_DEVICES="0,1"

# ─── Log setup ────────────────────────────────────────────────────────
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${EXPERIMENT}_fold${FOLD}_${TIMESTAMP}.log"

# ─── Print configuration ─────────────────────────────────────────────
echo "========================================"
echo "nnU-Net Ablation Training"
echo "========================================"
echo "Experiment : $DESC"
echo "Trainer    : $TRAINER"
echo "Dataset    : $DATASET_ID"
echo "Fold       : $FOLD"
echo "GPUs       : $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Log        : $LOG_FILE"
echo "Node       : $(hostname)"
echo "Time       : $(date)"
echo "========================================"
echo ""
echo "Results will be saved to:"
echo "  $nnUNet_results/Dataset${DATASET_ID}_*/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/"
echo "========================================"

# ─── Auto-detect resume ───────────────────────────────────────────────
export MASTER_PORT="$MASTER_PORT"

RESUME_FLAG=""
CHECKPOINT_DIR="$nnUNet_results/Dataset${DATASET_ID}_SpineAbnormal_Original/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}"
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
    echo ">> Checkpoint found — resuming training (--c)"
    RESUME_FLAG="--c"
else
    echo ">> No checkpoint found — starting fresh"
fi

# ─── Run training ─────────────────────────────────────────────────────
if [ "$TRAINER" = "nnUNetTrainer" ]; then
    nnUNetv2_train "$DATASET_ID" 3d_fullres "$FOLD" \
        -num_gpus "$NUM_GPUS" $RESUME_FLAG 2>&1 | tee "$LOG_FILE"
else
    nnUNetv2_train "$DATASET_ID" 3d_fullres "$FOLD" \
        -tr "$TRAINER" \
        -num_gpus "$NUM_GPUS" $RESUME_FLAG 2>&1 | tee "$LOG_FILE"
fi

# ─── Done ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "✓ Training completed: $DESC"
echo "  Log    : $LOG_FILE"
echo "  Results: $nnUNet_results/Dataset${DATASET_ID}_*/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/"
echo "========================================"
