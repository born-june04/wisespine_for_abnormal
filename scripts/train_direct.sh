#!/bin/bash
# Direct Training Script (No SLURM)
# Run on already allocated node

set -e

# Activate environment
source ~/.bashrc
conda activate py311

# Create directories
mkdir -p logs outputs/training

# Parse arguments
USE_ENHANCED_FRACTURE=""
RUN_NAME="original"
if [[ "$@" == *"--use_enhanced_fracture"* ]]; then
    USE_ENHANCED_FRACTURE="--use_enhanced_fracture"
    RUN_NAME="enhanced"
fi

# Print configuration
echo "========================================"
echo "TotalSegmentator Fine-tuning (Direct)"
echo "========================================"
echo "Node: $(hostname)"
echo "Run name: $RUN_NAME"
echo "Enhanced Fracture: ${USE_ENHANCED_FRACTURE:-false}"
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"
echo "========================================"

# Data paths
DATA_ROOT="/gscratch/scrubbed/june0604/wisespine_for_abnormal/data/raw/verse"
OUTPUT_DIR="/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/training"

# Training parameters
EPOCHS=100
BATCH_SIZE=2
LR=1e-4
VAL_INTERVAL=1

# Augmentation parameters
HARDWARE_PROB=0.5
FRACTURE_PROB=0.5
SCREW_PROB=0.8
ROD_PROB=0.8
ARTIFACT_STRENGTH=0.7

# Detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
else
    echo "ERROR: nvidia-smi not found. Please run this on a GPU node."
    echo "Hint: Use 'srun -p gpu-a40 --gres=gpu:2 --pty bash' to get GPU access"
    exit 1
fi

if [ $NUM_GPUS -eq 0 ]; then
    echo "ERROR: No GPUs detected. Please run this on a GPU node."
    exit 1
fi

echo "Detected $NUM_GPUS GPUs"

# Log file
LOG_FILE="logs/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Run training
if [ $NUM_GPUS -gt 1 ]; then
    echo "Running with torchrun (multi-GPU)..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        training/train_totalseg.py \
        --data_root $DATA_ROOT \
        --output_dir $OUTPUT_DIR \
        $USE_ENHANCED_FRACTURE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --val_interval $VAL_INTERVAL \
        --hardware_prob $HARDWARE_PROB \
        --fracture_prob $FRACTURE_PROB \
        --screw_prob $SCREW_PROB \
        --rod_prob $ROD_PROB \
        --artifact_strength $ARTIFACT_STRENGTH \
        --num_workers 4 2>&1 | tee $LOG_FILE
else
    echo "Running with single GPU..."
    python training/train_totalseg.py \
        --data_root $DATA_ROOT \
        --output_dir $OUTPUT_DIR \
        $USE_ENHANCED_FRACTURE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --val_interval $VAL_INTERVAL \
        --hardware_prob $HARDWARE_PROB \
        --fracture_prob $FRACTURE_PROB \
        --screw_prob $SCREW_PROB \
        --rod_prob $ROD_PROB \
        --artifact_strength $ARTIFACT_STRENGTH \
        --num_workers 4 2>&1 | tee $LOG_FILE
fi

echo "========================================"
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
echo "========================================"

