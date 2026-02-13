#!/bin/bash
# Run ablation study directly (no SLURM)

echo "========================================"
echo "Running Ablation Study (Direct)"
echo "========================================"
echo ""

# Check if on GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Make sure you're on a GPU node."
    exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs on $(hostname)"
echo ""

# Run both ablations sequentially
echo "Starting Ablation 1: Original Fracture Augmentation"
echo "----------------------------------------"
bash scripts/train_direct.sh
echo ""

echo "Starting Ablation 2: Enhanced Fracture Augmentation"
echo "----------------------------------------"
bash scripts/train_direct.sh --use_enhanced_fracture
echo ""

echo "========================================"
echo "✓ Ablation study completed!"
echo "========================================"
echo "Check results in: outputs/training/"
echo "View logs in: logs/"

