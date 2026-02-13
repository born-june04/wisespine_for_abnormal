#!/bin/bash
# Fine-tune TotalSegmentator on augmented VerSe data

set -e

source ~/.bashrc
conda activate py311

mkdir -p logs outputs/nnunet

# Parse arguments
USE_ENHANCED_FRACTURE=""
RUN_NAME="original"
DATASET_ID=500

if [[ "$@" == *"--use_enhanced_fracture"* ]]; then
    USE_ENHANCED_FRACTURE="--use_enhanced_fracture"
    RUN_NAME="enhanced"
    DATASET_ID=501
fi

echo "========================================"
echo "TotalSegmentator Fine-tuning"
echo "========================================"
echo "Mode: $RUN_NAME"
echo "Dataset ID: $DATASET_ID"
echo "========================================"

# Limit OpenBLAS threads to avoid resource issues
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8

DATA_ROOT="/gscratch/scrubbed/june0604/wisespine_for_abnormal/data/raw/verse"
OUTPUT_DIR="/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/nnunet"
LOG_FILE="logs/finetune_totalseg_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

python training/finetune_totalseg.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --dataset_id $DATASET_ID \
    $USE_ENHANCED_FRACTURE \
    --max_samples 10 \
    --augment_per_sample 4 2>&1 | tee $LOG_FILE

echo ""
echo "========================================"
echo "Log saved: $LOG_FILE"
echo "========================================"

