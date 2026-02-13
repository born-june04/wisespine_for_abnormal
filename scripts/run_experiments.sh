#!/bin/bash

# Training script for WiseSpine Abnormal CT project
# This script demonstrates how to train the model with different augmentation strategies

# Configuration
DATA_ROOT="/gscratch/scrubbed/june0604/wisespine_for_abnormal/data/raw/verse"
OUTPUT_DIR="/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs"
BATCH_SIZE=2
NUM_EPOCHS=50
LEARNING_RATE=1e-4
NUM_WORKERS=4

# Experiment 1: Baseline (No augmentation)
echo "=========================================="
echo "Experiment 1: Baseline (No augmentation)"
echo "=========================================="
python training/train_ts.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --augmentation none \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS

# Experiment 2: Hardware augmentation only
echo "=========================================="
echo "Experiment 2: Hardware augmentation"
echo "=========================================="
python training/train_ts.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --augmentation hardware \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS

# Experiment 3: Fracture augmentation only
echo "=========================================="
echo "Experiment 3: Fracture augmentation"
echo "=========================================="
python training/train_ts.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --augmentation fracture \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS

# Experiment 4: Both augmentations
echo "=========================================="
echo "Experiment 4: Both augmentations"
echo "=========================================="
python training/train_ts.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --augmentation both \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

