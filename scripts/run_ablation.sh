#!/bin/bash
# Run ablation study: Original vs Enhanced fracture augmentation

# Ablation 1: Original fracture augmentation
echo "Submitting job: Original Fracture Augmentation"
sbatch scripts/slurm_train_totalseg.sh

# Ablation 2: Enhanced fracture augmentation
echo "Submitting job: Enhanced Fracture Augmentation"
sbatch scripts/slurm_train_totalseg.sh --use_enhanced_fracture

echo ""
echo "✓ Submitted 2 ablation jobs"
echo "  1. Original fracture (deformation field)"
echo "  2. Enhanced fracture (physical compression + wedge + sclerosis)"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs: tail -f logs/train_*.out"

