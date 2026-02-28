# WiseSpine for Abnormal CT

**Robust Spine Segmentation via Synthetic Augmentation of Normal CT Data**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

---

## Motivation

State-of-the-art spine segmentation models (TotalSegmentator / nnU-Net) achieve Dice > 0.95 on **normal** spine CT but degrade significantly on **abnormal** cases — surgical hardware, fractures, and deformities. Collecting and labeling large-scale abnormal CT datasets is expensive and ethically constrained, so we take a **synthetic augmentation** approach: augment normal VerSe data with realistic abnormalities and retrain nnU-Net.

## Core Question

> **Can synthetic augmentation of normal CT data produce a segmentation model that generalizes to real abnormal spine CT?**

This repository answers that question through a controlled ablation study comparing:

1. **Traditional CV fracture augmentation** — deformation fields, morphological operations
2. **Physics-based fracture augmentation** — direct compression resampling, cortical disruption, sclerosis, column shortening
3. **Surgical hardware augmentation** — pedicle screws, rods, bone cement, metal artifacts

---

## Ablation Study Design

| ID | Experiment | Hardware | Fracture | Dataset ID | Purpose |
|----|-----------|:--------:|:--------:|:----------:|---------|
| **A** | Baseline | - | - | 500 | Normal segmentation baseline |
| **B** | Hardware Only | Yes | - | 501 | Isolated hardware augmentation effect |
| **C** | Fracture Only (Trad.) | - | Traditional CV | 502 | Traditional CV fracture alone |
| **D** | Fracture Only (Phys.) | - | Physics-based | 503 | Physics-based fracture alone |
| **E** | Full (Traditional) | Yes | Traditional CV | 504 | Hardware + Traditional fracture |
| **F** | Full (Physics) | Yes | Physics-based | 505 | **Main method** — Hardware + Physics fracture |

### Key Comparisons

| Comparison | Analysis |
|-----------|----------|
| **A vs F** | Overall augmentation effect (main result) |
| **C vs D** | Traditional CV vs Physics-based fracture — direct comparison |
| **E vs F** | Impact of fracture method in the full pipeline |
| **B, C, D** | Per-component contribution |

---

## Current Results (1000 Epochs)

All models trained with nnU-Net 3d_fullres, fold 0, 2-GPU DDP. Validation on fold 0 (205 cases, all normal spine).

| ID | Method | EMA Dice | Last 50 avg | Last 200 avg | Trend |
|----|--------|:--------:|:-----------:|:------------:|:-----:|
| **A** | Baseline | **0.918** | 0.914 | 0.900 | +0.012 |
| **B** | Hardware Only | 0.869 | 0.867 | 0.841 | +0.021 |
| **C** | Fracture Trad. | 0.909 | 0.909 | 0.890 | +0.013 |
| **D** | Fracture Phys. | 0.891 | 0.882 | 0.860 | +0.024 |
| **E** | Full Trad. | 0.889 | 0.884 | 0.862 | +0.020 |
| **F** | Full Phys. | 0.871 | 0.867 | 0.840 | +0.022 |

> **Note:** All models show positive trends (+0.01~0.02), indicating convergence is not yet complete at 1000 epochs. Augmented models converge slower but are expected to close the gap with extended training. Current validation is on **normal** data only — the true effect will be measured on **abnormal test sets**.

![Convergence comparison — Dice and Loss curves for all 6 ablation models](figs/convergence_comparison.png)

---

## Next Steps

1. **Extend training** to 2000+ epochs (all models still improving)
2. **Evaluate on real abnormal CT** — the critical test of augmentation effectiveness
3. **Per-vertebra analysis** — identify which vertebral levels benefit most from augmentation
4. **Bridge to WiseSpine** — use findings to guide more sophisticated physics-based fracture simulation

---

## Quick Start

```bash
conda activate py311
cd /gscratch/scrubbed/june0604/wisespine_for_abnormal

# Baseline training (GPU 0,1)
bash scripts/train_nnunet.sh

# Ablation (GPU 2,3)
CUDA_VISIBLE_DEVICES="2,3" nnUNetv2_train 503 3d_fullres 0 --npz -num_gpus 2
```

## Repository Structure

```
wisespine_for_abnormal/
├── augmentation/
│   ├── surgical_hardware.py       # Pedicle screws, rods, cement, metal artifacts
│   ├── fractures.py               # Traditional CV fracture augmentation
│   └── fractures_enhanced.py      # Physics-based fracture augmentation
├── training/
│   ├── train_nnunet.py            # nnU-Net training pipeline with augmentation
│   ├── data_loader.py             # VerSe dataset loader
│   └── ...
├── evaluation/
│   └── evaluate_segmentation.py   # Dice / HD95 evaluation
├── scripts/                       # Shell scripts & visualization tools
├── docs/
│   └── augmentation_methods.md    # Detailed augmentation documentation
└── figs/                          # All figures for documentation
```

## Documentation

Detailed augmentation methods, physics explanation, and ablation strategy:

- **[docs/augmentation_methods.md](docs/augmentation_methods.md)** — Complete technical documentation

---

**Contact**: june0604@uw.edu
