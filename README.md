# WiseSpine for Abnormal CT

VerSe 데이터 augmentation을 통한 abnormal spine CT segmentation 프로젝트.

## 목표

TotalSegmentator (nnU-Net)를 surgical hardware, fractures 등 abnormal한 spine CT에 대해 robust하게 만든다.

## Quick Start

```bash
conda activate py311
cd /gscratch/scrubbed/june0604/wisespine_for_abnormal

# Baseline 학습 (GPU 0,1)
bash scripts/train_nnunet.sh

# Ablation 실행 (GPU 2,3)
CUDA_VISIBLE_DEVICES="2,3" nnUNetv2_train 503 3d_fullres 0 --npz -num_gpus 2
```

## 📚 Documentation

Augmentation 방법, 시각화, ablation 전략 모두 한 파일에 정리:

→ **[docs/augmentation_methods.md](docs/augmentation_methods.md)**

## 현재 상태

- ✅ Augmentation 구현 완료 (Hardware, Fracture Original, Fracture Enhanced)
- ✅ nnU-Net 데이터 파이프라인 구축
- 🔄 Baseline 학습 진행 중
- ⏳ Ablation 실험 대기 중

---

**Target**: MICCAI 2026 | **Contact**: june0604@uw.edu
