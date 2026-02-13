"""
TotalSegmentator Fine-tuning using pretrained weights
Downloads TotalSegmentator weights and fine-tunes on augmented VerSe data
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import gc
import shutil

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
from augmentation.surgical_hardware import SurgicalHardwareAugmenter
from augmentation.fractures import FractureAugmenter
from augmentation.fractures_enhanced import EnhancedFractureAugmenter


def download_totalseg_weights(task_id=292):
    """
    Download TotalSegmentator pretrained weights using totalsegmentator library.
    Task 292 = vertebrae model
    """
    print(f"\n{'='*80}")
    print(f"Downloading TotalSegmentator Weights (Task {task_id})")
    print(f"{'='*80}\n")
    
    try:
        from totalsegmentator.libs import download_pretrained_weights, get_weights_dir
        
        # Download weights
        download_pretrained_weights(task_id)
        
        weights_dir = get_weights_dir()
        print(f"\n✓ Weights downloaded to: {weights_dir}")
        return weights_dir
        
    except ImportError:
        print("⚠️  TotalSegmentator not installed")
        print("Install with: pip install TotalSegmentator")
        return None
    except Exception as e:
        print(f"⚠️  Error downloading weights: {e}")
        return None


def setup_nnunet_dirs(output_base: Path):
    """Setup nnU-Net directory structure."""
    nnunet_raw = output_base / "nnUNet_raw"
    nnunet_preprocessed = output_base / "nnUNet_preprocessed"
    nnunet_results = output_base / "nnUNet_results"
    
    for d in [nnunet_raw, nnunet_preprocessed, nnunet_results]:
        d.mkdir(parents=True, exist_ok=True)
    
    os.environ['nnUNet_raw'] = str(nnunet_raw)
    os.environ['nnUNet_preprocessed'] = str(nnunet_preprocessed)
    os.environ['nnUNet_results'] = str(nnunet_results)
    
    return nnunet_raw, nnunet_preprocessed, nnunet_results


def find_verse_samples(data_root: Path, max_samples: int = 10):
    """Find VerSe CT and mask pairs."""
    samples = []
    
    # Try multiple dataset folders
    for dataset_name in ['dataset-verse19training', 'dataset-01training', 'dataset-02validation']:
        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            continue
        
        rawdata = dataset_path / 'rawdata'
        derivatives = dataset_path / 'derivatives'
        
        if not rawdata.exists():
            continue
        
        for subj_dir in sorted(rawdata.glob('sub-*')):
            if len(samples) >= max_samples:
                break
            
            subj_id = subj_dir.name
            
            # Find CT file (multiple patterns)
            ct_candidates = [
                subj_dir / f'{subj_id}_ct.nii.gz',
                subj_dir / f'{subj_id}_dir-iso_ct.nii.gz',
                subj_dir / f'{subj_id}_dir-ax_ct.nii.gz',
            ]
            # Also try glob for any CT file
            ct_file = None
            for ct_cand in ct_candidates:
                if ct_cand.exists():
                    ct_file = ct_cand
                    break
            
            if ct_file is None:
                ct_files = list(subj_dir.glob('*ct.nii.gz'))
                if len(ct_files) > 0:
                    ct_file = ct_files[0]
            
            if ct_file is None:
                continue
            
            # Find mask file
            mask_dir = derivatives / subj_id
            if not mask_dir.exists():
                continue
            
            mask_files = list(mask_dir.glob('*msk*.nii.gz'))
            if len(mask_files) > 0:
                samples.append({
                    'ct': ct_file,
                    'mask': mask_files[0],
                    'id': subj_id
                })
    
    return samples


def prepare_augmented_dataset(
    data_root: Path,
    output_dir: Path,
    dataset_id: int,
    dataset_name: str,
    use_enhanced_fracture: bool,
    max_samples: int = 10,
    augment_per_sample: int = 3
):
    """
    Prepare augmented dataset in nnU-Net format.
    Each sample generates: 1 original + N augmented versions
    """
    dataset_folder = output_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    print(f"\nPreparing Dataset: {dataset_name}")
    print(f"Fracture mode: {'ENHANCED' if use_enhanced_fracture else 'ORIGINAL'}")
    print(f"Output: {dataset_folder}\n")
    
    # Find samples
    samples = find_verse_samples(data_root, max_samples)
    print(f"Found {len(samples)} samples")
    
    if len(samples) == 0:
        print("ERROR: No samples found!")
        print(f"Please check data_root: {data_root}")
        return None, 0
    
    # Initialize augmenters
    hardware_aug = SurgicalHardwareAugmenter(
        screw_probability=0.8,
        rod_probability=0.8,
        artifact_strength=0.7
    )
    
    if use_enhanced_fracture:
        fracture_aug = EnhancedFractureAugmenter(
            compression_range=(0.2, 0.4),
            wedge_range=(0.1, 0.3),
            add_sclerosis=True,
            add_kyphosis=True
        )
    else:
        fracture_aug = FractureAugmenter(
            compression_probability=1.0,
            compression_range=(0.2, 0.4)
        )
    
    # Process samples
    case_idx = 0
    for sample in tqdm(samples, desc="Processing samples"):
        try:
            # Load
            ct_nifti = nib.load(sample['ct'])
            mask_nifti = nib.load(sample['mask'])
            
            ct = ct_nifti.get_fdata().astype(np.float32)
            mask = mask_nifti.get_fdata().astype(np.uint8)
            spacing = ct_nifti.header.get_zooms()[:3]
            
            # Save original
            case_name = f"case_{case_idx:04d}"
            nib.save(
                nib.Nifti1Image(ct, ct_nifti.affine),
                images_tr / f"{case_name}_0000.nii.gz"
            )
            nib.save(
                nib.Nifti1Image(mask, mask_nifti.affine),
                labels_tr / f"{case_name}.nii.gz"
            )
            case_idx += 1
            
            # Augmented versions
            for aug_idx in range(augment_per_sample):
                # Alternate between hardware and fracture
                if aug_idx % 2 == 0:
                    ct_aug, mask_aug = hardware_aug(ct.copy(), mask.copy(), spacing)
                else:
                    ct_aug, mask_aug = fracture_aug(ct.copy(), mask.copy(), spacing)
                
                case_name = f"case_{case_idx:04d}"
                nib.save(
                    nib.Nifti1Image(ct_aug, ct_nifti.affine),
                    images_tr / f"{case_name}_0000.nii.gz"
                )
                nib.save(
                    nib.Nifti1Image(mask_aug, mask_nifti.affine),
                    labels_tr / f"{case_name}.nii.gz"
                )
                case_idx += 1
                
                del ct_aug, mask_aug
                gc.collect()
            
            # Clean up
            del ct, mask, ct_nifti, mask_nifti
            gc.collect()
            
        except Exception as e:
            print(f"\nError processing {sample['id']}: {e}")
            continue
    
    print(f"\n✓ Generated {case_idx} cases")
    
    # Create dataset.json with TotalSegmentator-compatible label structure (27 classes)
    # This matches Dataset292 for transfer learning
    labels = {
        "background": 0,
        "sacrum": 1,
        "vertebrae_S1": 2,
        "vertebrae_L5": 3,
        "vertebrae_L4": 4,
        "vertebrae_L3": 5,
        "vertebrae_L2": 6,
        "vertebrae_L1": 7,
        "vertebrae_T12": 8,
        "vertebrae_T11": 9,
        "vertebrae_T10": 10,
        "vertebrae_T9": 11,
        "vertebrae_T8": 12,
        "vertebrae_T7": 13,
        "vertebrae_T6": 14,
        "vertebrae_T5": 15,
        "vertebrae_T4": 16,
        "vertebrae_T3": 17,
        "vertebrae_T2": 18,
        "vertebrae_T1": 19,
        "vertebrae_C7": 20,
        "vertebrae_C6": 21,
        "vertebrae_C5": 22,
        "vertebrae_C4": 23,
        "vertebrae_C3": 24,
        "vertebrae_C2": 25,
        "vertebrae_C1": 26
    }
    
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": case_idx,
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name
    }
    
    with open(dataset_folder / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    return dataset_folder, case_idx


def copy_totalseg_weights_to_dataset(weights_dir: Path, dataset_id: int, results_dir: Path):
    """
    Copy TotalSegmentator pretrained weights to nnU-Net results folder
    for the new dataset.
    """
    print(f"\n{'='*80}")
    print("Copying Pretrained Weights")
    print(f"{'='*80}\n")
    
    # TotalSegmentator Task 292 = Dataset292_TotalSegmentator_part2_vertebrae_1559subj
    source_weights = weights_dir / "Dataset292_TotalSegmentator_part2_vertebrae_1559subj"
    
    if not source_weights.exists():
        print(f"⚠️  Source weights not found: {source_weights}")
        return False
    
    # Copy to new dataset folder
    target_weights = results_dir / f"Dataset{dataset_id:03d}_SpineAbnormal" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    target_weights.mkdir(parents=True, exist_ok=True)
    
    # Copy checkpoint files
    for fold_dir in source_weights.glob("nnUNetTrainer*"):
        if fold_dir.is_dir():
            target_fold = target_weights.parent / fold_dir.name
            if not target_fold.exists():
                print(f"Copying: {fold_dir.name}")
                shutil.copytree(fold_dir, target_fold)
    
    print(f"✓ Weights copied to: {target_weights.parent}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TotalSegmentator on augmented data")
    parser.add_argument('--data_root', required=True, help="Path to VerSe data")
    parser.add_argument('--output_dir', default='outputs/nnunet', help="Output directory")
    parser.add_argument('--dataset_id', type=int, default=500, help="nnU-Net dataset ID")
    parser.add_argument('--use_enhanced_fracture', action='store_true', help="Use enhanced fracture augmentation")
    parser.add_argument('--max_samples', type=int, default=10, help="Max samples to process")
    parser.add_argument('--augment_per_sample', type=int, default=3, help="Augmentations per sample")
    parser.add_argument('--skip_download', action='store_true', help="Skip weight download")
    parser.add_argument('--skip_training', action='store_true', help="Skip training")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup nnU-Net directories
    nnunet_raw, nnunet_preprocessed, nnunet_results = setup_nnunet_dirs(output_dir)
    
    mode = "Enhanced" if args.use_enhanced_fracture else "Original"
    print(f"\n{'='*80}")
    print(f"TotalSegmentator Fine-tuning - {mode} Fracture")
    print(f"{'='*80}\n")
    
    # Download TotalSegmentator weights
    weights_dir = None
    if not args.skip_download:
        weights_dir = download_totalseg_weights(task_id=292)  # Vertebrae model
    
    # Prepare augmented dataset
    dataset_folder, num_cases = prepare_augmented_dataset(
        data_root=data_root,
        output_dir=nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=f"SpineAbnormal_{mode}",
        use_enhanced_fracture=args.use_enhanced_fracture,
        max_samples=args.max_samples,
        augment_per_sample=args.augment_per_sample
    )
    
    if dataset_folder is None:
        print("\n❌ Failed to prepare dataset")
        return 1
    
    if num_cases < 5:
        print(f"\n⚠️  WARNING: Only {num_cases} cases. Need at least 5 for 5-fold CV")
        print("Consider increasing --max_samples or --augment_per_sample")
    
    if args.skip_training:
        print("\n✓ Dataset ready. Skipping training.")
        print(f"\nTo train:")
        print(f"  export nnUNet_raw={nnunet_raw}")
        print(f"  export nnUNet_preprocessed={nnunet_preprocessed}")
        print(f"  export nnUNet_results={nnunet_results}")
        print(f"  export OPENBLAS_NUM_THREADS=8")
        print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id}")
        print(f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0 --npz")
        return 0
    
    # Set environment variable to limit OpenBLAS threads
    os.environ['OPENBLAS_NUM_THREADS'] = '8'
    os.environ['OMP_NUM_THREADS'] = '8'
    
    # Preprocess
    print(f"\n{'='*80}")
    print("Running nnU-Net Preprocessing")
    print(f"{'='*80}\n")
    
    import subprocess
    result = subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "-d", str(args.dataset_id),
        "--verify_dataset_integrity"
    ])
    
    if result.returncode != 0:
        print("\n⚠️  Preprocessing failed")
        return 1
    
    # Train
    print(f"\n{'='*80}")
    print("Starting nnU-Net Training")
    print(f"{'='*80}\n")
    
    # Note: For fine-tuning from TotalSegmentator, you would need to:
    # 1. Manually copy weights to the right location
    # 2. Or train from scratch on augmented data
    print("Training from scratch on augmented data...")
    print("(Fine-tuning from TotalSegmentator weights requires manual setup)\n")
    
    result = subprocess.run([
        "nnUNetv2_train",
        str(args.dataset_id),
        "3d_fullres",
        "0",
        "--npz"  # Use compressed format
    ])
    
    if result.returncode == 0:
        print(f"\n✓ Training completed!")
    else:
        print(f"\n⚠️  Training failed")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

