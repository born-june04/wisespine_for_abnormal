"""
TotalSegmentator Fine-tuning - Simplified Version
Prepares augmented data for nnU-Net training without loading all at once
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

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
from augmentation.surgical_hardware import SurgicalHardwareAugmenter
from augmentation.fractures import FractureAugmenter
from augmentation.fractures_enhanced import EnhancedFractureAugmenter


def setup_nnunet_directories(output_base: Path):
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


def prepare_nnunet_dataset_simple(
    data_root: Path,
    output_dir: Path,
    dataset_id: int,
    dataset_name: str,
    use_enhanced_fracture: bool,
    max_samples: int = 10  # Limit to avoid memory issues
):
    """
    Prepare dataset in nnU-Net format.
    Processes one sample at a time to avoid memory issues.
    """
    dataset_folder = output_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    print(f"\nPreparing dataset: Dataset{dataset_id:03d}_{dataset_name}")
    print(f"Fracture mode: {'ENHANCED' if use_enhanced_fracture else 'ORIGINAL'}")
    
    # Find samples
    verse_root = data_root / 'dataset-verse19training'
    rawdata = verse_root / 'rawdata'
    derivatives = verse_root / 'derivatives'
    
    samples = []
    for subj_dir in sorted(rawdata.glob('sub-verse*')):
        subj_id = subj_dir.name
        ct_file = subj_dir / f'{subj_id}_ct.nii.gz'
        mask_dir = derivatives / subj_id
        
        if not ct_file.exists() or not mask_dir.exists():
            continue
        
        mask_files = list(mask_dir.glob('*msk.nii.gz'))
        if len(mask_files) > 0:
            samples.append({
                'ct': ct_file,
                'mask': mask_files[0],
                'id': subj_id
            })
    
    samples = samples[:max_samples]
    print(f"Found {len(samples)} samples (limited to {max_samples})")
    
    # Initialize augmenters
    hardware_aug = SurgicalHardwareAugmenter(
        screw_probability=0.7, rod_probability=0.7, artifact_strength=0.6
    )
    
    if use_enhanced_fracture:
        fracture_aug = EnhancedFractureAugmenter(
            compression_range=(0.2, 0.4),
            wedge_range=(0.1, 0.2),
            add_sclerosis=True,
            add_kyphosis=True
        )
    else:
        fracture_aug = FractureAugmenter(
            compression_probability=1.0,
            compression_range=(0.2, 0.4)
        )
    
    # Process each sample ONE AT A TIME
    case_idx = 0
    for sample in tqdm(samples, desc="Processing"):
        try:
            # Load
            ct_nifti = nib.load(sample['ct'])
            mask_nifti = nib.load(sample['mask'])
            
            ct = ct_nifti.get_fdata().astype(np.float32)
            mask = mask_nifti.get_fdata().astype(np.uint8)
            spacing = ct_nifti.header.get_zooms()[:3]
            
            # Original
            case_name = f"case_{case_idx:04d}"
            nib.save(nib.Nifti1Image(ct, ct_nifti.affine), images_tr / f"{case_name}_0000.nii.gz")
            nib.save(nib.Nifti1Image(mask, mask_nifti.affine), labels_tr / f"{case_name}.nii.gz")
            case_idx += 1
            
            # Hardware augmented
            ct_hw, mask_hw = hardware_aug(ct.copy(), mask.copy(), spacing)
            case_name = f"case_{case_idx:04d}"
            nib.save(nib.Nifti1Image(ct_hw, ct_nifti.affine), images_tr / f"{case_name}_0000.nii.gz")
            nib.save(nib.Nifti1Image(mask_hw, mask_nifti.affine), labels_tr / f"{case_name}.nii.gz")
            case_idx += 1
            del ct_hw, mask_hw
            
            # Fracture augmented
            ct_fr, mask_fr = fracture_aug(ct.copy(), mask.copy(), spacing)
            case_name = f"case_{case_idx:04d}"
            nib.save(nib.Nifti1Image(ct_fr, ct_nifti.affine), images_tr / f"{case_name}_0000.nii.gz")
            nib.save(nib.Nifti1Image(mask_fr, mask_nifti.affine), labels_tr / f"{case_name}.nii.gz")
            case_idx += 1
            del ct_fr, mask_fr
            
            # Clean up
            del ct, mask, ct_nifti, mask_nifti
            gc.collect()
            
        except Exception as e:
            print(f"\nError with {sample['id']}: {e}")
            continue
    
    print(f"\nProcessed {case_idx} cases (original + augmented)")
    
    # dataset.json
    labels = {"background": 0}
    for i in range(1, 25):
        labels[f"vertebra_{i}"] = i
    
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": case_idx,
        "file_ending": ".nii.gz"
    }
    
    with open(dataset_folder / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"✓ Dataset ready: {dataset_folder}")
    return dataset_folder, case_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', default='outputs/nnunet')
    parser.add_argument('--dataset_id', type=int, default=500)
    parser.add_argument('--use_enhanced_fracture', action='store_true')
    parser.add_argument('--max_samples', type=int, default=5, help='Max samples to avoid memory issues')
    parser.add_argument('--skip_training', action='store_true')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nnunet_raw, _, _ = setup_nnunet_directories(output_dir)
    
    mode = "Enhanced" if args.use_enhanced_fracture else "Original"
    print(f"\n{'='*80}")
    print(f"TotalSegmentator Fine-tuning - {mode} Fracture")
    print(f"{'='*80}\n")
    
    dataset_folder, num_cases = prepare_nnunet_dataset_simple(
        data_root=data_root,
        output_dir=nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=f"Spine_{mode}",
        use_enhanced_fracture=args.use_enhanced_fracture,
        max_samples=args.max_samples
    )
    
    if args.skip_training:
        print("\n✓ Data ready. Skipping training.")
        print(f"\nTo train manually:")
        print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id}")
        print(f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0")
        return
    
    # Run preprocessing
    print(f"\n{'='*80}")
    print("Running nnU-Net Preprocessing")
    print(f"{'='*80}\n")
    
    import subprocess
    result = subprocess.run([
        "nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id)
    ])
    
    if result.returncode != 0:
        print("\n⚠️  Preprocessing failed")
        return
    
    # Run training
    print(f"\n{'='*80}")
    print("Starting nnU-Net Training")
    print(f"{'='*80}\n")
    
    result = subprocess.run([
        "nnUNetv2_train", str(args.dataset_id), "3d_fullres", "0"
    ])
    
    if result.returncode == 0:
        print(f"\n✓ Training complete!")
    else:
        print(f"\n⚠️  Training failed")


if __name__ == '__main__':
    main()

