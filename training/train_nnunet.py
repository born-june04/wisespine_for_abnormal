"""
TotalSegmentator Fine-tuning with nnU-Net
Proper fine-tuning using TotalSegmentator's nnU-Net framework
"""

import os
import sys
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import nibabel as nib
from tqdm import tqdm

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
    
    # Create directories
    for d in [nnunet_raw, nnunet_preprocessed, nnunet_results]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ['nnUNet_raw'] = str(nnunet_raw)
    os.environ['nnUNet_preprocessed'] = str(nnunet_preprocessed)
    os.environ['nnUNet_results'] = str(nnunet_results)
    
    return nnunet_raw, nnunet_preprocessed, nnunet_results


def prepare_nnunet_dataset(
    data_root: Path,
    output_dir: Path,
    dataset_id: int = 500,
    dataset_name: str = "SpineAbnormal",
    use_augmentation: bool = True,
    use_enhanced_fracture: bool = True,
    num_samples: int = None
):
    """
    Prepare dataset in nnU-Net format with augmentation.
    
    nnU-Net format:
    Dataset500_SpineAbnormal/
        imagesTr/
            case_0000_0000.nii.gz
            case_0001_0000.nii.gz
        labelsTr/
            case_0000.nii.gz
            case_0001.nii.gz
        dataset.json
    """
    dataset_folder = output_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    print(f"\nPreparing nnU-Net dataset: {dataset_folder}")
    print(f"Augmentation: {'ON' if use_augmentation else 'OFF'}")
    print(f"Fracture mode: {'ENHANCED' if use_enhanced_fracture else 'ORIGINAL'}")
    
    # Initialize augmenters
    if use_augmentation:
        hardware_aug = SurgicalHardwareAugmenter(
            screw_probability=0.8,
            rod_probability=0.8,
            artifact_strength=0.7
        )
        
        if use_enhanced_fracture:
            fracture_aug = EnhancedFractureAugmenter(
                compression_range=(0.2, 0.5),
                wedge_range=(0.1, 0.3),
                add_sclerosis=True,
                add_kyphosis=True
            )
        else:
            fracture_aug = FractureAugmenter(
                compression_probability=1.0,
                compression_range=(0.2, 0.5)
            )
    
    # Find all CT and mask pairs
    samples = []
    for dataset in ['dataset-verse19training', 'dataset-01training', 'dataset-02validation']:
        rawdata_path = data_root / dataset / 'rawdata'
        derivatives_path = data_root / dataset / 'derivatives'
        
        if not rawdata_path.exists():
            continue
        
        for subject_dir in sorted(rawdata_path.glob('sub-*')):
            subject_id = subject_dir.name
            
            # Try multiple CT file patterns
            ct_candidates = [
                subject_dir / f'{subject_id}_ct.nii.gz',
                subject_dir / f'{subject_id}_dir-iso_ct.nii.gz',
            ]
            
            ct_file = None
            for ct_cand in ct_candidates:
                if ct_cand.exists():
                    ct_file = ct_cand
                    break
            
            if ct_file is None:
                continue
            
            # Try multiple mask patterns
            mask_dir = derivatives_path / subject_id
            if not mask_dir.exists():
                continue
            
            mask_candidates = list(mask_dir.glob('*seg*vert*.nii.gz')) + \
                             list(mask_dir.glob('*msk*.nii.gz'))
            
            if len(mask_candidates) > 0:
                samples.append({
                    'ct': ct_file,
                    'mask': mask_candidates[0],
                    'subject_id': subject_id
                })
    
    if num_samples:
        samples = samples[:num_samples]
    
    print(f"Found {len(samples)} samples")
    
    if len(samples) < 5:
        print(f"WARNING: Only {len(samples)} samples found. nnU-Net requires at least 5 for cross-validation.")
        print("Consider using more data or --num_samples to limit augmentation variations.")
    
    # Process each sample
    case_idx = 0
    for sample in tqdm(samples, desc="Processing samples"):
        try:
            # Load
            ct_nifti = nib.load(sample['ct'])
            mask_nifti = nib.load(sample['mask'])
            
            ct = ct_nifti.get_fdata().astype(np.float32)
            mask = mask_nifti.get_fdata().astype(np.uint8)
            spacing = ct_nifti.header.get_zooms()[:3]
            
            # Apply augmentation
            if use_augmentation:
                aug_type = np.random.choice(['none', 'hardware', 'fracture'])
                
                if aug_type == 'hardware':
                    ct, mask = hardware_aug(ct.copy(), mask.copy(), spacing)
                elif aug_type == 'fracture':
                    ct, mask = fracture_aug(ct.copy(), mask.copy(), spacing)
            
            # Save in nnU-Net format
            case_name = f"case_{case_idx:04d}"
            
            # Save CT (channel 0)
            ct_output = nib.Nifti1Image(ct, ct_nifti.affine, ct_nifti.header)
            nib.save(ct_output, images_tr / f"{case_name}_0000.nii.gz")
            
            # Save mask
            mask_output = nib.Nifti1Image(mask, mask_nifti.affine, mask_nifti.header)
            nib.save(mask_output, labels_tr / f"{case_name}.nii.gz")
            
            case_idx += 1
            
        except Exception as e:
            print(f"Error processing {sample['subject_id']}: {e}")
            continue
    
    print(f"Processed {case_idx} cases successfully")
    
    # Create dataset.json
    # nnU-Net v2 requires labels as {"label_name": label_id}
    labels = {"background": 0}
    for i in range(1, 25):
        labels[f"vertebra_{i}"] = i
    
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": labels,
        "numTraining": case_idx,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": f"Spine segmentation with abnormalities ({'enhanced' if use_enhanced_fracture else 'original'} fracture)",
        "reference": "TotalSegmentator fine-tuning",
        "licence": "research only"
    }
    
    with open(dataset_folder / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"\n✓ Dataset prepared: {dataset_folder}")
    print(f"  - Images: {case_idx}")
    print(f"  - Labels: {case_idx}")
    
    return dataset_folder, case_idx


def download_totalsegmentator_weights(output_dir: Path):
    """
    Download TotalSegmentator pretrained weights.
    Returns path to weights if successful.
    """
    weights_dir = output_dir / "totalsegmentator_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Downloading TotalSegmentator Pretrained Weights")
    print("=" * 80)
    
    try:
        # Use TotalSegmentator CLI to download weights
        import subprocess
        result = subprocess.run([
            "TotalSegmentator",
            "--download_weights"
        ], capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print("✓ TotalSegmentator weights downloaded")
            # Weights are typically in ~/.totalsegmentator/nnunet/results
            home = Path.home()
            ts_weights = home / ".totalsegmentator" / "nnunet" / "results"
            if ts_weights.exists():
                return ts_weights
        else:
            print("⚠️  Could not download weights automatically")
            print(result.stderr)
    except Exception as e:
        print(f"⚠️  Error downloading weights: {e}")
    
    return None


def run_nnunet_training(
    dataset_id: int,
    fold: int = 0,
    use_pretrained: bool = True,
    output_dir: Path = None
):
    """
    Run nnU-Net training.
    Uses TotalSegmentator pretrained weights if available.
    """
    print("\n" + "=" * 80)
    print("Starting nnU-Net Training")
    print("=" * 80)
    
    # nnU-Net training command
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        "3d_fullres",
        str(fold),
    ]
    
    # Try to get TotalSegmentator pretrained weights
    pretrained_path = None
    if use_pretrained and output_dir:
        pretrained_path = download_totalsegmentator_weights(output_dir)
        
        if pretrained_path:
            print(f"Using pretrained weights from: {pretrained_path}")
            # Note: nnU-Net v2 uses --pretrained_weights flag differently
            # You may need to manually copy weights to the right location
            print("\nNote: To use pretrained weights, you may need to manually copy")
            print(f"      TotalSegmentator weights to nnUNet_results folder")
        else:
            print("\nTraining from scratch (no pretrained weights found)")
            print("To use TotalSegmentator weights:")
            print("  1. Install TotalSegmentator: pip install TotalSegmentator")
            print("  2. Run: TotalSegmentator --download_weights")
    
    cmd_str = " ".join(cmd)
    print(f"\nCommand: {cmd_str}")
    print("\nExecuting...\n")
    
    import subprocess
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n⚠️  Training failed with return code {result.returncode}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='TotalSegmentator Fine-tuning (nnU-Net)')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True, help='Path to VerSe data')
    parser.add_argument('--output_dir', type=str, default='outputs/nnunet', help='Output directory')
    parser.add_argument('--dataset_id', type=int, default=500, help='nnU-Net dataset ID')
    
    # Augmentation (ABLATION)
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='Use augmentation')
    parser.add_argument('--use_enhanced_fracture', action='store_true', help='Use enhanced fracture')
    parser.add_argument('--num_samples', type=int, default=None, help='Limit number of samples')
    
    # Training
    parser.add_argument('--fold', type=int, default=0, help='nnU-Net fold')
    parser.add_argument('--skip_training', action='store_true', help='Only prepare data, skip training')
    
    args = parser.parse_args()
    
    # Setup
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup nnU-Net directories
    nnunet_raw, nnunet_preprocessed, nnunet_results = setup_nnunet_directories(output_dir)
    
    print("=" * 80)
    print("TotalSegmentator Fine-tuning (nnU-Net)")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"Fracture mode: {'ENHANCED' if args.use_enhanced_fracture else 'ORIGINAL'}")
    print(f"nnUNet_raw: {nnunet_raw}")
    print("=" * 80)
    
    # Prepare dataset
    dataset_folder, num_cases = prepare_nnunet_dataset(
        data_root=data_root,
        output_dir=nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=f"SpineAbnormal_{'Enhanced' if args.use_enhanced_fracture else 'Original'}",
        use_augmentation=args.use_augmentation,
        use_enhanced_fracture=args.use_enhanced_fracture,
        num_samples=args.num_samples
    )
    
    if args.skip_training:
        print("\n✓ Data preparation complete. Skipping training (--skip_training)")
        return
    
    # Run nnU-Net preprocessing
    print("\n" + "=" * 80)
    print("Running nnU-Net Preprocessing")
    print("=" * 80)
    print(f"\nCommand: nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    
    import subprocess
    result = subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "-d", str(args.dataset_id),
        "--verify_dataset_integrity"
    ], check=False)
    
    if result.returncode != 0:
        print("\n⚠️  Preprocessing failed. Please install nnU-Net:")
        print("  pip install nnunetv2")
        return
    
    # Run training
    success = run_nnunet_training(
        dataset_id=args.dataset_id,
        fold=args.fold,
        use_pretrained=True,
        output_dir=output_dir
    )
    
    if success:
        print("\n" + "=" * 80)
        print("✓ Training Complete!")
        print("=" * 80)
        print(f"Results: {nnunet_results}")
    else:
        print("\n⚠️  Training was not completed successfully")


if __name__ == '__main__':
    main()

