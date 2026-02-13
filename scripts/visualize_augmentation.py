"""
Fixed visualization following spine-rl-sim conventions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from matplotlib.patches import Rectangle
import sys

sys.path.append(str(Path(__file__).parent.parent))
from augmentation.surgical_hardware import SurgicalHardwareAugmenter
from augmentation.fractures import FractureAugmenter


def window_ct(ct_volume, window_center=400, window_width=1800):
    """Apply CT windowing."""
    min_hu = window_center - window_width / 2
    max_hu = window_center + window_width / 2
    ct_windowed = np.clip(ct_volume, min_hu, max_hu)
    ct_normalized = (ct_windowed - min_hu) / (max_hu - min_hu)
    return ct_normalized


def visualize_hardware_detail(ct_orig, ct_aug, mask_orig, mask_aug, subject_id, output_path):
    """
    Hardware visualization:
    Row 1: Original with mask overlay
    Row 2: Augmented with metal + changed mask overlay
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Find center from bone
    bone_mask = ct_orig > 200
    if bone_mask.any():
        coords = np.where(bone_mask)
        center = [int(np.mean(c)) for c in coords]
    else:
        center = [s//2 for s in ct_orig.shape]
    
    # Windows
    ct_orig_bone = window_ct(ct_orig, window_center=400, window_width=1800)
    ct_aug_bone = window_ct(ct_aug, window_center=400, window_width=1800)
    
    # Find metal and changed mask
    metal_mask = ct_aug > 5000
    num_metal = metal_mask.sum()
    
    # Find mask differences (hardware affected regions)
    mask_changed = (mask_orig != mask_aug)
    num_changed = mask_changed.sum()
    
    # Row 1: Original with original mask
    # Sagittal
    axes[0, 0].imshow(ct_orig_bone[:, :, center[2]].T, cmap='gray', origin='lower')
    if mask_orig[:, :, center[2]].any():
        axes[0, 0].contour(mask_orig[:, :, center[2]].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    axes[0, 0].set_title('Original + Mask (Green) - Sagittal', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Coronal
    axes[0, 1].imshow(ct_orig_bone[center[0], :, :], cmap='gray', origin='lower')
    if mask_orig[center[0], :, :].any():
        axes[0, 1].contour(mask_orig[center[0], :, :], colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    axes[0, 1].set_title('Original + Mask (Green) - Coronal', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Axial
    axes[0, 2].imshow(ct_orig_bone[:, center[1], :].T, cmap='gray', origin='lower')
    if mask_orig[:, center[1], :].any():
        axes[0, 2].contour(mask_orig[:, center[1], :].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    axes[0, 2].set_title('Original + Mask (Green) - Axial', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Augmented with metal (red), augmented mask (cyan), and changed mask (yellow)
    # Sagittal
    axes[1, 0].imshow(ct_aug_bone[:, :, center[2]].T, cmap='gray', origin='lower')
    if mask_aug[:, :, center[2]].any():
        axes[1, 0].contour(mask_aug[:, :, center[2]].T, colors='cyan', linewidths=1.5, origin='lower', alpha=0.6)
    if metal_mask[:, :, center[2]].any():
        axes[1, 0].contour(metal_mask[:, :, center[2]].T, colors='red', linewidths=2, origin='lower')
    if mask_changed[:, :, center[2]].any():
        axes[1, 0].contour(mask_changed[:, :, center[2]].T, colors='yellow', linewidths=2, origin='lower', alpha=0.8)
    axes[1, 0].set_title('Hardware: Mask (Cyan) + Metal (Red) + Changed (Yellow) - Sagittal', fontsize=9, fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    # Coronal
    axes[1, 1].imshow(ct_aug_bone[center[0], :, :], cmap='gray', origin='lower')
    if mask_aug[center[0], :, :].any():
        axes[1, 1].contour(mask_aug[center[0], :, :], colors='cyan', linewidths=1.5, origin='lower', alpha=0.6)
    if metal_mask[center[0], :, :].any():
        axes[1, 1].contour(metal_mask[center[0], :, :], colors='red', linewidths=2, origin='lower')
    if mask_changed[center[0], :, :].any():
        axes[1, 1].contour(mask_changed[center[0], :, :], colors='yellow', linewidths=2, origin='lower', alpha=0.8)
    axes[1, 1].set_title('Hardware: Mask (Cyan) + Metal (Red) + Changed (Yellow) - Coronal', fontsize=9, fontweight='bold', color='red')
    axes[1, 1].axis('off')
    
    # Axial
    axes[1, 2].imshow(ct_aug_bone[:, center[1], :].T, cmap='gray', origin='lower')
    if mask_aug[:, center[1], :].any():
        axes[1, 2].contour(mask_aug[:, center[1], :].T, colors='cyan', linewidths=1.5, origin='lower', alpha=0.6)
    if metal_mask[:, center[1], :].any():
        axes[1, 2].contour(metal_mask[:, center[1], :].T, colors='red', linewidths=2, origin='lower')
    if mask_changed[:, center[1], :].any():
        axes[1, 2].contour(mask_changed[:, center[1], :].T, colors='yellow', linewidths=2, origin='lower', alpha=0.8)
    axes[1, 2].set_title('Hardware: Mask (Cyan) + Metal (Red) + Changed (Yellow) - Axial', fontsize=9, fontweight='bold', color='red')
    axes[1, 2].axis('off')
    
    fig.suptitle(
        f'{subject_id} - Hardware Augmentation\n'
        f'Metal voxels: {num_metal}, Changed mask voxels: {num_changed}, Max HU: {ct_aug.max():.0f}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Hardware: {output_path.name} (metal voxels: {num_metal})")


def visualize_fracture_detail(ct_orig, ct_aug, mask_orig, mask_aug, subject_id, output_path):
    """
    Fracture visualization:
    Row 1: Original with original mask
    Row 2: Fractured with fractured mask
    Row 3: Difference overlay (shows deformation clearly)
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Find center from bone
    bone_mask = ct_orig > 200
    if bone_mask.any():
        coords = np.where(bone_mask)
        center = [int(np.mean(c)) for c in coords]
    else:
        center = [s//2 for s in ct_orig.shape]
    
    # Windows
    ct_orig_bone = window_ct(ct_orig, window_center=400, window_width=1800)
    ct_aug_bone = window_ct(ct_aug, window_center=400, window_width=1800)
    
    # Find fractured vertebra (highest deformation)
    diff = np.abs(ct_aug - ct_orig)
    
    fractured_label = None
    fractured_mask_orig = np.zeros_like(mask_orig, dtype=bool)
    fractured_mask_aug = np.zeros_like(mask_aug, dtype=bool)
    max_diff_val = 0
    
    for label in np.unique(mask_orig):
        if label == 0:
            continue
        vertebra_mask = (mask_orig == label)
        vertebra_diff = diff[vertebra_mask].mean()
        if vertebra_diff > max_diff_val:
            max_diff_val = vertebra_diff
            fractured_label = label
            fractured_mask_orig = (mask_orig == label)
            fractured_mask_aug = (mask_aug == label)
    
    # Create overlay: CT with red difference overlay
    diff_norm = np.clip(diff / 200.0, 0, 1)
    
    # Row 1: Original with original mask
    # Sagittal
    axes[0, 0].imshow(ct_orig_bone[:, :, center[2]].T, cmap='gray', origin='lower')
    if mask_orig[:, :, center[2]].any():
        axes[0, 0].contour(mask_orig[:, :, center[2]].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_orig[:, :, center[2]].any():
        axes[0, 0].contour(fractured_mask_orig[:, :, center[2]].T, colors='cyan', linewidths=2, origin='lower')
    axes[0, 0].set_title('Original: All Mask (Green) + Target (Cyan) - Sagittal', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Coronal
    axes[0, 1].imshow(ct_orig_bone[center[0], :, :], cmap='gray', origin='lower')
    if mask_orig[center[0], :, :].any():
        axes[0, 1].contour(mask_orig[center[0], :, :], colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_orig[center[0], :, :].any():
        axes[0, 1].contour(fractured_mask_orig[center[0], :, :], colors='cyan', linewidths=2, origin='lower')
    axes[0, 1].set_title('Original: All Mask (Green) + Target (Cyan) - Coronal', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Axial
    axes[0, 2].imshow(ct_orig_bone[:, center[1], :].T, cmap='gray', origin='lower')
    if mask_orig[:, center[1], :].any():
        axes[0, 2].contour(mask_orig[:, center[1], :].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_orig[:, center[1], :].any():
        axes[0, 2].contour(fractured_mask_orig[:, center[1], :].T, colors='cyan', linewidths=2, origin='lower')
    axes[0, 2].set_title('Original: All Mask (Green) + Target (Cyan) - Axial', fontsize=10, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Fractured with augmented mask
    # Sagittal
    axes[1, 0].imshow(ct_aug_bone[:, :, center[2]].T, cmap='gray', origin='lower')
    if mask_aug[:, :, center[2]].any():
        axes[1, 0].contour(mask_aug[:, :, center[2]].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_aug[:, :, center[2]].any():
        axes[1, 0].contour(fractured_mask_aug[:, :, center[2]].T, colors='cyan', linewidths=2, origin='lower')
    axes[1, 0].set_title(f'Fractured: All Mask (Green) + Deformed (Cyan) - Sagittal', fontsize=10, fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    # Coronal
    axes[1, 1].imshow(ct_aug_bone[center[0], :, :], cmap='gray', origin='lower')
    if mask_aug[center[0], :, :].any():
        axes[1, 1].contour(mask_aug[center[0], :, :], colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_aug[center[0], :, :].any():
        axes[1, 1].contour(fractured_mask_aug[center[0], :, :], colors='cyan', linewidths=2, origin='lower')
    axes[1, 1].set_title(f'Fractured: All Mask (Green) + Deformed (Cyan) - Coronal', fontsize=10, fontweight='bold', color='red')
    axes[1, 1].axis('off')
    
    # Axial
    axes[1, 2].imshow(ct_aug_bone[:, center[1], :].T, cmap='gray', origin='lower')
    if mask_aug[:, center[1], :].any():
        axes[1, 2].contour(mask_aug[:, center[1], :].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_aug[:, center[1], :].any():
        axes[1, 2].contour(fractured_mask_aug[:, center[1], :].T, colors='cyan', linewidths=2, origin='lower')
    axes[1, 2].set_title(f'Fractured: All Mask (Green) + Deformed (Cyan) - Axial', fontsize=10, fontweight='bold', color='red')
    axes[1, 2].axis('off')
    
    # Row 3: CT with Difference Overlay (RED = high deformation)
    # Sagittal
    axes[2, 0].imshow(ct_aug_bone[:, :, center[2]].T, cmap='gray', origin='lower')
    if diff_norm[:, :, center[2]].any():
        axes[2, 0].imshow(diff_norm[:, :, center[2]].T, cmap='Reds', alpha=0.5, origin='lower', vmin=0, vmax=1)
    axes[2, 0].set_title('Deformation Overlay (Red) - Sagittal', fontsize=11, fontweight='bold', color='blue')
    axes[2, 0].axis('off')
    
    # Coronal
    axes[2, 1].imshow(ct_aug_bone[center[0], :, :], cmap='gray', origin='lower')
    if diff_norm[center[0], :, :].any():
        axes[2, 1].imshow(diff_norm[center[0], :, :], cmap='Reds', alpha=0.5, origin='lower', vmin=0, vmax=1)
    axes[2, 1].set_title('Deformation Overlay (Red) - Coronal', fontsize=11, fontweight='bold', color='blue')
    axes[2, 1].axis('off')
    
    # Axial
    axes[2, 2].imshow(ct_aug_bone[:, center[1], :].T, cmap='gray', origin='lower')
    if diff_norm[:, center[1], :].any():
        axes[2, 2].imshow(diff_norm[:, center[1], :].T, cmap='Reds', alpha=0.5, origin='lower', vmin=0, vmax=1)
    axes[2, 2].set_title('Deformation Overlay (Red) - Axial', fontsize=11, fontweight='bold', color='blue')
    axes[2, 2].axis('off')
    
    label_text = f"Label {int(fractured_label)}" if fractured_label else "N/A"
    
    fig.suptitle(
        f'{subject_id} - Fracture Augmentation\n'
        f'Fractured: {label_text}, Mean deformation: {diff.mean():.1f} HU, Max: {diff.max():.1f} HU',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Fracture: {output_path.name} (label: {label_text}, mean diff: {diff.mean():.1f})")


if __name__ == '__main__':
    import gc
    
    data_root = Path('/gscratch/scrubbed/june0604/wisespine_for_abnormal/data/raw/verse')
    output_dir = Path('/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the single sample but generate multiple augmentations with different parameters
    ct_file = data_root / 'dataset-verse19training/rawdata/sub-verse046/sub-verse046_ct.nii.gz'
    mask_file = data_root / 'dataset-verse19training/derivatives/sub-verse046/sub-verse046_seg-vert_msk.nii.gz'
    
    if not ct_file.exists() or not mask_file.exists():
        print('Error: Sample files not found')
        exit(1)
    
    subject_id = 'sub-verse046'
    
    print(f'Loading {subject_id}...')
    ct_nifti = nib.load(ct_file)
    ct_original = ct_nifti.get_fdata().astype(np.float32)
    spacing = ct_nifti.header.get_zooms()[:3]
    mask_nifti = nib.load(mask_file)
    mask_original = mask_nifti.get_fdata().astype(np.uint8)
    
    # Generate multiple augmentations with varying parameters
    num_samples = 10
    
    print(f'\nGenerating {num_samples} hardware augmentations...')
    for idx in range(num_samples):
        # Vary augmentation parameters
        screw_prob = 0.8 + idx * 0.02  # 0.8 to 0.98
        artifact_strength = 0.5 + idx * 0.05  # 0.5 to 0.95
        
        hardware_aug = SurgicalHardwareAugmenter(
            screw_probability=screw_prob, 
            rod_probability=1.0, 
            artifact_strength=artifact_strength
        )
        ct_hardware, mask_hardware = hardware_aug(ct_original.copy(), mask_original.copy(), spacing)
        visualize_hardware_detail(
            ct_original, ct_hardware, mask_original, mask_hardware, 
            f'{subject_id}_var{idx}', 
            output_dir / f'hardware_{idx:02d}_{subject_id}.png'
        )
        del ct_hardware, mask_hardware
        gc.collect()
    
    print(f'\nGenerating {num_samples} fracture augmentations...')
    for idx in range(num_samples):
        # Vary fracture parameters
        compression_min = 0.2 + idx * 0.03  # 0.2 to 0.47
        compression_max = compression_min + 0.2
        
        fracture_aug = FractureAugmenter(
            compression_probability=1.0, 
            compression_range=(compression_min, compression_max)
        )
        ct_fracture, mask_fracture = fracture_aug(ct_original.copy(), mask_original.copy(), spacing)
        visualize_fracture_detail(
            ct_original, ct_fracture, mask_original, mask_fracture, 
            f'{subject_id}_var{idx}',
            output_dir / f'fracture_{idx:02d}_{subject_id}.png'
        )
        del ct_fracture, mask_fracture
        gc.collect()
    
    print(f'\n✅ Done! Generated {num_samples*2} visualizations in outputs/visualizations/')
    print(f'   - hardware_00-09_sub-verse046.png')
    print(f'   - fracture_00-09_sub-verse046.png')

