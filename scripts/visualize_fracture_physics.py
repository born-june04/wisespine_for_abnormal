"""
Enhanced visualization showing physical deformation in fractures
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from matplotlib.patches import Rectangle
import sys

sys.path.append(str(Path(__file__).parent.parent))
from augmentation.surgical_hardware import SurgicalHardwareAugmenter
from augmentation.fractures_enhanced import EnhancedFractureAugmenter


def window_ct(ct_volume, window_center=400, window_width=1800):
    """Apply CT windowing."""
    min_hu = window_center - window_width / 2
    max_hu = window_center + window_width / 2
    ct_windowed = np.clip(ct_volume, min_hu, max_hu)
    ct_normalized = (ct_windowed - min_hu) / (max_hu - min_hu)
    return ct_normalized


def compute_vertebra_height_profile(ct_volume, mask, label):
    """Compute vertebra height along superior-inferior axis."""
    vertebra_mask = (mask == label)
    
    # Get z-axis profile (height)
    z_profile = []
    z_indices = []
    
    for z in range(ct_volume.shape[2]):
        slice_mask = vertebra_mask[:, :, z]
        if slice_mask.any():
            # Measure anterior-posterior extent
            coords = np.argwhere(slice_mask)
            height = coords[:, 0].max() - coords[:, 0].min()
            z_profile.append(height)
            z_indices.append(z)
    
    return np.array(z_indices), np.array(z_profile)


def visualize_fracture_physics(ct_orig, ct_aug, mask_orig, mask_aug, subject_id, output_path):
    """
    Fracture visualization showing physical deformation:
    Row 1: Sagittal view comparison (original vs fractured)
    Row 2: Height profile comparison
    Row 3: Deformation map
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)
    
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
    
    # Find fractured vertebra
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
    
    diff_norm = np.clip(diff / 200.0, 0, 1)
    
    # Row 1: Sagittal views
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(ct_orig_bone[:, :, center[2]].T, cmap='gray', origin='lower')
    if mask_orig[:, :, center[2]].any():
        ax0.contour(mask_orig[:, :, center[2]].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_orig[:, :, center[2]].any():
        ax0.contour(fractured_mask_orig[:, :, center[2]].T, colors='cyan', linewidths=2, origin='lower')
    ax0.set_title('Original - Sagittal', fontsize=14, fontweight='bold')
    ax0.axis('off')
    
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(ct_aug_bone[:, :, center[2]].T, cmap='gray', origin='lower')
    if mask_aug[:, :, center[2]].any():
        ax1.contour(mask_aug[:, :, center[2]].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_aug[:, :, center[2]].any():
        ax1.contour(fractured_mask_aug[:, :, center[2]].T, colors='cyan', linewidths=2, origin='lower')
    ax1.set_title('Fractured - Sagittal', fontsize=14, fontweight='bold', color='red')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(ct_aug_bone[:, :, center[2]].T, cmap='gray', origin='lower')
    if diff_norm[:, :, center[2]].any():
        ax2.imshow(diff_norm[:, :, center[2]].T, cmap='Reds', alpha=0.6, origin='lower', vmin=0, vmax=1)
    ax2.set_title('Deformation Overlay', fontsize=14, fontweight='bold', color='blue')
    ax2.axis('off')
    
    # Row 2: Coronal views
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(ct_orig_bone[center[0], :, :], cmap='gray', origin='lower')
    if mask_orig[center[0], :, :].any():
        ax3.contour(mask_orig[center[0], :, :], colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_orig[center[0], :, :].any():
        ax3.contour(fractured_mask_orig[center[0], :, :], colors='cyan', linewidths=2, origin='lower')
    ax3.set_title('Original - Coronal', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(ct_aug_bone[center[0], :, :], cmap='gray', origin='lower')
    if mask_aug[center[0], :, :].any():
        ax4.contour(mask_aug[center[0], :, :], colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_aug[center[0], :, :].any():
        ax4.contour(fractured_mask_aug[center[0], :, :], colors='cyan', linewidths=2, origin='lower')
    ax4.set_title('Fractured - Coronal', fontsize=14, fontweight='bold', color='red')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(ct_aug_bone[center[0], :, :], cmap='gray', origin='lower')
    if diff_norm[center[0], :, :].any():
        ax5.imshow(diff_norm[center[0], :, :], cmap='Reds', alpha=0.6, origin='lower', vmin=0, vmax=1)
    ax5.set_title('Deformation Overlay', fontsize=14, fontweight='bold', color='blue')
    ax5.axis('off')
    
    # Row 3: Axial views
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.imshow(ct_orig_bone[:, center[1], :].T, cmap='gray', origin='lower')
    if mask_orig[:, center[1], :].any():
        ax6.contour(mask_orig[:, center[1], :].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_orig[:, center[1], :].any():
        ax6.contour(fractured_mask_orig[:, center[1], :].T, colors='cyan', linewidths=2, origin='lower')
    ax6.set_title('Original - Axial', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.imshow(ct_aug_bone[:, center[1], :].T, cmap='gray', origin='lower')
    if mask_aug[:, center[1], :].any():
        ax7.contour(mask_aug[:, center[1], :].T, colors='green', linewidths=1.5, origin='lower', alpha=0.7)
    if fractured_mask_aug[:, center[1], :].any():
        ax7.contour(fractured_mask_aug[:, center[1], :].T, colors='cyan', linewidths=2, origin='lower')
    ax7.set_title('Fractured - Axial', fontsize=14, fontweight='bold', color='red')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.imshow(ct_aug_bone[:, center[1], :].T, cmap='gray', origin='lower')
    if diff_norm[:, center[1], :].any():
        ax8.imshow(diff_norm[:, center[1], :].T, cmap='Reds', alpha=0.6, origin='lower', vmin=0, vmax=1)
    ax8.set_title('Deformation Overlay', fontsize=14, fontweight='bold', color='blue')
    ax8.axis('off')
    
    # Row 4: Height Profile Comparison (shows physical compression)
    ax9 = fig.add_subplot(gs[3, :])
    
    if fractured_label is not None:
        # Compute height profiles
        z_orig, h_orig = compute_vertebra_height_profile(ct_orig, mask_orig, fractured_label)
        z_aug, h_aug = compute_vertebra_height_profile(ct_aug, mask_aug, fractured_label)
        
        ax9.plot(z_orig, h_orig, 'g-o', linewidth=3, markersize=6, label='Original Height', alpha=0.8)
        ax9.plot(z_aug, h_aug, 'r-s', linewidth=3, markersize=6, label='Fractured Height', alpha=0.8)
        ax9.fill_between(z_orig, h_orig, alpha=0.2, color='green')
        ax9.fill_between(z_aug, h_aug, alpha=0.2, color='red')
        
        # Compute compression percentage
        avg_compression = ((h_orig.mean() - h_aug.mean()) / h_orig.mean()) * 100 if len(h_orig) > 0 and len(h_aug) > 0 else 0
        
        ax9.set_xlabel('Z-axis (Superior-Inferior)', fontsize=14, fontweight='bold')
        ax9.set_ylabel('Vertebra AP Extent (voxels)', fontsize=14, fontweight='bold')
        ax9.set_title(f'Height Profile: Label {int(fractured_label)} | Avg Compression: {avg_compression:.1f}%', 
                     fontsize=16, fontweight='bold')
        ax9.legend(fontsize=12, loc='upper right')
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'No fracture detected', ha='center', va='center', fontsize=14)
        ax9.axis('off')
    
    label_text = f"Label {int(fractured_label)}" if fractured_label else "N/A"
    compression_text = f"{avg_compression:.1f}%" if fractured_label and len(z_orig) > 0 else "N/A"
    
    fig.suptitle(
        f'{subject_id} - Fracture Augmentation (Physical Deformation)\n'
        f'Fractured vertebra: {label_text}, Mean HU change: {diff.mean():.1f}, Max: {diff.max():.1f}',
        fontsize=18, fontweight='bold', y=0.98
    )
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Fracture physics: {output_path.name} (label: {label_text}, compression: {compression_text})")


if __name__ == '__main__':
    import gc
    
    data_root = Path('/gscratch/scrubbed/june0604/wisespine_for_abnormal/data/raw/verse')
    output_dir = Path('/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ct_file = data_root / 'dataset-verse19training/rawdata/sub-verse046/sub-verse046_ct.nii.gz'
    mask_file = data_root / 'dataset-verse19training/derivatives/sub-verse046/sub-verse046_seg-vert_msk.nii.gz'
    
    subject_id = 'sub-verse046'
    
    print(f'Loading {subject_id}...')
    ct_nifti = nib.load(ct_file)
    ct_original = ct_nifti.get_fdata().astype(np.float32)
    spacing = ct_nifti.header.get_zooms()[:3]
    mask_nifti = nib.load(mask_file)
    mask_original = mask_nifti.get_fdata().astype(np.uint8)
    
    # Generate multiple fractures with varying severity
    num_samples = 5
    
    print(f'\nGenerating {num_samples} fracture physics visualizations...')
    for idx in range(num_samples):
        compression_min = 0.2 + idx * 0.1  # 0.2 to 0.6 (20% to 60% height loss)
        compression_max = compression_min + 0.1
        wedge_min = 0.1 + idx * 0.05  # 0.1 to 0.3
        wedge_max = wedge_min + 0.1
        
        fracture_aug = EnhancedFractureAugmenter(
            compression_range=(compression_min, compression_max),
            wedge_range=(wedge_min, wedge_max),
            add_sclerosis=True,
            add_kyphosis=True
        )
        ct_fracture, mask_fracture = fracture_aug(ct_original.copy(), mask_original.copy(), spacing)
        visualize_fracture_physics(
            ct_original, ct_fracture, mask_original, mask_fracture, 
            f'{subject_id}_physics{idx}',
            output_dir / f'fracture_physics_{idx:02d}_{subject_id}.png'
        )
        del ct_fracture, mask_fracture
        gc.collect()
    
    print(f'\n✅ Done! Generated {num_samples} physics visualizations')
    print(f'   - fracture_physics_00-04_sub-verse046.png')

