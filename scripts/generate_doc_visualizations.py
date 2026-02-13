"""
Generate rich documentation visualizations for augmentation methods.

Produces:
  1. Fracture Progression — 6-step compression series (sagittal + axial + mask)
  2. Hardware Augmentation — multi-view with mask overlay
  3. GIF animations for progressive augmentation

Output:  docs/images/
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import nibabel as nib
from PIL import Image
import io

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from augmentation.surgical_hardware import SurgicalHardwareAugmenter
from augmentation.fractures_enhanced import EnhancedFractureAugmenter

OUT_DIR = ROOT / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────
def window_ct(vol, center=400, width=1800):
    lo, hi = center - width / 2, center + width / 2
    return np.clip((vol - lo) / (hi - lo), 0, 1)

def find_sample():
    """Return (ct_vol, mask_vol, spacing, subject_id) for the first available sample."""
    data_root = ROOT / "data" / "raw" / "verse"
    for split_dir in ["dataset-01training", "dataset-02validation", "dataset-03test"]:
        raw = data_root / split_dir / "rawdata"
        deriv = data_root / split_dir / "derivatives"
        if not raw.exists():
            continue
        for sub_dir in sorted(raw.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                continue
            ct_files = list(sub_dir.glob("*_ct.nii.gz"))
            if not ct_files:
                continue
            mask_dir = deriv / sub_dir.name
            if not mask_dir.exists():
                continue
            mask_files = list(mask_dir.glob("*seg*msk*.nii.gz")) + list(mask_dir.glob("*msk*.nii.gz"))
            if not mask_files:
                continue
            ct_nii = nib.load(str(ct_files[0]))
            mask_nii = nib.load(str(mask_files[0]))
            ct = ct_nii.get_fdata().astype(np.float32)
            mask = mask_nii.get_fdata().astype(np.int16)
            spacing = tuple(ct_nii.header.get_zooms()[:3])
            print(f"Loaded {sub_dir.name}  shape={ct.shape}  spacing={spacing}")
            return ct, mask, spacing, sub_dir.name
    raise RuntimeError("No valid VerSe sample found")


def get_vertebra_slice_indices(mask, label):
    """Return (sag_idx, ax_idx, cor_idx) centred on <label>."""
    coords = np.argwhere(mask == label)
    if len(coords) == 0:
        return mask.shape[0]//2, mask.shape[1]//2, mask.shape[2]//2
    c = coords.mean(axis=0).astype(int)
    return int(c[0]), int(c[1]), int(c[2])


def make_mask_overlay(ct_win, mask, label, alpha=0.35):
    """Return RGB array with mask overlaid on CT in a single colour."""
    rgb = np.stack([ct_win]*3, axis=-1)
    where = mask == label
    rgb[where] = rgb[where] * (1-alpha) + np.array([0.2, 0.9, 0.4]) * alpha  # green
    return rgb


# ── 1. Fracture progression series ────────────────────────────────────
def generate_fracture_progression(ct, mask, spacing, subject_id):
    """Generate 6-step fracture progression: sagittal + axial, with mask overlay."""
    print("\n=== Generating fracture progression ===")

    # Find a good vertebra
    labels = sorted(set(np.unique(mask)) - {0})
    if not labels:
        print("  No vertebrae found, skipping")
        return

    # Pick a vertebra near the middle of the spine
    mid_label = labels[len(labels)//2]
    sag_i, _, ax_i = get_vertebra_slice_indices(mask, mid_label)
    print(f"  Target vertebra label={mid_label}  sag_slice={sag_i}  ax_slice={ax_i}")

    compressions = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    frames_sag = []
    frames_ax = []
    pil_frames = []

    fig, axes = plt.subplots(2, len(compressions), figsize=(3.5*len(compressions), 7))
    fig.suptitle(f"Fracture Compression Progression  |  {subject_id}  |  Vertebra L{mid_label}",
                 fontsize=14, fontweight='bold', y=0.98)

    for col, comp in enumerate(compressions):
        if comp == 0.0:
            aug_ct, aug_mask = ct.copy(), mask.copy()
        else:
            aug = EnhancedFractureAugmenter(
                compression_range=(comp, comp + 0.001),
                wedge_range=(comp * 0.4, comp * 0.4 + 0.001),
                add_sclerosis=(comp >= 0.2),
                add_kyphosis=(comp >= 0.2),
            )
            np.random.seed(42)
            aug_ct, aug_mask = aug(ct.copy(), mask.copy(), spacing)

        ct_win = window_ct(aug_ct)

        # Sagittal
        sag = ct_win[sag_i, :, :]
        sag_m = aug_mask[sag_i, :, :]
        sag_rgb = make_mask_overlay(sag, sag_m, mid_label)
        axes[0, col].imshow(sag_rgb.transpose(1, 0, 2), origin='lower', aspect='auto')
        axes[0, col].set_title(f"{int(comp*100)}% compression", fontsize=11)
        if col == 0:
            axes[0, col].set_ylabel("Sagittal", fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        # Axial
        ax = ct_win[:, :, ax_i]
        ax_m = aug_mask[:, :, ax_i]
        ax_rgb = make_mask_overlay(ax, ax_m, mid_label)
        axes[1, col].imshow(ax_rgb, origin='lower', aspect='auto')
        if col == 0:
            axes[1, col].set_ylabel("Axial", fontsize=12, fontweight='bold')
        axes[1, col].axis('off')

        # --- collect PIL frames for GIF ---
        fig_frame, ax_frame = plt.subplots(1, 2, figsize=(6, 3.2))
        fig_frame.suptitle(f"Compression {int(comp*100)}%", fontsize=13, fontweight='bold')
        ax_frame[0].imshow(sag_rgb.transpose(1, 0, 2), origin='lower', aspect='auto')
        ax_frame[0].set_title("Sagittal", fontsize=10)
        ax_frame[0].axis('off')
        ax_frame[1].imshow(ax_rgb, origin='lower', aspect='auto')
        ax_frame[1].set_title("Axial", fontsize=10)
        ax_frame[1].axis('off')
        fig_frame.tight_layout()
        buf = io.BytesIO()
        fig_frame.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig_frame)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert('RGB'))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "fracture_progression.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Save GIF
    gif_path = OUT_DIR / "fracture_progression.gif"
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=800, loop=0
    )
    print(f"  Saved {gif_path}")


# ── 2. Hardware augmentation — multi-view with mask ───────────────────
def generate_hardware_multiview(ct, mask, spacing, subject_id):
    """Generate 3-view hardware comparison with mask overlays."""
    print("\n=== Generating hardware multi-view ===")

    hw_aug = SurgicalHardwareAugmenter(
        screw_probability=1.0,
        rod_probability=1.0,
        cement_probability=0.6,
        artifact_strength=0.6,
    )

    np.random.seed(42)
    aug_ct, aug_mask = hw_aug(ct.copy(), mask.copy(), spacing)

    labels = sorted(set(np.unique(mask)) - {0})
    mid_label = labels[len(labels)//2]
    sag_i, cor_i, ax_i = get_vertebra_slice_indices(mask, mid_label)

    ct_win_orig = window_ct(ct)
    ct_win_aug = window_ct(aug_ct)

    # Metal mask: HU > 5000
    metal_mask = (aug_ct > 5000).astype(float)
    # Mask diff
    mask_diff = (mask != aug_mask).astype(float)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"Hardware Augmentation  |  {subject_id}", fontsize=15, fontweight='bold', y=0.98)

    view_names = ["Sagittal", "Coronal", "Axial"]
    slices_orig = [ct_win_orig[sag_i,:,:], ct_win_orig[:,cor_i,:], ct_win_orig[:,:,ax_i]]
    slices_aug  = [ct_win_aug[sag_i,:,:],  ct_win_aug[:,cor_i,:],  ct_win_aug[:,:,ax_i]]
    slices_metal= [metal_mask[sag_i,:,:],  metal_mask[:,cor_i,:],  metal_mask[:,:,ax_i]]
    slices_mdiff= [mask_diff[sag_i,:,:],   mask_diff[:,cor_i,:],   mask_diff[:,:,ax_i]]
    masks_orig  = [mask[sag_i,:,:],        mask[:,cor_i,:],        mask[:,:,ax_i]]
    masks_aug   = [aug_mask[sag_i,:,:],    aug_mask[:,cor_i,:],    aug_mask[:,:,ax_i]]

    col_titles = ["Original + Mask", "Augmented + Mask", "Metal Regions", "Mask Changes"]

    for row in range(3):
        # transpose for display
        def t(arr):
            if row < 2:
                return arr.T
            return arr

        # Col 0: original + mask overlay
        orig_rgb = np.stack([t(slices_orig[row])]*3, axis=-1)
        m = t((masks_orig[row] > 0).astype(float))
        orig_rgb[:,:,1] = np.clip(orig_rgb[:,:,1] + m * 0.25, 0, 1)
        axes[row, 0].imshow(orig_rgb, origin='lower', aspect='auto')
        if row == 0:
            axes[row, 0].set_title(col_titles[0], fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel(view_names[row], fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')

        # Col 1: augmented + mask overlay
        aug_rgb = np.stack([t(slices_aug[row])]*3, axis=-1)
        m2 = t((masks_aug[row] > 0).astype(float))
        aug_rgb[:,:,1] = np.clip(aug_rgb[:,:,1] + m2 * 0.25, 0, 1)
        axes[row, 1].imshow(aug_rgb, origin='lower', aspect='auto')
        if row == 0:
            axes[row, 1].set_title(col_titles[1], fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')

        # Col 2: metal regions (red overlay)
        met_rgb = np.stack([t(slices_aug[row])]*3, axis=-1)
        met = t(slices_metal[row])
        met_rgb[:,:,0] = np.clip(met_rgb[:,:,0] + met * 0.6, 0, 1)
        met_rgb[:,:,1] = met_rgb[:,:,1] * (1 - met * 0.4)
        met_rgb[:,:,2] = met_rgb[:,:,2] * (1 - met * 0.4)
        axes[row, 2].imshow(met_rgb, origin='lower', aspect='auto')
        if row == 0:
            axes[row, 2].set_title(col_titles[2], fontsize=11, fontweight='bold')
        axes[row, 2].axis('off')

        # Col 3: mask diff (yellow overlay)
        diff_rgb = np.stack([t(slices_aug[row])]*3, axis=-1)
        d = t(slices_mdiff[row])
        diff_rgb[:,:,0] = np.clip(diff_rgb[:,:,0] + d * 0.5, 0, 1)
        diff_rgb[:,:,1] = np.clip(diff_rgb[:,:,1] + d * 0.5, 0, 1)
        diff_rgb[:,:,2] = diff_rgb[:,:,2] * (1 - d * 0.6)
        axes[row, 3].imshow(diff_rgb, origin='lower', aspect='auto')
        if row == 0:
            axes[row, 3].set_title(col_titles[3], fontsize=11, fontweight='bold')
        axes[row, 3].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "hardware_multiview.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── 3. Hardware progression — screw only → +rod → +cement → +artifact ─
def generate_hardware_progression(ct, mask, spacing, subject_id):
    """Show additive hardware components step by step."""
    print("\n=== Generating hardware component progression ===")

    labels = sorted(set(np.unique(mask)) - {0})
    mid_label = labels[len(labels)//2]
    sag_i, _, ax_i = get_vertebra_slice_indices(mask, mid_label)

    configs = [
        ("Original", dict(screw_probability=0, rod_probability=0, cement_probability=0, artifact_strength=0)),
        ("+ Screws", dict(screw_probability=1, rod_probability=0, cement_probability=0, artifact_strength=0)),
        ("+ Rods",   dict(screw_probability=1, rod_probability=1, cement_probability=0, artifact_strength=0)),
        ("+ Cement", dict(screw_probability=1, rod_probability=1, cement_probability=1, artifact_strength=0)),
        ("+ Artifacts", dict(screw_probability=1, rod_probability=1, cement_probability=1, artifact_strength=0.7)),
    ]

    fig, axes = plt.subplots(2, len(configs), figsize=(3.5 * len(configs), 7))
    fig.suptitle(f"Hardware Component Progression  |  {subject_id}",
                 fontsize=14, fontweight='bold', y=0.98)

    pil_frames = []

    for col, (title, kw) in enumerate(configs):
        if col == 0:
            aug_ct = ct.copy()
        else:
            np.random.seed(42)
            hw = SurgicalHardwareAugmenter(**kw)
            aug_ct, _ = hw(ct.copy(), mask.copy(), spacing)

        ct_win = window_ct(aug_ct)

        sag = ct_win[sag_i, :, :].T
        ax  = ct_win[:, :, ax_i]

        axes[0, col].imshow(sag, origin='lower', aspect='auto', cmap='gray')
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')
        if col == 0:
            axes[0, col].set_ylabel("Sagittal", fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        axes[1, col].imshow(ax, origin='lower', aspect='auto', cmap='gray')
        if col == 0:
            axes[1, col].set_ylabel("Axial", fontsize=12, fontweight='bold')
        axes[1, col].axis('off')

        # GIF frame
        fig_f, ax_f = plt.subplots(1, 2, figsize=(6, 3.2))
        fig_f.suptitle(title, fontsize=13, fontweight='bold')
        ax_f[0].imshow(sag, origin='lower', aspect='auto', cmap='gray')
        ax_f[0].set_title("Sagittal"); ax_f[0].axis('off')
        ax_f[1].imshow(ax, origin='lower', aspect='auto', cmap='gray')
        ax_f[1].set_title("Axial"); ax_f[1].axis('off')
        fig_f.tight_layout()
        buf = io.BytesIO()
        fig_f.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig_f)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert('RGB'))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "hardware_progression.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {out_path}")

    gif_path = OUT_DIR / "hardware_progression.gif"
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=1000, loop=0
    )
    print(f"  Saved {gif_path}")


# ── 4. Original vs Enhanced fracture comparison ───────────────────────
def generate_fracture_comparison(ct, mask, spacing, subject_id):
    """Side-by-side: Original fracture vs Enhanced fracture (same vertebra)."""
    print("\n=== Generating original vs enhanced fracture comparison ===")

    from augmentation.fractures import FractureAugmenter

    labels = sorted(set(np.unique(mask)) - {0})
    mid_label = labels[len(labels) // 2]
    sag_i, _, ax_i = get_vertebra_slice_indices(mask, mid_label)

    np.random.seed(42)
    orig_aug = FractureAugmenter(compression_probability=1.0, wedge_probability=0.0, burst_probability=0.0)
    orig_ct, orig_mask = orig_aug(ct.copy(), mask.copy(), spacing)

    np.random.seed(42)
    enh_aug = EnhancedFractureAugmenter(
        compression_range=(0.35, 0.36), wedge_range=(0.15, 0.16),
        add_sclerosis=True, add_kyphosis=True
    )
    enh_ct, enh_mask = enh_aug(ct.copy(), mask.copy(), spacing)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Original vs Enhanced Fracture  |  {subject_id}", fontsize=14, fontweight='bold', y=0.98)

    titles_row = ["Original CT", "Original Fracture", "Enhanced Fracture"]
    cts = [ct, orig_ct, enh_ct]
    masks_list = [mask, orig_mask, enh_mask]

    for col in range(3):
        ct_win = window_ct(cts[col])

        # Sagittal
        sag = ct_win[sag_i, :, :]
        sag_rgb = make_mask_overlay(sag, masks_list[col][sag_i,:,:], mid_label)
        axes[0, col].imshow(sag_rgb.transpose(1, 0, 2), origin='lower', aspect='auto')
        axes[0, col].set_title(titles_row[col], fontsize=12, fontweight='bold')
        if col == 0:
            axes[0, col].set_ylabel("Sagittal", fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        # Coronal (find a good coronal slice)
        cor_i = mask.shape[1] // 2
        cor = ct_win[:, cor_i, :]
        cor_rgb = make_mask_overlay(cor, masks_list[col][:,cor_i,:], mid_label)
        axes[1, col].imshow(cor_rgb.transpose(1, 0, 2), origin='lower', aspect='auto')
        if col == 0:
            axes[1, col].set_ylabel("Coronal", fontsize=12, fontweight='bold')
        axes[1, col].axis('off')

        # Axial
        ax = ct_win[:, :, ax_i]
        ax_rgb = make_mask_overlay(ax, masks_list[col][:,:,ax_i], mid_label)
        axes[2, col].imshow(ax_rgb, origin='lower', aspect='auto')
        if col == 0:
            axes[2, col].set_ylabel("Axial", fontsize=12, fontweight='bold')
        axes[2, col].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "fracture_original_vs_enhanced.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────
def main():
    ct, mask, spacing, subject_id = find_sample()

    generate_fracture_progression(ct, mask, spacing, subject_id)
    generate_hardware_multiview(ct, mask, spacing, subject_id)
    generate_hardware_progression(ct, mask, spacing, subject_id)
    generate_fracture_comparison(ct, mask, spacing, subject_id)

    print(f"\n✅ All visualizations written to {OUT_DIR}")
    print("  Files:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            print(f"    {f.name}  ({f.stat().st_size / 1024:.0f} KB)")

if __name__ == "__main__":
    main()
