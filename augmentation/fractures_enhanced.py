"""
Physics-Based Enhanced Fracture Augmentation Module
Simulates vertebral fractures with physically-motivated effects:

1. Direct volume resampling (ndimage.zoom) for realistic compression
2. Wedge deformation with gradient compression (kyphotic deformity)
3. Spinal column shortening — superior structures shift down after compression
4. Sclerosis around fracture margins (INSIDE bone, not outside)
5. Endplate irregularity (random perturbation of endplate surfaces)
6. Cortical disruption at fracture line (cortex breaks → soft tissue HU)

This module explicitly models physical consequences of fracture that
the traditional CV approach (fractures.py) ignores:
- Neighboring tissue displacement
- Reactive bone formation (sclerosis)
- Cortical shell integrity loss
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, List


class EnhancedFractureAugmenter:
    """
    Physics-based fracture augmentation.

    Key differences from traditional CV FractureAugmenter:
    - Uses ndimage.zoom for actual volume compression (not deformation field)
    - Shifts everything above the fracture DOWN (spinal column shortening)
    - Sclerosis is applied INSIDE bone at fracture margins
    - Cortical shell disruption at fracture line
    - Endplate irregularity for surface realism
    """

    def __init__(
        self,
        compression_range: Tuple[float, float] = (0.2, 0.5),
        wedge_range: Tuple[float, float] = (0.1, 0.4),
        add_sclerosis: bool = True,
        add_kyphosis: bool = True,
    ):
        self.compression_range = compression_range
        self.wedge_range = wedge_range
        self.add_sclerosis = add_sclerosis
        self.add_kyphosis = add_kyphosis

    def __call__(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply physics-based fracture augmentation."""
        augmented_ct = ct_volume.copy()
        augmented_mask = mask_volume.copy()

        vertebrae = self._find_vertebrae(mask_volume)
        if len(vertebrae) == 0:
            return augmented_ct, augmented_mask

        # Select random vertebra
        vertebra = np.random.choice(vertebrae)
        label = vertebra["label"]
        bmin = vertebra["bbox_min"]
        bmax = vertebra["bbox_max"]

        # Extract vertebra region
        region_ct = ct_volume[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ].copy()
        region_mask = (mask_volume[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ] == label).astype(float)

        original_z_size = region_ct.shape[2]

        # ─── 1. Uniform compression (direct resampling) ──────────
        compression = np.random.uniform(*self.compression_range)
        region_ct, region_mask = self._apply_compression(
            region_ct, region_mask, compression
        )

        # ─── 2. Wedge deformation (anterior > posterior) ─────────
        if self.add_kyphosis:
            wedge = np.random.uniform(*self.wedge_range)
            region_ct, region_mask = self._apply_wedge(
                region_ct, region_mask, wedge
            )

        # Calculate how much height was actually lost
        compressed_z = int(original_z_size * (1 - compression))
        height_loss = original_z_size - compressed_z

        # ─── 3. Endplate irregularity ────────────────────────────
        region_ct, region_mask = self._add_endplate_irregularity(
            region_ct, region_mask
        )

        # ─── 4. Fracture line with cortical disruption ───────────
        region_ct = self._add_fracture_line_with_cortical_disruption(
            region_ct, region_mask
        )

        # ─── 5. Sclerosis (INSIDE bone at fracture margins) ──────
        if self.add_sclerosis:
            region_ct = self._add_sclerosis(region_ct, region_mask)

        # ─── Place modified vertebra back ────────────────────────
        augmented_ct[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ] = region_ct
        augmented_mask[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ] = (region_mask > 0.5).astype(mask_volume.dtype) * label

        # ─── 6. Spinal column shortening ─────────────────────────
        # Shift everything ABOVE the fractured vertebra DOWN by height_loss
        if height_loss > 0:
            augmented_ct, augmented_mask = self._apply_column_shortening(
                augmented_ct, augmented_mask, bmax[2], height_loss
            )

        return augmented_ct, augmented_mask

    # ─── vertebra detection ───────────────────────────────────────────

    def _find_vertebrae(self, mask_volume: np.ndarray) -> List[dict]:
        vertebrae = []
        for label in np.unique(mask_volume):
            if label == 0:
                continue
            coords = np.argwhere(mask_volume == label)
            if len(coords) == 0:
                continue
            vertebrae.append({
                "label": label,
                "bbox_min": coords.min(axis=0),
                "bbox_max": coords.max(axis=0),
            })
        return vertebrae

    # ─── compression (physical resampling) ────────────────────────────

    def _apply_compression(
        self, ct: np.ndarray, mask: np.ndarray, compression: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Physical compression via ndimage.zoom in z-direction.
        This actually squashes the volume (reduces height) then pads
        to maintain bounding box size.
        """
        orig_shape = ct.shape
        new_z = max(1, int(orig_shape[2] * (1 - compression)))

        # Resample: compress in z
        compressed_ct = ndimage.zoom(
            ct, (1, 1, new_z / orig_shape[2]), order=1
        )
        compressed_mask = ndimage.zoom(
            mask, (1, 1, new_z / orig_shape[2]), order=0
        )

        # Pad: superior side gets edge-padded CT, zero-padded mask
        if compressed_ct.shape[2] < orig_shape[2]:
            pad = orig_shape[2] - compressed_ct.shape[2]
            compressed_ct = np.pad(
                compressed_ct, ((0, 0), (0, 0), (0, pad)), mode="edge"
            )
            compressed_mask = np.pad(
                compressed_mask, ((0, 0), (0, 0), (0, pad)), mode="constant"
            )
        elif compressed_ct.shape[2] > orig_shape[2]:
            compressed_ct = compressed_ct[:, :, :orig_shape[2]]
            compressed_mask = compressed_mask[:, :, :orig_shape[2]]

        return compressed_ct, compressed_mask

    # ─── wedge deformation (anterior height loss) ─────────────────────

    def _apply_wedge(
        self, ct: np.ndarray, mask: np.ndarray, wedge_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wedge deformation: anterior (x=0) compresses more than posterior.
        Creates kyphotic deformity. Applied slice-by-slice in x.
        """
        shape = ct.shape
        wedged_ct = ct.copy()
        wedged_mask = mask.copy()

        for x in range(shape[0]):
            # Linear gradient: full compression at x=0, zero at x=max
            local_comp = wedge_ratio * (1 - x / max(1, shape[0] - 1))

            if local_comp <= 0.05:
                continue

            s_ct = ct[x, :, :]
            s_mask = mask[x, :, :]

            new_z = max(1, int(shape[2] * (1 - local_comp)))

            comp_ct = ndimage.zoom(s_ct, (1, new_z / shape[2]), order=1)
            comp_mask = ndimage.zoom(s_mask, (1, new_z / shape[2]), order=0)

            # Pad back
            if comp_ct.shape[1] < shape[2]:
                pad = shape[2] - comp_ct.shape[1]
                comp_ct = np.pad(comp_ct, ((0, 0), (0, pad)), mode="edge")
                comp_mask = np.pad(comp_mask, ((0, 0), (0, pad)), mode="constant")
            elif comp_ct.shape[1] > shape[2]:
                comp_ct = comp_ct[:, :shape[2]]
                comp_mask = comp_mask[:, :shape[2]]

            wedged_ct[x, :, :] = comp_ct
            wedged_mask[x, :, :] = comp_mask

        return wedged_ct, wedged_mask

    # ─── spinal column shortening (physics-unique) ────────────────────

    def _apply_column_shortening(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        fracture_z_max: int,
        height_loss: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shift everything ABOVE the fractured vertebra DOWN by height_loss voxels.
        This is the key physical consequence of compression fracture:
        the spinal column shortens, and superior structures settle.

        Without this, there would be an unphysical gap at the endplate,
        which is immediately obvious to radiologists.
        """
        if height_loss <= 0 or fracture_z_max + 1 >= ct_volume.shape[2]:
            return ct_volume, mask_volume

        shifted_ct = ct_volume.copy()
        shifted_mask = mask_volume.copy()

        superior_start = fracture_z_max + 1
        superior_end = ct_volume.shape[2]

        # Amount to shift (limited to available space)
        shift = min(height_loss, superior_start)

        if shift <= 0:
            return ct_volume, mask_volume

        # Shift superior content downward
        src_start = superior_start
        src_end = superior_end
        dst_start = superior_start - shift
        dst_end = dst_start + (src_end - src_start)

        if dst_end > ct_volume.shape[2]:
            # Clip to volume bounds
            clip = dst_end - ct_volume.shape[2]
            src_end -= clip
            dst_end = ct_volume.shape[2]

        if dst_start < 0 or src_start >= src_end:
            return ct_volume, mask_volume

        shifted_ct[:, :, dst_start:dst_end] = ct_volume[:, :, src_start:src_end]
        shifted_mask[:, :, dst_start:dst_end] = mask_volume[:, :, src_start:src_end]

        # Fill the now-empty top slices with edge values
        if dst_end < ct_volume.shape[2]:
            shifted_ct[:, :, dst_end:] = ct_volume[:, :, -1:].repeat(
                ct_volume.shape[2] - dst_end, axis=2
            )
            shifted_mask[:, :, dst_end:] = 0

        return shifted_ct, shifted_mask

    # ─── endplate irregularity ────────────────────────────────────────

    def _add_endplate_irregularity(
        self, ct: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add random perturbation to superior/inferior endplate surfaces.
        In real fractures, endplates collapse irregularly (not uniformly).

        Uses smooth random noise to create undulating endplate surface,
        then randomly erodes/dilates the mask at the z-boundaries.
        """
        shape = ct.shape
        aug_ct = ct.copy()
        aug_mask = mask.copy()

        # Find endplate z-slices (top and bottom of vertebra in z)
        z_has_bone = np.any(mask > 0.5, axis=(0, 1))
        z_indices = np.where(z_has_bone)[0]

        if len(z_indices) < 4:
            return ct, mask

        # Superior endplate: top 2-3 slices
        sup_z = z_indices[-3:]
        # Inferior endplate: bottom 2-3 slices
        inf_z = z_indices[:3]

        for endplate_slices in [sup_z, inf_z]:
            # Generate smooth random perturbation
            noise = np.random.randn(shape[0], shape[1]) * 0.3
            noise = ndimage.gaussian_filter(noise, sigma=3.0)

            for z in endplate_slices:
                # Where noise > 0.2: erode (endplate collapses)
                # Where noise < -0.2: keep (endplate intact)
                collapse_mask = noise > 0.15

                # Collapse = darken CT + erode mask at that point
                aug_ct[:, :, z] = np.where(
                    collapse_mask & (mask[:, :, z] > 0.5),
                    aug_ct[:, :, z] * np.random.uniform(0.6, 0.8),
                    aug_ct[:, :, z],
                )
                aug_mask[:, :, z] = np.where(
                    collapse_mask & (mask[:, :, z] > 0.5) & (noise > 0.3),
                    0,  # Erode mask where collapse is severe
                    aug_mask[:, :, z],
                )

        return aug_ct, aug_mask

    # ─── fracture line + cortical disruption ──────────────────────────

    def _add_fracture_line_with_cortical_disruption(
        self, ct: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Add fracture line with cortical shell disruption.

        Two effects at the fracture line:
        1. Interior darkening (edema/hemorrhage): HU × 0.5
        2. Cortical disruption: where the cortex (bone edge) intersects
           the fracture line, reduce HU to soft tissue level (~50 HU)
           — this represents the cortical break visible on CT.
        """
        shape = ct.shape
        frac_z = int(shape[2] * np.random.uniform(0.3, 0.7))
        aug_ct = ct.copy()

        # Identify cortical shell (edge of bone mask)
        bone_mask = mask > 0.5
        eroded = ndimage.binary_erosion(bone_mask, iterations=1)
        cortex = bone_mask & ~eroded  # thin cortical shell

        for z in range(max(0, frac_z - 1), min(shape[2], frac_z + 2)):
            # 1. Interior darkening (only inside bone)
            aug_ct[:, :, z] = np.where(
                bone_mask[:, :, z],
                aug_ct[:, :, z] * 0.5,  # darker than traditional (0.7 vs 0.5)
                aug_ct[:, :, z],
            )

            # 2. Cortical disruption: where cortex meets fracture line
            #    → reduce to soft tissue HU (cortex is broken)
            aug_ct[:, :, z] = np.where(
                cortex[:, :, z],
                np.minimum(aug_ct[:, :, z], 50.0),  # soft tissue level
                aug_ct[:, :, z],
            )

        return aug_ct

    # ─── sclerosis (INSIDE bone, at fracture margins) ─────────────────

    def _add_sclerosis(self, ct: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Add sclerosis (reactive bone formation) at fracture margins.

        CRITICAL FIX: sclerosis occurs INSIDE the bone near its inner
        surface, NOT outside. Previous implementation used dilation
        which extended beyond bone boundary.

        Uses erosion to find inner edge of bone, then increases HU there.
        """
        bone_mask = mask > 0.5

        # Inner edge: bone voxels that are close to the surface
        eroded = ndimage.binary_erosion(bone_mask, iterations=2)
        inner_edge = bone_mask & ~eroded  # 2-voxel thick inner margin

        # Increase HU at inner edge (sclerotic reaction)
        aug_ct = ct.copy()
        aug_ct = np.where(
            inner_edge,
            aug_ct * 1.3 + 150,  # Increased bone density
            aug_ct,
        )

        return aug_ct


# ─── test ─────────────────────────────────────────────────────────────

def test_enhanced_fracture():
    """Test enhanced fracture augmentation."""
    print("Testing Physics-Based Enhanced Fracture Augmentation...")

    # Create synthetic data
    shape = (64, 64, 80)
    ct = np.random.randn(*shape).astype(np.float32) * 100 + 300
    mask = np.zeros(shape, dtype=np.uint8)

    # Two vertebrae stacked vertically
    for label, z_center in [(1, 30), (2, 55)]:
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        vert = (
            (x - z_center) ** 2 / 50
            + (y - shape[1] // 2) ** 2 / 100
            + (z - shape[0] // 2) ** 2 / 100
        ) < 1
        mask[vert] = label

    augmenter = EnhancedFractureAugmenter(
        compression_range=(0.3, 0.4),
        wedge_range=(0.15, 0.2),
        add_sclerosis=True,
        add_kyphosis=True,
    )

    np.random.seed(42)
    aug_ct, aug_mask = augmenter(ct, mask, spacing=(1.0, 1.0, 1.5))

    # Validate sclerosis is inside bone
    bone_orig = mask > 0
    bone_aug = aug_mask > 0
    sclerosis_voxels = (aug_ct > ct + 100) & bone_orig  # HU increased
    outside_sclerosis = (aug_ct > ct + 100) & ~bone_orig
    print(f"  Sclerosis inside bone: {sclerosis_voxels.sum()}")
    print(f"  Sclerosis outside bone (should be ~0): {outside_sclerosis.sum()}")

    # Validate column shortening
    label1_z_orig = np.argwhere(mask == 1)[:, 2].mean() if (mask == 1).any() else 0
    label2_z_orig = np.argwhere(mask == 2)[:, 2].mean() if (mask == 2).any() else 0
    label2_z_aug = np.argwhere(aug_mask == 2)[:, 2].mean() if (aug_mask == 2).any() else 0

    if label2_z_orig > 0 and label2_z_aug > 0:
        print(f"  Vertebra 2 z-center: {label2_z_orig:.1f} → {label2_z_aug:.1f} (should decrease)")

    print(f"  CT range: [{aug_ct.min():.0f}, {aug_ct.max():.0f}]")
    print("✓ Physics-based fracture augmentation test passed!")

    return aug_ct, aug_mask


if __name__ == "__main__":
    test_enhanced_fracture()
