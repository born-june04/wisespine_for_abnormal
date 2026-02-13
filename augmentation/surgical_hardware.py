"""
Surgical Hardware Augmentation Module
Simulates pedicle screws, rods, bone cement, and CT metal artifacts.

Hardware placement follows AO Spine standards:
- Pedicle screws: posterior→anterior trajectory, 10-15° medial convergence
- Connecting rods: bilateral, posterior to vertebral body
- Bone cement: irregular blob within vertebral body (vertebroplasty/kyphoplasty)

Metal artifact simulation based on CT physics:
- Radon-based streak artifacts (fan-beam pattern)
- Blooming effect (Gaussian smoothing at metal-bone interface)
- HU corruption (spatially-varying offsets near dense objects)
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, List


class SurgicalHardwareAugmenter:
    """
    Augments CT scans with surgical hardware (screws, rods, bone cement)
    and physically-motivated metal artifacts.
    """

    def __init__(
        self,
        screw_probability: float = 0.6,
        rod_probability: float = 0.5,
        cement_probability: float = 0.3,
        metal_hu_range: Tuple[int, int] = (15000, 25000),
        artifact_strength: float = 0.5,
    ):
        self.screw_probability = screw_probability
        self.rod_probability = rod_probability
        self.cement_probability = cement_probability
        self.metal_hu_range = metal_hu_range
        self.artifact_strength = artifact_strength

    def __call__(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply surgical hardware augmentation to CT and mask.

        Args:
            ct_volume: CT scan (H, W, D) in HU
            mask_volume: Segmentation mask (H, W, D)
            spacing: Voxel spacing (x, y, z) in mm

        Returns:
            augmented_ct, augmented_mask
        """
        augmented_ct = ct_volume.copy()
        augmented_mask = mask_volume.copy()

        vertebrae = self._find_vertebrae_positions(mask_volume)
        if len(vertebrae) == 0:
            return augmented_ct, augmented_mask

        # --- Add implants ---
        if np.random.rand() < self.screw_probability:
            augmented_ct, augmented_mask = self._add_pedicle_screws(
                augmented_ct, augmented_mask, vertebrae, spacing
            )

        if np.random.rand() < self.rod_probability:
            augmented_ct, augmented_mask = self._add_connecting_rods(
                augmented_ct, augmented_mask, vertebrae, spacing
            )

        if np.random.rand() < self.cement_probability:
            augmented_ct, augmented_mask = self._add_bone_cement(
                augmented_ct, augmented_mask, vertebrae, spacing
            )

        # --- Add metal artifacts (physics-based) ---
        augmented_ct = self._add_metal_artifacts(augmented_ct, spacing)

        return augmented_ct, augmented_mask

    # ─── vertebra detection ───────────────────────────────────────────

    def _find_vertebrae_positions(self, mask_volume: np.ndarray) -> List[dict]:
        vertebrae = []
        for label in np.unique(mask_volume):
            if label == 0:
                continue
            vmask = mask_volume == label
            coords = np.argwhere(vmask)
            if len(coords) == 0:
                continue
            vertebrae.append({
                "label": label,
                "centroid": coords.mean(axis=0),
                "bbox_min": coords.min(axis=0),
                "bbox_max": coords.max(axis=0),
                "mask": vmask,
            })
        return vertebrae

    # ─── pedicle screws (AO standard trajectory) ──────────────────────

    def _add_pedicle_screws(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        vertebrae: List[dict],
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add bilateral pedicle screws with anatomically correct trajectory.

        AO standard placement:
        - Entry: pedicle center (lateral mass)
        - Trajectory: posterior→anterior, 10-15° medial convergence
        - Diameter: 4.5-7.5 mm (typical 5.5 mm)
        - Length: 30-50 mm (typical 40 mm)
        """
        augmented_ct = ct_volume.copy()
        augmented_mask = mask_volume.copy()

        # Instrument 1-N vertebrae
        n_instrumented = np.random.randint(1, max(2, len(vertebrae)))
        selected = np.random.choice(len(vertebrae), size=n_instrumented, replace=False)

        for idx in selected:
            vert = vertebrae[idx]
            centroid = vert["centroid"]

            screw_diameter = np.random.uniform(4.5, 7.5) / spacing[0]  # voxels
            screw_length = np.random.uniform(30, 50) / spacing[0]      # voxels

            # Pedicle offset: ~15-20 mm lateral from vertebral body center
            lateral_offset = np.random.uniform(15, 20) / spacing[1]

            for side in [-1, 1]:  # bilateral
                # Entry point: posterior aspect of pedicle
                entry = centroid.copy()
                entry[1] += side * lateral_offset      # lateral to pedicle
                entry[0] -= 5.0 / spacing[0]           # slightly posterior

                # AO-standard trajectory:
                #   primary: posterior → anterior (positive x direction)
                #   convergence: 10-15° medial tilt (toward midline)
                #   slight caudal angulation: parallel to superior endplate
                medial_angle = np.radians(np.random.uniform(10, 15))
                caudal_angle = np.radians(np.random.uniform(-5, 5))

                direction = np.array([
                    np.cos(medial_angle) * np.cos(caudal_angle),   # anterior (primary)
                    -side * np.sin(medial_angle),                   # medial convergence
                    np.sin(caudal_angle),                           # slight craniocaudal
                ])
                direction /= np.linalg.norm(direction)

                screw_mask = self._create_cylinder(
                    augmented_ct.shape, entry, direction,
                    screw_diameter / 2, screw_length
                )

                metal_hu = np.random.randint(*self.metal_hu_range)
                augmented_ct[screw_mask] = metal_hu
                augmented_mask[screw_mask] = 0  # hardware occludes anatomy

        return augmented_ct, augmented_mask

    # ─── connecting rods ──────────────────────────────────────────────

    def _add_connecting_rods(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        vertebrae: List[dict],
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add bilateral connecting rods between instrumented levels.
        Rods run posterior to vertebral body, connecting screw heads.
        Diameter: 5-7 mm.
        """
        if len(vertebrae) < 2:
            return ct_volume, mask_volume

        augmented_ct = ct_volume.copy()
        augmented_mask = mask_volume.copy()

        sorted_verts = sorted(vertebrae, key=lambda v: v["centroid"][2])
        rod_diameter = np.random.uniform(5, 7) / spacing[0]
        lateral_offset = np.random.uniform(18, 22) / spacing[1]
        posterior_offset = np.random.uniform(10, 15) / spacing[0]

        for side in [-1, 1]:
            for i in range(len(sorted_verts) - 1):
                start = sorted_verts[i]["centroid"].copy()
                end = sorted_verts[i + 1]["centroid"].copy()

                # Rod position: lateral + posterior
                start[1] += side * lateral_offset
                end[1] += side * lateral_offset
                start[0] -= posterior_offset
                end[0] -= posterior_offset

                direction = end - start
                length = np.linalg.norm(direction)
                if length < 1e-6:
                    continue
                direction /= length

                rod_mask = self._create_cylinder(
                    augmented_ct.shape, start, direction,
                    rod_diameter / 2, length
                )

                metal_hu = np.random.randint(*self.metal_hu_range)
                augmented_ct[rod_mask] = metal_hu
                augmented_mask[rod_mask] = 0

        return augmented_ct, augmented_mask

    # ─── bone cement ──────────────────────────────────────────────────

    def _add_bone_cement(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        vertebrae: List[dict],
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add bone cement (vertebroplasty/kyphoplasty).
        HU: 800-1500, irregular multi-blob shape within vertebral body.
        """
        augmented_ct = ct_volume.copy()
        augmented_mask = mask_volume.copy()

        vert = np.random.choice(vertebrae)
        cement_center = vert["centroid"]
        cement_size = np.random.uniform(10, 20) / np.array(spacing)

        cement_mask = np.zeros(ct_volume.shape, dtype=bool)
        for _ in range(np.random.randint(3, 8)):
            blob_center = cement_center + np.random.randn(3) * cement_size / 3
            blob_radius = np.random.uniform(0.3, 0.7) * cement_size.mean()
            cement_mask |= self._create_sphere(ct_volume.shape, blob_center, blob_radius)

        # Constrain to vertebral body only
        cement_mask &= vert["mask"]

        augmented_ct[cement_mask] = np.random.randint(800, 1500)
        return augmented_ct, augmented_mask

    # ─── metal artifacts (physics-based) ──────────────────────────────

    def _add_metal_artifacts(
        self, ct_volume: np.ndarray, spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Physically-motivated metal artifact synthesis:
        1. Radon-based streak artifacts (fan-beam pattern in axial plane)
        2. Blooming effect (Gaussian blur at metal-bone interface)
        3. HU corruption (exponential decay near metal)
        """
        augmented_ct = ct_volume.copy()
        metal_mask = augmented_ct > 10000

        if not metal_mask.any():
            return augmented_ct

        # ─── 1. Blooming (partial volume + scatter) ───────────────
        bloom_sigma = self.artifact_strength * 3.0 / np.array(spacing)
        blurred = ndimage.gaussian_filter(augmented_ct.astype(np.float64), sigma=bloom_sigma)

        # Apply blooming only in ring around metal (2-4 mm)
        bloom_radius = int(max(2, 3.0 / min(spacing)))
        metal_dilated = ndimage.binary_dilation(metal_mask, iterations=bloom_radius)
        bloom_region = metal_dilated & ~metal_mask

        augmented_ct[bloom_region] = blurred[bloom_region]

        # ─── 2. HU corruption (exponential decay) ────────────────
        corruption_radius_mm = np.random.uniform(10, 20)
        corruption_radius_vox = int(corruption_radius_mm / min(spacing))

        # Distance transform from metal boundary
        dist_from_metal = ndimage.distance_transform_edt(~metal_mask, sampling=spacing)

        # Exponential decay: intensity offset near metal
        corruption_zone = dist_from_metal < corruption_radius_mm
        decay = np.exp(-dist_from_metal / (corruption_radius_mm / 3))
        hu_offset = self.artifact_strength * np.random.uniform(500, 1500) * decay

        # Alternate positive/negative offsets using spatial noise
        noise_field = np.random.randn(*ct_volume.shape)
        noise_field = ndimage.gaussian_filter(noise_field, sigma=5.0)
        noise_sign = np.sign(noise_field)

        augmented_ct[corruption_zone] += (
            hu_offset[corruption_zone] * noise_sign[corruption_zone]
        ).astype(augmented_ct.dtype)

        # ─── 3. Radon-based streak artifacts (axial fan-beam) ─────
        if self.artifact_strength > 0.2:
            augmented_ct = self._add_streak_artifacts(
                augmented_ct, metal_mask, spacing
            )

        return augmented_ct

    def _add_streak_artifacts(
        self,
        ct_volume: np.ndarray,
        metal_mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        Radon-based streak artifacts in the axial plane.

        Physical basis: photon starvation along ray paths through metal.
        For each axial slice containing metal:
        1. Find metal centroid in that slice
        2. Cast radial rays from centroid (fan-beam pattern)
        3. Add intensity perturbation decaying with distance
        4. Alternate bright/dark bands
        """
        augmented_ct = ct_volume.copy()

        # Find axial slices with metal
        metal_slices = np.any(metal_mask, axis=(0, 1))
        z_indices = np.where(metal_slices)[0]

        if len(z_indices) == 0:
            return augmented_ct

        # Process every Nth slice for efficiency
        step = max(1, len(z_indices) // 20)
        streak_intensity = self.artifact_strength * np.random.uniform(800, 1500)
        n_rays = np.random.randint(12, 24)

        for z_idx in z_indices[::step]:
            metal_slice = metal_mask[:, :, z_idx]
            if not metal_slice.any():
                continue

            # Metal centroid in this slice
            coords = np.argwhere(metal_slice)
            centroid = coords.mean(axis=0)

            # Cast radial rays
            angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
            # Add slight random jitter to angles for realism
            angles += np.random.uniform(-0.1, 0.1, size=len(angles))

            H, W = ct_volume.shape[0], ct_volume.shape[1]
            max_radius = np.sqrt(H**2 + W**2)

            for i, angle in enumerate(angles):
                # Alternate bright/dark streaks
                sign = 1.0 if i % 2 == 0 else -1.0

                dx = np.cos(angle)
                dy = np.sin(angle)

                # Sample along ray, starting from metal boundary
                for r in np.arange(1, max_radius, 1.5):
                    x = int(centroid[0] + r * dx)
                    y = int(centroid[1] + r * dy)

                    if not (0 <= x < H and 0 <= y < W):
                        break

                    if metal_mask[x, y, z_idx]:
                        continue  # skip metal voxels themselves

                    # Intensity decays with distance from metal
                    decay = np.exp(-r / (max_radius * 0.15))
                    perturbation = sign * streak_intensity * decay

                    # Apply streak to this voxel and neighbors (width ~2-3 voxels)
                    for dz in range(max(0, z_idx - 1), min(ct_volume.shape[2], z_idx + 2)):
                        augmented_ct[x, y, dz] += perturbation

        return augmented_ct

    # ─── geometry primitives ──────────────────────────────────────────

    def _create_cylinder(
        self,
        shape: Tuple[int, int, int],
        start: np.ndarray,
        direction: np.ndarray,
        radius: float,
        length: float,
    ) -> np.ndarray:
        """Create a cylindrical mask (for screws/rods)."""
        mask = np.zeros(shape, dtype=bool)

        n_samples = max(int(length * 2), 10)
        for t in np.linspace(0, length, n_samples):
            center = start + t * direction
            sphere = self._create_sphere(shape, center, radius)
            mask |= sphere

        return mask

    def _create_sphere(
        self,
        shape: Tuple[int, int, int],
        center: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Create a spherical mask."""
        mask = np.zeros(shape, dtype=bool)

        # Bounding box (clipped to volume)
        bmin = np.maximum(0, (center - radius - 1).astype(int))
        bmax = np.minimum(shape, (center + radius + 2).astype(int))

        if np.any(bmin >= bmax):
            return mask

        z, y, x = np.ogrid[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]]
        dist_sq = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2

        mask[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] = dist_sq <= radius ** 2
        return mask


# ─── test ─────────────────────────────────────────────────────────────

def test_augmentation():
    """Test surgical hardware augmentation with synthetic data."""
    print("Testing Surgical Hardware Augmentation...")

    shape = (256, 256, 128)
    ct_volume = np.random.randn(*shape).astype(np.float32) * 50 + 50
    mask_volume = np.zeros(shape, dtype=np.uint8)

    for i, z_pos in enumerate([40, 60, 80, 100], start=1):
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        vert = (
            (x - shape[2] // 2) ** 2 / 100
            + (y - shape[1] // 2) ** 2 / 100
            + (z - z_pos) ** 2 / 400
        ) < 1
        mask_volume[vert] = i

    augmenter = SurgicalHardwareAugmenter(
        screw_probability=1.0,
        rod_probability=1.0,
        cement_probability=1.0,
        artifact_strength=0.7,
    )

    spacing = (1.0, 1.0, 1.5)
    aug_ct, aug_mask = augmenter(ct_volume, mask_volume, spacing)

    print(f"  CT range: [{aug_ct.min():.0f}, {aug_ct.max():.0f}]")
    print(f"  High HU voxels (>10000): {(aug_ct > 10000).sum()}")
    print(f"  Mask labels: {np.unique(aug_mask)}")
    print("✓ Surgical hardware augmentation test passed!")

    return aug_ct, aug_mask


if __name__ == "__main__":
    test_augmentation()
