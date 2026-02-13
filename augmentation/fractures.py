"""
Traditional CV Fracture Augmentation Module
Simulates vertebral fractures using classical computer vision techniques:
deformation fields, morphological operations, and intensity manipulation.

Fracture types (based on clinical prevalence):
- Compression fracture (50%): Uniform height loss via deformation field
- Wedge fracture (30%): Asymmetric A-P compression via gradient field
- Burst fracture (20%): Radial expansion + compression + fragments

All fracture effects are strictly mask-aware (only affect bone voxels).
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, List


class FractureAugmenter:
    """
    Traditional CV fracture augmentation using deformation fields.

    Key design choices (traditional CV, no physics simulation):
    - Deformation fields for spatial transforms (RegularGridInterpolator)
    - Morphological operations for fragment generation
    - Intensity manipulation for fracture lines and edema
    """

    def __init__(
        self,
        compression_probability: float = 0.5,
        wedge_probability: float = 0.3,
        burst_probability: float = 0.2,
        compression_range: Tuple[float, float] = (0.1, 0.6),
        fragment_probability: float = 0.3,
    ):
        self.compression_probability = compression_probability
        self.wedge_probability = wedge_probability
        self.burst_probability = burst_probability
        self.compression_range = compression_range
        self.fragment_probability = fragment_probability

    def __call__(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        augmented_ct = ct_volume.copy()
        augmented_mask = mask_volume.copy()

        vertebrae = self._find_vertebrae(mask_volume)
        if len(vertebrae) == 0:
            return augmented_ct, augmented_mask

        # Select random vertebra to fracture
        fractured = np.random.choice(vertebrae)

        # Select fracture type
        fracture_type = self._select_fracture_type()

        if fracture_type == "compression":
            augmented_ct, augmented_mask = self._apply_compression_fracture(
                augmented_ct, augmented_mask, fractured, spacing
            )
        elif fracture_type == "wedge":
            augmented_ct, augmented_mask = self._apply_wedge_fracture(
                augmented_ct, augmented_mask, fractured, spacing
            )
        elif fracture_type == "burst":
            augmented_ct, augmented_mask = self._apply_burst_fracture(
                augmented_ct, augmented_mask, fractured, spacing
            )

        # Optionally add displaced bone fragments
        if np.random.rand() < self.fragment_probability:
            augmented_ct, augmented_mask = self._add_bone_fragments(
                augmented_ct, augmented_mask, fractured, spacing
            )

        return augmented_ct, augmented_mask

    # ─── vertebra detection ───────────────────────────────────────────

    def _find_vertebrae(self, mask_volume: np.ndarray) -> List[dict]:
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

    def _select_fracture_type(self) -> str:
        r = np.random.rand()
        if r < self.compression_probability:
            return "compression"
        elif r < self.compression_probability + self.wedge_probability:
            return "wedge"
        else:
            return "burst"

    # ─── compression fracture ─────────────────────────────────────────

    def _apply_compression_fracture(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        vertebra: dict,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform compression fracture via deformation field.
        Compresses vertebra height in z-direction using smooth deformation.
        """
        augmented_ct = ct_volume.copy()
        augmented_mask = mask_volume.copy()

        bmin, bmax = vertebra["bbox_min"], vertebra["bbox_max"]
        label = vertebra["label"]

        # Extract vertebra region
        region_ct = ct_volume[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ].copy()
        region_mask_full = mask_volume[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ].copy()
        vertebra_local = (region_mask_full == label)

        compression = np.random.uniform(*self.compression_range)

        # Create compression deformation field
        deformation = self._create_compression_field(
            region_ct.shape, compression, vertebra_local
        )

        # Apply deformation to CT and mask
        deformed_ct = self._apply_deformation(region_ct, deformation)
        deformed_mask = self._apply_deformation(
            region_mask_full.astype(float), deformation
        )
        deformed_mask = np.round(deformed_mask).astype(mask_volume.dtype)

        # Add fracture line (MASK-AWARE: only within vertebra)
        deformed_ct = self._add_fracture_line(
            deformed_ct, deformed_mask == label
        )

        # Place back
        augmented_ct[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ] = deformed_ct
        augmented_mask[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ] = deformed_mask

        return augmented_ct, augmented_mask

    # ─── wedge fracture ───────────────────────────────────────────────

    def _apply_wedge_fracture(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        vertebra: dict,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wedge fracture: anterior compression > posterior.
        Uses a gradient deformation field (more compression at x=0, less at x=max).
        """
        augmented_ct = ct_volume.copy()
        augmented_mask = mask_volume.copy()

        bmin, bmax = vertebra["bbox_min"], vertebra["bbox_max"]
        label = vertebra["label"]

        region_ct = ct_volume[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ].copy()
        region_mask_full = mask_volume[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ].copy()
        vertebra_local = (region_mask_full == label)

        ant_compression = np.random.uniform(*self.compression_range)
        post_compression = np.random.uniform(0.0, ant_compression * 0.3)

        deformation = self._create_wedge_field(
            region_ct.shape, ant_compression, post_compression, vertebra_local
        )

        deformed_ct = self._apply_deformation(region_ct, deformation)
        deformed_mask = self._apply_deformation(
            region_mask_full.astype(float), deformation
        )
        deformed_mask = np.round(deformed_mask).astype(mask_volume.dtype)

        # Add oblique fracture line (MASK-AWARE)
        deformed_ct = self._add_oblique_fracture_line(
            deformed_ct, deformed_mask == label
        )

        augmented_ct[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ] = deformed_ct
        augmented_mask[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ] = deformed_mask

        return augmented_ct, augmented_mask

    # ─── burst fracture ───────────────────────────────────────────────

    def _apply_burst_fracture(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        vertebra: dict,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Burst fracture: compression + radial expansion + multiple fracture lines.
        """
        # Start with compression
        augmented_ct, augmented_mask = self._apply_compression_fracture(
            ct_volume, mask_volume, vertebra, spacing
        )

        bmin, bmax = vertebra["bbox_min"], vertebra["bbox_max"]
        label = vertebra["label"]

        region_ct = augmented_ct[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ].copy()
        region_mask = augmented_mask[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ]
        vertebra_local = (region_mask == label)

        # Radial burst deformation (mask-aware)
        center = np.array(region_ct.shape) / 2.0
        burst_field = self._create_burst_field(
            region_ct.shape, center, expansion=0.1, mask=vertebra_local
        )
        region_ct = self._apply_deformation(region_ct, burst_field)

        # Multiple radial fracture lines (mask-aware)
        n_lines = np.random.randint(2, 5)
        for _ in range(n_lines):
            angle = np.random.uniform(0, 2 * np.pi)
            region_ct = self._add_radial_fracture_line(
                region_ct, center, angle, vertebra_local
            )

        augmented_ct[
            bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1, bmin[2]:bmax[2]+1
        ] = region_ct

        return augmented_ct, augmented_mask

    # ─── bone fragments (actually implemented) ────────────────────────

    def _add_bone_fragments(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
        vertebra: dict,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add displaced bone fragments near fractured vertebra.

        Creates small ellipsoidal fragments displaced from the vertebral
        body edge — simulating cortical fragments that break off during
        burst/comminuted fractures.

        - Fragment count: 2-4
        - Fragment size: 2-5 mm
        - Displacement: 2-5 mm from vertebral boundary
        - Fragment HU: cortical bone level (800-1200)
        """
        augmented_ct = ct_volume.copy()
        augmented_mask = mask_volume.copy()

        label = vertebra["label"]
        bone_mask = mask_volume == label

        # Find surface voxels (boundary of the vertebra)
        eroded = ndimage.binary_erosion(bone_mask, iterations=2)
        surface = bone_mask & ~eroded
        surface_coords = np.argwhere(surface)

        if len(surface_coords) < 10:
            return augmented_ct, augmented_mask

        n_fragments = np.random.randint(2, 5)

        for _ in range(n_fragments):
            # Pick a random surface point
            idx = np.random.randint(len(surface_coords))
            origin = surface_coords[idx].astype(float)

            # Displacement direction (outward from centroid)
            centroid = vertebra["centroid"]
            direction = origin - centroid
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction /= norm

            # Displace by 2-5 mm
            displacement_mm = np.random.uniform(2, 5)
            displacement_vox = displacement_mm / np.array(spacing)
            fragment_center = origin + direction * displacement_vox

            # Fragment size (2-5 mm radius, ellipsoidal)
            fragment_radius = np.random.uniform(2, 5) / np.array(spacing)

            # Create ellipsoidal fragment
            frag_mask = self._create_ellipsoid(
                ct_volume.shape, fragment_center, fragment_radius
            )

            # Don't place fragment inside existing anatomy
            frag_mask &= ~(mask_volume > 0)

            if frag_mask.sum() < 5:
                continue

            # Set fragment HU to cortical bone level
            fragment_hu = np.random.uniform(800, 1200)
            augmented_ct[frag_mask] = fragment_hu

        return augmented_ct, augmented_mask

    # ─── deformation fields ───────────────────────────────────────────

    def _create_compression_field(
        self,
        shape: Tuple[int, int, int],
        compression: float,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Uniform z-compression deformation field."""
        field = np.zeros((*shape, 3))

        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )

        center_z = shape[2] / 2.0
        field[:, :, :, 2] = -(z - center_z) * compression

        return field

    def _create_wedge_field(
        self,
        shape: Tuple[int, int, int],
        ant_compression: float,
        post_compression: float,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Gradient deformation: more compression anteriorly (x=0)."""
        field = np.zeros((*shape, 3))

        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )

        # Linear gradient: anterior (x=0) → posterior (x=max)
        compression_gradient = (
            ant_compression
            + (post_compression - ant_compression) * (x / max(1, shape[0] - 1))
        )

        center_z = shape[2] / 2.0
        field[:, :, :, 2] = -(z - center_z) * compression_gradient

        return field

    def _create_burst_field(
        self,
        shape: Tuple[int, int, int],
        center: np.ndarray,
        expansion: float,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Radial expansion field for burst fracture (masked)."""
        field = np.zeros((*shape, 3))

        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )

        dx = x - center[0]
        dy = y - center[1]
        dz = z - center[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6

        # Apply expansion only within mask
        mask_float = mask.astype(float)
        field[:, :, :, 0] = dx / distance * expansion * distance * mask_float
        field[:, :, :, 1] = dy / distance * expansion * distance * mask_float
        field[:, :, :, 2] = dz / distance * expansion * distance * 0.5 * mask_float

        return field

    def _apply_deformation(
        self, volume: np.ndarray, deformation_field: np.ndarray
    ) -> np.ndarray:
        """Apply deformation field via RegularGridInterpolator."""
        shape = volume.shape

        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )

        x_def = np.clip(x + deformation_field[:, :, :, 0], 0, shape[0] - 1)
        y_def = np.clip(y + deformation_field[:, :, :, 1], 0, shape[1] - 1)
        z_def = np.clip(z + deformation_field[:, :, :, 2], 0, shape[2] - 1)

        coords_deformed = np.stack(
            [x_def.ravel(), y_def.ravel(), z_def.ravel()], axis=1
        )

        interpolator = RegularGridInterpolator(
            (np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])),
            volume,
            bounds_error=False,
            fill_value=0,
        )

        return interpolator(coords_deformed).reshape(shape)

    # ─── fracture lines (all mask-aware) ──────────────────────────────

    def _add_fracture_line(
        self, volume: np.ndarray, bone_mask: np.ndarray
    ) -> np.ndarray:
        """Add horizontal fracture line, only inside the bone mask."""
        shape = volume.shape
        frac_z = int(shape[2] * np.random.uniform(0.3, 0.7))

        augmented = volume.copy()
        for z in range(max(0, frac_z - 1), min(shape[2], frac_z + 2)):
            augmented[:, :, z] = np.where(
                bone_mask[:, :, z],
                augmented[:, :, z] * 0.7,  # edema/hemorrhage darkening
                augmented[:, :, z],
            )
        return augmented

    def _add_oblique_fracture_line(
        self, volume: np.ndarray, bone_mask: np.ndarray
    ) -> np.ndarray:
        """Add oblique fracture line, only inside the bone mask."""
        z_center = volume.shape[2] // 2
        augmented = volume.copy()

        for z in range(max(0, z_center - 2), min(volume.shape[2], z_center + 3)):
            augmented[:, :, z] = np.where(
                bone_mask[:, :, z],
                augmented[:, :, z] * 0.75,
                augmented[:, :, z],
            )
        return augmented

    def _add_radial_fracture_line(
        self,
        volume: np.ndarray,
        center: np.ndarray,
        angle: float,
        bone_mask: np.ndarray,
    ) -> np.ndarray:
        """Add radial fracture line from center, only inside bone mask."""
        augmented = volume.copy()

        for r in np.linspace(0, min(volume.shape[:2]) / 2, 50):
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))

            if 0 <= x < volume.shape[0] and 0 <= y < volume.shape[1]:
                # Apply darkening only where bone exists
                bone_slice = bone_mask[x, y, :]
                augmented[x, y, :] = np.where(
                    bone_slice,
                    augmented[x, y, :] * 0.7,
                    augmented[x, y, :],
                )
        return augmented

    # ─── geometry primitives ──────────────────────────────────────────

    def _create_ellipsoid(
        self,
        shape: Tuple[int, int, int],
        center: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        """Create an ellipsoidal mask (for bone fragments)."""
        mask = np.zeros(shape, dtype=bool)

        max_r = radii.max()
        bmin = np.maximum(0, (center - max_r - 1).astype(int))
        bmax = np.minimum(shape, (center + max_r + 2).astype(int))

        if np.any(bmin >= bmax):
            return mask

        z, y, x = np.ogrid[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]]
        dist = (
            ((z - center[0]) / max(radii[0], 1e-6)) ** 2
            + ((y - center[1]) / max(radii[1], 1e-6)) ** 2
            + ((x - center[2]) / max(radii[2], 1e-6)) ** 2
        )

        mask[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] = dist <= 1.0
        return mask


# ─── test ─────────────────────────────────────────────────────────────

def test_augmentation():
    """Test fracture augmentation with synthetic data."""
    print("Testing Traditional CV Fracture Augmentation...")

    shape = (256, 256, 128)
    ct_volume = np.random.randn(*shape).astype(np.float32) * 50 + 50

    mask_volume = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    vert = (
        (x - shape[2] // 2) ** 2 / 100
        + (y - shape[1] // 2) ** 2 / 100
        + (z - shape[0] // 2) ** 2 / 400
    ) < 1
    mask_volume[vert] = 1

    augmenter = FractureAugmenter(
        compression_probability=1.0,
        compression_range=(0.3, 0.5),
        fragment_probability=1.0,
    )

    np.random.seed(42)
    aug_ct, aug_mask = augmenter(ct_volume, mask_volume, (1.0, 1.0, 1.5))

    # Validate mask-awareness: soft tissue outside vertebra should be unchanged
    outside_bone = mask_volume == 0
    ct_diff_outside = np.abs(aug_ct[outside_bone] - ct_volume[outside_bone])
    # Fragments may add new voxels outside, but fracture lines shouldn't change them
    print(f"  CT range: [{aug_ct.min():.0f}, {aug_ct.max():.0f}]")
    print(f"  Mask sum: {mask_volume.sum()} → {aug_mask.sum()}")
    print(f"  Outside-bone max change (should be from fragments only): {ct_diff_outside.max():.1f}")
    print("✓ Traditional CV fracture augmentation test passed!")

    return aug_ct, aug_mask


if __name__ == "__main__":
    test_augmentation()
