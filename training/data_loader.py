"""
Data Loading Pipeline for VerSe Dataset with Augmentation
Supports ablation between original and enhanced fracture augmentation
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from augmentation.surgical_hardware import SurgicalHardwareAugmenter
from augmentation.fractures import FractureAugmenter
from augmentation.fractures_enhanced import EnhancedFractureAugmenter


class VerSeDataset(Dataset):
    """
    Dataset class for VerSe (Vertebrae Segmentation) with abnormal augmentation.
    Supports ablation study between original and enhanced fracture augmentation.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        augmentation_config: Optional[Dict] = None,
        use_augmentation: bool = True,
        use_enhanced_fracture: bool = True,  # NEW: ablation flag
        normalize: bool = True,
        cache_data: bool = False
    ):
        """
        Args:
            data_root: Path to VerSe data root (e.g., data/raw/verse)
            split: 'train', 'val', or 'test'
            augmentation_config: Configuration for augmentation
            use_augmentation: Whether to apply abnormal augmentation
            use_enhanced_fracture: If True, use EnhancedFractureAugmenter (ablation)
            normalize: Whether to normalize CT values
            cache_data: Whether to cache loaded data in memory
        """
        self.data_root = Path(data_root)
        self.split = split
        self.use_augmentation = use_augmentation
        self.use_enhanced_fracture = use_enhanced_fracture
        self.normalize = normalize
        self.cache_data = cache_data
        
        #map split to VerSe dataset folders
        self.split_mapping = {
            'train': ['dataset-verse19training', 'dataset-01training'],
            'val': ['dataset-verse19validation', 'dataset-02validation'],
            'test': ['dataset-verse19test', 'dataset-03test']
        }
        
        # Initialize augmenters
        if augmentation_config is None:
            augmentation_config = self._default_augmentation_config()
        
        self.augmentation_config = augmentation_config
        self.hardware_augmenter = SurgicalHardwareAugmenter(
            **augmentation_config.get('hardware', {})
        )
        
        # Choose fracture augmenter based on ablation flag
        if self.use_enhanced_fracture:
            self.fracture_augmenter = EnhancedFractureAugmenter(
                **augmentation_config.get('fracture_enhanced', {})
            )
        else:
            self.fracture_augmenter = FractureAugmenter(
                **augmentation_config.get('fracture_original', {})
            )
        
        # Load dataset file list
        self.samples = self._load_dataset()
        
        # Cache
        self.cache = {} if cache_data else None
        
        fracture_mode = "ENHANCED" if self.use_enhanced_fracture else "ORIGINAL"
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"  - Fracture augmentation mode: {fracture_mode}")
    
    def _default_augmentation_config(self) -> Dict:
        """Default augmentation configuration."""
        return {
            'hardware': {
                'screw_probability': 0.4,
                'rod_probability': 0.3,
                'artifact_strength': 0.6,
            },
            'fracture_original': {
                'compression_probability': 1.0,
                'compression_range': (0.2, 0.5),
                'wedge_probability': 0.3,
                'burst_probability': 0.2,
            },
            'fracture_enhanced': {
                'compression_range': (0.2, 0.5),
                'wedge_range': (0.1, 0.3),
                'add_sclerosis': True,
                'add_kyphosis': True,
            },
            'hardware_prob': 0.5,
            'fracture_prob': 0.5,
        }
        
        # Initialize augmenters
        if augmentation_config is None:
            augmentation_config = self._default_augmentation_config()
        
        self.augmentation_config = augmentation_config
        self.hardware_augmenter = SurgicalHardwareAugmenter(
            **augmentation_config.get('hardware', {})
        )
        
        # Choose fracture augmenter based on ablation flag
        if self.use_enhanced_fracture:
            self.fracture_augmenter = EnhancedFractureAugmenter(
                **augmentation_config.get('fracture_enhanced', {})
            )
        else:
            self.fracture_augmenter = FractureAugmenter(
                **augmentation_config.get('fracture_original', {})
            )
        
        # Load dataset file list
        self.samples = self._load_dataset()
        
        # Cache
        self.cache = {} if cache_data else None
        
        fracture_mode = "ENHANCED" if self.use_enhanced_fracture else "ORIGINAL"
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"  - Fracture augmentation mode: {fracture_mode}")
    
    def _default_augmentation_config(self) -> Dict:
        """Default augmentation configuration."""
        return {
            'hardware': {
                'screw_probability': 0.4,
                'rod_probability': 0.3,
                'cement_probability': 0.2,
                'metal_hu_range': (15000, 25000),
                'artifact_strength': 0.5,
            },
            'fracture': {
                'compression_probability': 0.3,
                'wedge_probability': 0.2,
                'burst_probability': 0.1,
                'compression_range': (0.1, 0.5),
                'fragment_probability': 0.2,
            },
            'augmentation_probability': 0.7,  # Apply augmentation to 70% of samples
        }
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load dataset file paths for the given split.
        
        Returns:
            List of dicts containing CT and mask paths
        """
        samples = []
        
        folders = self.split_mapping.get(self.split, [])
        
        for folder in folders:
            folder_path = self.data_root / folder
            
            if not folder_path.exists():
                print(f"Warning: {folder_path} does not exist, skipping...")
                continue
            
            # Find CT and mask files
            # VerSe structure: folder/rawdata/*.nii.gz and folder/derivatives/*.nii.gz
            rawdata_path = folder_path / 'rawdata'
            derivatives_path = folder_path / 'derivatives'
            
            if not rawdata_path.exists():
                # Try alternative structure
                rawdata_path = folder_path
                derivatives_path = folder_path
            
            # Find all CT files (check both flat structure and subdirectories)
            ct_files = sorted(rawdata_path.glob('*.nii.gz'))
            ct_files_subdir = sorted(rawdata_path.glob('*/*.nii.gz'))
            ct_files = ct_files + ct_files_subdir
            
            # Filter to only CT files (not masks)
            ct_files = [f for f in ct_files if '_ct.nii.gz' in f.name or 
                       (not any(x in f.name for x in ['_seg', '_msk', 'mask']))]
            
            for ct_file in ct_files:
                # Find corresponding mask
                # VerSe structure: rawdata/sub-xxx/sub-xxx_ct.nii.gz
                #                 derivatives/sub-xxx/sub-xxx_seg-vert_msk.nii.gz
                
                # Get subject name from the CT file
                if ct_file.parent.name.startswith('sub-'):
                    # Subject is the directory name
                    subject_name = ct_file.parent.name
                else:
                    # Subject is in the filename
                    subject_name = ct_file.stem.replace('.nii', '').split('_')[0]
                
                # Check derivatives subfolder
                derivatives_subfolder = derivatives_path / subject_name
                
                if derivatives_subfolder.exists():
                    mask_candidates = list(derivatives_subfolder.glob('*seg*.nii.gz')) + \
                                     list(derivatives_subfolder.glob('*msk*.nii.gz'))
                    
                    if len(mask_candidates) > 0:
                        mask_file = mask_candidates[0]
                        
                        samples.append({
                            'ct_path': str(ct_file),
                            'mask_path': str(mask_file),
                            'subject_id': subject_name
                        })
                    else:
                        print(f"Warning: No mask found for {ct_file.name}")
                else:
                    print(f"Warning: No derivatives folder for {subject_name}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dict with keys:
                - 'ct': CT volume tensor (1, H, W, D)
                - 'mask': Segmentation mask tensor (1, H, W, D)
                - 'spacing': Voxel spacing (3,)
                - 'subject_id': Subject identifier
                - 'augmented': Whether augmentation was applied
        """
        # Check cache
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        sample_info = self.samples[idx]
        
        # Load CT and mask
        ct_nifti = nib.load(sample_info['ct_path'])
        mask_nifti = nib.load(sample_info['mask_path'])
        
        ct_volume = ct_nifti.get_fdata().astype(np.float32)
        mask_volume = mask_nifti.get_fdata().astype(np.uint8)
        
        # Get spacing
        spacing = ct_nifti.header.get_zooms()[:3]
        
        # Apply augmentation (only during training)
        augmented = False
        if self.use_augmentation and self.split == 'train':
            # Get augmentation probabilities
            hardware_prob = self.augmentation_config.get('hardware_prob', 0.5)
            fracture_prob = self.augmentation_config.get('fracture_prob', 0.5)
            
            # Randomly choose augmentation type
            rand_val = np.random.rand()
            
            if rand_val < hardware_prob:
                # Hardware augmentation
                augmented = True
                ct_volume, mask_volume = self.hardware_augmenter(
                    ct_volume, mask_volume, spacing
                )
            elif rand_val < hardware_prob + fracture_prob:
                # Fracture augmentation
                augmented = True
                ct_volume, mask_volume = self.fracture_augmenter(
                    ct_volume, mask_volume, spacing
                )
        
        # Normalize CT values
        if self.normalize:
            ct_volume = self._normalize_ct(ct_volume)
        
        # Convert to torch tensors
        ct_tensor = torch.from_numpy(ct_volume).unsqueeze(0)  # (1, H, W, D)
        mask_tensor = torch.from_numpy(mask_volume).unsqueeze(0)  # (1, H, W, D)
        spacing_tensor = torch.tensor(spacing, dtype=torch.float32)
        
        sample = {
            'ct': ct_tensor,
            'mask': mask_tensor,
            'spacing': spacing_tensor,
            'subject_id': sample_info['subject_id'],
            'augmented': augmented
        }
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = sample
        
        return sample
    
    def _normalize_ct(self, ct_volume: np.ndarray) -> np.ndarray:
        """
        Normalize CT values to [0, 1] range.
        
        Standard CT window: [-1000, 1000] HU (soft tissue window)
        For bone: [-200, 1500] HU
        """
        # Clip to reasonable range
        ct_clipped = np.clip(ct_volume, -1000, 2000)
        
        # Normalize to [0, 1]
        ct_normalized = (ct_clipped + 1000) / 3000.0
        
        return ct_normalized


def create_data_loaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    augmentation_config: Optional[Dict] = None,
    use_augmentation: bool = True,
    use_enhanced_fracture: bool = True  # NEW: ablation flag
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        data_root: Path to VerSe data root
        batch_size: Batch size for training
        num_workers: Number of worker processes
        augmentation_config: Augmentation configuration
        use_augmentation: Whether to use augmentation
        use_enhanced_fracture: If True, use EnhancedFractureAugmenter (ablation)
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = VerSeDataset(
        data_root=data_root,
        split='train',
        augmentation_config=augmentation_config,
        use_augmentation=use_augmentation,
        use_enhanced_fracture=use_enhanced_fracture,  # Pass through
        normalize=True,
        cache_data=False
    )
    
    val_dataset = VerSeDataset(
        data_root=data_root,
        split='val',
        augmentation_config=augmentation_config,
        use_augmentation=False,  # No augmentation for validation
        use_enhanced_fracture=use_enhanced_fracture,  # Pass through
        normalize=True,
        cache_data=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def test_data_loader():
    """
    Test the data loading pipeline.
    """
    print("Testing Data Loader...")
    
    # Use actual data path
    data_root = '/gscratch/scrubbed/june0604/wisespine_for_abnormal/data/raw/verse'
    
    # Create dataset
    try:
        dataset = VerSeDataset(
            data_root=data_root,
            split='train',
            use_augmentation=True,
            normalize=True,
            cache_data=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Load one sample
            sample = dataset[0]
            
            print(f"CT shape: {sample['ct'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")
            print(f"Spacing: {sample['spacing']}")
            print(f"Subject ID: {sample['subject_id']}")
            print(f"Augmented: {sample['augmented']}")
            print(f"CT value range: [{sample['ct'].min():.3f}, {sample['ct'].max():.3f}]")
            print(f"Mask labels: {torch.unique(sample['mask'])}")
            
            print("✓ Data loader test passed!")
        else:
            print("Warning: No samples found in dataset")
            
    except Exception as e:
        print(f"Error testing data loader: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_loader()

