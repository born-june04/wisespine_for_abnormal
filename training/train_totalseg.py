"""
TotalSegmentator Fine-Tuning Training Script
Fine-tune TotalSegmentator on augmented abnormal spine CT data

Uses nnU-Net based TotalSegmentator model and distributed training.
Supports ablation study between original and enhanced fracture augmentation.
"""

import os
import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.data_loader import VerSeDataset, create_data_loaders


def setup_logging(output_dir: Path, rank: int = 0) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('totalseg_finetune')
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    
    if rank == 0:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_totalsegmentator_model(device: torch.device):
    """
    Load TotalSegmentator model.
    
    For now, we use a simplified 3D U-Net. In production, you would:
    1. Install TotalSegmentator: pip install TotalSegmentator
    2. Load pretrained weights: from totalsegmentator.python_api import TotalSegmentator
    3. Extract the nnU-Net model and fine-tune specific layers
    """
    from training.train_ts import UNet3D  # Use existing U-Net
    
    model = UNet3D(
        in_channels=1,
        out_channels=25,  # Background + 24 vertebrae
        base_features=32
    )
    
    return model.to(device)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice loss for segmentation.
    
    Args:
        pred: Predictions (B, C, H, W, D)
        target: Ground truth (B, C, H, W, D) one-hot encoded
        smooth: Smoothing factor
    """
    pred = torch.softmax(pred, dim=1)
    
    # Flatten
    pred_flat = pred.contiguous().view(pred.size(0), pred.size(1), -1)
    target_flat = target.contiguous().view(target.size(0), target.size(1), -1)
    
    # Dice coefficient
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()
    
    return dice_loss


def combined_loss(pred: torch.Tensor, target: torch.Tensor, weight_ce: float = 0.5) -> torch.Tensor:
    """Combined Cross-Entropy + Dice Loss."""
    ce_loss = nn.CrossEntropyLoss()(pred, target.argmax(dim=1))
    d_loss = dice_loss(pred, target)
    
    return weight_ce * ce_loss + (1 - weight_ce) * d_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_dice: float,
    output_dir: Path,
    is_best: bool = False,
    rank: int = 0
):
    """Save model checkpoint."""
    if rank != 0:
        return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
    }
    
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = output_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model (Dice: {best_dice:.4f})")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer: Optional[SummaryWriter] = None,
    rank: int = 0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        ct = batch['ct'].to(device)
        mask = batch['mask'].to(device)
        
        # Forward
        optimizer.zero_grad()
        pred = model(ct)
        loss = combined_loss(pred, mask)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Logging
        if rank == 0 and batch_idx % 10 == 0:
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )
            
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/batch_loss', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    if rank == 0 and writer is not None:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    return {'loss': avg_loss}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer: Optional[SummaryWriter] = None,
    rank: int = 0
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            
            pred = model(ct)
            loss = combined_loss(pred, mask)
            
            # Compute Dice score
            pred_soft = torch.softmax(pred, dim=1)
            dice = 1.0 - dice_loss(pred, mask)
            
            total_loss += loss.item()
            total_dice += dice.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
    
    if rank == 0:
        logger.info(f"Validation Epoch [{epoch}] Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
        
        if writer is not None:
            writer.add_scalar('val/loss', avg_loss, epoch)
            writer.add_scalar('val/dice', avg_dice, epoch)
    
    return {'loss': avg_loss, 'dice': avg_dice}


def main(args):
    """Main training function."""
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fracture_mode = "enhanced" if args.use_enhanced_fracture else "original"
    output_dir = Path(args.output_dir) / f"{timestamp}_{fracture_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, rank)
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("TotalSegmentator Fine-tuning for Abnormal Spine CT")
        logger.info("=" * 80)
        logger.info(f"Fracture augmentation mode: {fracture_mode.upper()}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"World size: {world_size}")
        
        # TensorBoard
        writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
        
        # Save config
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else:
        writer = None
    
    # Create datasets
    augmentation_config = {
        'hardware': {
            'screw_probability': args.screw_prob,
            'rod_probability': args.rod_prob,
            'artifact_strength': args.artifact_strength,
        },
        'fracture_original': {
            'compression_probability': 1.0,
            'compression_range': (0.2, 0.5),
        },
        'fracture_enhanced': {
            'compression_range': (0.2, 0.5),
            'wedge_range': (0.1, 0.3),
            'add_sclerosis': True,
            'add_kyphosis': True,
        },
        'hardware_prob': args.hardware_prob,
        'fracture_prob': args.fracture_prob,
    }
    
    train_loader, val_loader = create_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_config=augmentation_config,
        use_enhanced_fracture=args.use_enhanced_fracture
    )
    
    # Create model
    model = load_totalsegmentator_model(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    best_dice = 0.0
    
    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, logger, writer, rank
        )
        
        # Validate
        if epoch % args.val_interval == 0:
            val_metrics = validate(
                model, val_loader, device, epoch, logger, writer, rank
            )
            
            # Update scheduler
            scheduler.step(val_metrics['dice'])
            
            # Save checkpoint
            is_best = val_metrics['dice'] > best_dice
            if is_best:
                best_dice = val_metrics['dice']
            
            save_checkpoint(
                model, optimizer, epoch, best_dice, output_dir, is_best, rank
            )
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"Training completed! Best Dice: {best_dice:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        writer.close()
    
    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TotalSegmentator Fine-tuning')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True, help='Path to VerSe data root')
    parser.add_argument('--output_dir', type=str, default='outputs/training', help='Output directory')
    
    # Augmentation (ABLATION)
    parser.add_argument('--use_enhanced_fracture', action='store_true', help='Use enhanced fracture augmentation')
    parser.add_argument('--hardware_prob', type=float, default=0.5, help='Hardware augmentation probability')
    parser.add_argument('--fracture_prob', type=float, default=0.5, help='Fracture augmentation probability')
    parser.add_argument('--screw_prob', type=float, default=0.8, help='Screw probability')
    parser.add_argument('--rod_prob', type=float, default=0.8, help='Rod probability')
    parser.add_argument('--artifact_strength', type=float, default=0.7, help='Artifact strength')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval (epochs)')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)

