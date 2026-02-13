"""
TotalSegmentator Fine-tuning Training Script
Fine-tune TotalSegmentator on augmented abnormal spine CT data
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.data_loader import create_data_loaders


class UNet3D(nn.Module):
    """
    Simplified 3D U-Net for vertebrae segmentation.
    This is a lightweight implementation for fine-tuning.
    
    For actual TotalSegmentator integration, we would load pretrained weights.
    """
    
    def __init__(self, in_channels=1, out_channels=25, base_features=32):
        """
        Args:
            in_channels: Number of input channels (1 for CT)
            out_channels: Number of output classes (vertebrae labels)
            base_features: Base number of features
        """
        super(UNet3D, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(base_features * 16, base_features * 8, 2, stride=2)
        self.dec4 = self._conv_block(base_features * 16, base_features * 8)
        
        self.upconv3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_features * 2, base_features, 2, stride=2)
        self.dec1 = self._conv_block(base_features * 2, base_features)
        
        # Output
        self.out = nn.Conv3d(base_features, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool3d(2, 2)
    
    def _conv_block(self, in_channels, out_channels):
        """Convolutional block: Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.out(dec1)
        
        return out


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W, D)
            target: Ground truth labels (B, 1, H, W, D)
        """
        # Convert target to one-hot
        num_classes = pred.shape[1]
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.long(), 1)
        
        # Apply softmax to predictions
        pred_softmax = torch.softmax(pred, dim=1)
        
        # Calculate Dice per class
        dice_scores = []
        for c in range(num_classes):
            pred_c = pred_softmax[:, c, ...]
            target_c = target_one_hot[:, c, ...]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average Dice across classes
        dice_loss = 1.0 - torch.stack(dice_scores).mean()
        
        return dice_loss


def calculate_dice_score(pred, target, num_classes):
    """
    Calculate Dice score for evaluation.
    
    Args:
        pred: Predicted labels (B, H, W, D)
        target: Ground truth labels (B, H, W, D)
        num_classes: Number of classes
        
    Returns:
        Dice score per class
    """
    dice_scores = []
    
    for c in range(1, num_classes):  # Skip background
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().float()
        union = pred_c.sum().float() + target_c.sum().float()
        
        if union > 0:
            dice = (2.0 * intersection) / union
            dice_scores.append(dice.item())
        else:
            dice_scores.append(float('nan'))
    
    return dice_scores


class Trainer:
    """
    Trainer for TotalSegmentator fine-tuning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        output_dir: str,
        num_epochs: int = 50,
        log_interval: int = 10,
        save_interval: int = 5
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        
        # Metrics history
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            ct = batch['ct'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(ct)
            loss = self.criterion(output, mask)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to TensorBoard
            if batch_idx % self.log_interval == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        all_dice_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                ct = batch['ct'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Forward pass
                output = self.model(ct)
                loss = self.criterion(output, mask)
                val_loss += loss.item()
                
                # Calculate Dice score
                pred_labels = torch.argmax(output, dim=1)
                batch_dice = calculate_dice_score(
                    pred_labels.cpu(),
                    mask.cpu().squeeze(1),
                    num_classes=output.shape[1]
                )
                all_dice_scores.extend(batch_dice)
        
        avg_val_loss = val_loss / len(self.val_loader)
        avg_dice = np.nanmean(all_dice_scores)
        
        self.val_losses.append(avg_val_loss)
        self.val_dice_scores.append(avg_dice)
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/Loss', avg_val_loss, self.current_epoch)
        self.writer.add_scalar('Val/Dice', avg_dice, self.current_epoch)
        
        return avg_val_loss, avg_dice
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dice_scores': self.val_dice_scores
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with Dice: {self.best_val_dice:.4f}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_dice = self.validate()
            
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Dice: {val_dice:.4f}")
            
            # Save checkpoint
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
            
            if (epoch + 1) % self.save_interval == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        # Save final training metrics
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dice_scores': self.val_dice_scores,
            'best_val_dice': self.best_val_dice
        }
        
        with open(self.output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.writer.close()
        print(f"\nTraining complete! Best validation Dice: {self.best_val_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune TotalSegmentator on abnormal spine CT')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, 
                        default='/gscratch/scrubbed/june0604/wisespine_for_abnormal/data/raw/verse',
                        help='Path to VerSe data root')
    parser.add_argument('--output_dir', type=str,
                        default='/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs',
                        help='Output directory for models and logs')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (use small batch for 3D volumes)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--base_features', type=int, default=16,
                        help='Base number of features in U-Net')
    parser.add_argument('--num_classes', type=int, default=25,
                        help='Number of output classes')
    
    # Augmentation arguments
    parser.add_argument('--augmentation', type=str, default='both',
                        choices=['none', 'hardware', 'fracture', 'both'],
                        help='Type of augmentation to use')
    parser.add_argument('--augmentation_prob', type=float, default=0.7,
                        help='Probability of applying augmentation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'run_{timestamp}_aug_{args.augmentation}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(args), f)
    
    # Configure augmentation
    augmentation_config = {
        'hardware': {
            'screw_probability': 0.4 if args.augmentation in ['hardware', 'both'] else 0.0,
            'rod_probability': 0.3 if args.augmentation in ['hardware', 'both'] else 0.0,
            'cement_probability': 0.2 if args.augmentation in ['hardware', 'both'] else 0.0,
        },
        'fracture': {
            'compression_probability': 0.3 if args.augmentation in ['fracture', 'both'] else 0.0,
            'wedge_probability': 0.2 if args.augmentation in ['fracture', 'both'] else 0.0,
            'burst_probability': 0.1 if args.augmentation in ['fracture', 'both'] else 0.0,
        },
        'augmentation_probability': args.augmentation_prob if args.augmentation != 'none' else 0.0
    }
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_config=augmentation_config,
        use_augmentation=(args.augmentation != 'none')
    )
    
    # Create model
    print("Creating model...")
    model = UNet3D(
        in_channels=1,
        out_channels=args.num_classes,
        base_features=args.base_features
    ).to(device)
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = DiceLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=str(output_dir),
        num_epochs=args.num_epochs,
        log_interval=10,
        save_interval=5
    )
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()

