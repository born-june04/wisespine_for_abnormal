"""
Evaluation script for segmentation performance
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.data_loader import create_data_loaders
from training.train_ts import UNet3D, calculate_dice_score


def evaluate_model(model, data_loader, device, num_classes):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run on
        num_classes: Number of classes
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_dice_scores = []
    dice_per_class = {i: [] for i in range(1, num_classes)}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            output = model(ct)
            pred_labels = torch.argmax(output, dim=1)
            
            # Calculate Dice per sample
            for i in range(ct.shape[0]):
                dice_scores = calculate_dice_score(
                    pred_labels[i:i+1].cpu(),
                    mask[i:i+1].cpu().squeeze(1),
                    num_classes=num_classes
                )
                
                all_dice_scores.extend(dice_scores)
                
                # Store per-class Dice
                for class_idx, dice in enumerate(dice_scores, start=1):
                    if not np.isnan(dice):
                        dice_per_class[class_idx].append(dice)
    
    # Calculate statistics
    mean_dice = np.nanmean(all_dice_scores)
    std_dice = np.nanstd(all_dice_scores)
    
    per_class_stats = {}
    for class_idx, scores in dice_per_class.items():
        if len(scores) > 0:
            per_class_stats[f'class_{class_idx}'] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'count': len(scores)
            }
    
    results = {
        'mean_dice': float(mean_dice),
        'std_dice': float(std_dice),
        'per_class': per_class_stats
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str,
                        default='/gscratch/scrubbed/june0604/wisespine_for_abnormal/data/raw/verse',
                        help='Path to data root')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    model = UNet3D(in_channels=1, out_channels=25, base_features=16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load data
    print(f"Loading {args.split} data...")
    if args.split == 'train':
        train_loader, _, _ = create_data_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            use_augmentation=False
        )
        data_loader = train_loader
    elif args.split == 'val':
        _, val_loader, _ = create_data_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            use_augmentation=False
        )
        data_loader = val_loader
    else:
        _, _, test_loader = create_data_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            use_augmentation=False
        )
        data_loader = test_loader
    
    # Evaluate
    print("Evaluating...")
    results = evaluate_model(model, data_loader, device, num_classes=25)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Dice Score: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
    print("\nPer-class Dice Scores:")
    for class_name, stats in results['per_class'].items():
        print(f"  {class_name}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
    
    # Save results
    if args.output is None:
        output_path = Path(args.checkpoint).parent / f'evaluation_{args.split}.json'
    else:
        output_path = Path(args.output)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()

