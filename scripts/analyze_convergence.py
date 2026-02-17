"""
Extract training curves from nnU-Net checkpoints and check convergence.
Generates progress plots for all 6 ablation models.
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = '/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/nnunet/nnUNet_results/Dataset500_SpineAbnormal_Original'
OUTPUT_DIR = '/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/convergence_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

models = {
    'A. Baseline': 'nnUNetTrainer',
    'B. HardwareOnly': 'nnUNetTrainer_HardwareOnly',
    'C. FractureTrad': 'nnUNetTrainer_FractureTrad',
    'D. FracturePhys': 'nnUNetTrainer_FracturePhys',
    'E. FullTrad': 'nnUNetTrainer_FullTrad',
    'F. FullPhys': 'nnUNetTrainer_FullPhys',
}

# Extract training logs from each checkpoint
all_data = {}
for label, name in models.items():
    ckpt_path = f'{RESULTS_DIR}/{name}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
    if not os.path.exists(ckpt_path):
        print(f'{label}: checkpoint_final.pth NOT FOUND')
        continue
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    epoch = ckpt.get('current_epoch', '?')
    
    # nnU-Net logger stores lists of per-epoch metrics
    logger = ckpt.get('logging', {})
    
    print(f'\n=== {label} ({name}) ===')
    print(f'  Epoch: {epoch}')
    print(f'  Logger keys: {list(logger.keys()) if isinstance(logger, dict) else type(logger)}')
    
    # Try different possible key names
    train_loss = logger.get('train_losses', logger.get('train_loss', []))
    val_loss = logger.get('val_losses', logger.get('val_loss', []))
    ema_dice = logger.get('ema_fg_dice', logger.get('dice', []))
    pseudo_dice = logger.get('dice_per_class_or_region', [])
    
    print(f'  train_loss: {len(train_loss)} entries, last={train_loss[-1] if train_loss else "N/A"}')
    print(f'  val_loss: {len(val_loss)} entries, last={val_loss[-1] if val_loss else "N/A"}')
    print(f'  ema_dice: {len(ema_dice)} entries, last={ema_dice[-1] if ema_dice else "N/A"}')
    
    # Check convergence: compare last 50 vs last 200 epochs
    if len(ema_dice) > 200:
        last_50 = np.mean(ema_dice[-50:])
        last_200 = np.mean(ema_dice[-200:])
        last_500 = np.mean(ema_dice[-500:]) if len(ema_dice) >= 500 else np.mean(ema_dice)
        print(f'  Convergence: last50={last_50:.4f}, last200={last_200:.4f}, last500={last_500:.4f}')
        trend = ema_dice[-1] - np.mean(ema_dice[-100:-50])
        print(f'  Recent trend (last vs 50-100 ago): {trend:+.4f} {"(still improving)" if trend > 0.002 else "(converged)"}')
    
    all_data[label] = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'ema_dice': ema_dice,
        'epoch': epoch,
    }

# === Plot 1: Comparison of all models' EMA Dice ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Dice comparison
ax = axes[0, 0]
for label, data in all_data.items():
    if data['ema_dice']:
        ax.plot(data['ema_dice'], label=label, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('EMA Pseudo Dice')
ax.set_title('Validation Dice - All Models')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Dice comparison zoomed (last 300 epochs)
ax = axes[0, 1]
for label, data in all_data.items():
    if data['ema_dice'] and len(data['ema_dice']) > 300:
        ax.plot(range(700, 1000), data['ema_dice'][-300:], label=label, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('EMA Pseudo Dice')
ax.set_title('Validation Dice - Last 300 Epochs (Convergence)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Train loss comparison
ax = axes[1, 0]
for label, data in all_data.items():
    if data['train_loss']:
        ax.plot(data['train_loss'], label=label, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss - All Models')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Val loss comparison
ax = axes[1, 1]
for label, data in all_data.items():
    if data['val_loss']:
        ax.plot(data['val_loss'], label=label, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.set_title('Validation Loss - All Models')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/convergence_comparison.png', dpi=150)
print(f'\nSaved: {OUTPUT_DIR}/convergence_comparison.png')

# === Plot 2: Individual progress.png for each model ===
for label, data in all_data.items():
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if data['train_loss'] and data['val_loss']:
        ax1.plot(data['train_loss'], label='Train', alpha=0.7)
        ax1.plot(data['val_loss'], label='Val', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{label} - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    if data['ema_dice']:
        ax2.plot(data['ema_dice'], color='green', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('EMA Pseudo Dice')
        ax2.set_title(f'{label} - Dice (final: {data["ema_dice"][-1]:.4f})')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to model's fold directory
    trainer_name = models[label]
    fold_dir = f'{RESULTS_DIR}/{trainer_name}__nnUNetPlans__3d_fullres/fold_0'
    fig2.savefig(f'{fold_dir}/progress.png', dpi=100)
    print(f'Saved: {fold_dir}/progress.png')
    plt.close(fig2)

plt.close('all')
print('\nDone!')
