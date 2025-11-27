"""
Snake World Model - DIAMOND-Style Implementation
Training script for pixel-space EDM model.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import argparse
from tqdm.auto import tqdm
from torchvision.utils import save_image
import csv

# Import from src modules
from src.models.pixel_edm import PixelSpaceUNet, EDMPrecond, EDMSampler
from src.data.pixel_dataset import SnakePixelDataset, CONTEXT_FRAMES, ACTION_DIM
from src.utils.ema import EMA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model, context, target, action, precond, optimizer, scaler, device):
    """One EDM training step with AMP."""
    log_sigma = torch.randn(target.shape[0], device=device) * 1.2 - 1.2
    sigma = log_sigma.exp()
    noise = torch.randn_like(target)
    x_noisy = target + noise * sigma.view(-1, 1, 1, 1)
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        denoised = model(x_noisy, sigma, context, action)
        weight = precond.get_loss_weight(sigma).view(-1, 1, 1, 1)
        loss = (weight * (denoised - target) ** 2).mean()
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def validate(model, val_loader, sampler, device):
    """Validate by generating frames and measuring MSE."""
    model.eval()
    total_mse = 0
    total_cfg_diff = 0
    count = 0
    
    with torch.no_grad():
        for context, target, action, _ in val_loader:
            if count >= 10:
                break
            
            context = context.to(device)
            target = target.to(device)
            action = action.to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                generated = sampler.sample(model, target.shape, context, action, n_steps=3, device=device)
                null_action = torch.zeros_like(action)
                generated_uncond = sampler.sample(model, target.shape, context, null_action, n_steps=3, device=device)
            
            # MSE (in fp32 for accuracy)
            mse = F.mse_loss(generated.float(), target.float()).item()
            total_mse += mse
            
            cfg_diff = F.mse_loss(generated.float(), generated_uncond.float()).item()
            total_cfg_diff += cfg_diff
            
            count += 1
    
    model.train()
    return total_mse / count, total_cfg_diff / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="data_v5/images")
    parser.add_argument("--metadata", type=str, default="data_v5/metadata.csv")
    parser.add_argument("--output_dir", type=str, default="output/pixel_edm")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.csv")
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_mse', 'cfg_diff', 'is_best'])
    
    full_dataset = SnakePixelDataset(args.img_dir, args.metadata, cache=not args.no_cache)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print('Calculating sampler weights...')
    train_indices = train_dataset.indices
    train_samples_weights = []
    
    count_dead = 0
    count_eat = 0
    
    for idx in train_indices:
        sample = full_dataset.samples[idx]
        weight = 1.0
        
        # Balanced weighting: 5.0 for both major events
        if sample['is_eating']:
            weight = 5.0
            count_eat += 1
        elif sample['is_dead']:
            weight = 5.0
            count_dead += 1
        
        train_samples_weights.append(weight)
    
    print(f"Stats in Train Set: Dead={count_dead}, Eat={count_eat}")
    
    train_samples_weights = torch.DoubleTensor(train_samples_weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_samples_weights,
        num_samples=len(train_samples_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=False, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    model = PixelSpaceUNet(
        in_channels=3 + 3 * CONTEXT_FRAMES, out_channels=3,
        base_dim=128, cond_dim=512).to(DEVICE)
    
    ema = EMA(model)
    precond = EDMPrecond()
    edm_sampler = EDMSampler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_mse = float('inf')
    
    print("Starting Training...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for context, target, action, _ in pbar:
            context = context.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            action = action.to(DEVICE, non_blocking=True)
            if np.random.random() < 0.3:
                action = torch.zeros_like(action)
            
            loss = train_step(model, context, target, action, precond, optimizer, scaler, DEVICE)
            epoch_loss += loss
            ema.update(model)
            pbar.set_postfix(loss=f"{loss:.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation & Checkpointing
        if (epoch + 1) % 5 == 0:
            original_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.apply(model)
            val_mse, cfg_diff = validate(model, val_loader, edm_sampler, DEVICE)
            
            print(f"\n[Epoch {epoch}] Loss: {avg_loss:.4f} | Val MSE: {val_mse:.4f} | CFG: {cfg_diff:.4f}")
            
            is_best = val_mse < best_val_mse
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, avg_loss, val_mse, cfg_diff, is_best])
            
            # Save Checkpoint
            ckpt_dir = os.path.join(args.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_ep{epoch}.pt"))
            
            if is_best:
                best_val_mse = val_mse
                print("⭐ New Best!")
                best_dir = os.path.join(args.output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            
            model.load_state_dict(original_state)


if __name__ == "__main__":
    main()

