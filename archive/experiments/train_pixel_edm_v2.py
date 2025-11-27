"""
Snake World Model - DIAMOND-Style Implementation (V2)
Updates:
1. Handles 'is_eating' from metadata
2. Balanced weighting (5.0 for Death, 5.0 for Eating)
3. Saves checkpoints every 5 epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
import argparse
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import csv
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64
CONTEXT_FRAMES = 4
ACTION_DIM = 4

# ============================================================================
# EDM FORMULATION 
# ============================================================================

class EDMPrecond:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data
    
    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        return c_skip, c_out, c_in
    
    def get_loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2


class EDMSampler:
    def __init__(self, sigma_min=0.002, sigma_max=80, rho=7):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
    
    def get_sigmas(self, n_steps):
        step_indices = torch.arange(n_steps)
        t = step_indices / (n_steps - 1)
        sigmas = (self.sigma_max ** (1/self.rho) + t * (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho))) ** self.rho
        return torch.cat([sigmas, torch.zeros(1)]) 
    
    @torch.no_grad()
    def sample(self, model, shape, context, action, n_steps=3, device='cuda'):
        sigmas = self.get_sigmas(n_steps).to(device)
        x = torch.randn(shape, device=device) * sigmas[0]
        
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            denoised = model(x, sigma.expand(x.shape[0]), context, action)
            d = (x - denoised) / sigma
            x = x + d * (sigma_next - sigma)
        
        return x

# ============================================================================
# UNET & LAYERS
# ============================================================================

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, cond_dim=256):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2) 
    
    def forward(self, x, cond):
        x = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=256, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_ch, min(num_groups, out_ch), cond_dim)
        self.norm2 = AdaptiveGroupNorm(out_ch, min(num_groups, out_ch), cond_dim)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, cond):
        h = F.silu(self.norm1(self.conv1(x), cond))
        h = F.silu(self.norm2(self.conv2(h), cond))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    
    def forward(self, x, cond):
        h = self.res(x, cond)
        return self.down(h), h

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch * 2, out_ch, cond_dim)
    
    def forward(self, x, skip, cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, cond)

class PixelSpaceUNet(nn.Module):
    def __init__(self, in_channels=15, out_channels=3, base_dim=128, cond_dim=512):
        super().__init__()
        self.cond_dim = cond_dim
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.action_embed = nn.Sequential(
            nn.Linear(ACTION_DIM, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.down1 = DownBlock(base_dim, base_dim * 2, cond_dim)
        self.down2 = DownBlock(base_dim * 2, base_dim * 4, cond_dim)
        self.down3 = DownBlock(base_dim * 4, base_dim * 4, cond_dim)
        
        self.mid1 = ResBlock(base_dim * 4, base_dim * 4, cond_dim)
        self.mid_attn = SelfAttention(base_dim * 4, num_heads=8)
        self.mid2 = ResBlock(base_dim * 4, base_dim * 4, cond_dim)
        
        self.up1 = UpBlock(base_dim * 4, base_dim * 4, cond_dim)
        self.up2 = UpBlock(base_dim * 4, base_dim * 2, cond_dim)
        self.up3 = UpBlock(base_dim * 2, base_dim, cond_dim)
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_dim), nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, padding=1))
        
        self.precond = EDMPrecond(sigma_data=0.5)
    
    def forward(self, x_noisy, sigma, context, action):
        c_skip, c_out, c_in = self.precond.get_scalings(sigma.view(-1, 1, 1, 1))
        x_scaled = x_noisy * c_in
        x = torch.cat([x_scaled, context], dim=1)
        sigma_emb = self.sigma_embed(sigma.log().view(-1, 1))
        action_emb = self.action_embed(action)
        cond = sigma_emb + action_emb
        h = self.conv_in(x)
        h, s1 = self.down1(h, cond)
        h, s2 = self.down2(h, cond)
        h, s3 = self.down3(h, cond)
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)
        h = self.up1(h, s3, cond)
        h = self.up2(h, s2, cond)
        h = self.up3(h, s1, cond)
        out = self.conv_out(h)
        return c_skip * x_noisy + c_out * out

# ============================================================================
# EMA
# ============================================================================

class EMA:
    def __init__(self, model, decay=0.9999, warmup=1000):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        self.step += 1
        decay = min(self.decay, (1 + self.step) / (self.warmup + self.step))
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].lerp_(param.data, 1 - decay)
    
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
                
    def state_dict(self):
        return {'shadow': self.shadow, 'step': self.step}
    
    def load_state_dict(self, state):
        self.shadow = state['shadow']
        self.step = state['step']

# ============================================================================
# DATASET
# ============================================================================

class SnakePixelDataset(Dataset):
    def __init__(self, img_dir, metadata_file, cache=True):
        self.img_dir = img_dir
        self.df = pd.read_csv(metadata_file)
        self.cache = cache
        self.img_cache = {}
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Indexing dataset from {metadata_file}...")
        valid_files = set([f for f in os.listdir(img_dir) if f.endswith('.png')])
        grouped = self.df.groupby('episode_id')
        
        self.samples = []
        for episode_id, group in grouped:
            if len(group) < CONTEXT_FRAMES + 1:
                continue
            group = group.sort_values('frame_number')
            files = group['image_file'].values
            actions = group['action'].values
            is_dead_vals = group['is_dead'].values
            
            # Check if is_eating exists in CSV, else default False
            if 'is_eating' in group.columns:
                is_eating_vals = group['is_eating'].values
            else:
                is_eating_vals = [False] * len(group)
            
            for i in range(CONTEXT_FRAMES, len(files)):
                ctx_files = files[i-CONTEXT_FRAMES:i].tolist()
                target_file = files[i]
                if target_file in valid_files: # Simple check
                    self.samples.append({
                        "context": ctx_files,
                        "target": target_file,
                        "action": int(actions[i-1]),
                        "is_dead": bool(is_dead_vals[i]),
                        "is_eating": bool(is_eating_vals[i])
                    })
        
        print(f"Dataset: {len(self.samples)} samples")
        
        if self.cache:
            print("Caching images...")
            all_files = set()
            for s in self.samples:
                all_files.update(s['context'])
                all_files.add(s['target'])
            for f in tqdm(all_files):
                self._load_img(f)
    
    def _load_img(self, filename):
        if filename in self.img_cache:
            return self.img_cache[filename]
        img = Image.open(os.path.join(self.img_dir, filename)).convert('RGB')
        tensor = self.transform(img)
        if self.cache:
            self.img_cache[filename] = tensor
        return tensor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = torch.cat([self._load_img(f) for f in sample['context']], dim=0)
        target = self._load_img(sample['target'])
        action = torch.zeros(ACTION_DIM)
        action[sample['action']] = 1.0
        return context, target, action, sample['action']

# ============================================================================
# TRAINING
# ============================================================================

def train_step(model, context, target, action, precond, optimizer, scaler, device):
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
    model.eval()
    total_mse = 0
    total_cfg_diff = 0
    count = 0
    with torch.no_grad():
        for context, target, action, _ in val_loader:
            if count >= 10: break
            context = context.to(device)
            target = target.to(device)
            action = action.to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                generated = sampler.sample(model, target.shape, context, action, n_steps=3, device=device)
                null_action = torch.zeros_like(action)
                generated_uncond = sampler.sample(model, target.shape, context, null_action, n_steps=3, device=device)
            
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
    parser.add_argument("--output_dir", type=str, default="output/pixel_edm_v2")
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
        
        # New Balancing Strategy: 5.0 for both major events
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
            if np.random.random() < 0.3: action = torch.zeros_like(action)
            
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