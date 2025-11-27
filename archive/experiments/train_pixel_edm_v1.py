"""
Snake World Model - DIAMOND-Style Implementation

Following the DIAMOND paper exactly:
1. Works in PIXEL SPACE (64x64x3), NO VAE
2. Uses EDM formulation (not DDPM)
3. Action conditioning via Adaptive Group Normalization
4. Frame stacking (4 previous frames concatenated channel-wise)
5. Only 3 denoising steps needed

Paper: "Diffusion for World Modeling: Visual Details Matter in Atari"
https://arxiv.org/abs/2405.12399
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
# EDM FORMULATION (from Karras et al. 2022)
# ============================================================================

class EDMPrecond:
    """EDM preconditioning - the key to making diffusion work with few steps."""
    
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data
    
    def get_scalings(self, sigma):
        """
        Returns c_skip, c_out, c_in for network preconditioning.
        These ensure network input/output stay at unit variance.
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        return c_skip, c_out, c_in
    
    def get_loss_weight(self, sigma):
        """Weight for the loss at each noise level."""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2


class EDMSampler:
    """EDM deterministic sampler - works with just 3 steps!"""
    
    def __init__(self, sigma_min=0.002, sigma_max=80, rho=7):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
    
    def get_sigmas(self, n_steps):
        """Get sigma schedule for n steps."""
        step_indices = torch.arange(n_steps)
        t = step_indices / (n_steps - 1)
        sigmas = (self.sigma_max ** (1/self.rho) + t * (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho))) ** self.rho
        return torch.cat([sigmas, torch.zeros(1)])  # Add sigma=0 at the end
    
    @torch.no_grad()
    def sample(self, model, shape, context, action, n_steps=3, device='cuda'):
        """
        EDM deterministic (Euler) sampler.
        """
        sigmas = self.get_sigmas(n_steps).to(device)
        x = torch.randn(shape, device=device) * sigmas[0]
        
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Denoised prediction
            denoised = model(x, sigma.expand(x.shape[0]), context, action)
            
            # Euler step
            d = (x - denoised) / sigma
            x = x + d * (sigma_next - sigma)
        
        return x


# ============================================================================
# ADAPTIVE GROUP NORMALIZATION (key for action conditioning)
# ============================================================================

class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization - conditions the normalization on action.
    This is what DIAMOND uses for action conditioning.
    """
    def __init__(self, num_channels, num_groups=32, cond_dim=256):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)  # scale and shift
    
    def forward(self, x, cond):
        """
        x: (B, C, H, W)
        cond: (B, cond_dim) - combined action + sigma embedding
        """
        x = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


# ============================================================================
# UNET WITH ADAPTIVE GROUPNORM (DIAMOND-style)
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with Adaptive GroupNorm for conditioning."""
    
    def __init__(self, in_ch, out_ch, cond_dim=256, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_ch, min(num_groups, out_ch), cond_dim)
        self.norm2 = AdaptiveGroupNorm(out_ch, min(num_groups, out_ch), cond_dim)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h, cond)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h, cond)
        h = F.silu(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Self-attention for global reasoning (used at bottleneck)."""
    
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
        
        # Scaled dot-product attention
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
        return self.down(h), h  # Return skip connection


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch * 2, out_ch, cond_dim)  # *2 for skip concat
    
    def forward(self, x, skip, cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, cond)


class PixelSpaceUNet(nn.Module):
    """
    DIAMOND-style UNet working directly in pixel space.
    
    Input: noisy_next_frame (3 ch) + context_frames (4 * 3 = 12 ch) = 15 channels
    Output: denoised_next_frame (3 ch)
    
    Conditioning: action (one-hot) + sigma embedding → AdaptiveGroupNorm
    """
    
    def __init__(self, in_channels=15, out_channels=3, base_dim=128, cond_dim=512):
        super().__init__()
        
        self.cond_dim = cond_dim
        
        # Sigma embedding (log-scale)
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(ACTION_DIM, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        # Encoder: 64 -> 32 -> 16 -> 8
        self.down1 = DownBlock(base_dim, base_dim * 2, cond_dim)      # 64 -> 32
        self.down2 = DownBlock(base_dim * 2, base_dim * 4, cond_dim)  # 32 -> 16
        self.down3 = DownBlock(base_dim * 4, base_dim * 4, cond_dim)  # 16 -> 8
        
        # Bottleneck with Self-Attention for global reasoning
        self.mid1 = ResBlock(base_dim * 4, base_dim * 4, cond_dim)
        self.mid_attn = SelfAttention(base_dim * 4, num_heads=8)  # Global attention at 8x8
        self.mid2 = ResBlock(base_dim * 4, base_dim * 4, cond_dim)
        
        # Decoder: 8 -> 16 -> 32 -> 64
        self.up1 = UpBlock(base_dim * 4, base_dim * 4, cond_dim)   # 8 -> 16
        self.up2 = UpBlock(base_dim * 4, base_dim * 2, cond_dim)   # 16 -> 32
        self.up3 = UpBlock(base_dim * 2, base_dim, cond_dim)       # 32 -> 64
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, padding=1),
        )
        
        # EDM preconditioning
        self.precond = EDMPrecond(sigma_data=0.5)
    
    def forward(self, x_noisy, sigma, context, action):
        """
        x_noisy: (B, 3, 64, 64) - noisy next frame
        sigma: (B,) - noise level
        context: (B, 12, 64, 64) - 4 previous frames stacked
        action: (B, 4) - one-hot action
        
        Returns: denoised x_0 prediction
        """
        # Get preconditioning scalings
        c_skip, c_out, c_in = self.precond.get_scalings(sigma.view(-1, 1, 1, 1))
        
        # Scale input
        x_scaled = x_noisy * c_in
        
        # Concatenate context frames
        x = torch.cat([x_scaled, context], dim=1)  # (B, 15, 64, 64)
        
        # Conditioning: action + sigma
        sigma_emb = self.sigma_embed(sigma.log().view(-1, 1))
        action_emb = self.action_embed(action)
        cond = sigma_emb + action_emb  # (B, cond_dim)
        
        # UNet forward
        h = self.conv_in(x)
        
        h, s1 = self.down1(h, cond)
        h, s2 = self.down2(h, cond)
        h, s3 = self.down3(h, cond)
        
        # Bottleneck with attention
        h = self.mid1(h, cond)
        h = self.mid_attn(h)  # Self-attention for global reasoning
        h = self.mid2(h, cond)
        
        h = self.up1(h, s3, cond)
        h = self.up2(h, s2, cond)
        h = self.up3(h, s1, cond)
        
        out = self.conv_out(h)
        
        # Apply preconditioning to output
        denoised = c_skip * x_noisy + c_out * out
        
        return denoised


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Exponential Moving Average of model weights."""
    
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
        # Linear warmup of decay
        decay = min(self.decay, (1 + self.step) / (self.warmup + self.step))
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].lerp_(param.data, 1 - decay)
    
    def apply(self, model):
        """Apply EMA weights to model (for eval/inference)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        """Restore original weights (after eval)."""
        # This requires storing original weights - simpler to just reload
        pass
    
    def state_dict(self):
        return {'shadow': self.shadow, 'step': self.step}
    
    def load_state_dict(self, state):
        self.shadow = state['shadow']
        self.step = state['step']


# ============================================================================
# DATASET (loads raw images, no VAE!)
# ============================================================================

class SnakePixelDataset(Dataset):
    """Load raw pixel images for DIAMOND-style training."""
    
    def __init__(self, img_dir, metadata_file, cache=True):
        self.img_dir = img_dir
        self.df = pd.read_csv(metadata_file)
        self.cache = cache
        self.img_cache = {}
        
        # Standard normalization to [-1, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"Indexing dataset...")
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
            
            for i in range(CONTEXT_FRAMES, len(files)):
                ctx_files = files[i-CONTEXT_FRAMES:i].tolist()
                target_file = files[i]
                if target_file in valid_files and all(f in valid_files for f in ctx_files):
                    self.samples.append({
                        "context": ctx_files,
                        "target": target_file,
                        "action": int(actions[i-1]),
                        "is_dead": bool(is_dead_vals[i])
                    })
        
        print(f"Dataset: {len(self.samples)} samples")
        
        # Pre-cache if requested
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
        
        # Load context frames and stack
        context = torch.cat([self._load_img(f) for f in sample['context']], dim=0)  # (12, 64, 64)
        target = self._load_img(sample['target'])  # (3, 64, 64)
        
        # One-hot action
        action = torch.zeros(ACTION_DIM)
        action[sample['action']] = 1.0
        
        return context, target, action, sample['action']


# ============================================================================
# TRAINING
# ============================================================================

def train_step(model, context, target, action, precond, optimizer, scaler, device):
    """One EDM training step with AMP."""
    
    # Sample sigma from log-normal distribution (EDM's recommendation)
    log_sigma = torch.randn(target.shape[0], device=device) * 1.2 - 1.2
    sigma = log_sigma.exp()
    
    # Add noise
    noise = torch.randn_like(target)
    x_noisy = target + noise * sigma.view(-1, 1, 1, 1)
    
    # Forward with AMP
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        denoised = model(x_noisy, sigma, context, action)
        
        # EDM loss (weighted MSE)
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
            
            # Generate with BF16
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                generated = sampler.sample(
                    model,
                    target.shape,
                    context,
                    action,
                    n_steps=3,
                    device=device
                )
                
                # CFG diff (conditioned vs unconditioned)
                null_action = torch.zeros_like(action)
                generated_uncond = sampler.sample(
                    model,
                    target.shape,
                    context,
                    null_action,
                    n_steps=3,
                    device=device
                )
            
            # MSE (in fp32 for accuracy)
            mse = F.mse_loss(generated.float(), target.float()).item()
            total_mse += mse
            
            cfg_diff = F.mse_loss(generated.float(), generated_uncond.float()).item()
            total_cfg_diff += cfg_diff
            
            count += 1
    
    model.train()
    return total_mse / count, total_cfg_diff / count


def visualize(model, dataset, sampler, device, save_path, n_frames=32):
    """Generate a rollout visualization."""
    model.eval()
    
    # Get initial context from dataset
    context, _, _, _ = dataset[0]
    context = context.unsqueeze(0).to(device)  # (1, 12, 64, 64)
    
    frames = []
    actions = [0]*8 + [3]*8 + [1]*8 + [2]*8  # Up, Right, Down, Left
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(n_frames):
            action = torch.zeros(1, ACTION_DIM, device=device)
            action[0, actions[i % len(actions)]] = 1.0
            
            # Generate next frame
            generated = sampler.sample(
                model,
                (1, 3, IMG_SIZE, IMG_SIZE),
                context,
                action,
                n_steps=3,
                device=device
            )
            
            frames.append(generated.float())
            
            # Update context (shift window)
            context = torch.cat([context[:, 3:], generated], dim=1)
    
    # Save grid
    all_frames = torch.cat(frames, dim=0)
    save_image(all_frames * 0.5 + 0.5, save_path, nrow=8)
    print(f"Saved rollout to {save_path}")
    
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="data_v3/images")
    parser.add_argument("--metadata", type=str, default="data_v3/metadata.csv")
    parser.add_argument("--output_dir", type=str, default="output/pixel_edm")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)  # Larger batch for A100
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logging
    log_file = os.path.join(args.output_dir, "training_log.csv")
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_mse', 'cfg_diff', 'is_best'])
    
    # Dataset
    full_dataset = SnakePixelDataset(args.img_dir, args.metadata, cache=not args.no_cache)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print('Calculating sampler weights...')

# 1. Get indices that belong to the training set
    train_indices = train_dataset.indices
    
    # 2. Extract targets for just the training set
    # We assign weight 5.0 to death frames, 1.0 to normal frames
    # Since you have ~2.5% deaths, a weight of 5x brings it to ~13% effective representation
    train_samples_weights = []
    for idx in train_indices:
        sample = full_dataset.samples[idx]
        weight = 5.0 if sample['is_dead'] else 1.0
        train_samples_weights.append(weight)
        
    train_samples_weights = torch.DoubleTensor(train_samples_weights)
    
    # 3. Create Sampler
    # num_samples=len(train_samples_weights) ensures epoch size stays the same
    # replacement=True is MANDATORY for weighted sampling
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_samples_weights,
        num_samples=len(train_samples_weights),
        replacement=True
    )
    
    # 4. Pass sampler to DataLoader and turn shuffle=False (Mutually Exclusive)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,      # <--- Added Sampler
        shuffle=False,        # <--- MUST BE FALSE when using sampler
        num_workers=8, 
        pin_memory=True, 
        drop_last=True, 
        persistent_workers=True
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Model with increased capacity
    model = PixelSpaceUNet(
        in_channels=3 + 3 * CONTEXT_FRAMES,  # noisy + context
        out_channels=3,
        base_dim=128,   # Increased from 64
        cond_dim=512    # Increased from 256
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # EMA
    ema = EMA(model, decay=args.ema_decay, warmup=1000)
    
    precond = EDMPrecond(sigma_data=0.5)
    sampler = EDMSampler()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()  # For AMP
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"--- Training DIAMOND-style Pixel-Space EDM ---")
    print(f"    Device: {DEVICE}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Using: BF16 AMP, EMA, Self-Attention")
    
    best_val_mse = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for context, target, action, _ in pbar:
            context = context.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            action = action.to(DEVICE, non_blocking=True)
            
            # CFG dropout (30%)
            if np.random.random() < 0.3:
                action = torch.zeros_like(action)
            
            loss = train_step(model, context, target, action, precond, optimizer, scaler, DEVICE)
            epoch_loss += loss
            
            # Update EMA
            ema.update(model)
            
            pbar.set_postfix(loss=f"{loss:.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation (use EMA model)
        if (epoch + 1) % 5 == 0:
            # Save current weights
            original_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            # Apply EMA weights for validation
            ema.apply(model)
            
            val_mse, cfg_diff = validate(model, val_loader, sampler, DEVICE)
            
            print(f"\n[Epoch {epoch}] Train: {avg_loss:.4f} | Val MSE: {val_mse:.4f} | CFG Diff: {cfg_diff:.4f}")
            
            is_best = val_mse < best_val_mse
            
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, avg_loss, val_mse, cfg_diff, is_best])
            
            if is_best:
                best_val_mse = val_mse
                print("⭐ New Best!")
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                # Save EMA model (currently applied)
                torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
                torch.save(ema.state_dict(), os.path.join(save_path, "ema.pt"))
            
            # Visualize with EMA model
            if (epoch + 1) % 10 == 0:
                viz_path = os.path.join(args.output_dir, f"rollout_ep{epoch+1}.png")
                visualize(model, val_dataset.dataset, sampler, DEVICE, viz_path)
            
            # Restore original weights for continued training
            model.load_state_dict(original_state)
    
    print("Training Complete!")


if __name__ == "__main__":
    main()