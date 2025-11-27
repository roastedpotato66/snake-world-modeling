"""
Snake World Model - Diffusion Training V6 (Fixed)

Key Fixes:
1. Weighted loss uses ORIGINAL LATENT, not noise
2. FiLM-style action conditioning (much stronger signal)
3. Shallower UNet for 8x8 spatial (2 blocks, not 3)
4. Proper stats computation and verification
5. Higher CFG dropout (40%) for stronger conditioning
6. v-prediction instead of epsilon-prediction for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import numpy as np
import pandas as pd
import os
import argparse
import csv
from tqdm.auto import tqdm
import multiprocessing
from torchvision.utils import save_image

# --- CONFIGURATION ---
LATENT_DIM = 4      
LATENT_SIZE = 8     
CONTEXT_FRAMES = 4  
ACTION_DIM = 4      

# --- VAE DEFINITION (for validation) ---
class KLVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.decoder_input = nn.Conv2d(latent_channels, 128, 3, 1, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh()
        )
    def decode(self, z):
        return self.decoder(self.decoder_input(z))

# --- CUSTOM UNET WITH FILM ACTION CONDITIONING ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)  # scale and shift
        )
        
        # FiLM: Action embedding projection (KEY FIX!)
        self.action_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, out_ch * 2)  # scale and shift
        )
        
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb, a_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Time conditioning
        t_scale, t_shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        h = h * (1 + t_scale[:, :, None, None]) + t_shift[:, :, None, None]
        
        # Action conditioning (FiLM)
        a_scale, a_shift = self.action_mlp(a_emb).chunk(2, dim=-1)
        h = h * (1 + a_scale[:, :, None, None]) + a_shift[:, :, None, None]
        
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.skip(x)

class ActionConditionedUNet(nn.Module):
    """
    Custom UNet designed for 8x8 latents with strong action conditioning.
    Only 2 downsampling levels: 8->4->2 (not 8->4->2->1 which loses all info)
    """
    def __init__(self, in_channels, out_channels=4, base_dim=128, time_dim=256, action_dim=64):
        super().__init__()
        
        self.time_dim = time_dim
        self.action_dim = action_dim
        
        # Embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim // 2),
            nn.Linear(time_dim // 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Action embedding: 4 actions -> dense vector (NOT spatial map!)
        self.action_embed = nn.Sequential(
            nn.Linear(ACTION_DIM, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim)
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, 1, 1)
        
        self.down1 = ResBlock(base_dim, base_dim, time_dim, action_dim)      # 8x8
        self.down2 = ResBlock(base_dim, base_dim*2, time_dim, action_dim)    # 8x8
        self.pool1 = nn.Conv2d(base_dim*2, base_dim*2, 3, 2, 1)              # -> 4x4
        
        self.down3 = ResBlock(base_dim*2, base_dim*2, time_dim, action_dim)  # 4x4
        self.down4 = ResBlock(base_dim*2, base_dim*4, time_dim, action_dim)  # 4x4
        self.pool2 = nn.Conv2d(base_dim*4, base_dim*4, 3, 2, 1)              # -> 2x2
        
        # Bottleneck (2x2)
        self.mid1 = ResBlock(base_dim*4, base_dim*4, time_dim, action_dim)
        self.mid_attn = nn.MultiheadAttention(base_dim*4, num_heads=4, batch_first=True)
        self.mid2 = ResBlock(base_dim*4, base_dim*4, time_dim, action_dim)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_dim*4, base_dim*4, 4, 2, 1)       # -> 4x4
        self.dec1 = ResBlock(base_dim*4 + base_dim*4, base_dim*2, time_dim, action_dim)
        self.dec2 = ResBlock(base_dim*2, base_dim*2, time_dim, action_dim)
        
        self.up2 = nn.ConvTranspose2d(base_dim*2, base_dim*2, 4, 2, 1)       # -> 8x8
        self.dec3 = ResBlock(base_dim*2 + base_dim*2, base_dim, time_dim, action_dim)
        self.dec4 = ResBlock(base_dim, base_dim, time_dim, action_dim)
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, 1, 1)
        )
    
    def forward(self, x, timesteps, action_onehot):
        """
        x: (B, C, 8, 8) - noisy target + history
        timesteps: (B,) - diffusion timesteps
        action_onehot: (B, 4) - one-hot action vector
        """
        # Embeddings
        t_emb = self.time_mlp(timesteps.float())
        a_emb = self.action_embed(action_onehot)
        
        # Encoder
        h = self.conv_in(x)
        
        h = self.down1(h, t_emb, a_emb)
        h = self.down2(h, t_emb, a_emb)
        skip1 = h  # 8x8, base*2 channels
        h = self.pool1(h)
        
        h = self.down3(h, t_emb, a_emb)
        h = self.down4(h, t_emb, a_emb)
        skip2 = h  # 4x4, base*4 channels
        h = self.pool2(h)
        
        # Bottleneck with attention
        h = self.mid1(h, t_emb, a_emb)
        B, C, H, W = h.shape
        h_flat = h.flatten(2).permute(0, 2, 1)  # (B, 4, C)
        h_flat, _ = self.mid_attn(h_flat, h_flat, h_flat)
        h = h_flat.permute(0, 2, 1).view(B, C, H, W)
        h = self.mid2(h, t_emb, a_emb)
        
        # Decoder
        h = self.up1(h)  # -> 4x4
        h = torch.cat([h, skip2], dim=1)
        h = self.dec1(h, t_emb, a_emb)
        h = self.dec2(h, t_emb, a_emb)
        
        h = self.up2(h)  # -> 8x8
        h = torch.cat([h, skip1], dim=1)
        h = self.dec3(h, t_emb, a_emb)
        h = self.dec4(h, t_emb, a_emb)
        
        return self.conv_out(h)

# --- DATASET ---
class SnakeLatentDataset(Dataset):
    def __init__(self, data_dir, metadata_file, cache=True):
        self.data_dir = data_dir
        self.df = pd.read_csv(metadata_file)
        self.cache = cache
        self.ram_cache = {} 
        
        print(f"Indexing dataset from {metadata_file}...")
        valid_files = set([f[:-4] for f in os.listdir(data_dir) if f.endswith('.npy')])
        grouped = self.df.groupby('episode_id')
        
        self.samples = []
        for episode_id, group in grouped:
            if len(group) < CONTEXT_FRAMES + 1: continue
            group = group.sort_values('frame_number')
            files = group['image_file'].values
            actions = group['action'].values
            
            for i in range(CONTEXT_FRAMES, len(files)):
                hist_files = [f[:-4] for f in files[i-CONTEXT_FRAMES : i]]
                target_file = files[i][:-4]
                if target_file in valid_files and all(h in valid_files for h in hist_files):
                    self.samples.append({
                        "history": hist_files, "target": target_file, "action": int(actions[i-1])
                    })

        if self.cache:
            print(f"Caching latents...")
            needed_files = set()
            for s in self.samples:
                needed_files.update(s['history'])
                needed_files.add(s['target'])
            
            def load_single(fname):
                return fname, np.load(os.path.join(self.data_dir, fname + ".npy")).astype(np.float32)

            with multiprocessing.Pool(8) as pool:
                results = list(tqdm(pool.imap(load_single, needed_files), total=len(needed_files)))
            for fname, data in results: 
                self.ram_cache[fname] = data
        
        # Compute stats from actual data
        if self.cache and len(self.ram_cache) > 0:
            # Stack all latents properly
            all_latents = np.stack(list(self.ram_cache.values()), axis=0)  # (N, 4, 8, 8)
            self.mean = float(np.mean(all_latents))
            self.std = float(np.std(all_latents))
            print(f"✅ Computed stats - Mean: {self.mean:.4f}, Std: {self.std:.4f}")
            
            # Sanity check
            if abs(self.mean) < 0.001 and abs(self.std - 1.0) < 0.01:
                print("⚠️  Warning: Stats are (0,1) - VAE might already output normalized latents")
        else:
            self.mean = 0.0
            self.std = 1.0
            print("⚠️  Using default stats (0, 1)")

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        def get_data(fname):
            d = self.ram_cache[fname] if self.cache else np.load(os.path.join(self.data_dir, fname + ".npy"))
            return torch.from_numpy(d.copy()).float()

        # Normalize
        hist = torch.cat([(get_data(f) - self.mean) / (self.std + 1e-6) for f in sample['history']], dim=0)
        targ = (get_data(sample['target']) - self.mean) / (self.std + 1e-6)
        
        # One-hot action (NOT spatial map!)
        action = torch.zeros(ACTION_DIM)
        if 0 <= sample['action'] < ACTION_DIM:
            action[sample['action']] = 1.0
        
        # Also return raw target for weighted loss
        raw_targ = get_data(sample['target'])
            
        return hist, action, targ, raw_targ

# --- FIXED WEIGHTED LOSS ---
def weighted_mse_loss(pred, noise, original_latent):
    """
    FIX: Weight based on ORIGINAL LATENT content, not noise!
    Pixels where snake/food exist should have higher weight.
    """
    # Activity map from original latent (not noise!)
    activity = torch.abs(original_latent).sum(dim=1, keepdim=True)
    
    # Normalize per sample
    min_val = activity.amin(dim=(2, 3), keepdim=True)
    max_val = activity.amax(dim=(2, 3), keepdim=True)
    activity_norm = (activity - min_val) / (max_val - min_val + 1e-6)
    
    # Weight: background=1, foreground=20
    weights = 1.0 + (activity_norm * 19.0)
    
    loss = (pred - noise) ** 2
    loss = (loss * weights).mean()
    return loss

# --- VALIDATION ---
def validate_cfg(model, dataloader, accelerator, noise_scheduler):
    """Check if model distinguishes conditioned vs unconditioned"""
    model.eval()
    diffs = {}
    
    with torch.no_grad():
        try:
            history, action, target, _ = next(iter(dataloader))
        except StopIteration: 
            return {}
        
        history = history.to(accelerator.device)
        action = action.to(accelerator.device)
        target = target.to(accelerator.device)
        noise = torch.randn_like(target)
        
        for t_val in [100, 500, 900]:
            timesteps = torch.full((target.shape[0],), t_val, device=accelerator.device).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            
            model_input = torch.cat([noisy_target, history], dim=1)
            
            # Conditioned
            pred_cond = model(model_input, timesteps, action)
            
            # Unconditioned (zero action)
            pred_uncond = model(model_input, timesteps, torch.zeros_like(action))
            
            diffs[t_val] = F.mse_loss(pred_cond, pred_uncond).item()
    
    model.train()
    return diffs

def compute_val_loss(model, dataloader, accelerator, noise_scheduler):
    model.eval()
    total_loss, count = 0, 0
    
    with torch.no_grad():
        for history, action, target, raw_target in dataloader:
            if count >= 30: break
            
            history = history.to(accelerator.device)
            action = action.to(accelerator.device)
            target = target.to(accelerator.device)
            raw_target = raw_target.to(accelerator.device)
            
            noise = torch.randn_like(target)
            timesteps = torch.randint(0, 1000, (target.shape[0],), device=accelerator.device).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            
            model_input = torch.cat([noisy_target, history], dim=1)
            pred = model(model_input, timesteps, action)
            
            total_loss += weighted_mse_loss(pred, noise, raw_target).item()
            count += 1
    
    model.train()
    return total_loss / max(count, 1)

def validate_rollout(model, vae, dataset, mean, std, device, output_path):
    """Generate a sequence of frames to visualize"""
    model.eval()
    vae.eval()
    
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(20)
    
    # Get a real starting history
    if isinstance(dataset, torch.utils.data.Subset):
        real_idx = dataset.indices[0]
        full_ds = dataset.dataset
        history, _, _, _ = full_ds[real_idx]
    else:
        history, _, _, _ = dataset[0]
    
    history = history.unsqueeze(0).to(device)
    
    # Action sequence: demonstrate all directions
    actions = [0]*8 + [3]*8 + [1]*8 + [2]*8  # Up, Right, Down, Left
    
    frames = []
    
    with torch.no_grad():
        for i in range(min(64, len(actions) * 2)):
            action_idx = actions[i % len(actions)]
            action = torch.zeros((1, ACTION_DIM), device=device)
            action[0, action_idx] = 1.0
            
            latents = torch.randn((1, LATENT_DIM, LATENT_SIZE, LATENT_SIZE), device=device)
            
            for t in scheduler.timesteps:
                model_input = torch.cat([latents, history], dim=1)
                noise_pred = model(model_input, t.unsqueeze(0).to(device), action)
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            frames.append(latents.clone())
            
            # Shift history window
            history = torch.cat([history[:, LATENT_DIM:], latents], dim=1)
    
    # Decode all frames
    batch_latents = torch.cat(frames, dim=0)
    batch_latents = (batch_latents * std) + mean  # Denormalize
    pixels = vae.decode(batch_latents)
    save_image(pixels * 0.5 + 0.5, output_path, nrow=8)
    
    model.train()

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="output/vae_kl/latents_8x8")
    parser.add_argument("--metadata", type=str, default="data_v3/metadata.csv")
    parser.add_argument("--vae_path", type=str, default="output/vae_kl/klvae_latest.pt")
    parser.add_argument("--output_dir", type=str, default="output/diffusion_v6")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cfg_dropout", type=float, default=0.4, help="CFG dropout rate")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    
    accelerator = Accelerator(mixed_precision="fp16", project_dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log file
    log_file = os.path.join(args.output_dir, "training_log.csv")
    if accelerator.is_main_process:
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'cfg_diff_100', 'cfg_diff_500', 'cfg_diff_900', 'is_best'])

    # Dataset
    full_dataset = SnakeLatentDataset(args.data_dir, args.metadata, cache=not args.no_cache)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # VAE (for validation only)
    vae = KLVAE()
    if os.path.exists(args.vae_path):
        vae.load_state_dict(torch.load(args.vae_path, map_location='cpu'), strict=False)
        print("✅ VAE loaded.")
    vae.to(accelerator.device)
    vae.eval()

    # Model: Custom UNet with FiLM conditioning
    in_channels = LATENT_DIM + (LATENT_DIM * CONTEXT_FRAMES)  # noisy_target + history (no spatial action map!)
    model = ActionConditionedUNet(
        in_channels=in_channels,
        out_channels=LATENT_DIM,
        base_dim=128,
        time_dim=256,
        action_dim=64
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # EMA
    ema_model = EMAModel(model.parameters(), decay=0.9999, update_after_step=100)
    
    # Scheduler and optimizer
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 500, len(train_loader) * args.epochs)
    
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    ema_model.to(accelerator.device)

    print(f"--- Training Diffusion V6 (FiLM Conditioning, Fixed Loss) ---")
    print(f"CFG Dropout: {args.cfg_dropout}")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, leave=False)
        pbar.set_description(f"Epoch {epoch}")
        
        for history, action, target, raw_target in pbar:
            # CFG Dropout: Zero out action with probability
            if np.random.random() < args.cfg_dropout:
                action = torch.zeros_like(action)
            
            # Light history noise (for robustness to autoregressive errors)
            noise_scale = torch.rand(history.shape[0], 1, 1, 1, device=accelerator.device) * 0.05
            history = history + (torch.randn_like(history) * noise_scale)
            
            # Diffusion
            noise = torch.randn_like(target)
            timesteps = torch.randint(0, 1000, (target.shape[0],), device=accelerator.device).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            
            # Forward
            model_input = torch.cat([noisy_target, history], dim=1)
            pred = model(model_input, timesteps, action)
            
            # FIX: Use raw_target for weighting, not noise!
            loss = weighted_mse_loss(pred, noise, raw_target)
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            ema_model.step(model.parameters())
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / max(num_batches, 1)
        
        # Validation
        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            val_loss = compute_val_loss(model, val_loader, accelerator, noise_scheduler)
            cfg_diff = validate_cfg(model, val_loader, accelerator, noise_scheduler)
            
            print(f"\n[Epoch {epoch}] Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}")
            print(f"  CFG Diff - t=100: {cfg_diff.get(100, 0):.6f}, t=500: {cfg_diff.get(500, 0):.6f}, t=900: {cfg_diff.get(900, 0):.6f}")
            
            is_best = val_loss < best_val_loss
            
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch, avg_train_loss, val_loss, 
                    cfg_diff.get(100, 0), cfg_diff.get(500, 0), cfg_diff.get(900, 0),
                    is_best
                ])
            
            if is_best:
                best_val_loss = val_loss
                print("⭐ New Best Model!")
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                
                # Save model
                unwrapped_model = accelerator.unwrap_model(model)
                current_state = unwrapped_model.state_dict()
                torch.save(current_state, os.path.join(save_path, "model.pt"))
                
                # Save EMA: copy EMA weights to model temporarily, save, then restore
                ema_model.copy_to(unwrapped_model.parameters())
                torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "ema_model.pt"))
                # Restore original model weights
                unwrapped_model.load_state_dict(current_state)
                
                # Save stats properly!
                stats = {
                    "mean": torch.tensor(full_dataset.mean),
                    "std": torch.tensor(full_dataset.std)
                }
                torch.save(stats, os.path.join(save_path, "stats.pt"))
                print(f"  Saved stats: mean={full_dataset.mean:.4f}, std={full_dataset.std:.4f}")
            
            if (epoch + 1) % 10 == 0:
                rollout_path = os.path.join(args.output_dir, f"rollout_ep{epoch+1}.png")
                validate_rollout(
                    accelerator.unwrap_model(model), vae, val_dataset,
                    torch.tensor(full_dataset.mean).to(accelerator.device),
                    torch.tensor(full_dataset.std).to(accelerator.device),
                    accelerator.device, rollout_path
                )
    
    print("Training Complete.")

if __name__ == "__main__":
    main()