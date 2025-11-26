"""
Snake World Model - Comprehensive Diagnostic Tool

This script diagnoses issues across the entire pipeline:
1. VAE encoding/decoding quality
2. Latent space statistics
3. Dataset sample visualization  
4. Diffusion model behavior at different timesteps
5. Action conditioning effectiveness
6. Gradient flow analysis

Run: python diagnose_pipeline.py --model_path output/diffusion_v6/best_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import pandas as pd
from collections import defaultdict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# MODEL DEFINITIONS (copy from your training code)
# ============================================================================

class KLVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.SiLU(),
        )
        self.to_moments = nn.Conv2d(128, 2 * latent_channels, 3, 1, 1)
        self.decoder_input = nn.Conv2d(latent_channels, 128, 3, 1, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh()
        )
    
    def encode(self, x):
        moments = self.to_moments(self.encoder(x))
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean, std
    
    def decode(self, z):
        return self.decoder(self.decoder_input(z))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        self.action_mlp = nn.Sequential(nn.SiLU(), nn.Linear(action_dim, out_ch * 2))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb, a_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        t_s, t_sh = self.time_mlp(t_emb).chunk(2, dim=-1)
        h = h * (1 + t_s[:, :, None, None]) + t_sh[:, :, None, None]
        a_s, a_sh = self.action_mlp(a_emb).chunk(2, dim=-1)
        h = h * (1 + a_s[:, :, None, None]) + a_sh[:, :, None, None]
        h = F.silu(h)
        h = self.norm2(self.conv2(h))
        return F.silu(h) + self.skip(x)

class ActionConditionedUNet(nn.Module):
    def __init__(self, in_channels, out_channels=4, base_dim=128, time_dim=256, action_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim // 2),
            nn.Linear(time_dim // 2, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.action_embed = nn.Sequential(
            nn.Linear(4, action_dim), nn.SiLU(),
            nn.Linear(action_dim, action_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, 1, 1)
        self.down1 = ResBlock(base_dim, base_dim, time_dim, action_dim)
        self.down2 = ResBlock(base_dim, base_dim*2, time_dim, action_dim)
        self.pool1 = nn.Conv2d(base_dim*2, base_dim*2, 3, 2, 1)
        self.down3 = ResBlock(base_dim*2, base_dim*2, time_dim, action_dim)
        self.down4 = ResBlock(base_dim*2, base_dim*4, time_dim, action_dim)
        self.pool2 = nn.Conv2d(base_dim*4, base_dim*4, 3, 2, 1)
        self.mid1 = ResBlock(base_dim*4, base_dim*4, time_dim, action_dim)
        self.mid_attn = nn.MultiheadAttention(base_dim*4, 4, batch_first=True)
        self.mid2 = ResBlock(base_dim*4, base_dim*4, time_dim, action_dim)
        self.up1 = nn.ConvTranspose2d(base_dim*4, base_dim*4, 4, 2, 1)
        self.dec1 = ResBlock(base_dim*8, base_dim*2, time_dim, action_dim)
        self.dec2 = ResBlock(base_dim*2, base_dim*2, time_dim, action_dim)
        self.up2 = nn.ConvTranspose2d(base_dim*2, base_dim*2, 4, 2, 1)
        self.dec3 = ResBlock(base_dim*4, base_dim, time_dim, action_dim)
        self.dec4 = ResBlock(base_dim, base_dim, time_dim, action_dim)
        self.conv_out = nn.Sequential(nn.GroupNorm(8, base_dim), nn.SiLU(), nn.Conv2d(base_dim, out_channels, 3, 1, 1))
    
    def forward(self, x, timesteps, action):
        t = self.time_mlp(timesteps.float())
        a = self.action_embed(action)
        h = self.conv_in(x)
        h = self.down1(h, t, a); h = self.down2(h, t, a); s1 = h; h = self.pool1(h)
        h = self.down3(h, t, a); h = self.down4(h, t, a); s2 = h; h = self.pool2(h)
        h = self.mid1(h, t, a)
        B, C, H, W = h.shape
        hf = h.flatten(2).permute(0,2,1)
        hf, _ = self.mid_attn(hf, hf, hf)
        h = hf.permute(0,2,1).view(B,C,H,W)
        h = self.mid2(h, t, a)
        h = torch.cat([self.up1(h), s2], 1); h = self.dec1(h, t, a); h = self.dec2(h, t, a)
        h = torch.cat([self.up2(h), s1], 1); h = self.dec3(h, t, a); h = self.dec4(h, t, a)
        return self.conv_out(h)

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def load_models(args):
    """Load VAE and diffusion model"""
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    
    # VAE
    vae = KLVAE().to(DEVICE)
    if os.path.exists(args.vae_path):
        vae.load_state_dict(torch.load(args.vae_path, map_location=DEVICE), strict=False)
        print(f"✅ VAE loaded from {args.vae_path}")
    else:
        print(f"❌ VAE not found at {args.vae_path}")
        return None, None, None, None
    vae.eval()
    
    # Diffusion
    model = None
    mean, std = torch.tensor(0.0), torch.tensor(1.0)
    
    if args.model_path and os.path.exists(args.model_path):
        model_file = os.path.join(args.model_path, "model.pt")
        if os.path.exists(model_file):
            model = ActionConditionedUNet(in_channels=20, out_channels=4).to(DEVICE)
            model.load_state_dict(torch.load(model_file, map_location=DEVICE))
            print(f"✅ Diffusion model loaded from {model_file}")
            model.eval()
        
        stats_file = os.path.join(args.model_path, "stats.pt")
        if os.path.exists(stats_file):
            stats = torch.load(stats_file, map_location=DEVICE)
            mean = stats.get("mean", torch.tensor(0.0)).float()
            std = stats.get("std", torch.tensor(1.0)).float()
            print(f"✅ Stats: mean={mean.item():.4f}, std={std.item():.4f}")
    
    return vae, model, mean, std

def diagnose_vae(vae, args, save_dir):
    """Test 1: VAE encoding/decoding quality"""
    print("\n" + "="*60)
    print("TEST 1: VAE QUALITY")
    print("="*60)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Load some sample images
    img_dir = os.path.join(args.data_dir, "..", "images") if "latents" in args.data_dir else args.data_dir
    if not os.path.exists(img_dir):
        img_dir = "data_v3/images"
    
    if not os.path.exists(img_dir):
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])[:16]
    
    images = []
    for f in img_files:
        img = Image.open(os.path.join(img_dir, f)).convert('RGB')
        images.append(transform(img))
    
    images = torch.stack(images).to(DEVICE)
    
    with torch.no_grad():
        mean, std = vae.encode(images)
        z = mean + std * torch.randn_like(mean)
        recon = vae.decode(z)
    
    # Visualize
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(8):
        # Original
        img = images[i].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
        axes[0, i].imshow(img.clip(0,1))
        axes[0, i].axis('off')
        axes[0, i].set_title('Original' if i==0 else '')
        
        # Reconstruction
        rec = recon[i].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
        axes[1, i].imshow(rec.clip(0,1))
        axes[1, i].axis('off')
        axes[1, i].set_title('Recon' if i==0 else '')
        
        # Latent channel 0
        lat = z[i, 0].cpu().numpy()
        axes[2, i].imshow(lat, cmap='viridis')
        axes[2, i].axis('off')
        axes[2, i].set_title('Latent[0]' if i==0 else '')
        
        # Latent channel 1
        lat = z[i, 1].cpu().numpy()
        axes[3, i].imshow(lat, cmap='viridis')
        axes[3, i].axis('off')
        axes[3, i].set_title('Latent[1]' if i==0 else '')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_vae_quality.png"), dpi=150)
    plt.close()
    
    # Statistics
    print(f"\n📊 Latent Statistics:")
    print(f"   Mean: {z.mean().item():.4f}")
    print(f"   Std:  {z.std().item():.4f}")
    print(f"   Min:  {z.min().item():.4f}")
    print(f"   Max:  {z.max().item():.4f}")
    
    # Per-channel stats
    print(f"\n📊 Per-Channel Statistics:")
    for c in range(4):
        ch = z[:, c]
        print(f"   Channel {c}: mean={ch.mean().item():+.3f}, std={ch.std().item():.3f}, range=[{ch.min().item():.2f}, {ch.max().item():.2f}]")
    
    # Reconstruction error
    mse = F.mse_loss(recon, images).item()
    print(f"\n📊 Reconstruction MSE: {mse:.6f}")
    
    if mse > 0.1:
        print("   ⚠️  HIGH reconstruction error - VAE may not be well trained")
    else:
        print("   ✅ Reconstruction quality looks reasonable")

def diagnose_latent_dataset(args, save_dir):
    """Test 2: Analyze saved latent files"""
    print("\n" + "="*60)
    print("TEST 2: LATENT DATASET ANALYSIS")
    print("="*60)
    
    latent_dir = args.data_dir
    if not os.path.exists(latent_dir):
        print(f"❌ Latent directory not found: {latent_dir}")
        return
    
    files = [f for f in os.listdir(latent_dir) if f.endswith('.npy')][:1000]
    print(f"   Found {len(files)} latent files (analyzing first 1000)")
    
    # Load samples
    latents = []
    for f in files[:100]:
        lat = np.load(os.path.join(latent_dir, f))
        latents.append(lat)
    
    latents = np.stack(latents)
    print(f"   Latent shape: {latents.shape}")
    
    # Statistics
    print(f"\n📊 Dataset Latent Statistics:")
    print(f"   Mean: {latents.mean():.4f}")
    print(f"   Std:  {latents.std():.4f}")
    print(f"   Min:  {latents.min():.4f}")
    print(f"   Max:  {latents.max():.4f}")
    
    # Check for anomalies
    if abs(latents.mean()) < 0.01 and abs(latents.std() - 1.0) < 0.1:
        print("   ⚠️  Latents appear pre-normalized to N(0,1)")
        print("   ⚠️  This means stats.pt should also be (0,1)")
    
    # Visualize distribution
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Overall histogram
    axes[0,0].hist(latents.flatten(), bins=100, density=True, alpha=0.7)
    axes[0,0].set_title('Overall Latent Distribution')
    axes[0,0].axvline(0, color='r', linestyle='--', label='zero')
    
    # Per-channel histograms
    colors = ['r', 'g', 'b', 'orange']
    for c in range(4):
        axes[0,1].hist(latents[:,c].flatten(), bins=50, density=True, alpha=0.5, label=f'Ch{c}', color=colors[c])
    axes[0,1].set_title('Per-Channel Distribution')
    axes[0,1].legend()
    
    # Spatial variance (where is the action?)
    spatial_var = latents.var(axis=0).mean(axis=0)  # (8, 8)
    im = axes[1,0].imshow(spatial_var, cmap='hot')
    axes[1,0].set_title('Spatial Variance (where info is)')
    plt.colorbar(im, ax=axes[1,0])
    
    # Sample latents
    axes[1,1].imshow(latents[0, 0], cmap='viridis')
    axes[1,1].set_title('Sample Latent (Ch 0)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_latent_dataset.png"), dpi=150)
    plt.close()

def diagnose_diffusion_denoising(model, vae, mean, std, args, save_dir):
    """Test 3: Check what diffusion model outputs at various timesteps"""
    print("\n" + "="*60)
    print("TEST 3: DIFFUSION DENOISING BEHAVIOR")
    print("="*60)
    
    if model is None:
        print("❌ No diffusion model loaded")
        return
    
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Create a simple test input
    batch_size = 4
    history = torch.randn(batch_size, 16, 8, 8).to(DEVICE) * 0.1  # Low noise history
    target = torch.randn(batch_size, 4, 8, 8).to(DEVICE)  # Random target
    
    # Test at different timesteps
    timesteps_to_test = [50, 200, 500, 800, 950]
    
    fig, axes = plt.subplots(len(timesteps_to_test), 6, figsize=(15, 3*len(timesteps_to_test)))
    
    for row, t_val in enumerate(timesteps_to_test):
        timesteps = torch.tensor([t_val] * batch_size, device=DEVICE)
        noise = torch.randn_like(target)
        noisy = scheduler.add_noise(target, noise, timesteps)
        
        # Predict with different actions
        predictions = []
        for action_idx in range(4):
            action = torch.zeros(batch_size, 4, device=DEVICE)
            action[:, action_idx] = 1.0
            
            with torch.no_grad():
                inp = torch.cat([noisy, history], dim=1)
                pred = model(inp, timesteps, action)
            predictions.append(pred)
        
        # Also predict with zero action
        with torch.no_grad():
            inp = torch.cat([noisy, history], dim=1)
            pred_uncond = model(inp, timesteps, torch.zeros(batch_size, 4, device=DEVICE))
        
        # Visualize
        axes[row, 0].imshow(noisy[0, 0].cpu().numpy(), cmap='viridis')
        axes[row, 0].set_title(f't={t_val}\nNoisy Input')
        axes[row, 0].axis('off')
        
        for i, (pred, name) in enumerate(zip(predictions, ['UP', 'DOWN', 'LEFT', 'RIGHT'])):
            axes[row, i+1].imshow(pred[0, 0].cpu().numpy(), cmap='viridis')
            diff_from_uncond = F.mse_loss(pred, pred_uncond).item()
            axes[row, i+1].set_title(f'{name}\ndiff={diff_from_uncond:.5f}')
            axes[row, i+1].axis('off')
        
        axes[row, 5].imshow(pred_uncond[0, 0].cpu().numpy(), cmap='viridis')
        axes[row, 5].set_title('NO ACTION')
        axes[row, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_diffusion_denoising.png"), dpi=150)
    plt.close()
    
    # Quantitative analysis
    print("\n📊 Action Sensitivity Analysis:")
    for t_val in [100, 500, 900]:
        timesteps = torch.tensor([t_val] * batch_size, device=DEVICE)
        noise = torch.randn_like(target)
        noisy = scheduler.add_noise(target, noise, timesteps)
        
        with torch.no_grad():
            inp = torch.cat([noisy, history], dim=1)
            pred_uncond = model(inp, timesteps, torch.zeros(batch_size, 4, device=DEVICE))
            
            diffs = []
            for action_idx in range(4):
                action = torch.zeros(batch_size, 4, device=DEVICE)
                action[:, action_idx] = 1.0
                pred = model(inp, timesteps, action)
                diffs.append(F.mse_loss(pred, pred_uncond).item())
        
        avg_diff = np.mean(diffs)
        print(f"   t={t_val}: avg_diff={avg_diff:.6f} (per action: {[f'{d:.6f}' for d in diffs]})")
        
        if avg_diff < 0.001:
            print(f"   ⚠️  Model ignores actions at t={t_val}!")

def diagnose_action_gradients(model, args, save_dir):
    """Test 4: Check if action embeddings receive gradients"""
    print("\n" + "="*60)
    print("TEST 4: ACTION GRADIENT FLOW")
    print("="*60)
    
    if model is None:
        print("❌ No diffusion model loaded")
        return
    
    model.train()  # Enable gradients
    
    # Create input
    batch_size = 4
    history = torch.randn(batch_size, 16, 8, 8, device=DEVICE, requires_grad=True)
    noisy = torch.randn(batch_size, 4, 8, 8, device=DEVICE, requires_grad=True)
    action = torch.zeros(batch_size, 4, device=DEVICE, requires_grad=True)
    action.data[:, 0] = 1.0
    timesteps = torch.tensor([500] * batch_size, device=DEVICE)
    
    # Forward
    inp = torch.cat([noisy, history], dim=1)
    pred = model(inp, timesteps, action)
    
    # Backward
    loss = pred.sum()
    loss.backward()
    
    # Check gradients
    print("\n📊 Gradient Magnitudes:")
    
    action_grad = action.grad
    if action_grad is not None:
        print(f"   Action input grad: {action_grad.abs().mean().item():.6f}")
    else:
        print("   ⚠️  Action has NO gradient!")
    
    # Check action embedding layers
    for name, param in model.named_parameters():
        if 'action' in name and param.grad is not None:
            grad_mag = param.grad.abs().mean().item()
            print(f"   {name}: {grad_mag:.6f}")
            if grad_mag < 1e-7:
                print(f"      ⚠️  Very small gradient!")
    
    model.eval()

def diagnose_full_generation(model, vae, mean, std, args, save_dir):
    """Test 5: Full generation pipeline"""
    print("\n" + "="*60)
    print("TEST 5: FULL GENERATION TEST")
    print("="*60)
    
    if model is None:
        print("❌ No diffusion model loaded")
        return
    
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(20)
    
    # Initialize with zeros (not random!)
    history = torch.zeros(1, 16, 8, 8, device=DEVICE)
    
    all_frames = []
    action_sequence = [0]*4 + [3]*4 + [1]*4 + [2]*4  # Up, Right, Down, Left
    
    print("\n📊 Generating frames with different actions...")
    
    for frame_idx, action_idx in enumerate(action_sequence):
        action = torch.zeros(1, 4, device=DEVICE)
        action[0, action_idx] = 1.0
        
        latents = torch.randn(1, 4, 8, 8, device=DEVICE)
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                inp = torch.cat([latents, history], dim=1)
                noise_pred = model(inp, t.unsqueeze(0).to(DEVICE), action)
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        denorm = latents * std.to(DEVICE) + mean.to(DEVICE)
        with torch.no_grad():
            pixels = vae.decode(denorm)
        
        img = pixels[0].cpu().permute(1,2,0).numpy() * 0.5 + 0.5
        all_frames.append(img.clip(0, 1))
        
        # Update history
        history = torch.cat([history[:, 4:], latents], dim=1)
        
        if frame_idx < 4:
            print(f"   Frame {frame_idx}: action={['UP','DOWN','LEFT','RIGHT'][action_idx]}, latent_range=[{latents.min():.2f}, {latents.max():.2f}]")
    
    # Visualize
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, img in enumerate(all_frames):
        row = i // 8
        col = i % 8
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        action_name = ['U','D','L','R'][action_sequence[i]]
        axes[row, col].set_title(f'{i}:{action_name}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "05_full_generation.png"), dpi=150)
    plt.close()
    
    # Check if frames are all the same
    frame_diffs = []
    for i in range(1, len(all_frames)):
        diff = np.abs(all_frames[i] - all_frames[0]).mean()
        frame_diffs.append(diff)
    
    avg_diff = np.mean(frame_diffs)
    print(f"\n📊 Frame Diversity:")
    print(f"   Avg diff from frame 0: {avg_diff:.4f}")
    
    if avg_diff < 0.01:
        print("   ⚠️  All frames look nearly identical!")
        print("   ⚠️  Model is not generating diverse outputs")
    elif avg_diff < 0.05:
        print("   ⚠️  Low diversity - model may not respond to actions well")
    else:
        print("   ✅ Reasonable frame diversity")

def diagnose_dataset_samples(args, save_dir):
    """Test 6: Visualize actual training samples"""
    print("\n" + "="*60)
    print("TEST 6: TRAINING DATA SAMPLES")
    print("="*60)
    
    metadata_path = args.metadata
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found: {metadata_path}")
        return
    
    df = pd.read_csv(metadata_path)
    print(f"   Total records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Action distribution
    if 'action' in df.columns:
        action_counts = df['action'].value_counts().sort_index()
        print(f"\n📊 Action Distribution:")
        for action, count in action_counts.items():
            pct = count / len(df) * 100
            action_name = ['UP', 'DOWN', 'LEFT', 'RIGHT'][int(action)] if action < 4 else 'UNKNOWN'
            print(f"   {action_name}: {count} ({pct:.1f}%)")
    
    # Episode lengths
    if 'episode_id' in df.columns:
        ep_lengths = df.groupby('episode_id').size()
        print(f"\n📊 Episode Statistics:")
        print(f"   Num episodes: {len(ep_lengths)}")
        print(f"   Avg length: {ep_lengths.mean():.1f}")
        print(f"   Min length: {ep_lengths.min()}")
        print(f"   Max length: {ep_lengths.max()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/diffusion_v6/best_model")
    parser.add_argument("--vae_path", type=str, default="output/vae_kl/klvae_latest.pt")
    parser.add_argument("--data_dir", type=str, default="output/vae_kl/latents_8x8")
    parser.add_argument("--metadata", type=str, default="data_v3/metadata.csv")
    parser.add_argument("--output_dir", type=str, default="diagnostic_output")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n🔍 SNAKE WORLD MODEL DIAGNOSTIC TOOL")
    print(f"   Saving results to: {args.output_dir}")
    
    # Load models
    vae, model, mean, std = load_models(args)
    
    # Run all diagnostics
    if vae is not None:
        diagnose_vae(vae, args, args.output_dir)
    
    diagnose_latent_dataset(args, args.output_dir)
    diagnose_dataset_samples(args, args.output_dir)
    
    if model is not None:
        diagnose_diffusion_denoising(model, vae, mean, std, args, args.output_dir)
        diagnose_action_gradients(model, args, args.output_dir)
        diagnose_full_generation(model, vae, mean, std, args, args.output_dir)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print(f"Check {args.output_dir}/ for visualizations")
    print("="*60)

if __name__ == "__main__":
    main()