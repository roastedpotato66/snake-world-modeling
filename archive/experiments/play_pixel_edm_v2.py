"""
Snake World Model - Play (DIAMOND-style Pixel-Space EDM)
Updated to load REAL data for initialization to fix "Red Screen" OOD issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
import time
import glob
import random
from PIL import Image
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64
CONTEXT_FRAMES = 4
ACTION_DIM = 4

# ============================================================================
# MODEL DEFINITIONS (must match training exactly)
# ============================================================================

class EDMPrecond:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data
    
    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        return c_skip, c_out, c_in

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
    def sample(self, model, shape, context, action, n_steps=3, device='cuda', cfg_scale=1.5):
        sigmas = self.get_sigmas(n_steps).to(device)
        x = torch.randn(shape, device=device) * sigmas[0]
        
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # CFG
            if cfg_scale > 1.0:
                denoised_cond = model(x, sigma.expand(x.shape[0]), context, action)
                denoised_uncond = model(x, sigma.expand(x.shape[0]), context, torch.zeros_like(action))
                denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
            else:
                denoised = model(x, sigma.expand(x.shape[0]), context, action)
            
            d = (x - denoised) / sigma
            x = x + d * (sigma_next - sigma)
        
        return x

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, cond_dim=512):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)
    
    def forward(self, x, cond):
        x = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=512, num_groups=32):
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
# REAL DATA LOADING
# ============================================================================

def get_random_real_context(data_dir):
    """
    Loads 4 consecutive frames from a random valid episode in the data directory.
    """
    print(f"Searching for episode data in {data_dir}...")
    
    # Try to find images
    all_files = glob.glob(os.path.join(data_dir, "*.png"))
    if not all_files:
        raise ValueError(f"No .png files found in {data_dir}. Please check path.")
        
    # Group by episode ID (assumed filename format: {worker}_{uuid}_{frame}.png)
    # Actually, simpler approach: Just pick a random file, parse its episode ID, 
    # and try to find the next 4 frames.
    
    max_attempts = 100
    for _ in range(max_attempts):
        start_file = random.choice(all_files)
        basename = os.path.basename(start_file)
        
        # Example: 0_abc123_00000.png
        try:
            parts = basename.rsplit('_', 1) # ["0_abc123", "00000.png"]
            ep_id = parts[0]
            frame_num = int(parts[1].split('.')[0])
        except:
            continue
            
        # Construct expected filenames for context
        # We need frames i, i+1, i+2, i+3 to predict i+4
        # Or just i, i+1, i+2, i+3 as context
        
        context_files = []
        valid_sequence = True
        
        for i in range(CONTEXT_FRAMES):
            fname = f"{ep_id}_{frame_num + i:05d}.png"
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                valid_sequence = False
                break
            context_files.append(fpath)
            
        if valid_sequence:
            print(f"✅ Found valid seed sequence starting at: {basename}")
            
            # Load and process
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            tensors = []
            for f in context_files:
                img = Image.open(f).convert('RGB')
                tensors.append(transform(img))
                
            # Stack along channel dim: (12, 64, 64)
            context = torch.cat(tensors, dim=0)
            return context.unsqueeze(0).to(DEVICE) # (1, 12, 64, 64)
            
    raise RuntimeError("Could not find a valid consecutive sequence of 4 frames after 100 attempts.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/pixel_edm_v2/best_model/model.pt")
    # Point this to your REAL training images folder
    parser.add_argument("--data_dir", type=str, default="data_v5/images") 
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    
    print("Loading model...")
    model = PixelSpaceUNet(
        in_channels=3 + 3 * CONTEXT_FRAMES,
        out_channels=3,
        base_dim=128,
        cond_dim=512
    ).to(DEVICE)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print(f"✅ Model loaded from {args.model_path}")
    else:
        print(f"❌ Model not found: {args.model_path}")
        return
    model.eval()
    
    sampler = EDMSampler()
    
    # Load Real Context
    try:
        context = get_random_real_context(args.data_dir)
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Fallback: Creating blank context (Expect Red Screen / OOD behavior)")
        context = torch.zeros(1, 12, IMG_SIZE, IMG_SIZE).to(DEVICE)

    current_action = 0
    
    print("\n" + "="*40)
    print("🐍 SNAKE WORLD MODEL (DIAMOND-style)")
    print("="*40)
    print("WASD or Arrow Keys to move")
    print("R to reset (Loads new random seed from data), ESC to quit")
    print(f"CFG Scale: {args.cfg_scale}")
    print("="*40)
    
    cv2.namedWindow("Snake", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Snake", 512, 512)
    
    frame_time = 1.0 / args.fps
    
    while True:
        start = time.time()
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
            
        if key == ord('r'):
            print("Resetting with new seed...")
            try:
                context = get_random_real_context(args.data_dir)
                print("Reset Complete!")
            except:
                print("Reset Failed.")
        
        if key in [ord('w'), 82]: current_action = 0
        elif key in [ord('s'), 84]: current_action = 1
        elif key in [ord('a'), 81]: current_action = 2
        elif key in [ord('d'), 83]: current_action = 3
        
        action = torch.zeros(1, ACTION_DIM, device=DEVICE)
        action[0, current_action] = 1.0
        
        with torch.no_grad():
            generated = sampler.sample(
                model,
                (1, 3, IMG_SIZE, IMG_SIZE),
                context,
                action,
                n_steps=args.steps,
                device=DEVICE,
                cfg_scale=args.cfg_scale
            )
        
        # Display
        img = generated[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        cv2.putText(img, action_names[current_action], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Snake", img)
        
        # Shift context: Drop oldest frame (first 3 chans), add new frame (generated)
        context = torch.cat([context[:, 3:], generated], dim=1)
        
        elapsed = time.time() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()