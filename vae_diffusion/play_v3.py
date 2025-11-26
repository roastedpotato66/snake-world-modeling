"""
Snake World Model - Interactive Play V6 (Fixed)

Key Fixes:
1. Loads custom ActionConditionedUNet (not diffusers UNet2D)
2. Proper history initialization from real encoded frames
3. Correct stats loading and denormalization
4. Better keyboard handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from diffusers import DDIMScheduler
import os
import time
from PIL import Image
from torchvision import transforms

# --- CONFIG ---
LATENT_SIZE = 8
LATENT_DIM = 4
ACTION_DIM = 4
CONTEXT_FRAMES = 4
IMG_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- VAE ---
class KLVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        # Encoder (needed for initialization)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.SiLU(),
        )
        self.to_moments = nn.Conv2d(128, 2 * latent_channels, 3, 1, 1)
        
        # Decoder
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
        return mean + std * torch.randn_like(mean)
    
    def decode(self, z):
        return self.decoder(self.decoder_input(z))

# --- CUSTOM UNET (must match training!) ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
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
        t_scale, t_shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        h = h * (1 + t_scale[:, :, None, None]) + t_shift[:, :, None, None]
        a_scale, a_shift = self.action_mlp(a_emb).chunk(2, dim=-1)
        h = h * (1 + a_scale[:, :, None, None]) + a_shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)

class ActionConditionedUNet(nn.Module):
    def __init__(self, in_channels, out_channels=4, base_dim=128, time_dim=256, action_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.action_dim = action_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim // 2),
            nn.Linear(time_dim // 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.action_embed = nn.Sequential(
            nn.Linear(ACTION_DIM, action_dim),
            nn.SiLU(),
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
        self.mid_attn = nn.MultiheadAttention(base_dim*4, num_heads=4, batch_first=True)
        self.mid2 = ResBlock(base_dim*4, base_dim*4, time_dim, action_dim)
        
        self.up1 = nn.ConvTranspose2d(base_dim*4, base_dim*4, 4, 2, 1)
        self.dec1 = ResBlock(base_dim*4 + base_dim*4, base_dim*2, time_dim, action_dim)
        self.dec2 = ResBlock(base_dim*2, base_dim*2, time_dim, action_dim)
        
        self.up2 = nn.ConvTranspose2d(base_dim*2, base_dim*2, 4, 2, 1)
        self.dec3 = ResBlock(base_dim*2 + base_dim*2, base_dim, time_dim, action_dim)
        self.dec4 = ResBlock(base_dim, base_dim, time_dim, action_dim)
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, 1, 1)
        )
    
    def forward(self, x, timesteps, action_onehot):
        t_emb = self.time_mlp(timesteps.float())
        a_emb = self.action_embed(action_onehot)
        
        h = self.conv_in(x)
        h = self.down1(h, t_emb, a_emb)
        h = self.down2(h, t_emb, a_emb)
        skip1 = h
        h = self.pool1(h)
        
        h = self.down3(h, t_emb, a_emb)
        h = self.down4(h, t_emb, a_emb)
        skip2 = h
        h = self.pool2(h)
        
        h = self.mid1(h, t_emb, a_emb)
        B, C, H, W = h.shape
        h_flat = h.flatten(2).permute(0, 2, 1)
        h_flat, _ = self.mid_attn(h_flat, h_flat, h_flat)
        h = h_flat.permute(0, 2, 1).view(B, C, H, W)
        h = self.mid2(h, t_emb, a_emb)
        
        h = self.up1(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.dec1(h, t_emb, a_emb)
        h = self.dec2(h, t_emb, a_emb)
        
        h = self.up2(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec3(h, t_emb, a_emb)
        h = self.dec4(h, t_emb, a_emb)
        
        return self.conv_out(h)

def create_initial_frame():
    """Create a synthetic starting frame (snake in center)"""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    cell = IMG_SIZE // 16
    
    # Draw snake body (white)
    for i, (x, y) in enumerate([(8, 8), (8, 9), (8, 10)]):
        color = (255, 0, 0) if i == 0 else (255, 255, 255)  # Blue head, white body
        cv2.rectangle(img, (x*cell+1, y*cell+1), ((x+1)*cell-2, (y+1)*cell-2), color, -1)
    
    # Draw food (red)
    fx, fy = 12, 4
    cv2.rectangle(img, (fx*cell+1, fy*cell+1), ((fx+1)*cell-2, (fy+1)*cell-2), (0, 0, 255), -1)
    
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="output/vae_kl/klvae_latest.pt")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=15, help="DDIM steps per frame")
    args = parser.parse_args()

    print("Loading VAE...")
    vae = KLVAE(latent_channels=4).to(DEVICE)
    vae.load_state_dict(torch.load(args.vae_path, map_location=DEVICE), strict=False)
    vae.eval()

    print("Loading Diffusion Model...")
    in_channels = LATENT_DIM + (LATENT_DIM * CONTEXT_FRAMES)
    model = ActionConditionedUNet(in_channels=in_channels, out_channels=LATENT_DIM).to(DEVICE)
    
    model_file = os.path.join(args.model_path, "model.pt")
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        print("✅ Model loaded from model.pt")
    else:
        print(f"❌ Model not found at {model_file}")
        return
    model.eval()
    
    # Load stats
    stats_path = os.path.join(args.model_path, "stats.pt")
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location=DEVICE)
        mean = stats["mean"].float().to(DEVICE)
        std = stats["std"].float().to(DEVICE)
        print(f"✅ Stats: Mean={mean.item():.4f}, Std={std.item():.4f}")
    else:
        print("⚠️  stats.pt not found, using defaults")
        mean, std = torch.tensor(0.0).to(DEVICE), torch.tensor(1.0).to(DEVICE)

    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(args.steps)

    # Initialize with real encoded frames
    print("Initializing game state...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    init_frame = create_initial_frame()
    init_tensor = transform(Image.fromarray(cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB)))
    init_tensor = init_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        init_latent = vae.encode(init_tensor)
        # Normalize latent
        init_latent = (init_latent - mean) / (std + 1e-6)
    
    # Stack for history
    history = init_latent.repeat(1, CONTEXT_FRAMES, 1, 1)
    
    current_action = 0  # Start moving up
    
    print("\n" + "="*40)
    print("🐍 SNAKE WORLD MODEL")
    print("="*40)
    print("Controls:")
    print("  ↑ W : Up")
    print("  ↓ S : Down")
    print("  ← A : Left")
    print("  → D : Right")
    print("  R   : Reset")
    print("  ESC : Quit")
    print("="*40)
    
    cv2.namedWindow("Snake World Model", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Snake World Model", 512, 512)

    frame_time = 1.0 / args.fps
    
    while True:
        start_time = time.time()
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            # Reset
            with torch.no_grad():
                init_latent = vae.encode(init_tensor)
                init_latent = (init_latent - mean) / (std + 1e-6)
            history = init_latent.repeat(1, CONTEXT_FRAMES, 1, 1)
            current_action = 0
            print("🔄 Reset!")
        
        # WASD and arrow keys
        if key == ord('w') or key == 82:  # W or Up
            current_action = 0
        elif key == ord('s') or key == 84:  # S or Down
            current_action = 1
        elif key == ord('a') or key == 81:  # A or Left
            current_action = 2
        elif key == ord('d') or key == 83:  # D or Right
            current_action = 3
        
        # Prepare action
        action = torch.zeros((1, ACTION_DIM), device=DEVICE)
        action[0, current_action] = 1.0
        
        # Generate next frame
        latents = torch.randn((1, LATENT_DIM, LATENT_SIZE, LATENT_SIZE), device=DEVICE)
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                model_input = torch.cat([latents, history], dim=1)
                t_batch = t.unsqueeze(0).to(DEVICE)
                
                # CFG
                noise_cond = model(model_input, t_batch, action)
                noise_uncond = model(model_input, t_batch, torch.zeros_like(action))
                noise_pred = noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)
                
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        denorm_latents = (latents * (std + 1e-6)) + mean
        
        with torch.no_grad():
            pixels = vae.decode(denorm_latents)
        
        # Display
        img = pixels[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Add action indicator
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        cv2.putText(img, f"Action: {action_names[current_action]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Snake World Model", img)
        
        # Update history
        history = torch.cat([history[:, LATENT_DIM:], latents], dim=1)
        
        # FPS control
        elapsed = time.time() - start_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()