import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
from diffusers import UNet2DModel, DDIMScheduler
import os
import time

# --- CONFIG ---
LATENT_SIZE = 8
LATENT_DIM = 4
ACTION_DIM = 4
CONTEXT_FRAMES = 4
IMG_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- VAE (Required for Decoding) ---
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
    def sample(self): return self.mean + self.std * torch.randn_like(self.mean)

class KLVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super(KLVAE, self).__init__()
        # Encoder not needed for play
        self.encoder = nn.Identity() 
        self.to_moments = nn.Identity()

        # Decoder 
        self.decoder_input = nn.Conv2d(latent_channels, 128, 3, 1, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh()
        )

    def forward(self, x, sample_posterior=True): return self.decode(x)
    
    def decode(self, z):
        return self.decoder(self.decoder_input(z))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint folder (e.g. output/diffusion_v10/best_model)")
    parser.add_argument("--vae_path", type=str, default="output/vae_kl/klvae_latest.pt")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Higher = stricter adherence to keys")
    args = parser.parse_args()

    print("Loading VAE...")
    vae = KLVAE(latent_channels=4).to(DEVICE)
    try:
        vae.load_state_dict(torch.load(args.vae_path, map_location=DEVICE), strict=False)
    except:
        print("Warning: VAE load issue (keys missing?), trying strict=False")
    vae.eval()

    print("Loading Diffusion Model...")
    unet = UNet2DModel.from_pretrained(args.model_path).to(DEVICE)
    unet.eval()
    
    # Load Stats
    stats_path = os.path.join(args.model_path, "stats.pt")
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location=DEVICE)
        mean, std = stats["mean"], stats["std"]
        print(f"Loaded stats: Mean={mean:.4f}, Std={std:.4f}")
    else:
        print("Warning: stats.pt not found, using default.")
        mean, std = torch.tensor(0.0).to(DEVICE), torch.tensor(1.0).to(DEVICE)

    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(20) # 20 steps per frame for speed

    # Initialize State
    # Start with noise history
    history = torch.randn(1, LATENT_DIM * CONTEXT_FRAMES, LATENT_SIZE, LATENT_SIZE).to(DEVICE)
    current_action = 0 # UP
    
    print("\n--- CONTROLS ---")
    print("Arrow Keys: Move")
    print("R: Reset")
    print("ESC: Quit")
    
    cv2.namedWindow("Snake World Model", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Snake World Model", 512, 512)

    clock = time.time()
    
    while True:
        # 1. Handle Input
        key = cv2.waitKey(1)
        if key == 27: break # ESC
        if key == ord('r'): 
            history = torch.randn(1, LATENT_DIM * CONTEXT_FRAMES, LATENT_SIZE, LATENT_SIZE).to(DEVICE)
            print("Reset!")
        
        # Map keys to actions (Up=0, Down=1, Left=2, Right=3)
        # Note: Check your dataset mapping. Usually: Up=0, Down=1, Left=2, Right=3
        if key == 0: current_action = 0  # Up (Platform dependent, check arrow codes)
        elif key == 1: current_action = 1 # Down
        # Simple OpenCV arrow key mapping
        if key == 2490368: current_action = 0 # Up
        if key == 2621440: current_action = 1 # Down
        if key == 2424832: current_action = 2 # Left
        if key == 2555904: current_action = 3 # Right
        
        # 2. Prepare Action Map
        action_map = torch.zeros((1, ACTION_DIM, LATENT_SIZE, LATENT_SIZE), device=DEVICE)
        action_map[0, current_action, :, :] = 1.0
        
        # 3. Diffusion Generation (Next Frame)
        # Start from random noise
        latents = torch.randn((1, LATENT_DIM, LATENT_SIZE, LATENT_SIZE), device=DEVICE)
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                # CFG: Predict noise with AND without action
                # Model Input: [Noisy_Target, History, Action]
                
                # Conditional
                cond_input = torch.cat([latents, history, action_map], dim=1)
                noise_cond = unet(cond_input, t).sample
                
                # Unconditional (Null Action)
                uncond_input = torch.cat([latents, history, torch.zeros_like(action_map)], dim=1)
                noise_uncond = unet(uncond_input, t).sample
                
                # Combine
                noise_pred = noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)
                
                # Step
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 4. Decode & Display
        # Denormalize
        denorm_latents = (latents * std) + mean
        
        # VAE Decode
        with torch.no_grad():
            pixels = vae.decode(denorm_latents)
        
        # Convert to numpy image
        img = pixels[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1) # [-1, 1] -> [0, 1]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Upscale for visibility
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Snake World Model", img)
        
        # 5. Update History
        # Slide window: Remove oldest 4 channels, add new latents
        history = torch.cat([history[:, LATENT_DIM:], latents], dim=1)
        
        # FPS Control
        dt = time.time() - clock
        if dt < 1.0 / args.fps:
            time.sleep((1.0 / args.fps) - dt)
        clock = time.time()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()