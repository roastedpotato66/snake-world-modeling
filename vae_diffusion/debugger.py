import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
import numpy as np
import os
import sys
import glob
from torchvision.utils import save_image
from collections import deque

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 64
LATENT_SIZE = 16
CONTEXT_FRAMES = 4

# Load Helpers
sys.path.append('.')
try:
    from train_vae import VQVAE, VectorQuantizerEMA
except ImportError:
    print("Error: train_vae.py not found.")
    sys.exit(1)

def load_real_sequence():
    """Loads a sequence of 5 frames (4 History + 1 Target)"""
    files = sorted(glob.glob("output/vae/latents/*.npy"))
    # Pick a sequence from the middle
    start_idx = 1000 
    frames = []
    print(f"Loading sequence starting at {files[start_idx]}...")
    for i in range(CONTEXT_FRAMES + 1):
        data = np.load(files[start_idx + i])
        frames.append(torch.from_numpy(data).unsqueeze(0).to(DEVICE))
    
    history = frames[:4]
    target = frames[4]
    
    # Fake action (just assuming moving right for test)
    action = torch.tensor([3]).to(DEVICE) 
    return history, target, action

def main():
    print(f"--- RUNNING DIAGNOSTICS ON {DEVICE} ---")
    
    # 1. Load Models
    vae = VQVAE(embedding_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load("output/vae/vqvae.pt", map_location=DEVICE))
    vae.eval()
    
    model_path = "output/diffusion_v2/latest_model"
    if not os.path.exists(model_path):
        checkpoints = sorted([d for d in os.listdir("output/diffusion_v2") if "checkpoint" in d])
        if checkpoints: model_path = os.path.join("output/diffusion_v2", checkpoints[-1], "unet")
            
    print(f"Loading UNet: {model_path}")
    unet = UNet2DModel.from_pretrained(model_path).to(DEVICE)
    unet.eval()
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Load Data
    history_frames, real_target, action = load_real_sequence()
    
    # --- TEST A: VAE INTEGRITY ---
    print("\n[TEST A] VAE Quantization Integrity")
    # Check if passing Real Target through Quantizer changes it significantly
    _, quantized_real, _ = vae.vq_layer(real_target)
    mse_vae = F.mse_loss(quantized_real, real_target).item()
    print(f"VAE Reconstruction Error (Latent Space): {mse_vae:.6f}")
    if mse_vae > 0.1:
        print(">> FAIL: VAE Quantizer is distorting data too much.")
    else:
        print(">> PASS: VAE is healthy.")

    # --- TEST B: OPEN LOOP PREDICTION (One Step) ---
    print("\n[TEST B] Open Loop Prediction (Perfect History)")
    
    # Setup Input
    context = torch.cat(history_frames, dim=1) # [1, 256, 16, 16]
    
    # Denoise Loop
    latents = torch.randn_like(real_target)
    scheduler.set_timesteps(50) # Use 50 steps
    
    with torch.no_grad():
        for t in scheduler.timesteps:
            model_input = torch.cat([latents, context], dim=1)
            noise_pred = unet(model_input, t, class_labels=action).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
    # Stats Check
    print(f"Real Target Stats -> Mean: {real_target.mean():.4f}, Std: {real_target.std():.4f}, Min: {real_target.min():.4f}, Max: {real_target.max():.4f}")
    print(f"Predicted Stats -> Mean: {latents.mean():.4f}, Std: {latents.std():.4f}, Min: {latents.min():.4f}, Max: {latents.max():.4f}")
    
    mse_pred = F.mse_loss(latents, real_target).item()
    print(f"Prediction MSE (vs Real): {mse_pred:.4f}")
    
    if latents.std() < 0.1:
        print(">> FAIL: Model collapse. Predicted output is flat/grey.")
    elif mse_pred > 1.0:
        print(">> FAIL: Prediction is wildly off.")
    else:
        print(">> PASS: Prediction stats look reasonable.")

    # --- TEST C: QUANTIZATION SNAP ---
    print("\n[TEST C] Quantization Snap Check")
    _, latents_snapped, _ = vae.vq_layer(latents)
    mse_snap = F.mse_loss(latents_snapped, latents).item()
    print(f"Quantization MSE (Pred vs Snapped): {mse_snap:.4f}")
    
    # --- VISUAL REPORT ---
    print("\nGenerating debug_report.png...")
    with torch.no_grad():
        # Decode images
        real_img = vae.decoder(real_target)
        pred_raw_img = vae.decoder(latents)
        pred_snap_img = vae.decoder(latents_snapped)
        
        # Visual comparison
        viz = torch.cat([real_img, pred_raw_img, pred_snap_img], dim=0)
        save_image(viz * 0.5 + 0.5, "debug_report.png", nrow=3)
        print("Saved. Order: [Real Target] | [Predicted Raw] | [Predicted Snapped]")

if __name__ == "__main__":
    main()