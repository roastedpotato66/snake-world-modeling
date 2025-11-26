import torch
import torch.nn as nn
import cv2
import numpy as np
from diffusers import UNet2DModel, DDIMScheduler, DDPMScheduler
from collections import deque
import time
import os
import sys
import glob

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 64
LATENT_SIZE = 16
CONTEXT_FRAMES = 4
FPS = 10 

# --- LOAD HELPERS ---
sys.path.append('.')
try:
    from train_vae import VQVAE, VectorQuantizerEMA
except ImportError:
    print("CRITICAL: train_vae.py not found. Please ensure it is in the same folder.")
    sys.exit(1)

def get_keyboard_action():
    import keyboard 
    if keyboard.is_pressed('up') or keyboard.is_pressed('w'): return 0
    if keyboard.is_pressed('down') or keyboard.is_pressed('s'): return 1
    if keyboard.is_pressed('left') or keyboard.is_pressed('a'): return 2
    if keyboard.is_pressed('right') or keyboard.is_pressed('d'): return 3
    return None

def load_real_start_state():
    """Loads a real game state from the training data to seed the memory."""
    files = glob.glob("output/vae/latents/*.npy")
    if not files:
        raise FileNotFoundError("No latent files found in output/vae/latents/. Train VAE first!")
    
    # Pick a random file, but preferably one from the start of an episode
    # Since we can't easily tell, just picking one is usually fine.
    # Ideally, pick a file you know is a 'start' state or just a clean middle state.
    target_file = files[0] 
    print(f"Seeding memory with: {target_file}")
    
    data = np.load(target_file) # Shape [64, 16, 16]
    tensor = torch.from_numpy(data).unsqueeze(0).to(DEVICE) # [1, 64, 16, 16]
    return tensor

def main():
    print(f"Loading Neural Engine on {DEVICE}...")
    
    # 1. Load VAE
    vae = VQVAE(embedding_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load("output/vae/vqvae.pt", map_location=DEVICE))
    vae.eval()
    
    # 2. Load UNet
    model_path = "output/diffusion_v2/latest_model"
    if not os.path.exists(model_path):
        checkpoints = sorted([d for d in os.listdir("output/diffusion_v2") if "checkpoint" in d])
        if checkpoints:
            model_path = os.path.join("output/diffusion_v2", checkpoints[-1], "unet")
    
    print(f"Loading UNet from {model_path}...")
    unet = UNet2DModel.from_pretrained(model_path).to(DEVICE)
    unet.eval()
    
    # 3. Scheduler
    # We switch to DDPM for stability first (slower but more accurate).
    # If this works, we can try DDIM again later.
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50) # Use 50 steps for quality check

    print("--- NEURAL SNAKE READY ---")
    
    # --- GAME STATE INITIALIZATION ---
    history_buffer = deque(maxlen=CONTEXT_FRAMES)
    
    # CRITICAL FIX: Load REAL data
    start_latent = load_real_start_state()
    
    # Fill history with copies of valid state
    for _ in range(CONTEXT_FRAMES):
        history_buffer.append(start_latent)
        
    last_action = 0 
    
    while True:
        loop_start = time.time()
        
        # 1. Input
        action = get_keyboard_action()
        if action is None: action = last_action
        
        # Reset
        import keyboard
        if keyboard.is_pressed('r'):
             start_latent = load_real_start_state() # Reseed with real data
             history_buffer.clear()
             for _ in range(CONTEXT_FRAMES): history_buffer.append(start_latent)
             print("Reset!")
        if keyboard.is_pressed('q'): break

        last_action = action
        action_tensor = torch.tensor([action], device=DEVICE)
        
        # 2. Prepare Context
        context = torch.cat(list(history_buffer), dim=1)
        
        # 3. Diffusion Generation
        latents = torch.randn(1, LATENT_DIM, LATENT_SIZE, LATENT_SIZE).to(DEVICE)
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                model_input = torch.cat([latents, context], dim=1)
                noise_pred = unet(model_input, t, class_labels=action_tensor).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 4. Quantization Snap
        # Force the output to snap to the nearest valid VAE token
        _, latents_quantized, _ = vae.vq_layer(latents)
        
        # 5. Update History
        history_buffer.append(latents_quantized)
        
        # 6. Render
        with torch.no_grad():
            pixels = vae.decoder(latents_quantized)
            
        img = (pixels[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("Neural Snake", img)
        cv2.waitKey(1)
        
        dt = time.time() - loop_start
        if dt < 1.0 / FPS: time.sleep((1.0 / FPS) - dt)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()