# Test if VAE is actually working
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import argparse
from PIL import Image
import numpy as np

# Import VAE classes from train_vae_v2
from train_vae_v2 import KLVAE, DiagonalGaussianDistribution, SimpleImageDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_frames_with_snake(data_dir, num_frames=8):
    """Load frames that have visible snake (non-black pixels)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = SimpleImageDataset(data_dir, transform=transform)
    
    # Load images and filter for ones with visible content (not all black)
    frames = []
    for i in range(min(len(dataset), num_frames * 10)):  # Check more than needed
        img, _ = dataset[i]
        # Check if image has non-black pixels (max value > -0.9 in normalized space)
        if img.max() > -0.9:
            frames.append(img)
            if len(frames) >= num_frames:
                break
    
    if len(frames) < num_frames:
        # If we didn't find enough, just take what we have
        print(f"Warning: Only found {len(frames)} frames with visible snake")
    
    # Stack into batch
    frames_tensor = torch.stack(frames).to(DEVICE)
    return frames_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/vae_kl/klvae_latest.pt")
    parser.add_argument("--data_dir", type=str, default="data_v3/images")
    parser.add_argument("--output", type=str, default="vae_test.png")
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()
    
    print(f"Loading VAE model from {args.model_path}...")
    vae = KLVAE(latent_channels=4).to(DEVICE)
    
    if os.path.exists(args.model_path):
        vae.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file not found at {args.model_path}")
        print("Testing with untrained model...")
    
    vae.eval()
    
    print(f"Loading frames from {args.data_dir}...")
    frames = load_frames_with_snake(args.data_dir, num_frames=args.num_frames)
    print(f"Loaded {frames.shape[0]} frames")
    
    with torch.no_grad():
        # Encode and decode
        latents = vae.encode(frames)
        reconstructed = vae.decoder(vae.decoder_input(latents))
        
        # Visual check - concatenate original and reconstructed
        comparison = torch.cat([frames, reconstructed], dim=0)
        
        # Denormalize from [-1, 1] to [0, 1] for visualization
        comparison = comparison * 0.5 + 0.5
        
        save_image(comparison, args.output, nrow=args.num_frames)
        print(f"Saved comparison image to {args.output}")
        print(f"Top row: Original frames")
        print(f"Bottom row: Reconstructed frames")

if __name__ == "__main__":
    main()

