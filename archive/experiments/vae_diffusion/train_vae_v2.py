import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self):
        return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])

class KLVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super(KLVAE, self).__init__()
        
        # --- Encoder (64x64 -> 8x8) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),      # 32
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),     # 16
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),    # 8
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.to_moments = nn.Conv2d(128, 2 * latent_channels, 3, 1, 1)

        # --- Decoder (8x8 -> 64x64) ---
        self.decoder_input = nn.Conv2d(latent_channels, 128, 3, 1, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            # FIX 1: Tanh Activation to force range [-1, 1]
            nn.Tanh() 
        )

    def forward(self, x, sample_posterior=True):
        moments = self.to_moments(self.encoder(x))
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mean
        x_recon = self.decoder(self.decoder_input(z))
        return x_recon, posterior

    def encode(self, x):
        moments = self.to_moments(self.encoder(x))
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample()

class SimpleImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.transform = transform
            
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform: image = self.transform(image)
            return image, self.img_files[idx]
        except:
            return torch.zeros(3, 64, 64), "error"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_v3/images")
    parser.add_argument("--output_dir", type=str, default="output/vae_kl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    # FIX 2: Increased Max KL Weight slightly, but we anneal it
    parser.add_argument("--kl_weight", type=float, default=0.00025) 
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = SimpleImageDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                          num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    model = KLVAE(latent_channels=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # Slightly higher LR
    scaler = GradScaler(enabled=(DEVICE=="cuda"))
    
    print(f"--- Training KL-VAE (Weighted Loss) ---")
    
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(dataloader)
        
        # FIX 3: KL Annealing (0.0 -> args.kl_weight over 10 epochs)
        # This lets the model learn to draw the snake BEFORE we force the latent space to be neat
        current_kl_weight = args.kl_weight * min(1.0, epoch / 10.0)
        
        for images, _ in loop:
            images = images.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast(enabled=(DEVICE=="cuda")):
                reconstruction, posterior = model(images)
                
                # --- FIX 4: WEIGHTED LOSS ---
                # 1. Identify "Interesting" pixels. Background is black (-1, -1, -1).
                #    If any channel is > -0.8, it's part of the game (snake, food, walls).
                #    We check max across channels.
                max_val, _ = torch.max(images, dim=1, keepdim=True)
                # Create mask: 1.0 for snake, 0.0 for background
                mask = (max_val > -0.9).float() 
                
                # 2. Assign weights: 1.0 for background, 50.0 for Snake
                #    This forces the model to care 50x more about the white dots.
                weights = 1.0 + (mask * 50.0) 
                
                # 3. Calculate Weighted MSE
                recon_loss = F.mse_loss(reconstruction, images, reduction='none')
                recon_loss = (recon_loss * weights).mean()
                
                kl_loss = torch.mean(posterior.kl())
                loss = recon_loss + (current_kl_weight * kl_loss)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(recon=recon_loss.item(), kl=kl_loss.item(), w=current_kl_weight)

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_imgs = images[:16]
                with autocast(enabled=(DEVICE=="cuda")):
                    recon, _ = model(test_imgs, sample_posterior=False) 
                comparison = torch.cat([test_imgs, recon], dim=0)
                save_image(comparison * 0.5 + 0.5, f"{args.output_dir}/recon_ep{epoch+1}.png", nrow=8)
            torch.save(model.state_dict(), f"{args.output_dir}/klvae_latest.pt")
    
    # --- PHASE 2: Generate Latents ---
    print("--- Pre-processing Data for Diffusion Phase ---")
    latents_dir = os.path.join(args.output_dir, "latents_8x8") 
    os.makedirs(latents_dir, exist_ok=True)
    
    seq_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    model.eval()
    with torch.no_grad():
        for images, filenames in tqdm(seq_loader):
            images = images.to(DEVICE, non_blocking=True)
            
            with autocast(enabled=(DEVICE=="cuda")):
                latents = model.encode(images) 
            
            latents_np = latents.cpu().numpy()
            for i, fname in enumerate(filenames):
                name = os.path.splitext(fname)[0]
                np.save(os.path.join(latents_dir, f"{name}.npy"), latents_np[i])
                
    print(f"Latents saved to {latents_dir}")
    print("Ready for Diffusion Training.")

if __name__ == "__main__":
    main()