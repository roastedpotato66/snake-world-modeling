# train_vae.py
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

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        
        # EMA buffers (not trainable parameters)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        
    def forward(self, inputs):
        # inputs: [B, Dim, H, W] -> [B, H, W, Dim]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Use EMA to update the embedding vectors
        if self.training:
            # 1. Update cluster size (how many times was this code used?)
            dw = torch.matmul(encodings.t(), flat_input)
            sync_cluster_size = torch.sum(encodings, dim=0)
            
            self.ema_cluster_size.data.mul_(self.decay).add_(sync_cluster_size, alpha=1 - self.decay)
            
            # Laplace smoothing to prevent division by zero
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size.data = (
                (self.ema_cluster_size.data + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            
            # 2. Update weights
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # 3. Normalize and assign
            self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices

class VQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 3, 1, 1), nn.ReLU()
        )
        self.vq_layer = VectorQuantizerEMA(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1), 
        )

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, _ = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon

    def encode(self, x):
        z = self.encoder(x)
        loss, quantized, _ = self.vq_layer(z)
        return quantized

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
            return image, self.img_files[idx] # Return filename to track ID
        except:
            return torch.zeros(3, 64, 64), "error"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_v2/images")
    parser.add_argument("--output_dir", type=str, default="output/vae")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = SimpleImageDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("--- Training VQ-VAE (EMA) ---")
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(dataloader)
        total_recon_loss = 0
        
        for images, _ in loop:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            
            vq_loss, reconstruction = model(images)
            recon_loss = F.mse_loss(reconstruction, images)
            
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            loop.set_postfix(recon=recon_loss.item())

        # Visual check
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                model.eval()
                test_imgs = images[:8]
                _, recon = model(test_imgs)
                comparison = torch.cat([test_imgs, recon], dim=0)
                save_image(comparison * 0.5 + 0.5, f"{args.output_dir}/recon_ep{epoch}.png", nrow=8)

    torch.save(model.state_dict(), f"{args.output_dir}/vqvae.pt")
    
    # --- PRE-PROCESS FOR PHASE 2 ---
    print("--- Pre-processing Data for Phase 2 ---")
    # We need ordered data now, not shuffled
    seq_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    latents_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for images, filenames in tqdm(seq_loader):
            images = images.to(DEVICE)
            # Encode to [B, 64, 16, 16] continuous latents
            latents = model.encode(images) 
            
            # Save as numpy arrays (much faster to load than PNGs)
            latents_np = latents.cpu().numpy()
            for i, fname in enumerate(filenames):
                name = os.path.splitext(fname)[0]
                np.save(os.path.join(latents_dir, f"{name}.npy"), latents_np[i])
                
    print(f"Latents saved to {latents_dir}")

if __name__ == "__main__":
    main()