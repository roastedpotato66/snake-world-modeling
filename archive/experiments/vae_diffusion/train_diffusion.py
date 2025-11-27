import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import numpy as np
import pandas as pd
import os
import argparse
from tqdm.auto import tqdm
from torchvision.utils import make_grid, save_image
import sys

# --- Configuration ---
LATENT_DIM = 64
LATENT_SIZE = 16
CONTEXT_FRAMES = 4

# Attempt to import VAE for visualization
sys.path.append('.')
try:
    from train_vae import VQVAE, VectorQuantizerEMA
except ImportError:
    print("Warning: Could not import VAE class. Visualization will be skipped.")
    VQVAE = None

class SnakeLatentDataset(Dataset):
    def __init__(self, data_dir, metadata_file, cache_in_ram=False):
        self.data_dir = data_dir
        self.df = pd.read_csv(metadata_file)
        self.samples = []
        self.cache_in_ram = cache_in_ram
        self.ram_cache = {}
        
        print("Indexing dataset sequences...")
        for episode_id, group in tqdm(self.df.groupby('episode_id')):
            if len(group) < CONTEXT_FRAMES + 1:
                continue
            
            group = group.sort_values('frame_number')
            indices = group.index.tolist()
            
            for i in range(CONTEXT_FRAMES, len(indices)):
                history_idx = indices[i-CONTEXT_FRAMES : i]
                target_idx = indices[i]
                
                action = group.loc[indices[i-1], 'action']
                hist_files = [os.path.splitext(group.loc[idx, 'image_file'])[0] for idx in history_idx]
                target_file = os.path.splitext(group.loc[target_idx, 'image_file'])[0]
                
                self.samples.append({
                    "history": hist_files,
                    "target": target_file,
                    "action": action
                })

        if self.cache_in_ram:
            print("Caching all latents to RAM (This speeds up training massively)...")
            # Get unique filenames to avoid loading duplicates
            unique_files = set()
            for s in self.samples:
                unique_files.update(s['history'])
                unique_files.add(s['target'])
            
            for fname in tqdm(unique_files):
                path = os.path.join(self.data_dir, fname + ".npy")
                self.ram_cache[fname] = np.load(path)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        history_arrays = []
        for fname in sample['history']:
            if self.cache_in_ram:
                data = self.ram_cache[fname]
            else:
                data = np.load(os.path.join(self.data_dir, fname + ".npy"))
            history_arrays.append(torch.from_numpy(data))
            
        if self.cache_in_ram:
            target_data = self.ram_cache[sample['target']]
        else:
            target_data = np.load(os.path.join(self.data_dir, sample['target'] + ".npy"))
        
        target_array = torch.from_numpy(target_data)
        history_tensor = torch.cat(history_arrays, dim=0) 
        
        return history_tensor, sample['action'], target_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="output/vae/latents")
    parser.add_argument("--metadata", type=str, default="data_v2/metadata.csv")
    parser.add_argument("--vae_path", type=str, default="output/vae/vqvae.pt")
    parser.add_argument("--output_dir", type=str, default="output/diffusion_v2")
    parser.add_argument("--batch_size", type=int, default=256) 
    parser.add_argument("--epochs", type=int, default=50) # Increased epochs
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cond_noise", type=float, default=0.05)
    parser.add_argument("--cache", action="store_true", help="Cache data in RAM")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    accelerator = Accelerator(mixed_precision="fp16", log_with="tensorboard", project_dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Dataset
    dataset = SnakeLatentDataset(args.data_dir, args.metadata, cache_in_ram=args.cache)
    
    # Optimized DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True, # Keeps workers alive (Fixes GPU drops)
        prefetch_factor=2        # Buffers next batch
    )
    
    # 2. Model
    model = UNet2DModel(
        sample_size=LATENT_SIZE,
        in_channels=LATENT_DIM + (LATENT_DIM * CONTEXT_FRAMES), 
        out_channels=LATENT_DIM,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        class_embed_type="timestep",
        num_class_embeds=5 
    )
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5) # Added Weight Decay
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=(len(dataloader) * args.epochs),
    )
    
    # Load VAE for Viz
    vae = None
    if VQVAE and os.path.exists(args.vae_path):
        vae = VQVAE(embedding_dim=LATENT_DIM).to(accelerator.device)
        vae.load_state_dict(torch.load(args.vae_path, map_location=accelerator.device))
        vae.eval()
        vae.requires_grad_(False)

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    # Resume training if needed
    if args.resume:
        print(f"Resuming from {args.resume}...")
        accelerator.load_state(args.resume)

    print(f"Starting Training: {args.epochs} Epochs, Batch Size {args.batch_size}")
    
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for history, action, target in progress_bar:
            # CFG
            if np.random.random() < 0.1:
                action = torch.full_like(action, 4)
            
            # Conditioning Noise
            if args.cond_noise > 0:
                history = history + (torch.randn_like(history) * args.cond_noise)
            
            noise = torch.randn_like(target)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (target.shape[0],), device=target.device).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            
            model_input = torch.cat([noisy_target, history], dim=1)
            noise_pred = model(model_input, timesteps, class_labels=action).sample
            
            loss = F.mse_loss(noise_pred, noise)
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix(loss=loss.item())
            
        # --- Checkpointing & Viz ---
        if accelerator.is_main_process:
            # Save Checkpoint Every 5 Epochs
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
                accelerator.save_state(save_path)
                print(f"Saved checkpoint to {save_path}")
                
                # Also save "best_model" format for easy loading later
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(os.path.join(args.output_dir, "latest_model"))

            # Visualization
            if vae is not None:
                model.eval()
                with torch.no_grad():
                    hist, act, real_tgt = next(iter(dataloader))
                    hist = hist[:4].to(accelerator.device)
                    act = act[:4].to(accelerator.device)
                    real_tgt = real_tgt[:4].to(accelerator.device)
                    
                    latents = torch.randn_like(real_tgt)
                    noise_scheduler.set_timesteps(20)
                    
                    for t in noise_scheduler.timesteps:
                        model_input = torch.cat([latents, hist], dim=1)
                        noise_pred = model(model_input, t, class_labels=act).sample
                        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                    
                    recon_img = vae.decoder(latents)
                    real_img = vae.decoder(real_tgt)
                    
                    # Visualize the immediate past frame (last 64 channels)
                    hist_img = vae.decoder(hist[:, -64:])
                    
                    viz = torch.cat([hist_img, recon_img, real_img], dim=0)
                    viz = (viz * 0.5 + 0.5).clamp(0, 1)
                    save_image(viz, os.path.join(args.output_dir, f"epoch_{epoch}_sample.png"), nrow=4)
            
    print("Training Complete.")

if __name__ == "__main__":
    main()