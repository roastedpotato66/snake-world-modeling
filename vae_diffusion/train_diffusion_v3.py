# train_diffusion_v3.py
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
import sys

# --- CONFIGURATION ---
LATENT_DIM = 64
LATENT_SIZE = 16
CONTEXT_FRAMES = 4

class SnakeLatentDataset(Dataset):
    def __init__(self, data_dir, metadata_file, cache=True):
        self.data_dir = data_dir
        self.df = pd.read_csv(metadata_file)
        self.samples = []
        self.cache = cache
        self.ram_cache = {} # The Speed Booster
        
        print("Indexing dataset sequences...")
        valid_files = set([f[:-4] for f in os.listdir(data_dir) if f.endswith('.npy')])
        
        for episode_id, group in tqdm(self.df.groupby('episode_id')):
            if len(group) < CONTEXT_FRAMES + 1: continue
            group = group.sort_values('frame_number')
            indices = group.index.tolist()
            for i in range(CONTEXT_FRAMES, len(indices)):
                history_idx = indices[i-CONTEXT_FRAMES : i]
                target_idx = indices[i]
                hist_files = [os.path.splitext(group.loc[idx, 'image_file'])[0] for idx in history_idx]
                target_file = os.path.splitext(group.loc[target_idx, 'image_file'])[0]
                
                if all(f in valid_files for f in hist_files) and target_file in valid_files:
                    self.samples.append({
                        "history": hist_files,
                        "target": target_file,
                        "action": group.loc[indices[i-1], 'action']
                    })

        # 1. LOAD DATA INTO RAM
        if self.cache:
            print(f"Caching {len(valid_files)} latents to RAM... (This takes a minute)")
            # We only load unique files once
            all_files = set()
            for s in self.samples:
                all_files.update(s['history'])
                all_files.add(s['target'])
            
            temp_values = []
            for fname in tqdm(all_files):
                path = os.path.join(self.data_dir, fname + ".npy")
                data = np.load(path)
                self.ram_cache[fname] = data
                temp_values.append(data)
            
            # 2. COMPUTE STATS FROM CACHE (Fast)
            print("Computing Normalization Stats...")
            all_data_matrix = np.stack(temp_values)
            self.mean = torch.tensor(np.mean(all_data_matrix)).float()
            self.std = torch.tensor(np.std(all_data_matrix)).float()
            del temp_values # Free up memory
            del all_data_matrix
            
        else:
            # Slow fallback
            print("Computing Normalization Stats (Slow Mode)...")
            # ... (omitted for brevity, same as before) ...
            # Just forcing cache mode for this script as it is the 'fast' version
            raise ValueError("Please use --cache mode for this script.")

        print(f"Dataset Mean: {self.mean:.4f}")
        print(f"Dataset Std:  {self.std:.4f}")
        if self.std < 1e-6: self.std = torch.tensor(1.0)

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        history_arrays = []
        for fname in sample['history']:
            # Load from RAM
            data = torch.from_numpy(self.ram_cache[fname])
            data = (data - self.mean) / self.std
            history_arrays.append(data)
            
        target_data = torch.from_numpy(self.ram_cache[sample['target']])
        target_data = (target_data - self.mean) / self.std
        
        history_tensor = torch.cat(history_arrays, dim=0) 
        return history_tensor, sample['action'], target_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="output/vae/latents")
    parser.add_argument("--metadata", type=str, default="data_v2/metadata.csv")
    parser.add_argument("--output_dir", type=str, default="output/diffusion_v3")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    accelerator = Accelerator(mixed_precision="fp16", project_dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Enable Caching by default
    dataset = SnakeLatentDataset(args.data_dir, args.metadata, cache=True)
    torch.save({"mean": dataset.mean, "std": dataset.std}, os.path.join(args.output_dir, "stats.pt"))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=500, num_training_steps=(len(dataloader) * args.epochs))
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    
    print(f"Starting Training (Fast Mode) on {len(dataset)} samples...")
    
    for epoch in range(args.epochs):
        model.train()
        if accelerator.is_main_process: print(f"Epoch {epoch}")
        
        # No TQDM on inner loop to save console spam if moving fast, just verify speed on first few
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        
        for history, action, target in progress_bar:
            if np.random.random() < 0.1: action = torch.full_like(action, 4)
            history = history + (torch.randn_like(history) * 0.05)

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

        if accelerator.is_main_process and (epoch+1) % 5 == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(os.path.join(args.output_dir, "latest_model"))

if __name__ == "__main__":
    main()