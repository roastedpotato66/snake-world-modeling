import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import numpy as np
import pandas as pd
import os
import argparse
import csv
from tqdm.auto import tqdm
import multiprocessing
from torchvision.utils import save_image

# --- CONFIGURATION ---
LATENT_DIM = 4      
LATENT_SIZE = 8     
CONTEXT_FRAMES = 4  
ACTION_DIM = 4      

# --- VAE DEFINITION (Exact Copy for Loading Weights) ---
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
        # Encoder
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
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh() # Tanh force range [-1, 1]
        )

    def forward(self, x, sample_posterior=True):
        moments = self.to_moments(self.encoder(x))
        posterior = DiagonalGaussianDistribution(moments)
        z = posterior.sample() if sample_posterior else posterior.mean
        return self.decoder(self.decoder_input(z)), posterior

# --- DATASET ---
class SnakeLatentDataset(Dataset):
    def __init__(self, data_dir, metadata_file, cache=True):
        self.data_dir = data_dir
        self.df = pd.read_csv(metadata_file)
        self.cache = cache
        self.ram_cache = {} 
        
        print(f"Indexing dataset from {metadata_file}...")
        valid_files = set([f[:-4] for f in os.listdir(data_dir) if f.endswith('.npy')])
        grouped = self.df.groupby('episode_id')
        
        self.samples = []
        for episode_id, group in grouped:
            if len(group) < CONTEXT_FRAMES + 1: continue
            group = group.sort_values('frame_number')
            files = group['image_file'].values
            actions = group['action'].values
            
            for i in range(CONTEXT_FRAMES, len(files)):
                hist_files = [f[:-4] for f in files[i-CONTEXT_FRAMES : i]]
                target_file = files[i][:-4]
                if target_file in valid_files and all(h in valid_files for h in hist_files):
                    self.samples.append({
                        "history": hist_files, "target": target_file, "action": actions[i-1]
                    })

        if self.cache:
            print(f"Caching {len(valid_files)} latents (Parallel)...")
            needed_files = set()
            for s in self.samples:
                needed_files.update(s['history'])
                needed_files.add(s['target'])
            
            def load_single(fname):
                return fname, np.load(os.path.join(self.data_dir, fname + ".npy")).astype(np.float32)

            with multiprocessing.Pool(8) as pool:
                results = list(tqdm(pool.imap(load_single, needed_files), total=len(needed_files)))
            for fname, data in results: self.ram_cache[fname] = data
                
        if self.cache:
            all_vals = np.concatenate(list(self.ram_cache.values()), axis=0)
            self.mean = torch.tensor(np.mean(all_vals)).float()
            self.std = torch.tensor(np.std(all_vals)).float()
        else:
            self.mean = torch.tensor(0.0)
            self.std = torch.tensor(1.0)
        print(f"Mean: {self.mean:.4f}, Std: {self.std:.4f}")

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        def get_data(fname):
            d = self.ram_cache[fname] if self.cache else np.load(os.path.join(self.data_dir, fname + ".npy"))
            return torch.from_numpy(d).float()

        hist = torch.cat([(get_data(f) - self.mean)/self.std for f in sample['history']], dim=0)
        targ = (get_data(sample['target']) - self.mean)/self.std
        
        action_map = torch.zeros((ACTION_DIM, LATENT_SIZE, LATENT_SIZE))
        if 0 <= int(sample['action']) < ACTION_DIM:
            action_map[int(sample['action']), :, :] = 1.0
            
        return hist, action_map, targ

# --- VALIDATION ---
def validate_cfg(model, dataloader, accelerator, noise_scheduler):
    model.eval()
    diffs = {}
    with torch.no_grad():
        try:
            history, action, target = next(iter(dataloader))
        except StopIteration: return {}
            
        history, action, target = history.to(accelerator.device), action.to(accelerator.device), target.to(accelerator.device)
        noise = torch.randn_like(target)
        
        for t_val in [100, 500, 900]:
            timesteps = torch.tensor([t_val], device=accelerator.device).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            
            zero_action = torch.zeros_like(action)
            pred_cond = model(torch.cat([noisy_target, history, action], dim=1), timesteps).sample
            pred_uncond = model(torch.cat([noisy_target, history, zero_action], dim=1), timesteps).sample
            diffs[t_val] = F.mse_loss(pred_cond, pred_uncond).item()
    model.train()
    return diffs

def compute_val_loss(model, dataloader, accelerator, noise_scheduler):
    model.eval()
    total_loss, count = 0, 0
    with torch.no_grad():
        for history, action_map, target in dataloader:
            if count >= 30: break 
            history, action_map, target = history.to(accelerator.device), action_map.to(accelerator.device), target.to(accelerator.device)
            noise = torch.randn_like(target)
            timesteps = torch.randint(0, 1000, (target.shape[0],), device=target.device).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            pred = model(torch.cat([noisy_target, history, action_map], dim=1), timesteps).sample
            total_loss += F.mse_loss(pred, noise).item()
            count += 1
    model.train()
    return total_loss / count

def validate_rollout(model, vae, dataset, mean, std, device, output_path):
    model.eval()
    vae.eval()
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(20)
    
    # Get a starting frame from the dataset (Dataset or Subset)
    # Handle Subset indexing
    if isinstance(dataset, torch.utils.data.Subset):
        real_idx = dataset.indices[0]
        full_ds = dataset.dataset
        history, _, _ = full_ds[real_idx]
    else:
        history, _, _ = dataset[0]

    history = history.unsqueeze(0).to(device)
    
    # Fixed Pattern: Up(8) -> Right(8) -> Down(8) -> Left(8) -> Circle
    actions = [0]*8 + [3]*8 + [1]*8 + [2]*8
    actions = actions * 2 # 64 frames
    
    frames = []
    
    with torch.no_grad():
        for i in range(64):
            action_idx = actions[i % len(actions)]
            action_map = torch.zeros((1, ACTION_DIM, LATENT_SIZE, LATENT_SIZE), device=device)
            action_map[0, action_idx, :, :] = 1.0
            
            latents = torch.randn((1, LATENT_DIM, LATENT_SIZE, LATENT_SIZE), device=device)
            for t in scheduler.timesteps:
                model_input = torch.cat([latents, history, action_map], dim=1)
                noise_pred = model(model_input, t).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            frames.append(latents)
            history = torch.cat([history[:, LATENT_DIM:], latents], dim=1)

    batch_latents = torch.cat(frames, dim=0)
    batch_latents = (batch_latents * std.to(device)) + mean.to(device)
    pixels = vae.decoder(vae.decoder_input(batch_latents))
    save_image(pixels * 0.5 + 0.5, output_path, nrow=8)
    model.train()

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="output/vae_kl/latents_8x8")
    parser.add_argument("--metadata", type=str, default="data_v3/metadata.csv")
    parser.add_argument("--vae_path", type=str, default="output/vae_kl/klvae_latest.pt")
    parser.add_argument("--output_dir", type=str, default="output/diffusion_v4")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    
    accelerator = Accelerator(mixed_precision="fp16", project_dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Setup Log
    log_file = os.path.join(args.output_dir, "training_log.csv")
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'cfg_diff_500', 'best_loss'])

    # 2. Dataset
    full_dataset = SnakeLatentDataset(args.data_dir, args.metadata, cache=not args.no_cache)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                          num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True, persistent_workers=True)

    # 3. VAE (Strict loading attempt)
    vae = KLVAE()
    if os.path.exists(args.vae_path):
        try:
            vae.load_state_dict(torch.load(args.vae_path, map_location='cpu'))
            print("✅ VAE loaded.")
        except:
            print("⚠️ VAE load strict failed, trying non-strict...")
            vae.load_state_dict(torch.load(args.vae_path, map_location='cpu'), strict=False)
    else:
        print("❌ VAE NOT FOUND. Validation will look black.")
    vae.to(accelerator.device)
    vae.eval()

    # 4. Model
    in_channels = LATENT_DIM + (LATENT_DIM * CONTEXT_FRAMES) + ACTION_DIM
    model = UNet2DModel(sample_size=LATENT_SIZE, in_channels=in_channels, out_channels=LATENT_DIM,
                        layers_per_block=2, block_out_channels=(128, 256, 512),
                        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
                        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"))
    
    ema_model = EMAModel(model.parameters(), decay=0.9999, update_after_step=100,
                         model_cls=UNet2DModel, model_config=model.config)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 500, len(train_loader)*args.epochs)
    
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    ema_model.to(accelerator.device)

    print(f"--- Training Diffusion V8 (Robust) ---")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, leave=False)
        pbar.set_description(f"Epoch {epoch}")
        
        for history, action_map, target in pbar:
            # CFG Dropout (20%)
            if np.random.random() < 0.20:
                action_map = torch.zeros_like(action_map)

            # Robust History Noise (0.0 - 0.2)
            noise_scale = torch.rand(history.shape[0], 1, 1, 1, device=accelerator.device) * 0.20
            history = history + (torch.randn_like(history) * noise_scale)

            # Training Step
            noise = torch.randn_like(target)
            timesteps = torch.randint(0, 1000, (target.shape[0],), device=target.device).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            
            pred = model(torch.cat([noisy_target, history, action_map], dim=1), timesteps).sample
            loss = F.mse_loss(pred, noise)
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            ema_model.step(model.parameters())
            pbar.set_postfix(loss=loss.item())

        # VALIDATION BLOCK
        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            val_loss = compute_val_loss(model, val_loader, accelerator, noise_scheduler)
            cfg_diff = validate_cfg(model, val_loader, accelerator, noise_scheduler).get(500, 0)
            
            print(f"\n[Epoch {epoch}] Val Loss: {val_loss:.5f} | CFG Diff: {cfg_diff:.5f}")
            
            # Log
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, loss.item(), val_loss, cfg_diff, val_loss < best_val_loss])

            # Save Best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("⭐ Saving Best Model...")
                save_path = os.path.join(args.output_dir, "best_model")
                accelerator.unwrap_model(model).save_pretrained(save_path)
                ema_model.save_pretrained(os.path.join(save_path, "ema"))
                torch.save({"mean": full_dataset.mean, "std": full_dataset.std}, os.path.join(save_path, "stats.pt"))

            # Rollout (Every 10 epochs)
            if (epoch + 1) % 10 == 0:
                rollout_path = os.path.join(args.output_dir, f"rollout_ep{epoch+1}.png")
                validate_rollout(accelerator.unwrap_model(model), vae, val_dataset, full_dataset.mean, full_dataset.std, accelerator.device, rollout_path)
                
                # Checkpoint
                save_path = os.path.join(args.output_dir, f"checkpoint_ep{epoch+1}")
                accelerator.unwrap_model(model).save_pretrained(save_path)
                ema_model.save_pretrained(os.path.join(save_path, "ema"))

    print("Training Complete.")

if __name__ == "__main__":
    main()