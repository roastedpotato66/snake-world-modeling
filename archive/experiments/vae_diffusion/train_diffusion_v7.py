"""
Snake World Model - Diffusion Training V7

CRITICAL FIXES:
1. x0-prediction instead of epsilon-prediction (model predicts CLEAN latent, not noise)
2. Action classifier auxiliary loss (forces model to be action-aware)
3. Reconstruction loss on decoded frames (grounds the latent space)
4. Much simpler architecture with cross-attention for action conditioning
5. Lower noise levels (SNR-weighted) to preserve structure

The key insight: In epsilon-prediction, the model can ignore actions because
noise is independent of actions. In x0-prediction, actions directly affect
the target, forcing the model to learn the relationship.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers import DDPMScheduler, DDIMScheduler
from accelerate import Accelerator
import numpy as np
import pandas as pd
import os
import argparse
import csv
from tqdm.auto import tqdm
import multiprocessing
from torchvision.utils import save_image

# --- CONFIG ---
LATENT_DIM = 4      
LATENT_SIZE = 8     
CONTEXT_FRAMES = 4  
ACTION_DIM = 4      

# --- VAE (for visualization) ---
class KLVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.decoder_input = nn.Conv2d(latent_channels, 128, 3, 1, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh()
        )
    def decode(self, z):
        return self.decoder(self.decoder_input(z))

# --- SIMPLE BUT EFFECTIVE ARCHITECTURE ---
class SimpleWorldModel(nn.Module):
    """
    A simpler architecture that CAN'T ignore actions.
    
    Key design choices:
    1. Action is concatenated at EVERY layer (not just injected via FiLM)
    2. Uses cross-attention between history and action
    3. Predicts x0 directly (clean latent), not noise
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # History encoder (4 frames * 4 channels = 16 channels)
        self.history_encoder = nn.Sequential(
            nn.Conv2d(LATENT_DIM * CONTEXT_FRAMES, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Noisy input encoder
        self.noisy_encoder = nn.Sequential(
            nn.Conv2d(LATENT_DIM, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Action embedding - make it STRONG
        self.action_embed = nn.Sequential(
            nn.Linear(ACTION_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Cross-attention: history attends to action
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Main processing blocks - action is CONCATENATED at each stage
        self.block1 = self._make_block(hidden_dim * 3, hidden_dim)  # history + noisy + action
        self.block2 = self._make_block(hidden_dim + hidden_dim, hidden_dim)  # prev + action
        self.block3 = self._make_block(hidden_dim + hidden_dim, hidden_dim)  # prev + action
        
        # Output head
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, LATENT_DIM, 3, 1, 1),
        )
        
        # Action classifier head (auxiliary task)
        self.action_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )
    
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    
    def get_timestep_embedding(self, timesteps, dim=256):
        half = dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward(self, noisy_latent, history, action, timesteps):
        """
        Args:
            noisy_latent: (B, 4, 8, 8) - noisy version of target
            history: (B, 16, 8, 8) - 4 previous frames
            action: (B, 4) - one-hot action
            timesteps: (B,) - diffusion timesteps
        
        Returns:
            x0_pred: (B, 4, 8, 8) - predicted CLEAN latent
            action_logits: (B, 4) - auxiliary action prediction
        """
        B = noisy_latent.shape[0]
        
        # Encode inputs
        h_hist = self.history_encoder(history)  # (B, hidden, 8, 8)
        h_noisy = self.noisy_encoder(noisy_latent)  # (B, hidden, 8, 8)
        
        # Embed action and time
        a_emb = self.action_embed(action)  # (B, hidden)
        t_emb = self.time_embed(self.get_timestep_embedding(timesteps))  # (B, hidden)
        
        # Combine time into action embedding
        a_emb = a_emb + t_emb
        
        # Cross-attention: history queries, action is key/value
        # Reshape for attention
        h_hist_flat = h_hist.flatten(2).permute(0, 2, 1)  # (B, 64, hidden)
        a_emb_exp = a_emb.unsqueeze(1).expand(-1, 64, -1)  # (B, 64, hidden)
        
        h_attended, _ = self.cross_attn(h_hist_flat, a_emb_exp, a_emb_exp)
        h_attended = h_attended.permute(0, 2, 1).view(B, self.hidden_dim, 8, 8)
        
        # Make action into spatial feature map
        a_spatial = a_emb[:, :, None, None].expand(-1, -1, 8, 8)  # (B, hidden, 8, 8)
        
        # Block 1: Combine history, noisy, and action
        x = torch.cat([h_attended, h_noisy, a_spatial], dim=1)
        x = self.block1(x)
        
        # Block 2: Add action again
        x = torch.cat([x, a_spatial], dim=1)
        x = self.block2(x)
        
        # Block 3: Add action again (force it to use action!)
        x = torch.cat([x, a_spatial], dim=1)
        x = self.block3(x)
        
        # Output
        x0_pred = self.output(x)
        
        # Auxiliary action classification (from the final features)
        action_logits = self.action_classifier(x)
        
        return x0_pred, action_logits

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
                        "history": hist_files, "target": target_file, "action": int(actions[i-1])
                    })

        if self.cache:
            print(f"Caching latents...")
            needed_files = set()
            for s in self.samples:
                needed_files.update(s['history'])
                needed_files.add(s['target'])
            
            def load_single(fname):
                return fname, np.load(os.path.join(self.data_dir, fname + ".npy")).astype(np.float32)

            with multiprocessing.Pool(8) as pool:
                results = list(tqdm(pool.imap(load_single, needed_files), total=len(needed_files)))
            for fname, data in results: 
                self.ram_cache[fname] = data
        
        # Compute stats
        if self.cache and len(self.ram_cache) > 0:
            all_latents = np.stack(list(self.ram_cache.values()), axis=0)
            self.mean = float(np.mean(all_latents))
            self.std = float(np.std(all_latents))
        else:
            self.mean, self.std = 0.0, 1.0
        
        print(f"Dataset: {len(self.samples)} samples, mean={self.mean:.4f}, std={self.std:.4f}")

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        def get_data(fname):
            d = self.ram_cache[fname] if self.cache else np.load(os.path.join(self.data_dir, fname + ".npy"))
            return torch.from_numpy(d.copy()).float()

        hist = torch.cat([get_data(f) for f in sample['history']], dim=0)
        target = get_data(sample['target'])
        
        # One-hot action
        action = torch.zeros(ACTION_DIM)
        action[sample['action']] = 1.0
        
        # Also return action index for classification loss
        action_idx = sample['action']
            
        return hist, action, target, action_idx

# --- TRAINING ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="output/vae_kl/latents_8x8")
    parser.add_argument("--metadata", type=str, default="data_v3/metadata.csv")
    parser.add_argument("--vae_path", type=str, default="output/vae_kl/klvae_latest.pt")
    parser.add_argument("--output_dir", type=str, default="output/diffusion_v7")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--action_loss_weight", type=float, default=0.5)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    
    accelerator = Accelerator(mixed_precision="fp16")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logging
    log_file = os.path.join(args.output_dir, "training_log.csv")
    if accelerator.is_main_process:
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'recon_loss', 'action_loss', 'action_acc', 'val_loss', 'cfg_diff'])

    # Dataset
    full_dataset = SnakeLatentDataset(args.data_dir, args.metadata, cache=not args.no_cache)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # VAE for visualization
    vae = KLVAE()
    if os.path.exists(args.vae_path):
        vae.load_state_dict(torch.load(args.vae_path, map_location='cpu'), strict=False)
    vae.to(accelerator.device)
    vae.eval()

    # Model
    model = SimpleWorldModel(hidden_dim=256)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Scheduler - use fewer timesteps for x0 prediction
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",  # Better for x0 prediction
        prediction_type="sample",  # Predict x0, not epsilon!
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    print(f"--- Training World Model V7 (x0-prediction + Action Classifier) ---")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        
        epoch_recon_loss = 0
        epoch_action_loss = 0
        epoch_action_correct = 0
        epoch_total = 0
        
        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        pbar.set_description(f"Epoch {epoch}")
        
        for history, action, target, action_idx in pbar:
            # CFG dropout (30%)
            if np.random.random() < 0.3:
                action = torch.zeros_like(action)
                action_idx = torch.full_like(action_idx, -100)  # Ignore in CE loss
            
            # Add noise to target
            noise = torch.randn_like(target)
            # Use lower noise levels more often (important for x0 prediction)
            timesteps = torch.randint(0, 500, (target.shape[0],), device=accelerator.device).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            
            # Forward
            x0_pred, action_logits = model(noisy_target, history, action, timesteps)
            
            # Losses
            # 1. Reconstruction loss (MSE on predicted x0 vs true x0)
            recon_loss = F.mse_loss(x0_pred, target)
            
            # 2. Action classification loss (forces model to be action-aware)
            action_loss = F.cross_entropy(action_logits, action_idx, ignore_index=-100)
            
            # Combined loss
            loss = recon_loss + args.action_loss_weight * action_loss
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Metrics
            epoch_recon_loss += recon_loss.item()
            epoch_action_loss += action_loss.item()
            
            # Action accuracy (only for non-dropped samples)
            valid_mask = action_idx != -100
            if valid_mask.sum() > 0:
                pred_actions = action_logits[valid_mask].argmax(dim=-1)
                epoch_action_correct += (pred_actions == action_idx[valid_mask]).sum().item()
                epoch_total += valid_mask.sum().item()
            
            pbar.set_postfix(recon=recon_loss.item(), act=action_loss.item())
        
        # Epoch stats
        n_batches = len(train_loader)
        avg_recon = epoch_recon_loss / n_batches
        avg_action = epoch_action_loss / n_batches
        action_acc = epoch_action_correct / max(epoch_total, 1)
        
        # Validation
        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            val_count = 0
            
            # CFG diff test
            cfg_diffs = []
            
            with torch.no_grad():
                for history, action, target, action_idx in val_loader:
                    if val_count >= 20: break
                    
                    noise = torch.randn_like(target)
                    timesteps = torch.full((target.shape[0],), 250, device=accelerator.device).long()
                    noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
                    
                    x0_pred, _ = model(noisy_target, history, action, timesteps)
                    val_loss += F.mse_loss(x0_pred, target).item()
                    
                    # CFG diff
                    x0_uncond, _ = model(noisy_target, history, torch.zeros_like(action), timesteps)
                    cfg_diffs.append(F.mse_loss(x0_pred, x0_uncond).item())
                    
                    val_count += 1
            
            val_loss /= val_count
            cfg_diff = np.mean(cfg_diffs)
            
            print(f"\n[Epoch {epoch}] Recon: {avg_recon:.4f} | Action Loss: {avg_action:.4f} | "
                  f"Action Acc: {action_acc:.2%} | Val: {val_loss:.4f} | CFG Diff: {cfg_diff:.4f}")
            
            # Log
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, avg_recon + avg_action, avg_recon, avg_action, action_acc, val_loss, cfg_diff])
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("⭐ New Best!")
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_path, "model.pt"))
                torch.save({"mean": full_dataset.mean, "std": full_dataset.std}, os.path.join(save_path, "stats.pt"))
            
            # Visualize rollout
            if (epoch + 1) % 10 == 0:
                visualize_rollout(accelerator.unwrap_model(model), vae, val_dataset, 
                                  full_dataset.mean, full_dataset.std,
                                  accelerator.device, 
                                  os.path.join(args.output_dir, f"rollout_ep{epoch+1}.png"))
            
            model.train()
    
    print("Training Complete!")

def visualize_rollout(model, vae, dataset, mean, std, device, save_path):
    """Generate frames autoregressively"""
    model.eval()
    
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
    )
    scheduler.set_timesteps(10)  # Few steps for x0 prediction
    
    # Get starting history
    if hasattr(dataset, 'dataset'):
        real_idx = dataset.indices[0]
        history, _, _, _ = dataset.dataset[real_idx]
    else:
        history, _, _, _ = dataset[0]
    
    history = history.unsqueeze(0).to(device)
    
    frames = []
    actions = [0]*8 + [3]*8 + [1]*8 + [2]*8
    
    with torch.no_grad():
        for i in range(32):
            action = torch.zeros(1, ACTION_DIM, device=device)
            action[0, actions[i % len(actions)]] = 1.0
            
            # Start from noise
            latent = torch.randn(1, LATENT_DIM, LATENT_SIZE, LATENT_SIZE, device=device)
            
            # Denoise
            for t in scheduler.timesteps:
                x0_pred, _ = model(latent, history, action, t.unsqueeze(0).to(device))
                latent = scheduler.step(x0_pred, t, latent, return_dict=True).prev_sample
            
            frames.append(latent)
            history = torch.cat([history[:, LATENT_DIM:], latent], dim=1)
    
    # Decode
    all_latents = torch.cat(frames, dim=0)
    pixels = vae.decode(all_latents)
    save_image(pixels * 0.5 + 0.5, save_path, nrow=8)
    print(f"Saved rollout to {save_path}")

if __name__ == "__main__":
    main()