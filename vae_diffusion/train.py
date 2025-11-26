import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from PIL import Image
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from accelerate import Accelerator
from torchvision.utils import make_grid
import math

# --- Configuration ---
IMG_SIZE = 64

class SnakeDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        curr_path = os.path.join(self.img_dir, row['current_image'])
        next_path = os.path.join(self.img_dir, row['next_image'])
        
        curr_img = Image.open(curr_path).convert("RGB")
        next_img = Image.open(next_path).convert("RGB")
        
        curr_tensor = self.transform(curr_img)
        next_tensor = self.transform(next_img)
        
        action = torch.tensor(row['action'], dtype=torch.long)
        
        return curr_tensor, action, next_tensor

def evaluate(model, scheduler, dataloader, device, epoch, output_dir, noise_aug_strength=0.0):
    """
    Evaluates using the EMA model.
    """
    model.eval()
    scheduler.set_timesteps(20)
    
    # Get batch
    curr_img, action, real_next = next(iter(dataloader))
    curr_img = curr_img[:8].to(device) # Show 8 examples
    action = action[:8].to(device)
    real_next = real_next[:8].to(device)
    
    noisy_latents = torch.randn_like(curr_img).to(device)
    
    # Add slight noise to conditioning during eval (mimic inference drift)
    if noise_aug_strength > 0:
        cond_noise = torch.randn_like(curr_img) * noise_aug_strength
        curr_img_cond = curr_img + cond_noise
    else:
        curr_img_cond = curr_img

    for t in scheduler.timesteps:
        model_input = torch.cat([noisy_latents, curr_img_cond], dim=1)
        with torch.no_grad():
            noise_pred = model(model_input, t, class_labels=action).sample
        noisy_latents = scheduler.step(noise_pred, t, noisy_latents).prev_sample

    # Viz
    generated = (noisy_latents / 2 + 0.5).clamp(0, 1)
    real = (real_next / 2 + 0.5).clamp(0, 1)
    input_view = (curr_img / 2 + 0.5).clamp(0, 1)
    
    viz = torch.cat([input_view, generated, real], dim=0)
    grid = make_grid(viz, nrow=8, padding=2)
    
    save_path = os.path.join(output_dir, "samples", f"epoch_{epoch:03d}_ema.png")
    transforms.ToPILImage()(grid).save(save_path)
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output/pro")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--save_interval", type=int, default=3)
    parser.add_argument("--noise_aug", type=float, default=0.1) # Aggressive noise training
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # 1. Data with Split
    full_dataset = SnakeDataset(
        csv_file=os.path.join(args.data_path, "metadata.csv"),
        img_dir=os.path.join(args.data_path, "images")
    )
    
    # 95% Train, 5% Validation
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 2. Model
    model = UNet2DModel(
        sample_size=IMG_SIZE,
        in_channels=6, 
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        class_embed_type="timestep", 
        num_class_embeds=4 
    )
    
    # 3. EMA Model (The Secret Weapon)
    # Keeps a smooth copy of the model. decay=0.999 means it changes slowly.
    ema_model = EMAModel(model.parameters(), decay=0.999, model_cls=UNet2DModel, model_config=model.config)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    eval_scheduler = DDIMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 4. LR Scheduler (Cosine)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * args.epochs),
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    ema_model.to(accelerator.device)

    print(f"Starting PRO training for {args.epochs} epochs...")
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        # --- TRAINING ---
        for curr_img, action, next_img in train_loader:
            noise = torch.randn_like(next_img)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (curr_img.shape[0],), device=next_img.device).long()
            noisy_next_img = noise_scheduler.add_noise(next_img, noise, timesteps)
            
            # Noise Augmentation
            aug_strength = torch.rand(curr_img.shape[0], 1, 1, 1, device=curr_img.device) * args.noise_aug
            cond_noise = torch.randn_like(curr_img) * aug_strength
            dirty_curr_img = curr_img + cond_noise
            
            model_input = torch.cat([noisy_next_img, dirty_curr_img], dim=1)
            
            noise_pred = model(model_input, timesteps, class_labels=action).sample
            loss = F.mse_loss(noise_pred, noise)
            
            accelerator.backward(loss)
            
            # Gradient Clipping (Prevents Explosions)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update EMA
            ema_model.step(model.parameters())
            
            progress_bar.update(1)
            progress_bar.set_postfix(mse=loss.item(), lr=lr_scheduler.get_last_lr()[0])

        # --- VALIDATION ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for curr_img, action, next_img in val_loader:
                noise = torch.randn_like(next_img)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (curr_img.shape[0],), device=next_img.device).long()
                noisy_next_img = noise_scheduler.add_noise(next_img, noise, timesteps)
                model_input = torch.cat([noisy_next_img, curr_img], dim=1)
                noise_pred = model(model_input, timesteps, class_labels=action).sample
                val_losses.append(F.mse_loss(noise_pred, noise).item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.5f}")

        if accelerator.is_main_process:
            # Save Best EMA Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ema_model.save_pretrained(os.path.join(args.output_dir, "best_model"))
                print("New Best Model Saved!")
            
            # Periodic Visual Eval
            if epoch % args.save_interval == 0:
                # Evaluate using EMA model (higher quality)
                # We manually copy EMA weights to a temporary model for inference
                temp_model = UNet2DModel.from_config(model.config).to(accelerator.device)
                ema_model.copy_to(temp_model.parameters())
                evaluate(temp_model, eval_scheduler, val_loader, accelerator.device, epoch, args.output_dir, noise_aug_strength=0.05)
                del temp_model

    print("Training Complete!")

if __name__ == "__main__":
    main()