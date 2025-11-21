import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from PIL import Image
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from accelerate import Accelerator
from torchvision.utils import make_grid

# --- Configuration ---
IMG_SIZE = 64

class SnakeDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        # Transform: Scale to [-1, 1] for Diffusion
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load Images
        curr_path = os.path.join(self.img_dir, row['current_image'])
        next_path = os.path.join(self.img_dir, row['next_image'])
        
        curr_img = Image.open(curr_path).convert("RGB")
        next_img = Image.open(next_path).convert("RGB")
        
        curr_tensor = self.transform(curr_img)
        next_tensor = self.transform(next_img)
        
        action = torch.tensor(row['action'], dtype=torch.long)
        
        return curr_tensor, action, next_tensor

def evaluate(model, scheduler, dataloader, device, epoch, output_dir):
    """
    Runs inference on a few samples to visually check progress.
    Does NOT require playing the game.
    """
    model.eval()
    scheduler.set_timesteps(20) # Fast inference steps
    
    # Get a small batch of real data
    curr_img, action, real_next = next(iter(dataloader))
    curr_img = curr_img[:4].to(device) # Take top 4
    action = action[:4].to(device)
    real_next = real_next[:4].to(device)
    
    # Start from random noise
    noisy_latents = torch.randn_like(curr_img).to(device)
    
    # Denoising Loop
    for t in scheduler.timesteps:
        # 1. Concatenate Current Frame (Conditioning) + Noisy Latent
        # Input shape becomes (B, 6, 64, 64)
        model_input = torch.cat([noisy_latents, curr_img], dim=1)
        
        # 2. Predict Noise
        with torch.no_grad():
            noise_pred = model(model_input, t, class_labels=action).sample
            
        # 3. Step (Remove Noise)
        noisy_latents = scheduler.step(noise_pred, t, noisy_latents).prev_sample

    # Post-process for visualization
    # [-1, 1] -> [0, 1]
    generated = (noisy_latents / 2 + 0.5).clamp(0, 1)
    real = (real_next / 2 + 0.5).clamp(0, 1)
    input_view = (curr_img / 2 + 0.5).clamp(0, 1)
    
    # Stack images: Top Row = Input, Middle = AI Gen, Bottom = Real Target
    viz = torch.cat([input_view, generated, real], dim=0)
    grid = make_grid(viz, nrow=4, padding=2)
    
    # Save
    save_path = os.path.join(output_dir, "samples", f"epoch_{epoch:03d}.png")
    transforms.ToPILImage()(grid).save(save_path)
    model.train()

def main():
    parser = argparse.ArgumentParser(description="Train Neural Snake World Model")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data folder containing images/ and metadata.csv")
    parser.add_argument("--output_dir", type=str, default="output", help="Where to save model and samples")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every X epochs")
    args = parser.parse_args()

    # Setup Accelerate (Handles GPU placement & mixed precision automatically)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    # Create Dirs
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # 1. Load Dataset
    dataset = SnakeDataset(
        csv_file=os.path.join(args.data_path, "metadata.csv"),
        img_dir=os.path.join(args.data_path, "images")
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 2. Initialize Model
    # Note: in_channels=6 because we stack (Noisy Latent [3] + Current Frame [3])
    model = UNet2DModel(
        sample_size=IMG_SIZE,
        in_channels=6, 
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D", 
            "DownBlock2D", 
            "AttnDownBlock2D", 
            "AttnDownBlock2D"
        ),
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D"
        ),
        # --- FIX IS HERE ---
        # Changed "identity" to "timestep" so it learns embeddings for 0,1,2,3
        class_embed_type="timestep", 
        num_class_embeds=4 
    )
    
    # 3. Schedulers
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    eval_scheduler = DDIMScheduler(num_train_timesteps=1000) # Faster for testing

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Prepare everything with Accelerate
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training Loop
    print(f"Starting training on {accelerator.device}...")
    print(f"Dataset size: {len(dataset)}")
    
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for curr_img, action, next_img in dataloader:
            # Sample noise
            noise = torch.randn_like(next_img)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (curr_img.shape[0],), device=next_img.device).long()
            
            # Add noise to the TARGET image (next_frame)
            noisy_next_img = noise_scheduler.add_noise(next_img, noise, timesteps)
            
            # CONDITIONING: Concatenate (Noisy Next) + (Clean Current)
            # This tells the model: "Given this current state, what noise was added to the next state?"
            model_input = torch.cat([noisy_next_img, curr_img], dim=1)
            
            # Predict
            noise_pred = model(model_input, timesteps, class_labels=action).sample
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backprop
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(mse=loss.item())
            global_step += 1

        # --- Validation & Saving ---
        if accelerator.is_main_process:
            if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
                # 1. Run Visual Evaluation
                print(f"Running visual evaluation for Epoch {epoch}...")
                evaluate(model, eval_scheduler, dataloader, accelerator.device, epoch, args.output_dir)
                
                # 2. Save Checkpoint
                save_path = os.path.join(args.output_dir, "checkpoints", f"model_epoch_{epoch}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

    print("Training Complete!")

if __name__ == "__main__":
    main()