"""
Snake World Model - Play (DIAMOND-style Pixel-Space EDM)
Interactive inference script with real data initialization.
"""

import torch
import numpy as np
import cv2
import argparse
import os
import time
import glob
import random
from PIL import Image
from torchvision import transforms

# Import from src modules
from src.models.pixel_edm import PixelSpaceUNet, EDMSampler, CONTEXT_FRAMES, ACTION_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64


def get_random_real_context(data_dir):
    """
    Loads 4 consecutive frames from a random valid episode in the data directory.
    """
    print(f"Searching for episode data in {data_dir}...")
    
    # Try to find images
    all_files = glob.glob(os.path.join(data_dir, "*.png"))
    if not all_files:
        raise ValueError(f"No .png files found in {data_dir}. Please check path.")
    
    max_attempts = 100
    for _ in range(max_attempts):
        start_file = random.choice(all_files)
        basename = os.path.basename(start_file)
        
        # Example: 0_abc123_00000.png
        try:
            parts = basename.rsplit('_', 1)  # ["0_abc123", "00000.png"]
            ep_id = parts[0]
            frame_num = int(parts[1].split('.')[0])
        except:
            continue
        
        # Construct expected filenames for context
        context_files = []
        valid_sequence = True
        
        for i in range(CONTEXT_FRAMES):
            fname = f"{ep_id}_{frame_num + i:05d}.png"
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                valid_sequence = False
                break
            context_files.append(fpath)
        
        if valid_sequence:
            print(f"✅ Found valid seed sequence starting at: {basename}")
            
            # Load and process
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            tensors = []
            for f in context_files:
                img = Image.open(f).convert('RGB')
                tensors.append(transform(img))
            
            # Stack along channel dim: (12, 64, 64)
            context = torch.cat(tensors, dim=0)
            return context.unsqueeze(0).to(DEVICE)  # (1, 12, 64, 64)
    
    raise RuntimeError("Could not find a valid consecutive sequence of 4 frames after 100 attempts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/pixel_edm/best_model/model.pt")
    parser.add_argument("--data_dir", type=str, default="data_v5/images")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    
    print("Loading model...")
    model = PixelSpaceUNet(
        in_channels=3 + 3 * CONTEXT_FRAMES,
        out_channels=3,
        base_dim=128,
        cond_dim=512
    ).to(DEVICE)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print(f"✅ Model loaded from {args.model_path}")
    else:
        print(f"❌ Model not found: {args.model_path}")
        return
    
    model.eval()
    
    sampler = EDMSampler()
    
    # Load Real Context
    try:
        context = get_random_real_context(args.data_dir)
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Fallback: Creating blank context (Expect Red Screen / OOD behavior)")
        context = torch.zeros(1, 12, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    current_action = 0
    
    print("\n" + "="*40)
    print("🐍 SNAKE WORLD MODEL (DIAMOND-style)")
    print("="*40)
    print("WASD or Arrow Keys to move")
    print("R to reset (Loads new random seed from data), ESC to quit")
    print(f"CFG Scale: {args.cfg_scale}")
    print("="*40)
    
    cv2.namedWindow("Snake", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Snake", 512, 512)
    
    frame_time = 1.0 / args.fps
    
    while True:
        start = time.time()
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        
        if key == ord('r'):
            print("Resetting with new seed...")
            try:
                context = get_random_real_context(args.data_dir)
                print("Reset Complete!")
            except:
                print("Reset Failed.")
        
        if key in [ord('w'), 82]:
            current_action = 0
        elif key in [ord('s'), 84]:
            current_action = 1
        elif key in [ord('a'), 81]:
            current_action = 2
        elif key in [ord('d'), 83]:
            current_action = 3
        
        action = torch.zeros(1, ACTION_DIM, device=DEVICE)
        action[0, current_action] = 1.0
        
        with torch.no_grad():
            generated = sampler.sample(
                model,
                (1, 3, IMG_SIZE, IMG_SIZE),
                context,
                action,
                n_steps=args.steps,
                device=DEVICE,
                cfg_scale=args.cfg_scale
            )
        
        # Display
        img = generated[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        cv2.putText(img, action_names[current_action], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Snake", img)
        
        # Shift context: Drop oldest frame (first 3 chans), add new frame (generated)
        context = torch.cat([context[:, 3:], generated], dim=1)
        
        elapsed = time.time() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

