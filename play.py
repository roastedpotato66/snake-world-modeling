import torch
from diffusers import UNet2DModel, DDIMScheduler
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
import keyboard
import time
import os

# --- Config ---
MODEL_PATH = "output/pro/best_model" 
IMG_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model ---
print(f"Loading Neural Engine from {MODEL_PATH}...")

model = UNet2DModel.from_pretrained("output/pro/best_model").to(DEVICE)
model.to(DEVICE)
model.eval()

scheduler = DDIMScheduler(num_train_timesteps=1000)
# INCREASED STEPS: 15 is more stable than 10.
scheduler.set_timesteps(15) 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def cleanup_frame(tensor_frame):
    """
    THE DRIFT FIX:
    Takes the AI's 'float' output (which has tiny noise) and snaps it 
    to the nearest valid game colors. This resets the noise counter to 0.
    """
    # 1. Denormalize to [0, 1]
    img = (tensor_frame / 2 + 0.5).clamp(0, 1)
    
    # 2. Hard Thresholding (Quantization)
    # Snake is high contrast. We can just round the values.
    # Everything > 0.5 becomes 1.0 (White/Color), < 0.5 becomes 0.0 (Black)
    # This removes the gray "sludge" noise.
    img = torch.round(img) 
    
    # 3. Renormalize to [-1, 1] for the model
    return (img - 0.5) * 2

def tensor_to_cv2(tensor):
    image = (tensor / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def get_action():
    if keyboard.is_pressed('up'): return 0
    if keyboard.is_pressed('down'): return 1
    if keyboard.is_pressed('left'): return 2
    if keyboard.is_pressed('right'): return 3
    return None

def main():
    print("Starting Neural Snake... (Press 'Q' to quit)")
    
    # Seed with a real image to ensure clean start
    if os.path.exists("data/images"):
        start_img_name = os.listdir("data/images")[0]
        pil_img = Image.open(f"data/images/{start_img_name}").convert("RGB")
        current_frame = transform(pil_img).unsqueeze(0).to(DEVICE)
    else:
        current_frame = torch.zeros((1, 3, 64, 64)).to(DEVICE)
    
    last_action = 0
    
    while True:
        action = get_action()
        if action is None: action = last_action
        last_action = action
        
        if keyboard.is_pressed('q'): break

        action_tensor = torch.tensor([action], device=DEVICE)
        noise = torch.randn_like(current_frame)
        latents = noise
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                model_input = torch.cat([latents, current_frame], dim=1)
                noise_pred = model(model_input, t, class_labels=action_tensor).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # --- THE MAGIC FIX ---
        # Instead of using the raw output, we clean it up.
        next_frame_clean = cleanup_frame(latents)
        
        # Render
        visual = tensor_to_cv2(next_frame_clean[0])
        visual_large = cv2.resize(visual, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Neural Snake (AI Generated)", visual_large)
        cv2.waitKey(1)
        
        # Update Loop
        current_frame = next_frame_clean
        
        # Death Check
        avg_color = np.mean(visual, axis=(0,1))
        # If we see too much Red (Death Color), reset
        if avg_color[2] > 50 and avg_color[1] < 50: 
            print("Death Detected! Resetting...")
            time.sleep(1)
            if os.path.exists("data/images"):
                random_img = np.random.choice(os.listdir("data/images"))
                pil_img = Image.open(f"data/images/{random_img}").convert("RGB")
                current_frame = transform(pil_img).unsqueeze(0).to(DEVICE)

if __name__ == "__main__":
    main()