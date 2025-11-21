import pygame
import random
import os
import numpy as np
import pandas as pd
import uuid
import argparse
import time
import shutil
from multiprocessing import Pool, cpu_count

# --- Configuration ---
IMG_SIZE = 64
GRID_SIZE = 16
CELL_SIZE = IMG_SIZE // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255) # Body
RED = (255, 0, 0)      # Food
GREEN = (0, 255, 0)    # Body
BLUE = (0, 0, 255)     # Head (New!)
GRAY = (100, 100, 100) # Dead Body
DARK_RED = (50, 0, 0)  # Dead Background

# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

class SnakeGame:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.head = [GRID_SIZE // 2, GRID_SIZE // 2]
        self.body = [[self.head[0], self.head[1]], 
                     [self.head[0], self.head[1]+1], 
                     [self.head[0], self.head[1]+2]]
        self.direction = ACTION_UP
        self.score = 0
        self.food = self._spawn_food()
        self.dead = False
        return self.render_surface()

    def _spawn_food(self):
        while True:
            food = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
            if food not in self.body:
                return food

    def step(self, action):
        if self.dead:
            return True, self.score

        # Prevent 180 turns
        if (action == ACTION_UP and self.direction == ACTION_DOWN) or \
           (action == ACTION_DOWN and self.direction == ACTION_UP) or \
           (action == ACTION_LEFT and self.direction == ACTION_RIGHT) or \
           (action == ACTION_RIGHT and self.direction == ACTION_LEFT):
            action = self.direction 
        
        self.direction = action
        x, y = self.head
        if action == ACTION_UP: y -= 1
        elif action == ACTION_DOWN: y += 1
        elif action == ACTION_LEFT: x -= 1
        elif action == ACTION_RIGHT: x += 1
        
        new_head = [x, y]

        # Check Collisions
        if (x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE or new_head in self.body):
            self.dead = True
            return True, self.score

        self.head = new_head
        self.body.insert(0, new_head)
        
        if self.head == self.food:
            self.score += 1
            self.food = self._spawn_food()
        else:
            self.body.pop()
            
        return False, self.score

    def render_surface(self):
        surface = pygame.Surface((IMG_SIZE, IMG_SIZE))
        
        # Colors
        bg_color = DARK_RED if self.dead else BLACK
        surface.fill(bg_color)
        
        # Draw Food (slightly smaller for style)
        rect_x = self.food[0] * CELL_SIZE
        rect_y = self.food[1] * CELL_SIZE
        # Inset by 1 pixel on all sides to create a gap
        rect = pygame.Rect(rect_x+1, rect_y+1, CELL_SIZE-2, CELL_SIZE-2)
        pygame.draw.rect(surface, RED, rect)
        
        # Draw Body
        for i, segment in enumerate(self.body):
            rect_x = segment[0] * CELL_SIZE
            rect_y = segment[1] * CELL_SIZE
            # Inset by 1 pixel to create grid lines
            rect = pygame.Rect(rect_x+1, rect_y+1, CELL_SIZE-2, CELL_SIZE-2)
            
            if self.dead:
                color = GRAY
            else:
                # Index 0 is Head -> BLUE, Others -> WHITE
                color = BLUE if i == 0 else WHITE
                
            pygame.draw.rect(surface, color, rect)
            
        return surface

def get_bot_action(game, epsilon=0.05):
    # Random move (suicide/exploration)
    if random.random() < epsilon:
        return random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])

    head = game.head
    food = game.food
    
    # Prefer moves towards food
    possible_moves = []
    if food[0] < head[0]: possible_moves.append(ACTION_LEFT)
    elif food[0] > head[0]: possible_moves.append(ACTION_RIGHT)
    if food[1] < head[1]: possible_moves.append(ACTION_UP)
    elif food[1] > head[1]: possible_moves.append(ACTION_DOWN)
    random.shuffle(possible_moves)
    
    all_moves = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
    remaining = [m for m in all_moves if m not in possible_moves]
    random.shuffle(remaining)
    possible_moves.extend(remaining)
    
    for move in possible_moves:
        if is_safe(game, move): 
            return move
            
    return possible_moves[0]

def is_safe(game, action):
    x, y = game.head
    if action == ACTION_UP: y -= 1
    elif action == ACTION_DOWN: y += 1
    elif action == ACTION_LEFT: x -= 1
    elif action == ACTION_RIGHT: x += 1
    
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE: return False
    if [x, y] in game.body[:-1]: return False
    
    if (action == ACTION_UP and game.direction == ACTION_DOWN) or \
       (action == ACTION_DOWN and game.direction == ACTION_UP) or \
       (action == ACTION_LEFT and game.direction == ACTION_RIGHT) or \
       (action == ACTION_RIGHT and game.direction == ACTION_LEFT): return False
       
    return True

def generate_worker(args):
    target_count, output_dir, worker_id = args
    os.environ["SDL_VIDEODRIVER"] = "dummy" 
    pygame.init()
    
    game = SnakeGame()
    records = []
    img_dir = os.path.join(output_dir, "images")
    
    current_surface = game.render_surface()
    current_img_name = f"{worker_id}_{uuid.uuid4().hex}.png"
    pygame.image.save(current_surface, os.path.join(img_dir, current_img_name))

    frames_generated = 0
    while frames_generated < target_count:
        action = get_bot_action(game, epsilon=0.05)
        is_dead, _ = game.step(action)
        
        next_surface = game.render_surface()
        next_img_name = f"{worker_id}_{uuid.uuid4().hex}.png"
        pygame.image.save(next_surface, os.path.join(img_dir, next_img_name))
        
        records.append({
            "current_image": current_img_name,
            "action": action,
            "next_image": next_img_name
        })
        
        frames_generated += 1
        
        if is_dead:
            game.reset()
            current_surface = game.render_surface()
            current_img_name = f"{worker_id}_{uuid.uuid4().hex}.png"
            pygame.image.save(current_surface, os.path.join(img_dir, current_img_name))
        else:
            current_surface = next_surface
            current_img_name = next_img_name

        if frames_generated % 2000 == 0:
             print(f"Worker {worker_id}: {frames_generated}/{target_count}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, f"metadata_{worker_id}.csv"), index=False)

def run_test_mode():
    pygame.init()
    screen = pygame.display.set_mode((512, 512))
    pygame.display.set_caption("Neural Snake - Visual Test (Blue Head + Gaps)")
    clock = pygame.time.Clock()
    game = SnakeGame()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r: game.reset()

        if not game.dead:
            action = get_bot_action(game, epsilon=0.05)
            game.step(action)
        else:
            time.sleep(0.5)
            game.reset()
            
        surface_64 = game.render_surface()
        surface_scaled = pygame.transform.scale(surface_64, (512, 512))
        screen.blit(surface_scaled, (0,0))
        pygame.display.flip()
        
        clock.tick(10) 
        
    pygame.quit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run visually to check bot behavior")
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--zip", action="store_true", help="Zip the output folder for Colab")
    args = parser.parse_args()

    if args.test:
        run_test_mode()
        return

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(os.path.join(args.output, "images"))

    num_workers = max(1, cpu_count() - 1) if args.parallel else 1
    frames_per_worker = args.count // num_workers
    worker_args = [(frames_per_worker, args.output, i) for i in range(num_workers)]
    
    print(f"Generating {args.count} frames (Blue Head + Grid Gaps) using {num_workers} workers...")
    start = time.time()
    
    if args.parallel:
        with Pool(num_workers) as p: p.map(generate_worker, worker_args)
    else:
        generate_worker(worker_args[0])

    print("Combining metadata...")
    all_dfs = []
    for i in range(num_workers):
        path = os.path.join(args.output, f"metadata_{i}.csv")
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path))
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(os.path.join(args.output, "metadata.csv"), index=False)
    
    print(f"Generation complete in {time.time()-start:.2f}s")

    if args.zip:
        print("Zipping dataset for Colab upload...")
        shutil.make_archive("snake_dataset", 'zip', args.output)
        print("Created snake_dataset.zip")

if __name__ == "__main__":
    main()