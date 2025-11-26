# data_gen.py v3
import pygame
import random
import os
import pandas as pd
import uuid
import argparse
import time
import shutil
from multiprocessing import Pool, cpu_count
import numpy as np

# --- Configuration ---
IMG_SIZE = 64
GRID_SIZE = 16
CELL_SIZE = IMG_SIZE // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255) # Body
RED = (255, 0, 0)       # Food
BLUE = (0, 0, 255)      # Head
GRAY = (50, 50, 50)     # Dead Body
DARK_RED = (40, 0, 0)   # Dead Background

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
        """
        Returns: (is_dead, is_eating, score)
        """
        if self.dead:
            return True, False, self.score

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
            return True, False, self.score

        self.head = new_head
        self.body.insert(0, new_head)
        
        is_eating = False
        if self.head == self.food:
            self.score += 1
            self.food = self._spawn_food()
            is_eating = True
        else:
            self.body.pop()
            
        return False, is_eating, self.score

    def render_surface(self):
        surface = pygame.Surface((IMG_SIZE, IMG_SIZE))
        
        # Colors
        bg_color = DARK_RED if self.dead else BLACK
        surface.fill(bg_color)
        
        # Draw Food
        rect_x = self.food[0] * CELL_SIZE
        rect_y = self.food[1] * CELL_SIZE
        rect = pygame.Rect(rect_x+1, rect_y+1, CELL_SIZE-2, CELL_SIZE-2)
        pygame.draw.rect(surface, RED, rect)
        
        # Draw Body
        for i, segment in enumerate(self.body):
            rect_x = segment[0] * CELL_SIZE
            rect_y = segment[1] * CELL_SIZE
            rect = pygame.Rect(rect_x+1, rect_y+1, CELL_SIZE-2, CELL_SIZE-2)
            
            if self.dead:
                color = GRAY
            else:
                color = BLUE if i == 0 else WHITE
                
            pygame.draw.rect(surface, color, rect)
            
        return surface

def get_bot_action(game, epsilon=0.3):
    # to reach the food. We need "competent" gameplay to generate "eating" frames.
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
    
    # Prevent immediate 180 (neck break)
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
    
    # Episode ID ensures we don't stack frames across game resets
    episode_id = f"{worker_id}_{uuid.uuid4().hex[:8]}"
    frame_idx = 0
    
    frames_generated = 0
    
    # Save initial state
    current_surface = game.render_surface()
    img_name = f"{episode_id}_{frame_idx:05d}.png"
    pygame.image.save(current_surface, os.path.join(img_dir, img_name))

    while frames_generated < target_count:
        # 1. Decide Action
        action = get_bot_action(game, epsilon=0.3)
        
        # 2. Step Game
        is_dead, is_eating, _ = game.step(action)
        
        # 3. Save Record (State t, Action t)
        records.append({
            "episode_id": episode_id,
            "frame_number": frame_idx,
            "image_file": img_name,
            "action": action,
            "is_dead": is_dead,
            "is_eating": is_eating, # <--- Tracking this now
        })
        
        frames_generated += 1
        frame_idx += 1
        
        # 4. Render New State
        next_surface = game.render_surface()
        
        if is_dead:
            # Save the death frame
            img_name = f"{episode_id}_{frame_idx:05d}.png"
            pygame.image.save(next_surface, os.path.join(img_dir, img_name))
            
            # Save one last record for the death state
            records.append({
                "episode_id": episode_id,
                "frame_number": frame_idx,
                "image_file": img_name,
                "action": 0,
                "is_dead": True,
                "is_eating": False
            })

            # RESET GAME
            game.reset()
            episode_id = f"{worker_id}_{uuid.uuid4().hex[:8]}"
            frame_idx = 0
            
            # Save new initial state
            current_surface = game.render_surface()
            img_name = f"{episode_id}_{frame_idx:05d}.png"
            pygame.image.save(current_surface, os.path.join(img_dir, img_name))
            
        else:
            # Continue episode
            img_name = f"{episode_id}_{frame_idx:05d}.png"
            pygame.image.save(next_surface, os.path.join(img_dir, img_name))

        if frames_generated % 5000 == 0:
             print(f"Worker {worker_id}: {frames_generated}/{target_count}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, f"metadata_{worker_id}.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200000)
    parser.add_argument("--output", type=str, default="data_v5")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(os.path.join(args.output, "images"))

    num_workers = min(args.workers, cpu_count())
    frames_per_worker = args.count // num_workers
    worker_args = [(frames_per_worker, args.output, i) for i in range(num_workers)]
    
    print(f"Generating {args.count} frames... (Workers: {num_workers})")
    start = time.time()
    
    if args.parallel:
        with Pool(num_workers) as p: 
            p.map(generate_worker, worker_args)
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
        final_df = final_df.sort_values(by=['episode_id', 'frame_number'])
        final_df.to_csv(os.path.join(args.output, "metadata.csv"), index=False)
    
    print(f"Generation complete in {time.time()-start:.2f}s")
    print(f"Dataset saved to: {args.output}")

if __name__ == "__main__":
    main()