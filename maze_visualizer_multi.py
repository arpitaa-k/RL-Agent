# """
# Maze Multi-Algorithm FINAL Comparative Simulation — 3 Agents in One Maze (25x25)

# Key Changes:
# 1. All three agents now start in the SAME CELL (0, 0), slightly offset for visibility.
# 2. Grid size is 25x25.
# 3. CELL size and AGENT_RADIUS are reduced for visualization.
# 4. Max steps increased to 400 for larger maze navigation.
# 5. Includes the ARROW_FONT_SIZE constant fix.

# Controls (in Pygame window):
# - SPACE : Start the simultaneous animation from the start area.
# - ESC or close window : Quit

# Requires pygame: pip install pygame
# """
# import pygame
# import sys
# import time
# import random
# import math
# import numpy as np

# # --- ENVIRONMENT & RL ALGORITHMS ---

# class MazeEnv:
#     def __init__(self, grid, start_rect, goal, traps=None, slip_prob=0.0, step_penalty=-0.01):
#         self.grid = np.array(grid)
#         self.rows, self.cols = self.grid.shape
#         self.start_rect = start_rect
#         self.goal = tuple(goal)
#         self.traps = set(traps) if traps else set()
#         self.slip_prob = slip_prob
#         self.step_penalty = step_penalty
#         self.actions = [(-1,0),(0,1),(1,0),(0,-1)]
#         self.reset()

#     def in_bounds(self, s):
#         r,c = s
#         return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] == 0

#     def step(self, action, current_pos):
#         actual_action = action
#         if random.random() < self.slip_prob:
#             actual_action = random.randrange(4) 
            
#         dr, dc = self.actions[actual_action]
#         nr, nc = current_pos[0] + dr, current_pos[1] + dc
        
#         new_pos = (nr, nc)
#         if not self.in_bounds(new_pos):
#             new_pos = current_pos 
            
#         reward = self.step_penalty
#         done = False
        
#         if new_pos == self.goal:
#             reward = 1.0
#             done = True
#         elif new_pos in self.traps:
#             reward = -1.0 
#             done = True
            
#         return new_pos, reward, done

#     def reset(self, start_idx=0):
#         # This will now always return the same coordinate (0, 0) during training
#         self.pos = self.start_rect[start_idx] 
#         return self.pos

# def state_to_idx(state, shape):
#     r,c = state
#     return r * shape[1] + c

# def epsilon_greedy(Q, s_idx, nA, eps):
#     if random.random() < eps:
#         return random.randrange(nA)
#     qvals = Q[s_idx]
#     maxv = max(qvals)
#     candidates = [i for i,v in enumerate(qvals) if v == maxv]
#     return random.choice(candidates)

# def softmax_action_from_Q(Q, s_idx, nA, tau=1.0):
#     q = np.array(Q[s_idx], dtype=float)
#     exps = np.exp((q - np.max(q)) / max(tau, 1e-6))
#     probs = exps / np.sum(exps)
#     if np.isnan(probs).any() or np.isinf(probs).any():
#         probs = np.ones(nA) / nA
#     return np.random.choice(range(nA), p=probs)

# def run_q_learning(env, episodes, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=None, max_steps=400, policy='eps'):
#     nS = env.rows * env.cols
#     nA = 4
#     Q = [[0.0]*nA for _ in range(nS)]
    
#     num_starts = len(env.start_rect) 
    
#     for ep in range(episodes):
#         # We cycle through start_rect to pick a policy_data index, 
#         # but since all start_rect points are now (0,0), 
#         # this ensures each agent is equally trained starting at (0,0)
#         start_idx = ep % num_starts
#         s = env.reset(start_idx=start_idx) 
#         s_idx = state_to_idx(s, env.grid.shape)
        
#         for t in range(max_steps):
#             if policy == 'eps':
#                 a = epsilon_greedy(Q, s_idx, nA, eps)
#             else:
#                 a = softmax_action_from_Q(Q, s_idx, nA, tau=eps)
                
#             s2, r, done = env.step(a, s) 
#             s2_idx = state_to_idx(s2, env.grid.shape)
            
#             best_next = max(Q[s2_idx]) 
#             Q[s_idx][a] += alpha * (r + gamma * best_next - Q[s_idx][a])
#             s, s_idx = s2, s2_idx
#             if done:
#                 break
#         if eps_decay:
#             eps = eps_decay(ep, eps)
#     return Q

# def run_sarsa(env, episodes, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=None, max_steps=400, policy='eps'):
#     nS = env.rows * env.cols
#     nA = 4
#     Q = [[0.0]*nA for _ in range(nS)]
    
#     num_starts = len(env.start_rect)
    
#     for ep in range(episodes):
#         start_idx = ep % num_starts
#         s = env.reset(start_idx=start_idx)
#         s_idx = state_to_idx(s, env.grid.shape)
        
#         if policy == 'eps':
#             a = epsilon_greedy(Q, s_idx, nA, eps)
#         else:
#             a = softmax_action_from_Q(Q, s_idx, nA, tau=eps)
            
#         for t in range(max_steps):
#             s2, r, done = env.step(a, s)
#             s2_idx = state_to_idx(s2, env.grid.shape)
            
#             if policy == 'eps':
#                 a2 = epsilon_greedy(Q, s2_idx, nA, eps)
#             else:
#                 a2 = softmax_action_from_Q(Q, s2_idx, nA, tau=eps)
                
#             Q[s_idx][a] += alpha * (r + gamma * Q[s2_idx][a2] - Q[s_idx][a])
#             s, s_idx, a = s2, s2_idx, a2
#             if done:
#                 break
#         if eps_decay:
#             eps = eps_decay(ep, eps)
#     return Q

# def extract_greedy_policy(Q, env):
#     policy = {}
#     for r in range(env.rows):
#         for c in range(env.cols):
#             s = (r,c)
#             idx = state_to_idx(s, env.grid.shape)
#             if env.grid[r,c] == 1 or s == env.goal or s in env.traps:
#                 policy[s] = None
#                 continue
            
#             best_action = max(range(4), key=lambda a: Q[idx][a])
#             policy[s] = best_action
            
#     return policy

# # ----------------------------------------------------
# # *** 25x25 GRID FUNCTION ***
# # ----------------------------------------------------
# def build_complex_labyrinth_25x25():
#     # 25x25 Highly Branched Labyrinth grid
#     grid_str = [
#         "S000000000000000000000000",
#         "0111110111110111110111110",
#         "0100000000000000000000010",
#         "0101111111111111111111010",
#         "0000000000000000000000010",
#         "0111111111111111111111010",
#         "0000000000000000000000010",
#         "0111111111111111111111110",
#         "0000000000000000000000000",
#         "1111111111111111111111110",
#         "0000000000000000000000000",
#         "0111111111111111111111110",
#         "0000000000000000000000000",
#         "0111111111111111111111110",
#         "0000000000000000000000000",
#         "0111111111111111111111110",
#         "0000000000000000000000000",
#         "0111111111111111111111110",
#         "0000000000000000000000000",
#         "0111111111111111111111110",
#         "0000000000000000000000000",
#         "0111111111111111111111110",
#         "0000000000000000000000000",
#         "0111111111111111111111110",
#         "000000000000G0000T0000000"
#     ]
#     grid = np.array([[1 if c == '1' else 0 for c in row] for row in grid_str])
    
#     # Start area is the top-left corner (0, 0)
#     start_r, start_c = 0, 0
#     # All three agents start at the same (0, 0) coordinate
#     start_rect = [(start_r, start_c), (start_r, start_c), (start_r, start_c)] 
    
#     # Goal (G) is at the bottom right corner (24, 24)
#     goal = (24, grid_str[24].index('G'))
    
#     traps = []
#     for r, row in enumerate(grid_str):
#         for c, char in enumerate(row):
#             if char == 'T':
#                 traps.append((r, c)) # Trap (T) is at (24, 18)
                
#     slip_prob = 0.05
#     env = MazeEnv(grid, start_rect, goal, traps=traps, slip_prob=slip_prob)
#     return env

# # --- PYGAME VISUALIZATION ---

# CELL = 25 # Reduced cell size for 25x25 grid
# MARGIN = 1
# AGENT_RADIUS = 5 
# ARROW_FONT_SIZE = 12 
# ARROW_MAP = {0:'\u2191', 1:'\u2192', 2:'\u2193', 3:'\u2190'} 

# # Colors for the three agents
# COLOR_BG = (255, 255, 255)
# COLOR_WALL = (0, 0, 0)           
# COLOR_OUTLINE = (0, 0, 0)
# COLOR_Q = (255, 0, 0)            # Red (Q-learning)
# COLOR_SARSA = (0, 150, 0)        # Green (SARSA)
# COLOR_SOFTMAX = (0, 0, 255)      # Blue (Softmax-Q)
# COLOR_TEXT = (0, 0, 0)           

# def draw_text_in_cell(screen, text, font, r, c, color, offset_y=0):
#     x = c * (CELL + MARGIN) + MARGIN
#     y = r * (CELL + MARGIN) + MARGIN
#     text_surface = font.render(text, True, color)
#     text_rect = text_surface.get_rect(center=(x + CELL // 2, y + CELL // 2 + offset_y))
#     screen.blit(text_surface, text_rect)

# def draw_grid(screen, env, policies=None, animating=False):
#     rows, cols = env.rows, env.cols
#     screen.fill(COLOR_BG)

#     font_small = pygame.font.SysFont(None, 14)

#     for r in range(rows):
#         for c in range(cols):
#             x = c * (CELL + MARGIN) + MARGIN
#             y = r * (CELL + MARGIN) + MARGIN
#             rect = pygame.Rect(x, y, CELL, CELL)
#             s = (r,c)
            
#             # Draw Cell Background
#             if env.grid[r,c] == 1:
#                 pygame.draw.rect(screen, COLOR_WALL, rect)
#             else:
#                 pygame.draw.rect(screen, COLOR_BG, rect)
            
#             # Draw Special Markers/Labels (Black Outlines)
#             # Only mark (0, 0) since all agents start there
#             if s == env.start_rect[0]: 
#                 pygame.draw.rect(screen, COLOR_OUTLINE, rect.inflate(-3,-3), 2)
#                 draw_text_in_cell(screen, "S", font_small, r, c, COLOR_TEXT)
            
#             if s == env.goal:
#                 pygame.draw.rect(screen, COLOR_OUTLINE, rect.inflate(-3,-3), 2)
#                 draw_text_in_cell(screen, "G", font_small, r, c, COLOR_TEXT)
            
#             if s in env.traps:
#                 pygame.draw.rect(screen, COLOR_OUTLINE, rect.inflate(-3,-3), 2)
#                 draw_text_in_cell(screen, "T", font_small, r, c, COLOR_TEXT)
                
#             # Draw Policy Arrow only when not animating
#             if policies and not animating:
#                 policy = policies[0][0] 
#                 if s in policy and policy[s] is not None:
#                     action = policy[s]
#                     arrow_char = ARROW_MAP.get(action, '?')
#                     font_arrow = pygame.font.SysFont(None, ARROW_FONT_SIZE, bold=True)
#                     draw_text_in_cell(screen, arrow_char, font_arrow, r, c, COLOR_TEXT)


#     # Draw legend and instructions below the grid
#     font = pygame.font.SysFont(None, 24, bold=True)
#     font_small = pygame.font.SysFont(None, 18)
    
#     legend_y = env.rows*(CELL+MARGIN) + 8
    
#     # Draw Legend
#     text = font.render("Algorithms:", True, COLOR_TEXT)
#     screen.blit(text, (10, legend_y))
    
#     pygame.draw.circle(screen, COLOR_Q, (110, legend_y + 12), AGENT_RADIUS)
#     text_q = font_small.render("Q-learning (Red)", True, COLOR_Q)
#     screen.blit(text_q, (125, legend_y + 4))

#     pygame.draw.circle(screen, COLOR_SARSA, (250, legend_y + 12), AGENT_RADIUS)
#     text_sarsa = font_small.render("SARSA (Green)", True, COLOR_SARSA)
#     screen.blit(text_sarsa, (265, legend_y + 4))

#     pygame.draw.circle(screen, COLOR_SOFTMAX, (390, legend_y + 12), AGENT_RADIUS)
#     text_softmax = font_small.render("Softmax-Q (Blue)", True, COLOR_SOFTMAX)
#     screen.blit(text_softmax, (405, legend_y + 4))
    
#     # Draw Instructions
#     inst_y = legend_y + 25
#     inst_text = font.render("Press SPACE to Run Simulation | ESC to Quit", True, COLOR_TEXT)
#     screen.blit(inst_text, (10, inst_y))


# def follow_greedy_path(env, policy, start_pos, max_steps=1500):
#     s = start_pos
#     path = [s]
#     for _ in range(max_steps):
#         if s == env.goal or s in env.traps:
#             break
#         a = policy.get(s, None)
#         if a is None:
#             break
        
#         dr,dc = env.actions[a]
#         ns = (s[0]+dr, s[1]+dc)
        
#         if not env.in_bounds(ns):
#             break 
            
#         path.append(ns)
#         s = ns
        
#     return path

# def animate_all_agents(screen, env, all_paths, colors, clock, speed=16):
    
#     agent_positions = [path[0] for path in all_paths]
#     path_indices = [0] * len(all_paths)
    
#     max_path_len = max(len(path) for path in all_paths)
    
#     steps = speed
    
#     # Define offsets for agents within a cell for all drawing operations
#     offsets = [(-CELL // 4, -CELL // 4), (0, 0), (CELL // 4, CELL // 4)]
    
#     for i in range(max_path_len - 1):
        
#         interpolations = []
#         for j in range(len(all_paths)):
#             offset_x, offset_y = offsets[j] # Use agent-specific offset
            
#             if path_indices[j] < len(all_paths[j]) - 1:
#                 s = all_paths[j][path_indices[j]]
#                 s2 = all_paths[j][path_indices[j] + 1]
                
#                 # Calculate start and end points using the agent's offset
#                 start_x = s[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
#                 start_y = s[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
#                 end_x = s2[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
#                 end_y = s2[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
                
#                 interpolations.append({'start': (start_x, start_y), 'end': (end_x, end_y), 'active': True})
#             else:
#                 s = all_paths[j][-1]
#                 x = s[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
#                 y = s[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
#                 interpolations.append({'start': (x, y), 'end': (x, y), 'active': False})

#         for t in range(steps):
#             draw_grid(screen, env, animating=True)
            
#             for j in range(len(all_paths)):
#                 interp = interpolations[j]
                
#                 ix = interp['start'][0] + (interp['end'][0] - interp['start'][0]) * (t+1)/steps
#                 iy = interp['start'][1] + (interp['end'][1] - interp['start'][1]) * (t+1)/steps
                
#                 pygame.draw.circle(screen, colors[j], (int(ix), int(iy)), AGENT_RADIUS)
            
#             pygame.display.flip()
#             clock.tick(60)
            
#         for j in range(len(all_paths)):
#             if path_indices[j] < len(all_paths[j]) - 1:
#                 path_indices[j] += 1
                
#     # Final draw after all paths are complete
#     draw_grid(screen, env, policies=None, animating=False) 
#     for j in range(len(all_paths)):
#         offset_x, offset_y = offsets[j]
#         last = all_paths[j][-1]
#         lx = last[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
#         ly = last[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
#         pygame.draw.circle(screen, colors[j], (int(lx), int(ly)), AGENT_RADIUS)
        
#     pygame.display.flip()


# # -----------------------------
# # Main Execution
# # -----------------------------
# def main():
#     env = build_complex_labyrinth_25x25()
    
#     # --- Training ---
#     print("----------------------------------------------------------")
#     print("🚀 Training Algorithms (4,000 episodes for comparative difference)")
#     print("----------------------------------------------------------")
    
#     TRAINING_EPISODES = 1000
#     def eps_decay(ep, e): return max(0.01, e * (1 - ep/2000)) 
#     random.seed(0); np.random.seed(0)

#     Q_q = run_q_learning(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=eps_decay, policy='eps')
#     policy_q = extract_greedy_policy(Q_q, env)
#     print("✅ Trained: Q-learning (Red)")

#     Q_sarsa = run_sarsa(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=eps_decay, policy='eps')
#     policy_sarsa = extract_greedy_policy(Q_sarsa, env)
#     print("✅ Trained: SARSA (Green)")

#     Q_q_soft = run_q_learning(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=2.0, eps_decay=eps_decay, policy='softmax')
#     policy_q_soft = extract_greedy_policy(Q_q_soft, env)
#     print("✅ Trained: Q-learning (Softmax) (Blue)")
    
#     print("----------------------------------------------------------")

#     # Group policies and colors
#     policy_data = [
#         # All agents now share the start position (0, 0) in the policy_data structure
#         (policy_q, COLOR_Q, env.start_rect[0]),       
#         (policy_sarsa, COLOR_SARSA, env.start_rect[1]), 
#         (policy_q_soft, COLOR_SOFTMAX, env.start_rect[2]) 
#     ]
    
#     # --- Calculate Paths (used for animation) ---
#     all_paths = [follow_greedy_path(env, p, start) for p, c, start in policy_data]
    
#     # --- Pygame Initialization ---
#     rows, cols = env.rows, env.cols
#     win_w = cols * (CELL + MARGIN) + MARGIN
#     win_h = rows * (CELL + MARGIN) + MARGIN + 60

#     pygame.init()
#     screen = pygame.display.set_mode((win_w, win_h))
#     pygame.display.set_caption("RL Triple Agent Comparative Simulation (25x25) - Shared Start")
#     clock = pygame.time.Clock()
    
#     draw_grid(screen, env, policies=policy_data, animating=False) 
    
#     # Draw initial agent positions with offsets
#     initial_positions = []
#     # Offsets: Red (Top-Left), Green (Center), Blue (Bottom-Right)
#     offsets = [(-CELL // 4, -CELL // 4), (0, 0), (CELL // 4, CELL // 4)]
    
#     for i, (p, color, start_pos) in enumerate(policy_data):
#         # Center of the cell
#         sx = start_pos[1] * (CELL + MARGIN) + MARGIN + CELL//2
#         sy = start_pos[0] * (CELL + MARGIN) + MARGIN + CELL//2
        
#         # Apply offset for visual separation
#         offset_x, offset_y = offsets[i]
        
#         pygame.draw.circle(screen, color, (sx + offset_x, sy + offset_y), AGENT_RADIUS)
#         initial_positions.append((sx + offset_x, sy + offset_y, color))
    
#     pygame.display.flip()

#     # --- Main Pygame Loop ---
#     simulation_ran = False
    
#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
#                 pygame.quit(); sys.exit(0)
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_SPACE:
#                     if not simulation_ran:
#                         # Run the simulation animation
#                         print("Starting simulation...")
#                         colors = [c for p, c, s in policy_data]
#                         # animate_all_agents now handles the per-agent offset for smooth movement
#                         animate_all_agents(screen, env, all_paths, colors, clock, speed=16)
#                         simulation_ran = True
#                     else:
#                         # Reset agents back to initial offset position
#                         draw_grid(screen, env, policies=policy_data, animating=False)
#                         for sx, sy, color in initial_positions:
#                             pygame.draw.circle(screen, color, (sx, sy), AGENT_RADIUS)
#                         pygame.display.flip()
#                         simulation_ran = False

#         clock.tick(30)

# if __name__ == '__main__':
#     main()


"""
Maze Multi-Algorithm FINAL Comparative Simulation — 5 Agents in One Maze (25x25)

Key Changes:
1. All five agents now start in the SAME CELL (0, 0), offset for visibility.
2. Adds Double Q-Learning and Deep Q-Learning (DQN).
3. Grid size is 25x25.
4. CELL size and AGENT_RADIUS are reduced for visualization.
5. Max steps increased to 400 for larger maze navigation.
6. Includes the ARROW_FONT_SIZE constant fix.

Controls (in Pygame window):
- SPACE : Start the simultaneous animation from the start area.
- ESC or close window : Quit

Requires:
- pygame: pip install pygame
- torch:  pip install torch
"""
import pygame
import sys
import time
import random
import math
import numpy as np
from collections import deque

# --- New Imports for DQN ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- ENVIRONMENT & RL ALGORITHMS ---

class MazeEnv:
    def __init__(self, grid, start_rect, goal, traps=None, slip_prob=0.0, step_penalty=-0.01):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.start_rect = start_rect # This will be a list of (0,0)
        self.goal = tuple(goal)
        self.traps = set(traps) if traps else set()
        self.slip_prob = slip_prob
        self.step_penalty = step_penalty
        self.actions = [(-1,0),(0,1),(1,0),(0,-1)] # 0:Up, 1:Right, 2:Down, 3:Left
        self.reset()

    def in_bounds(self, s):
        r,c = s
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] == 0

    def step(self, action, current_pos):
        actual_action = action
        # Handle stochasticity (slipping)
        if random.random() < self.slip_prob:
            actual_action = random.randrange(4) 
            
        dr, dc = self.actions[actual_action]
        nr, nc = current_pos[0] + dr, current_pos[1] + dc
        
        new_pos = (nr, nc)
        if not self.in_bounds(new_pos):
            new_pos = current_pos # Bumped into wall, stay in place
            
        reward = self.step_penalty
        done = False
        
        if new_pos == self.goal:
            reward = 1.0
            done = True
        elif new_pos in self.traps:
            reward = -1.0 
            done = True
            
        return new_pos, reward, done

    def reset(self, start_idx=0):
        # All agents start at the same coordinate (0, 0)
        self.pos = self.start_rect[start_idx] 
        return self.pos

def state_to_idx(state, shape):
    r,c = state
    return r * shape[1] + c

# --- Policy Functions ---

def epsilon_greedy(Q, s_idx, nA, eps):
    if random.random() < eps:
        return random.randrange(nA)
    qvals = Q[s_idx]
    maxv = max(qvals)
    # Handle ties randomly
    candidates = [i for i,v in enumerate(qvals) if v == maxv]
    return random.choice(candidates)

def softmax_action_from_Q(Q, s_idx, nA, tau=1.0):
    q = np.array(Q[s_idx], dtype=float)
    # Softmax probability calculation
    exps = np.exp((q - np.max(q)) / max(tau, 1e-6))
    probs = exps / np.sum(exps)
    if np.isnan(probs).any() or np.isinf(probs).any():
        probs = np.ones(nA) / nA # Fallback to uniform
    return np.random.choice(range(nA), p=probs)

# --- 1. Q-Learning ---

def run_q_learning(env, episodes, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=None, max_steps=400, policy='eps'):
    nS = env.rows * env.cols
    nA = 4
    Q = [[0.0]*nA for _ in range(nS)]
    
    num_starts = len(env.start_rect) 
    
    for ep in range(episodes):
        start_idx = ep % num_starts
        s = env.reset(start_idx=start_idx) 
        s_idx = state_to_idx(s, env.grid.shape)
        
        for t in range(max_steps):
            if policy == 'eps':
                a = epsilon_greedy(Q, s_idx, nA, eps)
            else:
                a = softmax_action_from_Q(Q, s_idx, nA, tau=eps)
                
            s2, r, done = env.step(a, s) 
            s2_idx = state_to_idx(s2, env.grid.shape)
            
            # Q-Learning update: Uses max(Q(s', a'))
            best_next = max(Q[s2_idx]) 
            Q[s_idx][a] += alpha * (r + gamma * best_next - Q[s_idx][a])
            
            s, s_idx = s2, s2_idx
            if done:
                break
        if eps_decay:
            eps = eps_decay(ep, eps)
    return Q

# --- 2. SARSA ---

def run_sarsa(env, episodes, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=None, max_steps=400, policy='eps'):
    nS = env.rows * env.cols
    nA = 4
    Q = [[0.0]*nA for _ in range(nS)]
    
    num_starts = len(env.start_rect)
    
    for ep in range(episodes):
        start_idx = ep % num_starts
        s = env.reset(start_idx=start_idx)
        s_idx = state_to_idx(s, env.grid.shape)
        
        if policy == 'eps':
            a = epsilon_greedy(Q, s_idx, nA, eps)
        else:
            a = softmax_action_from_Q(Q, s_idx, nA, tau=eps)
            
        for t in range(max_steps):
            s2, r, done = env.step(a, s)
            s2_idx = state_to_idx(s2, env.grid.shape)
            
            # SARSA update: Uses Q(s', a')
            if policy == 'eps':
                a2 = epsilon_greedy(Q, s2_idx, nA, eps)
            else:
                a2 = softmax_action_from_Q(Q, s2_idx, nA, tau=eps)
                
            Q[s_idx][a] += alpha * (r + gamma * Q[s2_idx][a2] - Q[s_idx][a])
            
            s, s_idx, a = s2, s2_idx, a2
            if done:
                break
        if eps_decay:
            eps = eps_decay(ep, eps)
    return Q

# --- 3. Double Q-Learning (NEW) ---

def run_double_q_learning(env, episodes, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=None, max_steps=400):
    nS = env.rows * env.cols
    nA = 4
    Q_A = [[0.0]*nA for _ in range(nS)]
    Q_B = [[0.0]*nA for _ in range(nS)]
    
    num_starts = len(env.start_rect)
    
    for ep in range(episodes):
        start_idx = ep % num_starts
        s = env.reset(start_idx=start_idx)
        
        for t in range(max_steps):
            s_idx = state_to_idx(s, env.grid.shape)
            
            # Policy derived from the sum of both Q-tables
            Q_sum = [[Q_A[i][j] + Q_B[i][j] for j in range(nA)] for i in range(nS)]
            a = epsilon_greedy(Q_sum, s_idx, nA, eps)
            
            s2, r, done = env.step(a, s)
            s2_idx = state_to_idx(s2, env.grid.shape)
            
            if random.random() < 0.5:
                # Update Q_A using Q_B's value estimate
                best_action = max(range(nA), key=lambda a_prime: Q_A[s2_idx][a_prime])
                Q_A[s_idx][a] += alpha * (r + gamma * Q_B[s2_idx][best_action] - Q_A[s_idx][a])
            else:
                # Update Q_B using Q_A's value estimate
                best_action = max(range(nA), key=lambda a_prime: Q_B[s2_idx][a_prime])
                Q_B[s_idx][a] += alpha * (r + gamma * Q_A[s2_idx][best_action] - Q_B[s_idx][a])
                
            s = s2
            if done:
                break
        if eps_decay:
            eps = eps_decay(ep, eps)
            
    # Return the average/sum of Q-tables for the final policy
    Q_final = [[Q_A[i][j] + Q_B[i][j] for j in range(nA)] for i in range(nS)]
    return Q_final

# --- 4. Deep Q-Learning (DQN) (NEW) ---

# Neural network model
class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=64):
        super(QNetwork, self).__init__()
        # Input is (row, col)
        self.layer1 = nn.Linear(n_states, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Helper to normalize state for the network
def process_state_for_nn(state, env):
    # Normalize state (row, col) to be between 0 and 1
    r, c = state
    return torch.tensor([r / (env.rows - 1), c / (env.cols - 1)], dtype=torch.float32)

def run_dqn(env, episodes, gamma=0.99, eps=1.0, eps_decay=None, max_steps=400,
            buffer_size=10000, batch_size=64, target_update=10):
    
    nA = 4
    nS_features = 2 # We use (row, col) as features
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = QNetwork(nS_features, nA).to(device)
    target_net = QNetwork(nS_features, nA).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayBuffer(buffer_size)
    
    num_starts = len(env.start_rect)

    for ep in range(episodes):
        start_idx = ep % num_starts
        s = env.reset(start_idx=start_idx)
        state_tensor = process_state_for_nn(s, env).to(device)

        for t in range(max_steps):
            # Epsilon-greedy action selection
            if random.random() < eps:
                a = random.randrange(nA)
            else:
                with torch.no_grad():
                    a = policy_net(state_tensor).argmax().item()
            
            s2, r, done = env.step(a, s)
            
            next_state_tensor = process_state_for_nn(s2, env).to(device)
            
            # Store transition in replay buffer
            memory.push(state_tensor, a, r, next_state_tensor, done)
            
            s = s2
            state_tensor = next_state_tensor
            
            # Perform one step of the optimization
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                
                # Unpack the batch
                state_batch = torch.stack([s for (s,a,r,s_n,d) in transitions]).to(device)
                action_batch = torch.tensor([a for (s,a,r,s_n,d) in transitions], dtype=torch.long).to(device)
                reward_batch = torch.tensor([r for (s,a,r,s_n,d) in transitions], dtype=torch.float32).to(device)
                next_state_batch = torch.stack([s_n for (s,a,r,s_n,d) in transitions]).to(device)
                done_batch = torch.tensor([d for (s,a,r,s_n,d) in transitions], dtype=torch.bool).to(device)

                # Q-values for current states
                q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
                
                # Q-values for next states from target network
                with torch.no_grad():
                    next_q_values = target_net(next_state_batch).max(1)[0]
                
                # Set next_q_value to 0 for terminal states
                next_q_values[done_batch] = 0.0
                
                # Compute the expected Q values (target)
                expected_q_values = (next_q_values * gamma) + reward_batch
                
                # Compute loss
                loss = F.mse_loss(q_values.squeeze(), expected_q_values)
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break
        
        if eps_decay:
            eps = eps_decay(ep, eps)
        
        # Update the target network
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
    return policy_net

# --- Policy Extraction ---

def extract_greedy_policy(Q, env):
    # For tabular methods (Q-Learning, SARSA, Double Q)
    policy = {}
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r,c)
            idx = state_to_idx(s, env.grid.shape)
            if env.grid[r,c] == 1 or s == env.goal or s in env.traps:
                policy[s] = None
                continue
            
            best_action = max(range(4), key=lambda a: Q[idx][a])
            policy[s] = best_action
            
    return policy

def extract_greedy_policy_from_dqn(policy_net, env):
    # For DQN (NEW)
    policy = {}
    device = next(policy_net.parameters()).device
    
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r,c)
            if env.grid[r,c] == 1 or s == env.goal or s in env.traps:
                policy[s] = None
                continue
            
            state_tensor = process_state_for_nn(s, env).to(device)
            with torch.no_grad():
                best_action = policy_net(state_tensor).argmax().item()
            policy[s] = best_action
            
    return policy

# ----------------------------------------------------
# *** 25x25 GRID FUNCTION ***
# ----------------------------------------------------
def build_complex_labyrinth_25x25():
    # 25x25 Highly Branched Labyrinth grid
    grid_str = [
        "S000000000000000000000000",
        "0111110111110111110111110",
        "0100000000000000000000010",
        "0101111111111111111111010",
        "0000000000000000000000010",
        "0111111111111111111111010",
        "0000000000000000000000010",
        "0111111111111111111111110",
        "0000000000000000000000000",
        "1111111111111111111111110",
        "0000000000000000000000000",
        "0111111111111111111111110",
        "0000000000000000000000000",
        "0111111111111111111111110",
        "0000000000000000000000000",
        "0111111111111111111111110",
        "0000000000000000000000000",
        "0111111111111111111111110",
        "0000000000000000000000000",
        "0111111111111111111111110",
        "0000000000000000000000000",
        "0111111111111111111111110",
        "0000000000000000000000000",
        "0111111111111111111111110",
        "000000000000G0000T0000000"
    ]
    grid = np.array([[1 if c == '1' else 0 for c in row] for row in grid_str])
    
    # Start area is the top-left corner (0, 0)
    start_r, start_c = 0, 0
    # All five agents start at the same (0, 0) coordinate
    start_rect = [(start_r, start_c)] * 5
    
    # Goal (G)
    goal = (24, grid_str[24].index('G'))
    
    traps = []
    for r, row in enumerate(grid_str):
        for c, char in enumerate(row):
            if char == 'T':
                traps.append((r, c)) # Trap (T)
                
    slip_prob = 0.05
    env = MazeEnv(grid, start_rect, goal, traps=traps, slip_prob=slip_prob)
    return env

# --- PYGAME VISUALIZATION ---

CELL = 25 # Reduced cell size for 25x25 grid
MARGIN = 1
AGENT_RADIUS = 5 
ARROW_FONT_SIZE = 12 
ARROW_MAP = {0:'\u2191', 1:'\u2192', 2:'\u2193', 3:'\u2190'} 

# Colors for the five agents
COLOR_BG = (255, 255, 255)
COLOR_WALL = (0, 0, 0) 
COLOR_OUTLINE = (0, 0, 0)
COLOR_TEXT = (0, 0, 0) 
COLOR_Q = (255, 0, 0)           # Red (Q-learning)
COLOR_SARSA = (0, 150, 0)       # Green (SARSA)
COLOR_SOFTMAX = (0, 0, 255)     # Blue (Softmax-Q)
COLOR_DOUBLE_Q = (255, 120, 0)  # Orange (Double Q)
COLOR_DQN = (128, 0, 128)       # Purple (DQN)

# Offsets for agents in the same cell (X pattern)
# (Q-learn, SARSA, Softmax-Q, Double-Q, DQN)
AGENT_OFFSETS = [(-8, -8), (8, -8), (0, 0), (-8, 8), (8, 8)]

def draw_text_in_cell(screen, text, font, r, c, color, offset_y=0):
    x = c * (CELL + MARGIN) + MARGIN
    y = r * (CELL + MARGIN) + MARGIN
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x + CELL // 2, y + CELL // 2 + offset_y))
    screen.blit(text_surface, text_rect)

def draw_grid(screen, env, policies=None, animating=False):
    rows, cols = env.rows, env.cols
    screen.fill(COLOR_BG)

    font_small = pygame.font.SysFont(None, 14)

    for r in range(rows):
        for c in range(cols):
            x = c * (CELL + MARGIN) + MARGIN
            y = r * (CELL + MARGIN) + MARGIN
            rect = pygame.Rect(x, y, CELL, CELL)
            s = (r,c)
            
            # Draw Cell Background
            if env.grid[r,c] == 1:
                pygame.draw.rect(screen, COLOR_WALL, rect)
            else:
                pygame.draw.rect(screen, COLOR_BG, rect)
            
            # Draw Special Markers/Labels (Black Outlines)
            if s == env.start_rect[0]: 
                pygame.draw.rect(screen, COLOR_OUTLINE, rect.inflate(-3,-3), 2)
                draw_text_in_cell(screen, "S", font_small, r, c, COLOR_TEXT)
            
            if s == env.goal:
                pygame.draw.rect(screen, COLOR_OUTLINE, rect.inflate(-3,-3), 2)
                draw_text_in_cell(screen, "G", font_small, r, c, COLOR_TEXT)
            
            if s in env.traps:
                pygame.draw.rect(screen, COLOR_OUTLINE, rect.inflate(-3,-3), 2)
                draw_text_in_cell(screen, "T", font_small, r, c, COLOR_TEXT)
                
            # Draw Policy Arrow only when not animating
            # This just shows the first agent's policy (Q-learning) for clarity
            if policies and not animating:
                policy = policies[0][0] 
                if s in policy and policy[s] is not None:
                    action = policy[s]
                    arrow_char = ARROW_MAP.get(action, '?')
                    font_arrow = pygame.font.SysFont(None, ARROW_FONT_SIZE, bold=True)
                    draw_text_in_cell(screen, arrow_char, font_arrow, r, c, COLOR_TEXT)


    # Draw legend and instructions below the grid
    font = pygame.font.SysFont(None, 24, bold=True)
    font_small = pygame.font.SysFont(None, 18)
    
    legend_y = env.rows*(CELL+MARGIN) + 8
    
    # --- Draw Legend (Updated for 5 agents) ---
    text = font.render("Algorithms:", True, COLOR_TEXT)
    screen.blit(text, (10, legend_y))
    
    # Row 1
    pygame.draw.circle(screen, COLOR_Q, (110, legend_y + 12), AGENT_RADIUS)
    text_q = font_small.render("Q-learning (Red)", True, COLOR_Q)
    screen.blit(text_q, (125, legend_y + 4))

    pygame.draw.circle(screen, COLOR_SARSA, (250, legend_y + 12), AGENT_RADIUS)
    text_sarsa = font_small.render("SARSA (Green)", True, COLOR_SARSA)
    screen.blit(text_sarsa, (265, legend_y + 4))

    pygame.draw.circle(screen, COLOR_SOFTMAX, (390, legend_y + 12), AGENT_RADIUS)
    text_softmax = font_small.render("Softmax-Q (Blue)", True, COLOR_SOFTMAX)
    screen.blit(text_softmax, (405, legend_y + 4))
    
    # Row 2
    legend_y_2 = legend_y + 25
    pygame.draw.circle(screen, COLOR_DOUBLE_Q, (110, legend_y_2 + 12), AGENT_RADIUS)
    text_double = font_small.render("Double Q (Orange)", True, COLOR_DOUBLE_Q)
    screen.blit(text_double, (125, legend_y_2 + 4))
    
    pygame.draw.circle(screen, COLOR_DQN, (250, legend_y_2 + 12), AGENT_RADIUS)
    text_dqn = font_small.render("DQN (Purple)", True, COLOR_DQN)
    screen.blit(text_dqn, (265, legend_y_2 + 4))

    
    # Draw Instructions
    inst_y = legend_y + 50
    inst_text = font.render("Press SPACE to Run Simulation | ESC to Quit", True, COLOR_TEXT)
    screen.blit(inst_text, (10, inst_y))


def follow_greedy_path(env, policy, start_pos, max_steps=1500):
    s = start_pos
    path = [s]
    for _ in range(max_steps):
        if s == env.goal or s in env.traps:
            break
        a = policy.get(s, None)
        if a is None:
            # Policy doesn't know what to do (e.g., trapped or untrained)
            break
        
        dr,dc = env.actions[a]
        ns = (s[0]+dr, s[1]+dc)
        
        if not env.in_bounds(ns):
            break # Policy tries to run into a wall
            
        path.append(ns)
        s = ns
        
    return path

def animate_all_agents(screen, env, all_paths, colors, clock, speed=16):
    
    path_indices = [0] * len(all_paths)
    max_path_len = max(len(path) for path in all_paths)
    steps = speed
    
    # Use the globally defined offsets
    offsets = AGENT_OFFSETS
    
    for i in range(max_path_len - 1):
        
        interpolations = []
        for j in range(len(all_paths)):
            offset_x, offset_y = offsets[j] # Use agent-specific offset
            
            if path_indices[j] < len(all_paths[j]) - 1:
                s = all_paths[j][path_indices[j]]
                s2 = all_paths[j][path_indices[j] + 1]
                
                # Calculate start and end points using the agent's offset
                start_x = s[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
                start_y = s[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
                end_x = s2[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
                end_y = s2[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
                
                interpolations.append({'start': (start_x, start_y), 'end': (end_x, end_y), 'active': True})
            else:
                # Agent has finished its path, stay at its end point
                s = all_paths[j][-1]
                x = s[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
                y = s[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
                interpolations.append({'start': (x, y), 'end': (x, y), 'active': False})

        for t in range(steps):
            draw_grid(screen, env, animating=True)
            
            for j in range(len(all_paths)):
                interp = interpolations[j]
                
                ix = interp['start'][0] + (interp['end'][0] - interp['start'][0]) * (t+1)/steps
                iy = interp['start'][1] + (interp['end'][1] - interp['start'][1]) * (t+1)/steps
                
                pygame.draw.circle(screen, colors[j], (int(ix), int(iy)), AGENT_RADIUS)
            
            pygame.display.flip()
            clock.tick(60)
            
        # Move to next step in path
        for j in range(len(all_paths)):
            if path_indices[j] < len(all_paths[j]) - 1:
                path_indices[j] += 1
                
    # Final draw after all paths are complete
    draw_grid(screen, env, policies=None, animating=False) 
    for j in range(len(all_paths)):
        offset_x, offset_y = offsets[j]
        last = all_paths[j][-1]
        lx = last[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
        ly = last[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
        pygame.draw.circle(screen, colors[j], (int(lx), int(ly)), AGENT_RADIUS)
        
    pygame.display.flip()


# -----------------------------
# Main Execution
# -----------------------------
def main():
    env = build_complex_labyrinth_25x25()
    
    # --- Training ---
    print("----------------------------------------------------------")
    print(f"🚀 Training 5 Algorithms ({'1,000'} episodes each)")
    print("   (Note: DQN will likely be undertrained at 1k episodes)")
    print("----------------------------------------------------------")
    
    TRAINING_EPISODES = 5000
    # Epsilon decay: Linear decay to 1% over 2000 episodes
    def eps_decay(ep, e): return max(0.01, e * (1 - ep/2000))
    # DQN Epsilon decay: Faster decay
    def dqn_eps_decay(ep, e): return max(0.01, e * (1 - ep/800))
    
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # 1. Q-Learning (Red)
    Q_q = run_q_learning(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=eps_decay, policy='eps')
    policy_q = extract_greedy_policy(Q_q, env)
    print("✅ Trained: Q-learning (Red)")

    # 2. SARSA (Green)
    Q_sarsa = run_sarsa(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=eps_decay, policy='eps')
    policy_sarsa = extract_greedy_policy(Q_sarsa, env)
    print("✅ Trained: SARSA (Green)")

    # 3. Q-Learning (Softmax) (Blue)
    Q_q_soft = run_q_learning(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=2.0, eps_decay=eps_decay, policy='softmax')
    policy_q_soft = extract_greedy_policy(Q_q_soft, env)
    print("✅ Trained: Q-learning (Softmax) (Blue)")

    # 4. Double Q-Learning (Orange)
    Q_double = run_double_q_learning(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=eps_decay)
    policy_double_q = extract_greedy_policy(Q_double, env)
    print("✅ Trained: Double Q-Learning (Orange)")

    # 5. Deep Q-Learning (Purple)
    dqn_net = run_dqn(env, TRAINING_EPISODES, gamma=0.99, eps=1.0, eps_decay=dqn_eps_decay, max_steps=400)
    policy_dqn = extract_greedy_policy_from_dqn(dqn_net, env)
    print("✅ Trained: Deep Q-Learning (DQN) (Purple)")
    
    print("----------------------------------------------------------")

    # Group policies and colors
    policy_data = [
        # All agents share the start position (0, 0)
        (policy_q, COLOR_Q, env.start_rect[0]), 
        (policy_sarsa, COLOR_SARSA, env.start_rect[1]), 
        (policy_q_soft, COLOR_SOFTMAX, env.start_rect[2]),
        (policy_double_q, COLOR_DOUBLE_Q, env.start_rect[3]),
        (policy_dqn, COLOR_DQN, env.start_rect[4])
    ]
    
    # --- Calculate Paths (used for animation) ---
    all_paths = [follow_greedy_path(env, p, start) for p, c, start in policy_data]
    
    # --- Pygame Initialization ---
    rows, cols = env.rows, env.cols
    win_w = cols * (CELL + MARGIN) + MARGIN
    win_h = rows * (CELL + MARGIN) + MARGIN + 80 # Increased height for new legend row

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("RL 5-Agent Comparative Simulation (25x25) - Shared Start")
    clock = pygame.time.Clock()
    
    draw_grid(screen, env, policies=policy_data, animating=False) 
    
    # Draw initial agent positions with offsets
    initial_positions = []
    offsets = AGENT_OFFSETS # Use global offsets
    
    for i, (p, color, start_pos) in enumerate(policy_data):
        # Center of the cell
        sx = start_pos[1] * (CELL + MARGIN) + MARGIN + CELL//2
        sy = start_pos[0] * (CELL + MARGIN) + MARGIN + CELL//2
        
        # Apply offset for visual separation
        offset_x, offset_y = offsets[i]
        
        final_x, final_y = sx + offset_x, sy + offset_y
        pygame.draw.circle(screen, color, (final_x, final_y), AGENT_RADIUS)
        initial_positions.append((final_x, final_y, color))
    
    pygame.display.flip()

    # --- Main Pygame Loop ---
    simulation_ran = False
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not simulation_ran:
                        # Run the simulation animation
                        print("Starting simulation...")
                        colors = [c for p, c, s in policy_data]
                        animate_all_agents(screen, env, all_paths, colors, clock, speed=16)
                        simulation_ran = True
                        print("Simulation complete. Press SPACE to reset.")
                    else:
                        # Reset agents back to initial offset position
                        draw_grid(screen, env, policies=policy_data, animating=False)
                        for sx, sy, color in initial_positions:
                            pygame.draw.circle(screen, color, (sx, sy), AGENT_RADIUS)
                        pygame.display.flip()
                        simulation_ran = False

        clock.tick(30)

if __name__ == '__main__':
    main()