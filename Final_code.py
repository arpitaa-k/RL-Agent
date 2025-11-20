import pygame
import sys
import time
import random
import math
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- CONFIGURATION ---
GRID_SIZE = 5  # Changed to 5x5
CELL_SIZE = 100 # Much larger cells for 5x5 visibility
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE + 2
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + 120 # Extra space for legend
FPS = 30

# --- ENVIRONMENT ---

class MazeEnv:
    def __init__(self, grid, start_rect, goal, traps=None, slip_prob=0.0, step_penalty=-0.01):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.start_rect = start_rect
        self.goal = tuple(goal)
        self.traps = set(traps) if traps else set()
        self.slip_prob = slip_prob
        self.step_penalty = step_penalty
        self.actions = [(-1,0),(0,1),(1,0),(0,-1)] # Up, Right, Down, Left
        self.reset()

    def in_bounds(self, s):
        r,c = s
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] == 0

    def step(self, action, current_pos):
        actual_action = action
        if random.random() < self.slip_prob:
            actual_action = random.randrange(4) 
            
        dr, dc = self.actions[actual_action]
        nr, nc = current_pos[0] + dr, current_pos[1] + dc
        
        new_pos = (nr, nc)
        if not self.in_bounds(new_pos):
            new_pos = current_pos 
            
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
        self.pos = self.start_rect[start_idx] 
        return self.pos

# --- HELPERS ---

def state_to_idx(state, shape):
    r,c = state
    return r * shape[1] + c

def epsilon_greedy(Q, s_idx, nA, eps):
    if random.random() < eps:
        return random.randrange(nA)
    qvals = Q[s_idx]
    maxv = max(qvals)
    candidates = [i for i,v in enumerate(qvals) if v == maxv]
    return random.choice(candidates)

def softmax_action_from_Q(Q, s_idx, nA, tau=1.0):
    q = np.array(Q[s_idx], dtype=float)
    exps = np.exp((q - np.max(q)) / max(tau, 1e-6))
    probs = exps / np.sum(exps)
    return np.random.choice(range(nA), p=probs)

# --- NEW: REPORTING FUNCTION ---
def print_report(name, stats):
    print(f"\n--- {name} Report ---")
    if stats['first_win_ep']:
        print(f"  > First Reached Goal: Episode {stats['first_win_ep']} (took {stats['first_win_steps']} steps)")
    else:
        print(f"  > FAILED to reach goal in training time.")
    
    print(f"  > Steps at Episode 10: {stats['ep_10_steps']} (Learning check)")
    print(f"  > Steps at Final Episode: {stats['final_steps']} (Optimality check)")
    print(f"  > Total Wins: {stats['total_wins']}")

# --- ALGORITHMS WITH METRICS ---

def run_tabular_algo(env, episodes, algo_type='q_learning', alpha=0.6, gamma=0.99, eps=0.6, eps_decay=None, max_steps=100):
    nS = env.rows * env.cols
    nA = 4
    Q = [[0.0]*nA for _ in range(nS)]
    # For Double Q
    Q_B = [[0.0]*nA for _ in range(nS)] if algo_type == 'double_q' else None
    
    # Metric tracking
    stats = {
        'first_win_ep': None,
        'first_win_steps': 0,
        'ep_10_steps': 0,
        'final_steps': 0,
        'total_wins': 0
    }
    
    num_starts = len(env.start_rect)
    
    for ep in range(episodes):
        start_idx = ep % num_starts
        s = env.reset(start_idx=start_idx)
        s_idx = state_to_idx(s, env.grid.shape)
        steps_this_ep = 0
        
        for t in range(max_steps):
            # Action Selection
            if algo_type == 'double_q':
                Q_sum = [[Q[i][j] + Q_B[i][j] for j in range(nA)] for i in range(nS)]
                a = epsilon_greedy(Q_sum, s_idx, nA, eps)
            elif algo_type == 'softmax':
                a = softmax_action_from_Q(Q, s_idx, nA, tau=eps)
            else: # Q-learning and SARSA (epsilon greedy)
                a = epsilon_greedy(Q, s_idx, nA, eps)
                
            s2, r, done = env.step(a, s)
            s2_idx = state_to_idx(s2, env.grid.shape)
            steps_this_ep += 1
            
            # Updates
            if algo_type == 'q_learning' or algo_type == 'softmax':
                best_next = max(Q[s2_idx])
                Q[s_idx][a] += alpha * (r + gamma * best_next - Q[s_idx][a])
                s, s_idx = s2, s2_idx
                
            elif algo_type == 'sarsa':
                if algo_type == 'softmax': # Shouldn't happen based on logic above, but for safety
                     a2 = softmax_action_from_Q(Q, s2_idx, nA, tau=eps)
                else:
                     a2 = epsilon_greedy(Q, s2_idx, nA, eps)
                Q[s_idx][a] += alpha * (r + gamma * Q[s2_idx][a2] - Q[s_idx][a])
                s, s_idx, a = s2, s2_idx, a2
                
            elif algo_type == 'double_q':
                if random.random() < 0.5:
                    best_a = max(range(nA), key=lambda x: Q[s2_idx][x])
                    Q[s_idx][a] += alpha * (r + gamma * Q_B[s2_idx][best_a] - Q[s_idx][a])
                else:
                    best_a = max(range(nA), key=lambda x: Q_B[s2_idx][x])
                    Q_B[s_idx][a] += alpha * (r + gamma * Q[s2_idx][best_a] - Q_B[s_idx][a])
                s, s_idx = s2, s2_idx

            if done:
                if r > 0: # Reached Goal
                    stats['total_wins'] += 1
                    if stats['first_win_ep'] is None:
                        stats['first_win_ep'] = ep + 1
                        stats['first_win_steps'] = steps_this_ep
                break
        
        # Record Stats
        if ep == 9: stats['ep_10_steps'] = steps_this_ep
        if ep == episodes - 1: stats['final_steps'] = steps_this_ep
        
        if eps_decay: eps = eps_decay(ep, eps)

    if algo_type == 'double_q':
        Q = [[Q[i][j] + Q_B[i][j] for j in range(nA)] for i in range(nS)]
        
    return Q, stats

# --- DQN IMPLEMENTATION ---

class QNetwork(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_output)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def run_dqn(env, episodes, gamma=0.99, eps=1.0, eps_decay=None, max_steps=100):
    nA = 4
    policy_net = QNetwork(2, nA)
    target_net = QNetwork(2, nA)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.005)
    memory = deque(maxlen=2000)
    batch_size = 32
    
    stats = {'first_win_ep': None, 'first_win_steps': 0, 'ep_10_steps': 0, 'final_steps': 0, 'total_wins': 0}

    for ep in range(episodes):
        start_idx = ep % len(env.start_rect)
        s = env.reset(start_idx)
        state_t = torch.tensor([s[0]/env.rows, s[1]/env.cols], dtype=torch.float32)
        steps_this_ep = 0
        
        for t in range(max_steps):
            if random.random() < eps:
                a = random.randrange(nA)
            else:
                with torch.no_grad(): a = policy_net(state_t).argmax().item()
            
            s2, r, done = env.step(a, s)
            next_state_t = torch.tensor([s2[0]/env.rows, s2[1]/env.cols], dtype=torch.float32)
            memory.append((state_t, a, r, next_state_t, done))
            
            state_t = next_state_t
            s = s2
            steps_this_ep += 1
            
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                b_s, b_a, b_r, b_ns, b_d = zip(*batch)
                
                b_s = torch.stack(b_s)
                b_a = torch.tensor(b_a).unsqueeze(1)
                b_r = torch.tensor(b_r, dtype=torch.float32)
                b_ns = torch.stack(b_ns)
                b_d = torch.tensor(b_d, dtype=torch.bool)
                
                curr_q = policy_net(b_s).gather(1, b_a).squeeze()
                next_q = target_net(b_ns).max(1)[0]
                next_q[b_d] = 0.0
                target_q = b_r + gamma * next_q
                
                loss = F.mse_loss(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                if r > 0:
                    stats['total_wins'] += 1
                    if stats['first_win_ep'] is None:
                        stats['first_win_ep'] = ep + 1
                        stats['first_win_steps'] = steps_this_ep
                break
        
        if ep == 9: stats['ep_10_steps'] = steps_this_ep
        if ep == episodes - 1: stats['final_steps'] = steps_this_ep
        if eps_decay: eps = eps_decay(ep, eps)
        if ep % 20 == 0: target_net.load_state_dict(policy_net.state_dict())
        
    return policy_net, stats

# --- MAIN & VISUALIZATION ---

def build_5x5_maze():
    # 5x5 Grid: 0=Path, 1=Wall
    # S = Start (0,0), G = Goal (4,4)
    grid_str = [
        "S0000",
        "01010",
        "01000",
        "1010T",
        "0001G"
    ]
    grid = np.array([[1 if c == '1' else 0 for c in row] for row in grid_str])
    start_rect = [(0,0)] * 5 # 5 agents starting at top left
    goal = (4, 4)
    env = MazeEnv(grid, start_rect, goal)
    return env

def extract_policy(Q, env, is_dqn=False):
    policy = {}
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r,c] == 1: continue
            if (r,c) == env.goal: continue
            
            if is_dqn:
                st = torch.tensor([r/env.rows, c/env.cols], dtype=torch.float32)
                with torch.no_grad(): a = Q(st).argmax().item()
            else:
                idx = state_to_idx((r,c), env.grid.shape)
                a = max(range(4), key=lambda x: Q[idx][x])
            policy[(r,c)] = a
    return policy

def main():
    env = build_5x5_maze()
    EPISODES = 20
    decay = lambda ep, e: max(0.01, e * 0.99)
    
    print(f"Training 5 Agents on 5x5 Maze for {EPISODES} episodes...")
    
    # Train all
    Q_q, stats_q = run_tabular_algo(env, EPISODES, 'q_learning', eps_decay=decay)
    print_report("Q-Learning (Red)", stats_q)
    
    Q_s, stats_s = run_tabular_algo(env, EPISODES, 'sarsa', eps_decay=decay)
    print_report("SARSA (Green)", stats_s)
    
    Q_sm, stats_sm = run_tabular_algo(env, EPISODES, 'softmax', eps=2.0, eps_decay=decay)
    print_report("Softmax (Blue)", stats_sm)
    
    Q_dq, stats_dq = run_tabular_algo(env, EPISODES, 'double_q', eps_decay=decay)
    print_report("Double Q (Orange)", stats_dq)
    
    net_dqn, stats_dqn = run_dqn(env, EPISODES, eps_decay=decay)
    print_report("DQN (Purple)", stats_dqn)
    
    # Extract Policies
    policies = [
        (extract_policy(Q_q, env), (255,0,0)),
        (extract_policy(Q_s, env), (0,255,0)),
        (extract_policy(Q_sm, env), (0,0,255)),
        (extract_policy(Q_dq, env), (255,165,0)),
        (extract_policy(net_dqn, env, True), (128,0,128))
    ]
    
    # Visualization
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("5x5 RL Comparative Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    running = True
    agents_pos = [env.start_rect[0]] * 5
    simulating = False
    path_indices = [0] * 5
    paths = []
    
    # Pre-calculate paths
    for p, c in policies:
        path = [env.start_rect[0]]
        curr = path[0]
        for _ in range(30): # shorter max steps for 5x5
            if curr == env.goal: break
            if curr not in p: break
            dr, dc = env.actions[p[curr]]
            nxt = (curr[0]+dr, curr[1]+dc)
            if env.in_bounds(nxt):
                curr = nxt
                path.append(curr)
            else: break
        paths.append(path)

    while running:
        screen.fill((255,255,255))
        
        # Draw Grid
        for r in range(env.rows):
            for c in range(env.cols):
                rect = (c*CELL_SIZE+1, r*CELL_SIZE+1, CELL_SIZE-2, CELL_SIZE-2)
                color = (0,0,0) if env.grid[r,c] == 1 else (240,240,240)
                if (r,c) == env.goal: color = (200,255,200)
                if (r,c) == env.start_rect[0]: color = (200,200,255)
                pygame.draw.rect(screen, color, rect)
                if (r,c) == env.goal: 
                    screen.blit(font.render("G", True, (0,0,0)), (c*CELL_SIZE+40, r*CELL_SIZE+35))
        
        # Draw Agents
        offsets = [(-20,-20), (20,-20), (0,0), (-20,20), (20,20)]
        for i, (pos, color) in enumerate(zip(agents_pos, [p[1] for p in policies])):
            c, r = pos[1], pos[0]
            ox, oy = offsets[i]
            pygame.draw.circle(screen, color, (c*CELL_SIZE + CELL_SIZE//2 + ox, r*CELL_SIZE + CELL_SIZE//2 + oy), 15)

        # Instructions
        inst = font.render("SPACE to Restart Simulation", True, (0,0,0))
        screen.blit(inst, (20, SCREEN_HEIGHT - 40))
        
        # --- LEGEND ---
        legend_y = SCREEN_HEIGHT - 80
        legend_data = [
            ("Q-Learn (Red)", (255,0,0), 20),
            ("SARSA (Grn)", (0,255,0), 160),
            ("Softmax (Blu)", (0,0,255), 290),
            ("DoubleQ (Org)", (255,165,0), 430),
            ("DQN (Pur)", (128,0,128), 570)
        ]
        # Check fit on screen (5x5 is small, so we might need to wrap)
        # But 500px width should fit.
        for text, col, x_pos in legend_data:
            if x_pos < SCREEN_WIDTH - 50:
                 pygame.draw.circle(screen, col, (x_pos, legend_y), 8)
                 lbl = font.render(text, True, (0,0,0))
                 screen.blit(lbl, (x_pos + 15, legend_y - 8))

        # Animation Logic
        if simulating:
            done_cnt = 0
            for i in range(5):
                if path_indices[i] < len(paths[i]) - 1:
                    path_indices[i] += 1
                    agents_pos[i] = paths[i][path_indices[i]]
                else:
                    done_cnt += 1
            if done_cnt == 5: simulating = False
            time.sleep(0.15)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                agents_pos = [env.start_rect[0]] * 5
                path_indices = [0] * 5
                simulating = True

        pygame.display.flip()
        clock.tick(FPS)
        
    pygame.quit()

if __name__ == "__main__":
    main()