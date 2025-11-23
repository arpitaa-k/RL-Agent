"""
RL 5-Agent Comparative Simulation — Guaranteed Success Version (Option A)

Features:
- Tabular: Q-Learning, SARSA, Softmax-Q, Double Q
- Deep: DQN with local patch + coords + distance
- BFS fallback to guarantee each agent reaches the Goal (G)
- Visualization: pygame (5 agent offsets)
- Slip set to 0.0 for deterministic final paths (training still uses small randomness if desired)
- Reproducible via seeds

Requirements:
pip install pygame torch numpy
"""

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

# ---------------------- ENVIRONMENT ----------------------

class MazeEnv:
    def __init__(self, grid, start_rect, goal, traps=None,
                 slip_prob=0.0, step_penalty=-0.01, revisit_penalty=-0.02):
        self.grid = np.array(grid, dtype=np.int8)
        self.rows, self.cols = self.grid.shape
        self.start_rect = list(start_rect)
        self.goal = tuple(goal)
        self.traps = set(traps) if traps else set()
        self.slip_prob = slip_prob
        self.step_penalty = step_penalty
        self.revisit_penalty = revisit_penalty
        self.actions = [(-1,0),(0,1),(1,0),(0,-1)]  # up, right, down, left
        self.reset()

    def in_bounds(self, s):
        r,c = s
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] == 0

    def reset(self, start_idx=0):
        self.pos = tuple(self.start_rect[start_idx])
        self.visited = set([self.pos])
        return self.pos

    def step(self, action, current_pos):
        # deterministic unless slip_prob > 0
        actual_action = action
        if random.random() < self.slip_prob:
            actual_action = random.randrange(4)
        dr, dc = self.actions[actual_action]
        nr, nc = current_pos[0] + dr, current_pos[1] + dc
        new_pos = (nr, nc)
        if not self.in_bounds(new_pos):
            new_pos = current_pos  # bump into wall, stay
        reward = self.step_penalty
        done = False
        if new_pos == self.goal:
            reward = 1.0
            done = True
        elif new_pos in self.traps:
            reward = -1.0
            done = True
        else:
            if new_pos in self.visited:
                reward += self.revisit_penalty
            self.visited.add(new_pos)
        return new_pos, reward, done

# ---------------------- UTILITIES ----------------------

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
    probs = exps / (np.sum(exps) + 1e-12)
    if np.isnan(probs).any() or np.isinf(probs).any():
        probs = np.ones(nA) / nA
    return np.random.choice(range(nA), p=probs)

# BFS shortest path (avoids walls and traps) - used as fallback
def bfs_shortest_path(env, start, goal):
    if start == goal:
        return [start]
    q = deque([start])
    visited = {start: None}
    while q:
        cur = q.popleft()
        for dr,dc in env.actions:
            nxt = (cur[0]+dr, cur[1]+dc)
            if 0 <= nxt[0] < env.rows and 0 <= nxt[1] < env.cols:
                if env.grid[nxt] == 0 and nxt not in visited and nxt not in env.traps:
                    visited[nxt] = cur
                    q.append(nxt)
                    if nxt == goal:
                        # reconstruct
                        path = [goal]
                        p = goal
                        while visited[p] is not None:
                            p = visited[p]
                            path.append(p)
                        return list(reversed(path))
    return None

# ---------------------- TABULAR ALGORITHMS ----------------------

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
            best_next = max(Q[s2_idx])
            Q[s_idx][a] += alpha * (r + gamma * best_next - Q[s_idx][a])
            s, s_idx = s2, s2_idx
            if done:
                break
        if eps_decay:
            eps = eps_decay(ep, eps)
    return Q

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
            # Sum Q for action selection
            Q_sum = [ [Q_A[i][j] + Q_B[i][j] for j in range(nA)] for i in range(nS) ]
            a = epsilon_greedy(Q_sum, s_idx, nA, eps)
            s2, r, done = env.step(a, s)
            s2_idx = state_to_idx(s2, env.grid.shape)
            if random.random() < 0.5:
                # update Q_A using argmax from Q_A but value from Q_B
                a_max = max(range(nA), key=lambda a_prime: Q_A[s2_idx][a_prime])
                Q_A[s_idx][a] += alpha * (r + gamma * Q_B[s2_idx][a_max] - Q_A[s_idx][a])
            else:
                a_max = max(range(nA), key=lambda a_prime: Q_B[s2_idx][a_prime])
                Q_B[s_idx][a] += alpha * (r + gamma * Q_A[s2_idx][a_max] - Q_B[s_idx][a])
            s = s2
            if done:
                break
        if eps_decay:
            eps = eps_decay(ep, eps)
    Q_final = [[Q_A[i][j] + Q_B[i][j] for j in range(nA)] for i in range(nS)]
    return Q_final

def extract_greedy_policy(Q, env):
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

# ---------------------- DQN ----------------------

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=[256,256]):
        super(QNetwork, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def get_local_patch(env, state, patch_size=5):
    r,c = state
    p = patch_size
    half = p // 2
    patch = np.ones((p,p), dtype=np.float32)  # walls default=1
    for i in range(-half, half+1):
        for j in range(-half, half+1):
            rr, cc = r + i, c + j
            if 0 <= rr < env.rows and 0 <= cc < env.cols:
                if (rr,cc) == env.goal:
                    patch[i+half, j+half] = 0.5
                elif (rr,cc) in env.traps:
                    patch[i+half, j+half] = -1.0
                else:
                    patch[i+half, j+half] = float(env.grid[rr, cc])  # 0 or 1
    return patch.flatten()

def process_state_for_nn(state, env, patch_size=5):
    r,c = state
    coords = np.array([r / (env.rows - 1), c / (env.cols - 1)], dtype=np.float32)
    patch = get_local_patch(env, state, patch_size=patch_size)
    # distance to goal (normalized)
    dist = np.array([abs(r - env.goal[0]) / (env.rows - 1), abs(c - env.goal[1]) / (env.cols - 1)], dtype=np.float32)
    return np.concatenate([coords, dist, patch]).astype(np.float32)

def run_dqn(env, episodes, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_steps=30000,
            max_steps=400, buffer_size=50000, batch_size=64, lr=1e-3, target_update=200,
            patch_size=5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nA = 4
    sample_state = process_state_for_nn((0,0), env, patch_size=patch_size)
    input_dim = sample_state.shape[0]
    policy_net = QNetwork(input_dim, nA, hidden=[256,256]).to(device)
    target_net = QNetwork(input_dim, nA, hidden=[256,256]).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(buffer_size)
    steps_done = 0
    num_starts = len(env.start_rect)
    def get_epsilon(steps):
        if steps >= eps_decay_steps:
            return eps_end
        frac = steps / eps_decay_steps
        return eps_start + frac * (eps_end - eps_start)
    for ep in range(episodes):
        start_idx = ep % num_starts
        s = env.reset(start_idx=start_idx)
        state_vec = process_state_for_nn(s, env, patch_size=patch_size)
        for t in range(max_steps):
            eps = get_epsilon(steps_done)
            steps_done += 1
            if random.random() < eps:
                a = random.randrange(nA)
            else:
                with torch.no_grad():
                    x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
                    a = int(policy_net(x).argmax().item())
            s2, r, done = env.step(a, s)
            next_state_vec = process_state_for_nn(s2, env, patch_size=patch_size)
            memory.push(state_vec, a, r, next_state_vec, done)
            state_vec = next_state_vec
            s = s2
            # learn step
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                state_batch = torch.tensor(np.stack(transitions.state), dtype=torch.float32).to(device)
                action_batch = torch.tensor(transitions.action, dtype=torch.long).unsqueeze(1).to(device)
                reward_batch = torch.tensor(transitions.reward, dtype=torch.float32).to(device)
                next_state_batch = torch.tensor(np.stack(transitions.next_state), dtype=torch.float32).to(device)
                done_batch = torch.tensor(transitions.done, dtype=torch.bool).to(device)
                q_values = policy_net(state_batch).gather(1, action_batch).squeeze()
                with torch.no_grad():
                    next_q_values = target_net(next_state_batch).max(1)[0]
                next_q_values[done_batch] = 0.0
                expected_q = reward_batch + gamma * next_q_values
                loss = F.mse_loss(q_values, expected_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            if done:
                break
        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    return policy_net

def extract_greedy_policy_from_dqn(policy_net, env, patch_size=5, device=None):
    if device is None:
        device = next(policy_net.parameters()).device
    policy = {}
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r,c)
            if env.grid[r,c] == 1 or s == env.goal or s in env.traps:
                policy[s] = None
                continue
            state_vec = process_state_for_nn(s, env, patch_size=patch_size)
            with torch.no_grad():
                x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
                best_action = int(policy_net(x).argmax().item())
            policy[s] = best_action
    return policy

# ---------------------- VISUALIZATION ----------------------

CELL = 25
MARGIN = 1
AGENT_RADIUS = 5
ARROW_FONT_SIZE = 12
ARROW_MAP = {0:'\u2191', 1:'\u2192', 2:'\u2193', 3:'\u2190'}

COLOR_BG = (255,255,255)
COLOR_WALL = (0,0,0)
COLOR_OUTLINE = (0,0,0)
COLOR_TEXT = (0,0,0)
COLOR_Q = (255,0,0)
COLOR_SARSA = (0,150,0)
COLOR_SOFTMAX = (0,0,255)
COLOR_DOUBLE_Q = (255,120,0)
COLOR_DQN = (128,0,128)

AGENT_OFFSETS = [(-8,-8), (8,-8), (0,0), (-8,8), (8,8)]

def draw_text_in_cell(screen, text, font, r, c, color, offset_y=0):
    x = c * (CELL + MARGIN) + MARGIN
    y = r * (CELL + MARGIN) + MARGIN
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x + CELL//2, y + CELL//2 + offset_y))
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
            if env.grid[r,c] == 1:
                pygame.draw.rect(screen, COLOR_WALL, rect)
            else:
                pygame.draw.rect(screen, COLOR_BG, rect)
            if s == env.start_rect[0]:
                pygame.draw.rect(screen, COLOR_OUTLINE, rect.inflate(-3,-3), 2)
                draw_text_in_cell(screen, "S", font_small, r, c, COLOR_TEXT)
            if s == env.goal:
                pygame.draw.rect(screen, COLOR_OUTLINE, rect.inflate(-3,-3), 2)
                draw_text_in_cell(screen, "G", font_small, r, c, COLOR_TEXT)
            if s in env.traps:
                pygame.draw.rect(screen, COLOR_OUTLINE, rect.inflate(-3,-3), 2)
                draw_text_in_cell(screen, "T", font_small, r, c, COLOR_TEXT)
            if policies and not animating:
                policy = policies[0][0]
                if s in policy and policy[s] is not None:
                    action = policy[s]
                    arrow_char = ARROW_MAP.get(action, '?')
                    font_arrow = pygame.font.SysFont(None, ARROW_FONT_SIZE, bold=True)
                    draw_text_in_cell(screen, arrow_char, font_arrow, r, c, COLOR_TEXT)
    # legend
    font = pygame.font.SysFont(None, 24, bold=True)
    font_small = pygame.font.SysFont(None, 18)
    legend_y = env.rows*(CELL+MARGIN) + 8
    screen.blit(font.render("Algorithms:", True, COLOR_TEXT), (10, legend_y))
    pygame.draw.circle(screen, COLOR_Q, (110, legend_y + 12), AGENT_RADIUS)
    screen.blit(font_small.render("Q-learning (Red)", True, COLOR_Q), (125, legend_y + 4))
    pygame.draw.circle(screen, COLOR_SARSA, (250, legend_y + 12), AGENT_RADIUS)
    screen.blit(font_small.render("SARSA (Green)", True, COLOR_SARSA), (265, legend_y + 4))
    pygame.draw.circle(screen, COLOR_SOFTMAX, (390, legend_y + 12), AGENT_RADIUS)
    screen.blit(font_small.render("Softmax-Q (Blue)", True, COLOR_SOFTMAX), (405, legend_y + 4))
    legend_y_2 = legend_y + 25
    pygame.draw.circle(screen, COLOR_DOUBLE_Q, (110, legend_y_2 + 12), AGENT_RADIUS)
    screen.blit(font_small.render("Double Q (Orange)", True, COLOR_DOUBLE_Q), (125, legend_y_2 + 4))
    pygame.draw.circle(screen, COLOR_DQN, (250, legend_y_2 + 12), AGENT_RADIUS)
    screen.blit(font_small.render("DQN (Purple)", True, COLOR_DQN), (265, legend_y_2 + 4))
    inst_y = legend_y + 50
    inst_text = font.render("Press SPACE to Run Simulation | ESC to Quit", True, COLOR_TEXT)
    screen.blit(inst_text, (10, inst_y))

def follow_greedy_path(env, policy, start_pos, max_steps=2000):
    s = start_pos
    path = [s]
    for _ in range(max_steps):
        if s == env.goal or s in env.traps:
            break
        a = policy.get(s, None)
        if a is None:
            break
        dr,dc = env.actions[a]
        ns = (s[0]+dr, s[1]+dc)
        if not env.in_bounds(ns):
            break
        path.append(ns)
        s = ns
    return path

def animate_all_agents(screen, env, all_paths, colors, clock, speed=16):
    # all_paths guaranteed to end at env.goal by BFS fallback
    path_indices = [0] * len(all_paths)
    max_path_len = max(len(path) for path in all_paths)
    steps = speed
    offsets = AGENT_OFFSETS
    for i in range(max_path_len - 1):
        interpolations = []
        for j in range(len(all_paths)):
            offset_x, offset_y = offsets[j]
            if path_indices[j] < len(all_paths[j]) - 1:
                s = all_paths[j][path_indices[j]]
                s2 = all_paths[j][path_indices[j] + 1]
                start_x = s[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
                start_y = s[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
                end_x = s2[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
                end_y = s2[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
                interpolations.append({'start': (start_x, start_y), 'end': (end_x, end_y), 'active': True})
            else:
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
        for j in range(len(all_paths)):
            if path_indices[j] < len(all_paths[j]) - 1:
                path_indices[j] += 1
    draw_grid(screen, env, policies=None, animating=False)
    for j in range(len(all_paths)):
        offset_x, offset_y = offsets[j]
        last = all_paths[j][-1]
        lx = last[1] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_x
        ly = last[0] * (CELL + MARGIN) + MARGIN + CELL//2 + offset_y
        pygame.draw.circle(screen, colors[j], (int(lx), int(ly)), AGENT_RADIUS)
    pygame.display.flip()

# ---------------------- MAIN & MAZE ----------------------

def build_guaranteed_maze():
    # Adjusted 25x25 maze with a clear path to G (guaranteed)
    grid_str = [
        "S000000000000000000000000",
        "0111110111110111110111110",
        "0100000100000000000000010",
        "0101110101111111111111010",
        "0001110100000000000000010",
        "0111110111111111111111010",
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
        "000000000000G000000000000"
    ]
    grid = np.array([[1 if c == '1' else 0 for c in row] for row in grid_str], dtype=np.int8)
    start_r, start_c = 0, 0
    start_rect = [(start_r, start_c)] * 5
    goal = (24, grid_str[24].index('G'))
    traps = []
    # no traps on guaranteed path; place a trap far from main corridor to keep challenge
    traps.append((24, 7))  # a trap but not blocking the main access to G
    slip_prob = 0.0
    env = MazeEnv(grid, start_rect, goal, traps=traps, slip_prob=slip_prob,
                  step_penalty=-0.01, revisit_penalty=-0.02)
    return env

def main():
    # reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = build_guaranteed_maze()

    # training hyperparams (moderate for demonstration; increase for better DQN)
    TRAINING_EPISODES = 10000    # 10k per method; increase for DQN if desired
    MAX_STEPS = 400

    def eps_decay(ep, e):
        return max(0.01, e * (1 - ep / (TRAINING_EPISODES * 1.5)))

    print("----------------------------------------------------------")
    print(f"🚀 Training 5 Algorithms ({TRAINING_EPISODES} episodes each)")
    print("Note: If running on CPU, DQN training may take time. Increase episodes for stronger policies.")
    print("----------------------------------------------------------")

    # 1 Q-Learning
    Q_q = run_q_learning(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=eps_decay, max_steps=MAX_STEPS, policy='eps')
    policy_q = extract_greedy_policy(Q_q, env)
    print("✅ Trained: Q-learning (Red)")

    # 2 SARSA
    Q_sarsa = run_sarsa(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=eps_decay, max_steps=MAX_STEPS, policy='eps')
    policy_sarsa = extract_greedy_policy(Q_sarsa, env)
    print("✅ Trained: SARSA (Green)")

    # 3 Softmax-Q (use Q-learning with softmax policy during action selection)
    Q_q_soft = run_q_learning(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=2.0, eps_decay=eps_decay, max_steps=MAX_STEPS, policy='softmax')
    policy_q_soft = extract_greedy_policy(Q_q_soft, env)
    print("✅ Trained: Q-learning (Softmax) (Blue)")

    # 4 Double Q-Learning
    Q_double = run_double_q_learning(env, TRAINING_EPISODES, alpha=0.6, gamma=0.99, eps=0.6, eps_decay=eps_decay, max_steps=MAX_STEPS)
    policy_double_q = extract_greedy_policy(Q_double, env)
    print("✅ Trained: Double Q-Learning (Orange)")

    # 5 DQN
    dqn_net = run_dqn(env,
                      episodes=TRAINING_EPISODES,
                      gamma=0.99,
                      eps_start=1.0, eps_end=0.05,
                      eps_decay_steps=TRAINING_EPISODES * 4,   # slow decay
                      max_steps=MAX_STEPS,
                      buffer_size=50000,
                      batch_size=64,
                      lr=1e-3,
                      target_update=200,
                      patch_size=5)
    policy_dqn = extract_greedy_policy_from_dqn(dqn_net, env, patch_size=5)
    print("✅ Trained: Deep Q-Learning (DQN) (Purple)")

    print("----------------------------------------------------------")
    # Group policies and compute paths
    policy_data = [
        (policy_q, COLOR_Q, env.start_rect[0]),
        (policy_sarsa, COLOR_SARSA, env.start_rect[1]),
        (policy_q_soft, COLOR_SOFTMAX, env.start_rect[2]),
        (policy_double_q, COLOR_DOUBLE_Q, env.start_rect[3]),
        (policy_dqn, COLOR_DQN, env.start_rect[4])
    ]

    # Compute greedy paths; if policy fails to reach G, replace with BFS shortest safe path
    all_paths = []
    for (pol, color, start) in policy_data:
        path = follow_greedy_path(env, pol, start, max_steps=2000)
        if len(path) > 0 and path[-1] == env.goal:
            all_paths.append(path)
        else:
            # fallback to BFS - guaranteed path avoiding walls and traps
            bfs_path = bfs_shortest_path(env, start, env.goal)
            if bfs_path is None:
                # As a last resort (shouldn't happen with this maze), use direct greedy path even if it ends in trap
                all_paths.append(path if len(path)>0 else [start, env.goal])
            else:
                all_paths.append(bfs_path)

    # Pygame initialization and drawing
    rows, cols = env.rows, env.cols
    win_w = cols * (CELL + MARGIN) + MARGIN
    win_h = rows * (CELL + MARGIN) + MARGIN + 80
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("RL 5-Agent Comparative Simulation (25x25) - Guaranteed Success (A)")
    clock = pygame.time.Clock()

    draw_grid(screen, env, policies=policy_data, animating=False)
    initial_positions = []
    for i, (p, color, start_pos) in enumerate(policy_data):
        sx = start_pos[1] * (CELL + MARGIN) + MARGIN + CELL//2
        sy = start_pos[0] * (CELL + MARGIN) + MARGIN + CELL//2
        offset_x, offset_y = AGENT_OFFSETS[i]
        final_x, final_y = sx + offset_x, sy + offset_y
        pygame.draw.circle(screen, color, (final_x, final_y), AGENT_RADIUS)
        initial_positions.append((final_x, final_y, color))
    pygame.display.flip()

    # Main loop
    simulation_ran = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not simulation_ran:
                        print("Starting simulation...")
                        colors = [c for p,c,s in policy_data]
                        animate_all_agents(screen, env, all_paths, colors, clock, speed=18)
                        simulation_ran = True
                        print("Simulation complete. Press SPACE to reset.")
                    else:
                        draw_grid(screen, env, policies=policy_data, animating=False)
                        for sx, sy, color in initial_positions:
                            pygame.draw.circle(screen, color, (sx, sy), AGENT_RADIUS)
                        pygame.display.flip()
                        simulation_ran = False
        clock.tick(30)

if __name__ == '__main__':
    main()
