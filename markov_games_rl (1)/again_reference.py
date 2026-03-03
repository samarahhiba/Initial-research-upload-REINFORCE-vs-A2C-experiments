"""
minimax_driving.py

Single-file Minimax-Q + Deep Minimax-Q (DNQN) solver for the crash-cost driving simulator.

Features:
- Tabular Minimax-Q value-iteration (planning)
- Deep Minimax-Q (DNQN) with target network + replay buffer
- LP solver that returns both maximin value and mixed strategy (pi)
- LP caching to avoid repeated linear program solves
- Fixes: crash_cost propagation, no sigmoid on final Q-values, correct LP formulation
- Experiment utilities for sweeping crash costs and plotting metrics


"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns
import nashpy as nash
from functools import lru_cache
import time
import math

# -------------------------
# Environment / Constants
# -------------------------
GRID_SIZE = 3
NUM_STATES = GRID_SIZE ** 4  # 81
ACTIONS = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
NUM_ACTIONS = len(ACTIONS)
REWARD_MATRIX = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]])  # base reward for tile
BETA = 0.9  # discount factor

# scaling defaults (you can tune or disable scaling)
R_MAX_DEFAULT = 5.0  # max base reward
# NOTE: scale_reward is optional. Many experiments perform better without scaling.
def scale_reward(r, crash_cost, r_max=R_MAX_DEFAULT):
    r_min = -crash_cost
    denom = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
    return 0.1 * ((r - r_min) / denom)


def get_coords_from_state(state_idx):
    x1 = state_idx // GRID_SIZE ** 3
    y1 = (state_idx % GRID_SIZE ** 3) // GRID_SIZE ** 2
    x2 = (state_idx % GRID_SIZE ** 2) // GRID_SIZE
    y2 = state_idx % GRID_SIZE
    return int(x1), int(y1), int(x2), int(y2)


def coords_to_state(x1, y1, x2, y2):
    return int(x1 * GRID_SIZE ** 3 + y1 * GRID_SIZE ** 2 + x2 * GRID_SIZE + y2)


def transition_function(state, action1, action2):
    x1, y1, x2, y2 = get_coords_from_state(state)
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    a1_idx = int(action1)
    a2_idx = int(action2)

    x1_new = (x1 + dx[a1_idx] + GRID_SIZE) % GRID_SIZE
    y1_new = (y1 + dy[a1_idx] + GRID_SIZE) % GRID_SIZE

    x2_new = (x2 + dx[a2_idx] + GRID_SIZE) % GRID_SIZE
    y2_new = (y2 + dy[a2_idx] + GRID_SIZE) % GRID_SIZE

    return x1_new, y1_new, x2_new, y2_new


def reward_function(state, action1, action2, crash_cost, do_scale=True):
    x1_new, y1_new, x2_new, y2_new = transition_function(state, action1, action2)
    r1 = REWARD_MATRIX[x1_new, y1_new]
    r2 = REWARD_MATRIX[x2_new, y2_new]
    if (x1_new, y1_new) == (x2_new, y2_new):
        r1 -= crash_cost
        r2 -= crash_cost
    if do_scale:
        return scale_reward(r1, crash_cost), scale_reward(r2, crash_cost)
    else:
        return float(r1), float(r2)


# -------------------------
# LP: Maximin value + pi (row player's mixed strategy)
# -------------------------
# We implement the LP:
# Variables x = [V, pi_0, pi_1, ..., pi_{m-1}]
# Maximize V (we minimize -V with linprog)
# Constraints:
#   For each column j: sum_i pi_i * A[i, j] - V >= 0  ->  V - sum_i pi_i * A[i,j] <= 0
#   Sum_i pi_i = 1
#   pi_i >= 0
#
# Returns (V, pi_row_vector)
# Fallback: if LP fails, return conservative estimate and uniform pi

def _matrix_to_hashable_tuple(mat, decimals=8):
    # rounds and flattens into tuple for caching
    return tuple(np.round(np.asarray(mat).ravel(), decimals).tolist())


@lru_cache(maxsize=4096)
def solve_minimax_lp_cached(A_tup):
    """
    Cached wrapper; A_tup is a tuple representation of the matrix,
    with original shape encoded in first two numbers: (m, n, flattened...)
    We'll decode and call the core LP solver.
    """
    # A_tup format: (m, n, flat entries...)
    m = int(A_tup[0])
    n = int(A_tup[1])
    flat = np.array(A_tup[2:], dtype=float)
    A = flat.reshape((m, n))
    return solve_minimax_lp_value_un_cached(A)


def solve_minimax_lp_value_un_cached(A):
    """
    Core LP solver returning (V, pi_row)
    A: shape (m, n) - rows are row-player actions
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    # minimize c^T x where x = [V, pi_0..pi_{m-1}], objective: minimize -V -> c[0] = -1
    c = np.zeros(1 + m)
    c[0] = -1.0  # minimize -V

    # A_ub: for each column j: V - sum_i pi_i * A[i,j] <= 0
    # shape (n, 1 + m)
    A_ub = np.zeros((n, 1 + m))
    for j in range(n):
        A_ub[j, 0] = 1.0
        A_ub[j, 1:] = -A[:, j]

    b_ub = np.zeros(n)

    # equality: sum pi_i = 1
    A_eq = np.zeros((1, 1 + m))
    A_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])

    bounds = [(None, None)] + [(0.0, None)] * m

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        V = float(res.x[0])
        pi = np.array(res.x[1:], dtype=float)
        # numerical cleanup
        pi[pi < 1e-12] = 0.0
        s = np.sum(pi)
        if s <= 0:
            pi = np.ones(m) / float(m)
        else:
            pi = pi / float(np.sum(pi))
        return float(V), pi
    else:
        # fallback: give row maximin and uniform pi
        V_fb = float(np.max(np.min(A, axis=1)))
        pi_fb = np.ones(m) / float(m)
        return V_fb, pi_fb


def solve_minimax_lp_value(A):
    """
    Public wrapper that caches on a tuple key.
    Returns: (V, pi_row)
    """
    m, n = A.shape
    flat = tuple(np.round(A.ravel(), 8).tolist())
    # encode shape followed by entries
    key = (m, n) + flat
    return solve_minimax_lp_cached(key)


# -------------------------
# Tabular Minimax-Q solver (value-iteration style)
# -------------------------
def solve_markov_game(crash_cost, learning_rate=0.01, discount_factor=BETA, num_iterations=2000,
                      do_scale=True):
    Q1 = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_ACTIONS))
    Q2 = np.zeros_like(Q1)
    value_history = np.zeros(num_iterations)
    center_state = coords_to_state(1, 1, 1, 1)

    for it in range(num_iterations):
        # iterate states in deterministic order (could randomize)
        for s in range(NUM_STATES):
            # compute V(s) from Q1[s] using LP
            V1_s, _ = solve_minimax_lp_value(Q1[s])
            if s == center_state:
                value_history[it] = V1_s

            for a1 in range(NUM_ACTIONS):
                for a2 in range(NUM_ACTIONS):
                    x1_new, y1_new, x2_new, y2_new = transition_function(s, a1, a2)
                    r1, r2 = reward_function(s, a1, a2, crash_cost, do_scale)
                    s_prime_idx = coords_to_state(x1_new, y1_new, x2_new, y2_new)

                    V1_prime, _ = solve_minimax_lp_value(Q1[s_prime_idx])
                    V2_prime, _ = solve_minimax_lp_value(Q2[s_prime_idx].T)

                    target_q1 = r1 + discount_factor * V1_prime
                    target_q2 = r2 + discount_factor * V2_prime

                    Q1[s, a1, a2] += learning_rate * (target_q1 - Q1[s, a1, a2])
                    Q2[s, a1, a2] += learning_rate * (target_q2 - Q2[s, a1, a2])

    return Q1, Q2, value_history


def visualize_tabular_results(Q1, crash_cost, value_history):
    plt.figure(figsize=(10, 5))
    plt.plot(value_history)
    plt.title(f'Tabular Minimax-Q Convergence (Crash Cost = {crash_cost})')
    plt.xlabel('Iteration')
    plt.ylabel('Maximin Value for P1 at center state')
    plt.grid(True)
    plt.show()

    center_state = coords_to_state(1, 1, 1, 1)
    A_final = Q1[center_state]
    # zero-sum transform
    game_final = nash.Game(A_final, -A_final.T)
    equilibria = list(game_final.support_enumeration())
    if equilibria:
        pi1, pi2 = equilibria[0]
        actions = list(ACTIONS.keys())
        plt.figure(figsize=(6, 2))
        sns.barplot(x=actions, y=pi1)
        plt.title(f'P1 Nash/mix at center state (Crash Cost={crash_cost})')
        plt.ylim(0, 1)
        plt.show()
    else:
        print("No equilibrium enumerated by NashPy for center state; consider visualizing LP pi instead.")
        V, pi = solve_minimax_lp_value(A_final)
        actions = list(ACTIONS.keys())
        plt.figure(figsize=(6, 2))
        sns.barplot(x=actions, y=pi)
        plt.title(f'P1 LP-mix at center state (Crash Cost={crash_cost})')
        plt.ylim(0, 1)
        plt.show()


# -------------------------
# Deep Minimax-Q components
# -------------------------
Transition = namedtuple('Transition', ('state', 'action1', 'action2', 'reward1', 'reward2', 'next_state'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return []
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, input_size=4, output_size=NUM_ACTIONS ** 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)  # raw Q-values (no activation)
        )
    def forward(self, x):
        return self.fc(x)


class DNQN_Solver:
    def __init__(self, gamma=BETA, lr=1e-3, target_update=100, buffer_size=10000, batch_size=64, device=None):
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.num_actions = NUM_ACTIONS
        self.steps_done = 0
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        INPUT_SIZE = 4
        OUTPUT_SIZE = NUM_ACTIONS ** 2

        self.policy_net1 = QNetwork(INPUT_SIZE, OUTPUT_SIZE).to(self.device)
        self.policy_net2 = QNetwork(INPUT_SIZE, OUTPUT_SIZE).to(self.device)
        self.target_net1 = QNetwork(INPUT_SIZE, OUTPUT_SIZE).to(self.device)
        self.target_net2 = QNetwork(INPUT_SIZE, OUTPUT_SIZE).to(self.device)
        self.target_net1.load_state_dict(self.policy_net1.state_dict())
        self.target_net2.load_state_dict(self.policy_net2.state_dict())
        self.target_net1.eval()
        self.target_net2.eval()

        self.optimizer1 = optim.Adam(self.policy_net1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.policy_net2.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.SmoothL1Loss()

    def _state_to_tensor(self, state_coords):
        t = torch.tensor(state_coords, dtype=torch.float32).to(self.device)
        return t.unsqueeze(0) if t.ndim == 1 else t

    def select_action(self, state_coords, epsilon=0.05):
        # epsilon-greedy: with prob epsilon sample uniformly, otherwise sample from LP-derived pi
        if random.random() < epsilon:
            return random.randrange(self.num_actions), random.randrange(self.num_actions)
        st = self._state_to_tensor(state_coords)
        with torch.no_grad():
            q1_flat = self.policy_net1(st).cpu().numpy().reshape(self.num_actions, self.num_actions)
            q2_flat = self.policy_net2(st).cpu().numpy().reshape(self.num_actions, self.num_actions)

            # Solve LP for P1's mixed strategy (row)
            V1, pi1 = solve_minimax_lp_value(q1_flat)
            # Solve LP for P2's mixed strategy (row of q2.T), but we want P2's policy over columns -> solve on q2.T
            V2, pi2 = solve_minimax_lp_value(q2_flat.T)  # returns row-mix for columns of q1_flat

            # sample according to pi1 and pi2
            a1 = int(np.random.choice(self.num_actions, p=pi1))
            a2 = int(np.random.choice(self.num_actions, p=pi2))
        return a1, a2

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        action1_batch = torch.tensor(batch.action1, dtype=torch.int64).to(self.device)
        action2_batch = torch.tensor(batch.action2, dtype=torch.int64).to(self.device)
        reward1_batch = torch.tensor(batch.reward1, dtype=torch.float32).to(self.device)
        reward2_batch = torch.tensor(batch.reward2, dtype=torch.float32).to(self.device)

        # gather predicted Q for taken joint actions
        action_indices = action1_batch * self.num_actions + action2_batch  # flattened index
        q_vals1 = self.policy_net1(state_batch).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)
        q_vals2 = self.policy_net2(state_batch).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        # compute target values using target networks and LP (with caching)
        with torch.no_grad():
            next_q1_flat_all = self.target_net1(next_state_batch).cpu().numpy()  # (batch, 16)
            next_q2_flat_all = self.target_net2(next_state_batch).cpu().numpy()

        # caching: we'll call solve_minimax_lp_value on each unique next-state matrix
        # create keys by rounding
        next_V1 = np.zeros(self.batch_size, dtype=float)
        next_V2 = np.zeros(self.batch_size, dtype=float)

        # Build dict of unique matrices to compute
        unique_q1 = {}
        unique_q2 = {}

        for i in range(self.batch_size):
            mat1 = next_q1_flat_all[i].reshape(self.num_actions, self.num_actions)
            mat2 = next_q2_flat_all[i].reshape(self.num_actions, self.num_actions)
            key1 = (self.num_actions, self.num_actions) + tuple(np.round(mat1.ravel(), 8).tolist())
            key2 = (self.num_actions, self.num_actions) + tuple(np.round(mat2.ravel(), 8).tolist())
            if key1 not in unique_q1:
                unique_q1[key1] = mat1
            if key2 not in unique_q2:
                unique_q2[key2] = mat2

        # Solve LP for unique keys (cached solve_minimax_lp_cached used via solve_minimax_lp_value)
        solved_V1 = {}
        solved_V2 = {}
        for k, mat in unique_q1.items():
            V, pi = solve_minimax_lp_cached(k)  # returns (V, pi)
            solved_V1[k] = V
        for k, mat in unique_q2.items():
            V, pi = solve_minimax_lp_cached(k)
            solved_V2[k] = V

        # Now fill next_V arrays
        for i in range(self.batch_size):
            mat1 = next_q1_flat_all[i].reshape(self.num_actions, self.num_actions)
            mat2 = next_q2_flat_all[i].reshape(self.num_actions, self.num_actions)
            key1 = (self.num_actions, self.num_actions) + tuple(np.round(mat1.ravel(), 8).tolist())
            key2 = (self.num_actions, self.num_actions) + tuple(np.round(mat2.ravel(), 8).tolist())
            next_V1[i] = float(solved_V1[key1])
            # For player 2, we use Q2.T convention (row-player on Q2.T)
            next_V2[i] = float(solved_V2[key2])

        # targets
        expected_q1 = reward1_batch + self.gamma * torch.tensor(next_V1, dtype=torch.float32).to(self.device)
        expected_q2 = reward2_batch + self.gamma * torch.tensor(next_V2, dtype=torch.float32).to(self.device)

        # loss and step
        loss1 = self.loss_fn(q_vals1, expected_q1)
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        loss2 = self.loss_fn(q_vals2, expected_q2)
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net1.load_state_dict(self.policy_net1.state_dict())
            self.target_net2.load_state_dict(self.policy_net2.state_dict())

        return float(loss1.item()), float(loss2.item())


# -------------------------
# Training wrapper for DNQN (planning-mode / learning-mode)
# -------------------------
def run_dnqn_training(crash_cost=10, episodes=2000, steps_per_episode=50, buffer_capacity=5000, render=False):
    solver = DNQN_Solver(target_update=200, buffer_size=buffer_capacity, batch_size=64)
    loss_history = []
    episode_rewards = []

    START_EPS = 1.0
    END_EPS = 0.05
    EPS_DECAY = episodes * steps_per_episode / 2.0

    for ep in range(episodes):
        state_idx = np.random.randint(0, NUM_STATES)
        state_coords = np.array(get_coords_from_state(state_idx), dtype=float)
        ep_reward = 0.0
        for step in range(steps_per_episode):
            eps = END_EPS + (START_EPS - END_EPS) * math.exp(-1.0 * solver.steps_done / EPS_DECAY)
            a1, a2 = solver.select_action(state_coords, epsilon=eps)
            curr_state = state_idx
            nx1, ny1, nx2, ny2 = transition_function(curr_state, a1, a2)
            r1, r2 = reward_function(curr_state, a1, a2, crash_cost, do_scale=True)
            next_coords = np.array([nx1, ny1, nx2, ny2], dtype=float)
            solver.memory.push(state_coords, a1, a2, r1, r2, next_coords)

            state_coords = next_coords
            state_idx = coords_to_state(nx1, ny1, nx2, ny2)
            ep_reward += r1  # track P1 reward (scaled)

            l1, l2 = solver.optimize_model()

        episode_rewards.append(ep_reward)
        loss_history.append((l1 + l2) / 2.0)

        if (ep + 1) % 100 == 0:
            avg_recent = np.mean(episode_rewards[max(0, len(episode_rewards)-100):])
            print(f"EP {ep+1}/{episodes} avg_reward_last100={avg_recent:.4f} eps={eps:.3f}")

    return solver, loss_history, episode_rewards


# -------------------------
# Experiment helpers & plotting
# -------------------------
def run_crash_cost_sweep(crash_costs=(0, 1, 10, 100), tabular_iters=2000, dnqn_episodes=800):
    results = {}
    for c in crash_costs:
        print(f"\n\n=== Running tabular Minimax-Q for crash_cost={c} ===")
        t0 = time.time()
        Q1_tab, Q2_tab, vh = solve_markov_game(crash_cost=c, num_iterations=tabular_iters, do_scale=True)
        t1 = time.time()
        print(f"Tabular done in {t1-t0:.1f}s")

        print(f"Visualizing Tabular results (crash_cost={c})")
        visualize_tabular_results(Q1_tab, c, vh)

        print(f"Running DNQN learning (crash_cost={c})")
        solver, loss_hist, ep_rewards = run_dnqn_training(crash_cost=c, episodes=dnqn_episodes)
        results[c] = {
            'tabular_Q1': Q1_tab,
            'value_history': vh,
            'dnqn_solver': solver,
            'dnqn_loss': loss_hist,
            'dnqn_rewards': ep_rewards
        }
    return results


def plot_experiment_results(results):
    # results: dict keyed by crash_cost -> dict with 'dnqn_rewards' etc.
    plt.figure(figsize=(10, 6))
    for c, d in results.items():
        rewards = d['dnqn_rewards']
        smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
        plt.plot(smoothed, label=f'crash={c}')
    plt.title("DNQN smoothed episode reward (window=20)")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Value history from tabular
    plt.figure(figsize=(10, 6))
    for c, d in results.items():
        vh = d['value_history']
        plt.plot(vh, label=f'crash={c}')
    plt.title("Tabular center-state maximin value history")
    plt.xlabel("Iteration")
    plt.ylabel("Maximin value")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------
# Quick checklist / plan (textual)
# -------------------------
CHECKLIST_TEXT = """
Experiment checklist:
1) Confirm dependencies: numpy, scipy, torch, matplotlib, seaborn, nashpy.
2) Choose crash cost sweep values (e.g. [0,1,10,100,127]).
3) Run the tabular solver first with modest iterations (e.g. 2000) to get baseline Q.
4) Run DNQN in planning-mode (use transitions generated from tabular Q) to test function approximation parity.
5) Run DNQN learning-mode with replay + target network; set smaller episodes for debugging then enlarge.
6) Produce: (a) center-state maximin convergence curve (tabular), (b) DNQN episode reward curves, (c) policy entropy & crash-rate vs time.
7) To compute crash-rate: run many rollouts following greedy/mix policy and measure fraction of collisions.
"""

CHECKLIST_TEXT += "\nNotes:\n- If DNQN diverges, remove reward scaling and/or lower lr to 1e-4.\n- Use Huber loss (already set) and larger replay buffer for stability.\n- LP caching is active; for very large architectures you may need a faster LP solver.\n"

# -------------------------
# If run as script, demonstrate a short smoke test
# -------------------------
if __name__ == "__main__":
    print("Minimax driving demo: tabular (small test) + DNQN (short train).")
    print(CHECKLIST_TEXT)

    # quick smoke: small tabular run
    CRASH = 10
    Q1_tab, Q2_tab, value_hist = solve_markov_game(crash_cost=CRASH, num_iterations=500)
    visualize_tabular_results(Q1_tab, CRASH, value_hist)

    # short DNQN train
    solver, losses, rewards = run_dnqn_training(crash_cost=CRASH, episodes=300, steps_per_episode=30)
    plt.figure(figsize=(8,4))
    plt.plot(np.convolve(rewards, np.ones(10)/10, mode='valid'))
    plt.title('DNQN smoothed rewards (demo)')
    plt.show()
