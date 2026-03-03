
from __future__ import annotations
import math, random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .minimax_lp import solve_minimax_cached

@dataclass
class Transition:
    s: int
    a1: int
    a2: int
    r: float
    s2: int
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf = []
        self.i = 0

    def push(self, tr: Transition):
        if len(self.buf) < self.capacity:
            self.buf.append(tr)
        else:
            self.buf[self.i] = tr
        self.i = (self.i + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)

class QNet(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(n_states, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, n_actions*n_actions)

    def forward(self, s_idx: torch.Tensor):
        x = self.embed(s_idx)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)  # linear output (no sigmoid)
        return q

def epsilon_by_step(step: int, eps_start: float, eps_end: float, decay_steps: int):
    if step >= decay_steps:
        return eps_end
    t = step/decay_steps
    return eps_start + t*(eps_end-eps_start)

@torch.no_grad()
def select_actions_from_Q(env, qnet: QNet, s: int, eps: float, device="cpu"):
    A = env.n_actions if hasattr(env,"n_actions") else env.n_actions
    if random.random() < eps:
        return random.randrange(A), random.randrange(A)
    s_t = torch.tensor([s], dtype=torch.long, device=device)
    q = qnet(s_t).view(A,A).cpu().numpy()
    # row player uses minimax mixed strategy; column player best-responds (for behavior we sample)
    v, pi_row = solve_minimax_cached(q)
    a1 = int(np.random.choice(A, p=pi_row))
    # simple behavior for player2: best response to sampled a1
    a2 = int(np.argmin(q[a1,:]))
    return a1,a2

def train_dqn_minimax(env, cfg, outdir, hidden=128):
    device = torch.device(cfg.device)
    A = env.n_actions if hasattr(env,"n_actions") else env.n_actions
    q = QNet(env.n_states, A, hidden=hidden).to(device)
    q_targ = QNet(env.n_states, A, hidden=hidden).to(device)
    q_targ.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    rb = ReplayBuffer(cfg.replay_size)

    log = []
    step = 0
    for ep in range(cfg.episodes):
        s = env.reset()
        ep_ret = 0.0
        for t in range(cfg.max_steps_per_episode):
            eps = epsilon_by_step(step, cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps)
            a1,a2 = select_actions_from_Q(env, q, s, eps, device=device)
            s2, r1, _, done, _ = env.step(a1,a2)
            rb.push(Transition(s,a1,a2,r1,s2,done))
            ep_ret += r1
            s = s2
            step += 1

            if len(rb) >= cfg.batch_size:
                batch = rb.sample(cfg.batch_size)
                s_b = torch.tensor([b.s for b in batch], dtype=torch.long, device=device)
                a1_b = torch.tensor([b.a1 for b in batch], dtype=torch.long, device=device)
                a2_b = torch.tensor([b.a2 for b in batch], dtype=torch.long, device=device)
                r_b = torch.tensor([b.r for b in batch], dtype=torch.float32, device=device)
                s2_b = torch.tensor([b.s2 for b in batch], dtype=torch.long, device=device)
                done_b = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device)

                q_sa = q(s_b).view(-1, A, A)
                q_sa = q_sa[torch.arange(cfg.batch_size), a1_b, a2_b]

                with torch.no_grad():
                    q2 = q_targ(s2_b).view(-1, A, A).cpu().numpy()
                    v2 = []
                    for i in range(cfg.batch_size):
                        if done_b[i].item() > 0.5:
                            v2.append(0.0)
                        else:
                            v, _ = solve_minimax_cached(q2[i])
                            v2.append(v)
                    v2 = torch.tensor(v2, dtype=torch.float32, device=device)

                y = r_b + cfg.gamma * (1.0 - done_b) * v2
                loss = F.mse_loss(q_sa, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()

                if step % cfg.target_update == 0:
                    q_targ.load_state_dict(q.state_dict())

            if done:
                break

        log.append({"episode": ep, "return_p1": float(ep_ret)})

    return q, log
