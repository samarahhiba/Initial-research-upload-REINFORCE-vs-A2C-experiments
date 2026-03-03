
from __future__ import annotations
import math, random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(n_states, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, n_actions)

    def forward(self, s_idx: torch.Tensor):
        x = self.embed(s_idx)
        x = F.relu(self.fc1(x))
        return self.logits(x)

    def dist(self, s_idx: torch.Tensor):
        logits = self.forward(s_idx)
        return torch.distributions.Categorical(logits=logits)

class ValueNet(nn.Module):
    def __init__(self, n_states: int, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(n_states, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, s_idx: torch.Tensor):
        x = self.embed(s_idx)
        x = F.relu(self.fc1(x))
        return self.v(x).squeeze(-1)

def rollout_episode(env, pi1: PolicyNet, pi2: PolicyNet, device, max_steps: int):
    s = env.reset()
    traj = []
    ret1 = 0.0
    for t in range(max_steps):
        s_t = torch.tensor([s], dtype=torch.long, device=device)
        d1 = pi1.dist(s_t)
        d2 = pi2.dist(s_t)
        a1 = int(d1.sample().item())
        a2 = int(d2.sample().item())
        logp1 = d1.log_prob(torch.tensor(a1, device=device))
        logp2 = d2.log_prob(torch.tensor(a2, device=device))

        s2, r1, r2, done, _ = env.step(a1,a2)
        traj.append((s, a1, a2, float(r1), float(r2), logp1, logp2))
        ret1 += r1
        s = s2
        if done:
            break
    return traj, ret1

def compute_returns(rews, gamma: float):
    G = 0.0
    out = []
    for r in reversed(rews):
        G = r + gamma*G
        out.append(G)
    out.reverse()
    return out

def train_reinforce(env, cfg, outdir, hidden=128, baseline="none", v_star_fn=None):
    """baseline: 'none' | 'vstar' (requires v_star_fn(state)->float)"""
    device = torch.device(cfg.device)
    A = env.n_actions if hasattr(env,"n_actions") else env.n_actions
    pi1 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    pi2 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    opt1 = torch.optim.Adam(pi1.parameters(), lr=cfg.lr)
    opt2 = torch.optim.Adam(pi2.parameters(), lr=cfg.lr)

    log = []
    for ep in range(cfg.episodes):
        traj, ep_ret1 = rollout_episode(env, pi1, pi2, device, cfg.max_steps_per_episode)
        rews1 = [x[3] for x in traj]
        rews2 = [x[4] for x in traj]
        G1 = compute_returns(rews1, cfg.gamma)
        G2 = compute_returns(rews2, cfg.gamma)

        # optional baseline
        b1 = []
        b2 = []
        for (s, *_rest) in traj:
            if baseline == "vstar" and v_star_fn is not None:
                v = float(v_star_fn(s))
                b1.append(v)
                b2.append(-v)
            else:
                b1.append(0.0); b2.append(0.0)

        loss1 = 0.0
        loss2 = 0.0
        for i,(s,a1,a2,r1,r2,logp1,logp2) in enumerate(traj):
            adv1 = (G1[i] - b1[i])
            adv2 = (G2[i] - b2[i])
            loss1 = loss1 + (-logp1 * adv1)
            loss2 = loss2 + (-logp2 * adv2)

        opt1.zero_grad(); opt2.zero_grad()
        loss1.backward(); loss2.backward()
        nn.utils.clip_grad_norm_(pi1.parameters(), 10.0)
        nn.utils.clip_grad_norm_(pi2.parameters(), 10.0)
        opt1.step(); opt2.step()

        log.append({"episode": ep, "return_p1": float(ep_ret1)})

    return (pi1,pi2), log

def train_a2c(env, cfg, outdir, hidden=128):
    device = torch.device(cfg.device)
    A = env.n_actions if hasattr(env,"n_actions") else env.n_actions
    pi1 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    pi2 = PolicyNet(env.n_states, A, hidden=hidden).to(device)
    V = ValueNet(env.n_states, hidden=hidden).to(device)

    opt_pi1 = torch.optim.Adam(pi1.parameters(), lr=cfg.lr)
    opt_pi2 = torch.optim.Adam(pi2.parameters(), lr=cfg.lr)
    opt_V = torch.optim.Adam(V.parameters(), lr=cfg.lr)

    log = []
    for ep in range(cfg.episodes):
        traj, ep_ret1 = rollout_episode(env, pi1, pi2, device, cfg.max_steps_per_episode)
        rews1 = [x[3] for x in traj]
        G1 = compute_returns(rews1, cfg.gamma)

        s_list = [x[0] for x in traj]
        s_t = torch.tensor(s_list, dtype=torch.long, device=device)
        Vpred = V(s_t)

        G1_t = torch.tensor(G1, dtype=torch.float32, device=device)
        adv1 = (G1_t - Vpred).detach()

        # actor losses
        loss_pi1 = 0.0
        loss_pi2 = 0.0
        for i,(_,a1,a2,_,_,logp1,logp2) in enumerate(traj):
            # For zero-sum, use opposite advantage for player2
            loss_pi1 = loss_pi1 + (-logp1 * adv1[i])
            loss_pi2 = loss_pi2 + (-logp2 * (-adv1[i]))

        # critic loss
        loss_V = F.mse_loss(Vpred, G1_t)

        opt_pi1.zero_grad(); opt_pi2.zero_grad(); opt_V.zero_grad()
        (loss_pi1 + loss_pi2 + loss_V).backward()
        nn.utils.clip_grad_norm_(list(pi1.parameters())+list(pi2.parameters())+list(V.parameters()), 10.0)
        opt_pi1.step(); opt_pi2.step(); opt_V.step()

        log.append({"episode": ep, "return_p1": float(ep_ret1)})

    return (pi1,pi2,V), log
