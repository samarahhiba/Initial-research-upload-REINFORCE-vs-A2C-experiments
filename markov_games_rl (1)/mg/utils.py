
import os, json, random
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: str | Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

@dataclass
class RunConfig:
    seed: int = 0
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50_000
    target_update: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    max_steps_per_episode: int = 25
    episodes: int = 5_000
    device: str = "cpu"

    def to_dict(self):
        return asdict(self)
