
# Markov Games: Minimax-DQN, REINFORCE, and A2C (RPS + Car-Bus)

This mini-project implements **zero-sum Markov-game RL** for:

1. **Rock–Paper–Scissors (RPS)**: stateless, single-step, 2-player zero-sum.
2. **Car–Bus game (3×3 grid, no wrap-around)**: state is joint positions (car_x, car_y, bus_x, bus_y).
   - Car (P1) tries to reach a goal square.
   - Bus (P2) tries to collide with the car.
   - Rewards are for **P1**; **P2 gets negative** (zero-sum).

Algorithms implemented:

- **Minimax-DQN** (DQN target network + replay, backup uses **minimax value** of the next-state stage game)
- **REINFORCE** (policy gradient, baseline optional in code)
- **A2C** (actor-critic; typically smoother learning curves than raw REINFORCE)

---

## Setup

Requires: Python 3.10+.

Suggested install:

```bash
pip install numpy torch scipy matplotlib
```

---

## Run

From the project root:

### RPS
```bash
python scripts/run_rps.py
```

### Car–Bus
```bash
python scripts/run_car_bus.py
```

All outputs go into `outputs/`.

---

## Output files (what each one means)

Each environment creates a folder:

- `outputs/rps/`
- `outputs/car_bus/`

Inside each folder:

### Common
- `config.json`  
  The hyperparameters used for the run (seed, gamma, learning rate, etc).

### DQN (minimax)
- `dqn_qnet.pt`  
  PyTorch weights for the learned Q network **Q(s,a1,a2)**.
- `dqn_log.csv`  
  Per-episode log with at least:
  - `episode`
  - `return_p1` (episode return for player 1)

### REINFORCE
- `reinforce_pi1.pt`  
  Policy network weights for player 1.
- `reinforce_pi2.pt`  
  Policy network weights for player 2.
- `reinforce_log.csv`  
  Per-episode log (`return_p1`).

### A2C
- `a2c_pi1.pt`  
  Actor weights for player 1.
- `a2c_pi2.pt`  
  Actor weights for player 2.
- `a2c_value.pt`  
  Critic/value network weights **V(s)** (for P1; P2 is -V in a zero-sum game).
- `a2c_log.csv`  
  Per-episode log (`return_p1`).

### Plot (the “photo” comparison)
- `a2c_vs_reinforce.png`  
  A matplotlib plot comparing **moving-average** episode returns:
  - A2C curve (usually smoother)
  - REINFORCE curve (usually higher variance)

---

## Code map

- `mg/envs.py`  
  RPS + CarBus environments.
- `mg/minimax_lp.py`  
  LP-based minimax solver for stage games.
- `mg/dqn.py`  
  Minimax-DQN (target net + replay).
- `mg/policy_grad.py`  
  REINFORCE + A2C.
- `mg/viz.py`  
  CSV logger + plot utility.
- `scripts/run_rps.py`, `scripts/run_car_bus.py`  
  Entry points.

---

## Notes

- Q-network output is **linear** (no sigmoid) to avoid saturating Q values.
- Minimax value is computed by solving an LP per state (cached for speed).
