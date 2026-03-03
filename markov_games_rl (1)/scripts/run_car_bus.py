import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from pathlib import Path
import numpy as np
import torch

from mg.envs import CarBusGame
from mg.utils import RunConfig, set_seed, ensure_dir, save_json
from mg.policy_grad import train_reinforce, train_a2c
from mg.viz import save_log_csv, plot_compare

def add_set_state(env: CarBusGame):
    # Adds method used by planning_minimax_q if you want planning later
    def _set_state_from_id(sid: int):
        g = env.grid_size
        by = sid % g
        sid //= g
        bx = sid % g
        sid //= g
        cy = sid % g
        cx = sid // g
        env.t = 0
        env.car = [int(cx), int(cy)]
        env.bus = [int(bx), int(by)]
    env._set_state_from_id = _set_state_from_id
    return env

def main():
    import os
    RUN_DQN = bool(int(os.environ.get('RUN_DQN','0')))
    cfg = RunConfig(seed=0, episodes=200, max_steps_per_episode=25, gamma=0.95, lr=1e-3, batch_size=128, device="cpu",
                    epsilon_decay_steps=15000, target_update=500)
    set_seed(cfg.seed)
    env = add_set_state(CarBusGame(grid_size=3, crash_cost=10.0, goal_reward=10.0, step_cost=0.1, max_steps=25))

    out = ensure_dir(Path("outputs")/"car_bus")
    save_json(out/"config.json", cfg.to_dict())

    # DQN (optional; enable with RUN_DQN=1)
    if RUN_DQN:
            qnet, log_dqn = train_dqn_minimax(env, cfg, out/"dqn")
            torch.save(qnet.state_dict(), out/"dqn_qnet.pt")
            save_log_csv(out/"dqn_log.csv", log_dqn)

    # REINFORCE (no baseline)
    (pi1,pi2), log_rein = train_reinforce(env, cfg, out/"reinforce", baseline="none")
    torch.save(pi1.state_dict(), out/"reinforce_pi1.pt")
    torch.save(pi2.state_dict(), out/"reinforce_pi2.pt")
    save_log_csv(out/"reinforce_log.csv", log_rein)

    # A2C
    (pi1a,pi2a,V), log_a2c = train_a2c(env, cfg, out/"a2c")
    torch.save(pi1a.state_dict(), out/"a2c_pi1.pt")
    torch.save(pi2a.state_dict(), out/"a2c_pi2.pt")
    torch.save(V.state_dict(), out/"a2c_value.pt")
    save_log_csv(out/"a2c_log.csv", log_a2c)

    plot_compare(log_a2c, log_rein, "A2C", "REINFORCE", out/"a2c_vs_reinforce.png", ma_window=50, title="Car-Bus: A2C vs REINFORCE (P1 return)")
    print("Done. Outputs in:", out.resolve())

if __name__ == "__main__":
    main()