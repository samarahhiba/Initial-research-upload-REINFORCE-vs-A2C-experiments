
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from mg.envs import RPSGame, CarBusGame
from mg.utils import RunConfig, set_seed, ensure_dir, save_json
from mg.policy_grad import train_reinforce, train_a2c
from mg.viz import save_log_csv, plot_compare

def run_env(env_name: str):
    if env_name == "rps":
        env = RPSGame()
        cfg = RunConfig(seed=0, episodes=400, max_steps_per_episode=1, gamma=0.95, lr=1e-3, device="cpu")
    else:
        env = CarBusGame(grid_size=3, crash_cost=10.0, goal_reward=10.0, step_cost=0.1, max_steps=25)
        cfg = RunConfig(seed=0, episodes=600, max_steps_per_episode=25, gamma=0.95, lr=1e-3, device="cpu")

    set_seed(cfg.seed)
    out = ensure_dir(Path("outputs")/f"{env_name}_pg_demo")
    save_json(out/"config.json", cfg.to_dict())

    (pi1,pi2), log_rein = train_reinforce(env, cfg, out/"reinforce", baseline="none")
    torch.save(pi1.state_dict(), out/"reinforce_pi1.pt")
    torch.save(pi2.state_dict(), out/"reinforce_pi2.pt")
    save_log_csv(out/"reinforce_log.csv", log_rein)

    (pi1a,pi2a,V), log_a2c = train_a2c(env, cfg, out/"a2c")
    torch.save(pi1a.state_dict(), out/"a2c_pi1.pt")
    torch.save(pi2a.state_dict(), out/"a2c_pi2.pt")
    torch.save(V.state_dict(), out/"a2c_value.pt")
    save_log_csv(out/"a2c_log.csv", log_a2c)

    plot_compare(log_a2c, log_rein, "A2C", "REINFORCE", out/"a2c_vs_reinforce.png",
                 ma_window=50, title=f"{env_name.upper()}: A2C vs REINFORCE (P1 return)")

def main():
    run_env("rps")
    run_env("car_bus")

if __name__ == "__main__":
    main()
