import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from pathlib import Path
import torch
from mg.envs import RPSGame
from mg.utils import RunConfig, set_seed, ensure_dir, save_json
from mg.policy_grad import train_reinforce, train_a2c
from mg.viz import save_log_csv, plot_compare

def main():
    import os
    RUN_DQN = bool(int(os.environ.get('RUN_DQN','0')))
    cfg = RunConfig(seed=0, episodes=120, max_steps_per_episode=1, gamma=0.95, lr=1e-3, batch_size=64, device="cpu")
    set_seed(cfg.seed)
    env = RPSGame()

    out = ensure_dir(Path("outputs")/"rps")
    save_json(out/"config.json", cfg.to_dict())

    # DQN (optional; enable with RUN_DQN=1)
    if RUN_DQN:
            qnet, log_dqn = train_dqn_minimax(env, cfg, out/"dqn")
            torch.save(qnet.state_dict(), out/"dqn_qnet.pt")
            save_log_csv(out/"dqn_log.csv", log_dqn)

    # REINFORCE
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

    # Plot A2C vs REINFORCE (A2C smoother typically)
    plot_compare(log_a2c, log_rein, "A2C", "REINFORCE", out/"a2c_vs_reinforce.png", ma_window=50, title="RPS: A2C vs REINFORCE (P1 return)")
    print("Done. Outputs in:", out.resolve())

if __name__ == "__main__":
    main()