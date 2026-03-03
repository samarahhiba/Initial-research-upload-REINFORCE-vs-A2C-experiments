
from __future__ import annotations
import numpy as np
from .minimax_lp import solve_minimax_cached

def planning_minimax_q(env, gamma: float = 0.95, iters: int = 10_000, tol: float = 1e-8):
    """Tabular planning: value-iteration over Q(s,a1,a2) using minimax on next state."""
    S = env.n_states
    A = env.n_actions if hasattr(env, "n_actions") else env.n_actions  # both players same
    Q = np.zeros((S, A, A), dtype=float)

    def V_of(s):
        M = Q[s]  # (A,A)
        v, _ = solve_minimax_cached(M)
        return v

    for it in range(iters):
        delta = 0.0
        for s in range(S):
            for a1 in range(A):
                for a2 in range(A):
                    # simulate one-step from state s: we need env that can step from arbitrary s
                    # For small games we can brute force by temporarily setting state.
                    if hasattr(env, "_set_state_from_id"):
                        env._set_state_from_id(s)
                    else:
                        # Stateless (RPS)
                        pass
                    s2, r1, _, done, _ = env.step(a1,a2)
                    target = r1 if done else (r1 + gamma * V_of(s2))
                    old = Q[s,a1,a2]
                    Q[s,a1,a2] = target
                    delta = max(delta, abs(old-target))
        if delta < tol:
            break
    return Q
