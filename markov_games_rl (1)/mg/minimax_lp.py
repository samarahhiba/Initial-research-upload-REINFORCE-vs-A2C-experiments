
from __future__ import annotations
import numpy as np
from functools import lru_cache
from scipy.optimize import linprog

def _sanitize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    assert M.ndim == 2
    return M

@lru_cache(maxsize=200_000)
def _solve_cached(key: bytes, n: int, m: int):
    M = np.frombuffer(key, dtype=np.float64).reshape(n,m)
    return solve_minimax(M)

def solve_minimax(M: np.ndarray):
    """Solve max_pi min_j sum_i pi_i M[i,j] for zero-sum.
    Returns (V, pi) where pi is row player's mixed strategy.
    """
    M = _sanitize(M)
    n, m = M.shape
    # Variables: pi_0..pi_{n-1}, v
    # Max v s.t. M^T pi >= v*1, sum pi=1, pi>=0
    # Convert to linprog minimization: minimize -v
    c = np.zeros(n+1)
    c[-1] = -1.0

    # Inequalities: -(M^T pi - v) <= 0  =>  -M^T pi + v <= 0
    A_ub = np.zeros((m, n+1))
    A_ub[:, :n] = -M.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(m)

    A_eq = np.zeros((1, n+1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)]*n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    pi = res.x[:n]
    pi = np.clip(pi, 0, 1)
    if pi.sum() <= 0:
        pi = np.ones(n)/n
    else:
        pi = pi/pi.sum()
    v = res.x[-1]
    return float(v), pi

def solve_minimax_cached(M: np.ndarray):
    M = np.asarray(M, dtype=np.float64)
    key = M.tobytes()
    return _solve_cached(key, M.shape[0], M.shape[1])
