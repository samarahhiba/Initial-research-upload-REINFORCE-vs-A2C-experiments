
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

def save_log_csv(path: str | Path, log: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(log[0].keys()) if log else ["episode","return_p1"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in log:
            w.writerow(row)

def moving_avg(x, k=50):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    w = np.ones(k)/k
    return np.convolve(x, w, mode="valid")

def plot_compare(log_a, log_b, label_a, label_b, outpath: str | Path, ma_window=50, title="Training return (P1)"):
    ra = [r["return_p1"] for r in log_a]
    rb = [r["return_p1"] for r in log_b]
    ma = moving_avg(ra, ma_window)
    mb = moving_avg(rb, ma_window)

    plt.figure()
    plt.plot(ma, label=label_a)
    plt.plot(mb, label=label_b)
    plt.legend()
    plt.title(title)
    plt.xlabel(f"episode (moving avg window={ma_window})")
    plt.ylabel("return")
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", dpi=160)
    plt.close()
