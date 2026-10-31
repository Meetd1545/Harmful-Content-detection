"""Temperature scaling (one temperature per dimension) and expected calibration error (ECE)."""
import numpy as np
from scipy.optimize import minimize_scalar

from config import UNKNOWN


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperatures(logits, labels):
    """logits, labels: arrays of shape (n, k); labels may contain -1 (ignored)."""
    temps = []
    for j in range(logits.shape[1]):
        m = labels[:, j] != UNKNOWN
        z, y = logits[m, j], labels[m, j]
        if len(y) == 0 or len(set(y)) < 2:
            temps.append(1.0)
            continue

        def nll(t):
            p = np.clip(sigmoid(z / t), 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        temps.append(float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x))
    return np.array(temps)


def apply_temperatures(logits, temps):
    return sigmoid(logits / temps)


def ece(probs, labels, n_bins=10):
    """Expected calibration error for binary probabilities (positive class), ignoring -1 labels."""
    m = labels != UNKNOWN
    p, y = probs[m], labels[m]
    if len(y) == 0:
        return float("nan")
    conf = np.where(p >= 0.5, p, 1 - p)
    pred = (p >= 0.5).astype(int)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if s.any():
            total += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
    return float(total)
