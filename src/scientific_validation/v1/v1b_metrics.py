"""V1-B metric and bounded-memory utilities.

No scientific decision logic belongs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import math

import numpy as np


def flatten_native(x: np.ndarray) -> np.ndarray:
    """Return per-sample native vectors for distance computation.

    This is an implementation-level vectorization of a single sample.
    It does NOT concatenate hierarchy levels.  For structured Z/S/S_tilde
    representations, the complete native sample is vectorized only to evaluate
    its within-level Euclidean distance.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    return x.reshape(x.shape[0], -1)


def native_pair_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance for paired samples at one hierarchy level."""
    aa = flatten_native(a).astype(np.float64, copy=False)
    bb = flatten_native(b).astype(np.float64, copy=False)
    if aa.shape != bb.shape:
        raise ValueError(f"Paired shapes differ: {aa.shape} vs {bb.shape}")
    return np.linalg.norm(aa - bb, axis=1)


def within_chunk_pair_distances(x: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise distances for one bounded chunk.

    Used only for bounded R1 distribution sampling.
    """
    v = flatten_native(x).astype(np.float64, copy=False)
    n = len(v)
    if n < 2:
        return np.empty(0, dtype=np.float64)
    out = []
    for i in range(n - 1):
        d = np.linalg.norm(v[i + 1:] - v[i], axis=1)
        out.append(d)
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def behavioral_window_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Frozen R2 normalized Frobenius distance.

    d_X = ||X_i-X_j||_F / (W*N)

    W=30, N=23 for CU/HDEG V1.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Window shapes differ: {x1.shape} vs {x2.shape}")
    return float(np.linalg.norm(
        np.asarray(x1, dtype=np.float64) -
        np.asarray(x2, dtype=np.float64)
    ) / float(x1.size))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without requiring sklearn."""
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance using scipy's unbiased sample implementation."""
    from scipy.stats import energy_distance as scipy_energy_distance
    return float(scipy_energy_distance(x, y))


def percentile_summary(values: np.ndarray) -> dict:
    """Robust R1 distribution summary."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {
            "count": 0,
            "median": None,
            "iqr": None,
            "q01": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
            "q99": None,
            "mean": None,
            "std": None,
        }
    qs = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(len(v)),
        "median": float(qs[3]),
        "iqr": float(qs[4] - qs[2]),
        "q01": float(qs[0]),
        "q05": float(qs[1]),
        "q25": float(qs[2]),
        "q50": float(qs[3]),
        "q75": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
    }


@dataclass
class Reservoir:
    """Deterministic bounded reservoir for R1 distribution evidence."""
    capacity: int
    seed: int = 42

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.rng = np.random.default_rng(self.seed)
        self.data = np.empty(self.capacity, dtype=np.float64)
        self.size = 0
        self.seen = 0

    def update(self, values: Iterable[float]) -> None:
        for value in values:
            value = float(value)
            if not math.isfinite(value):
                continue
            self.seen += 1
            if self.size < self.capacity:
                self.data[self.size] = value
                self.size += 1
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.capacity:
                    self.data[j] = value

    def values(self) -> np.ndarray:
        return self.data[:self.size].copy()


def moving_block_bootstrap_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate n indices using a moving-block bootstrap."""
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if block_length > n:
        block_length = n

    starts = np.arange(0, n - block_length + 1, dtype=np.int64)
    result = []
    total = 0
    while total < n:
        s = int(rng.choice(starts))
        block = np.arange(s, s + block_length, dtype=np.int64)
        result.append(block)
        total += len(block)
    return np.concatenate(result)[:n]


def bootstrap_energy_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_length: int = 30,
    replicates: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Moving-block bootstrap CI for Energy distance.

    Bootstrap is performed independently within each condition.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    observed = energy_distance(x, y)
    vals = np.empty(replicates, dtype=np.float64)

    for b in range(replicates):
        ix = moving_block_bootstrap_indices(len(x), block_length, rng)
        iy = moving_block_bootstrap_indices(len(y), block_length, rng)
        vals[b] = energy_distance(x[ix], y[iy])

    alpha = 1.0 - confidence
    lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "point_estimate": float(observed),
        "confidence_level": float(confidence),
        "interval_method": "percentile",
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "bootstrap_replicates": int(replicates),
        "block_length_windows": int(block_length),
        "bootstrap_seed": int(seed),
    }
