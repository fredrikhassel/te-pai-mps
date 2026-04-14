"""Numba-accelerated circuit sampling for TE-PAI."""

from numba import jit
import numpy as np
import multiprocessing as mp
import sys


def batch_sampling(probs, batch_size, n_workers=None, seed=None):
    """Sample batch_size circuits in parallel using multiprocessing.

    Uses 'fork' context on Unix (safe with numba) and falls back to
    sequential execution on platforms where fork is unavailable.

    Args:
        probs: Array of probability distributions for each step/term.
        batch_size: Number of circuits to sample.
        n_workers: Number of parallel workers (default: cpu_count - 2).
        seed: Optional int seed for reproducibility.  When provided every
              worker gets a deterministic sub-seed so results are identical
              across runs.
    """
    # Generate deterministic per-circuit seeds when a seed is given.
    if seed is not None:
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**63, size=batch_size).tolist()
        args = [(probs, s) for s in seeds]
    else:
        args = [(probs, None) for _ in range(batch_size)]

    workers = n_workers if n_workers is not None else max(1, mp.cpu_count() - 2)
    if sys.platform != "win32":
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            return pool.map(_sample_seeded, args)
    else:
        return [_sample_seeded(a) for a in args]


def _sample_seeded(args):
    """Wrapper that seeds numba's RNG before sampling."""
    probs, seed = args
    if seed is not None:
        _numba_seed(seed)
    return sample_from_prob(probs)


@jit(nopython=True)
def _numba_seed(seed):
    """Seed numba's internal RNG for reproducibility."""
    np.random.seed(seed)


@jit(nopython=True)
def custom_random_choice(prob):
    """Weighted random choice returning 1-based index."""
    r = np.random.random()
    cum_prob = 0.0
    for idx in range(len(prob)):
        cum_prob += prob[idx]
        if r < cum_prob:
            return idx + 1


@jit(nopython=True)
def sample_from_prob(probs):
    """Sample one circuit: for each step and term, draw from probability distribution.
    Returns list of (term_index, sampled_value) for non-identity gates only."""
    res = []
    for i in range(probs.shape[0]):
        res2 = []
        for j in range(probs.shape[1]):
            val = custom_random_choice(probs[i][j])
            if val != 1:
                res2.append((j, val))
        res.append(res2)
    return res
