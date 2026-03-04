"""Nonconformity score functions for CPSL.

Two methods for aggregating frame-level detector scores into a single
utterance-level nonconformity score:

- ``max``: s(x) = max_t f(x_t)
  Simple and interpretable; sensitive to the single most anomalous frame.

- ``logsumexp``: s(x) = (1/beta) log(Sum_t exp(beta * f(x_t))) - log(T)/beta
  Smooth differentiable approximation to max; beta controls sharpness.
  As beta -> inf, converges to max.  The -log(T)/beta term normalises
  for sequence length.

Reference
---------
Romano, Sesia, Candes. "Classification with Valid and Adaptive Coverage",
NeurIPS 2020.  (APS nonconformity design principles.)
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as _logsumexp


def max_score(frame_scores: np.ndarray) -> float:
    """Compute max-aggregated nonconformity score.

    Parameters
    ----------
    frame_scores : np.ndarray, shape (T,)
        Per-frame detector scores in [0, 1].

    Returns
    -------
    float
        Maximum frame score.
    """
    return float(np.max(frame_scores))


def logsumexp_score(frame_scores: np.ndarray, beta: float = 10.0) -> float:
    """Compute logsumexp-aggregated nonconformity score.

    s(x) = (1/beta) * log(Sum_t exp(beta * f(x_t))) - log(T) / beta

    Parameters
    ----------
    frame_scores : np.ndarray, shape (T,)
        Per-frame detector scores in [0, 1].
    beta : float
        Sharpness parameter.  Higher = closer to max.

    Returns
    -------
    float
        Smooth-max aggregated score.
    """
    T = len(frame_scores)
    return float(_logsumexp(beta * frame_scores) / beta - np.log(T) / beta)


def compute_nonconformity(
    frame_scores_list: list[np.ndarray],
    method: str = "max",
    beta: float = 10.0,
) -> np.ndarray:
    """Compute nonconformity scores for a batch of utterances.

    Parameters
    ----------
    frame_scores_list : list of np.ndarray
        Each element is shape (T_i,) with per-frame detector scores.
    method : str
        Aggregation method: ``"max"`` or ``"logsumexp"``.
    beta : float
        Sharpness parameter for logsumexp (ignored if method="max").

    Returns
    -------
    np.ndarray, shape (n_utterances,)
        Utterance-level nonconformity scores.

    Raises
    ------
    ValueError
        If method is not recognised.
    """
    if method == "max":
        return np.array([max_score(fs) for fs in frame_scores_list])
    elif method == "logsumexp":
        return np.array([logsumexp_score(fs, beta) for fs in frame_scores_list])
    else:
        raise ValueError(f"Unknown method: {method}. Use 'max' or 'logsumexp'.")
