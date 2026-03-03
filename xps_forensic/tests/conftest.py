"""Shared test fixtures for XPS-Forensic."""
import pytest
import numpy as np


@pytest.fixture
def rng():
    """Reproducible random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def dummy_frame_scores(rng):
    """Simulated frame-level detector scores for 10 utterances.
    Returns list of arrays, each shape (n_frames,) with values in [0,1].
    Odd-indexed utterances have a 'spoofed' segment in the middle.
    """
    scores = []
    for i in range(10):
        n_frames = rng.integers(100, 500)
        s = rng.uniform(0.1, 0.4, size=n_frames)
        if i % 2 == 1:
            start = n_frames // 3
            end = 2 * n_frames // 3
            s[start:end] = rng.uniform(0.7, 0.95, size=end - start)
        scores.append(s)
    return scores


@pytest.fixture
def dummy_labels():
    """Ground-truth utterance labels. 0=real, 1=partially_fake.
    Odd-indexed are partially_fake."""
    return [1 if i % 2 == 1 else 0 for i in range(10)]


@pytest.fixture
def dummy_segment_labels(dummy_frame_scores, rng):
    """Ground-truth frame-level binary labels (0=real, 1=fake)."""
    labels = []
    for i, s in enumerate(dummy_frame_scores):
        n = len(s)
        lab = np.zeros(n, dtype=int)
        if i % 2 == 1:
            lab[n // 3: 2 * n // 3] = 1
        labels.append(lab)
    return labels
