"""Shared loading and validation for frozen embedding caches."""

import numpy as np

from src.config import FEATURE_FILE, FEATURES_DIR, FROZEN_EMBEDDING_DIMS


def load_features(backbone, split):
    """Load finite, nonzero embeddings and return their L2-normalized rows."""
    x = np.load(FEATURES_DIR / FEATURE_FILE.format(backbone=backbone, split=split, kind="x"))
    y = np.load(FEATURES_DIR / FEATURE_FILE.format(backbone=backbone, split=split, kind="y"))
    if y.ndim != 1 or x.shape != (len(y), FROZEN_EMBEDDING_DIMS[backbone]):
        raise ValueError(f"Invalid cache shapes for {backbone}/{split}: {x.shape}, {y.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError(f"Non-finite or zero embeddings in {backbone}/{split}")
    return x / norms, y
