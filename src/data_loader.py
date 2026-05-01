"""
Data loading and preprocessing for the MNIST handwritten digit dataset.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, VALIDATION_SPLIT, NORMALIZE, FLATTEN


def load_mnist():
    """
    Load the MNIST dataset from TensorFlow/Keras.

    Falls back to scikit-learn's `fetch_openml` if TensorFlow is unavailable.

    Returns
    -------
    tuple
        (X_train_full, y_train_full, X_test, y_test) as NumPy arrays.
    """
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_SEED)
        print(f"[data] TensorFlow {tf.__version__} detected — using keras.datasets")
        (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    except ImportError:
        print("[data] TensorFlow not found — falling back to sklearn fetch_openml")
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X_all, y_all = mnist.data, mnist.target.astype(int)
        X_train_full, X_test = X_all[:60000], X_all[60000:]
        y_train_full, y_test = y_all[:60000], y_all[60000:]

    print(f"[data] Raw shapes  — train: {X_train_full.shape}, test: {X_test.shape}")
    print(f"[data] Pixel range — [{X_train_full.min()}, {X_train_full.max()}]")
    return X_train_full, y_train_full, X_test, y_test


def preprocess(X_train_full, y_train_full, X_test, y_test):
    """
    Normalize, flatten, and split the data.

    Parameters
    ----------
    X_train_full, y_train_full : np.ndarray
        Full training set (60 000 samples).
    X_test, y_test : np.ndarray
        Test set (10 000 samples).

    Returns
    -------
    dict
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # ── Normalize to [0, 1] float32 ─────────────────────────────
    if NORMALIZE:
        X_train_full = X_train_full.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        print("[preprocess] Pixel values normalized to [0, 1]")

    # ── Flatten 28×28 → 784 ─────────────────────────────────────
    if FLATTEN:
        if X_train_full.ndim == 3:
            X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
        if X_test.ndim == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)
        print("[preprocess] Images flattened to 784-d vectors")

    print(f"[preprocess] Shapes — train: {X_train_full.shape}, test: {X_test.shape}")

    # ── Stratified train / validation split ──────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SPLIT,
        stratify=y_train_full,
        random_state=RANDOM_SEED,
    )
    print(f"[preprocess] Split — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # ── Class distribution summary ───────────────────────────────
    unique, counts = np.unique(y_train, return_counts=True)
    print("[preprocess] Training class distribution:")
    for digit, count in zip(unique, counts):
        print(f"           digit {digit}: {count} samples ({count / len(y_train) * 100:.1f} %)")

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )
