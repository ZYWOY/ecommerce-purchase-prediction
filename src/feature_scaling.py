from __future__ import annotations

from pathlib import Path

import numpy as np


def fit_feature_scaler(X_seq: np.ndarray, X_static: np.ndarray) -> dict[str, np.ndarray]:
    seq_mean = X_seq.reshape(-1, X_seq.shape[-1]).mean(axis=0)
    seq_std = X_seq.reshape(-1, X_seq.shape[-1]).std(axis=0)
    static_mean = X_static.mean(axis=0)
    static_std = X_static.std(axis=0)

    seq_std = np.where(seq_std < 1e-6, 1.0, seq_std)
    static_std = np.where(static_std < 1e-6, 1.0, static_std)

    return {
        "seq_mean": seq_mean.astype(np.float32),
        "seq_std": seq_std.astype(np.float32),
        "static_mean": static_mean.astype(np.float32),
        "static_std": static_std.astype(np.float32),
    }


def transform_feature_arrays(
    X_seq: np.ndarray,
    X_static: np.ndarray,
    scaler: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    scaled_seq = (X_seq - scaler["seq_mean"].reshape(1, 1, -1)) / scaler["seq_std"].reshape(1, 1, -1)
    scaled_static = (X_static - scaler["static_mean"].reshape(1, -1)) / scaler["static_std"].reshape(1, -1)
    return scaled_seq.astype(np.float32), scaled_static.astype(np.float32)


def save_feature_scaler(path: str | Path, scaler: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        seq_mean=scaler["seq_mean"],
        seq_std=scaler["seq_std"],
        static_mean=scaler["static_mean"],
        static_std=scaler["static_std"],
    )


def load_feature_scaler(path: str | Path) -> dict[str, np.ndarray]:
    loaded = np.load(Path(path))
    return {
        "seq_mean": loaded["seq_mean"].astype(np.float32),
        "seq_std": loaded["seq_std"].astype(np.float32),
        "static_mean": loaded["static_mean"].astype(np.float32),
        "static_std": loaded["static_std"].astype(np.float32),
    }
