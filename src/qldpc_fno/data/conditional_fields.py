from __future__ import annotations

import numpy as np


def add_noise_channel(syndromes: np.ndarray, error_rates: np.ndarray) -> np.ndarray:
    """Append a ring-broadcast log-odds channel to binary syndrome fields."""
    syndrome_array = np.asarray(syndromes)
    if syndrome_array.ndim != 3 or syndrome_array.shape[0] == 0:
        raise ValueError("syndromes must have shape (positive shots, channels, ell)")
    if not np.all((syndrome_array == 0) | (syndrome_array == 1)):
        raise ValueError("syndromes must be binary")

    rate_array = np.asarray(error_rates)
    if rate_array.ndim != 1 or rate_array.shape[0] != syndrome_array.shape[0]:
        raise ValueError("there must be one error rate per shot")
    if not np.all(np.isfinite(rate_array)):
        raise ValueError("error rates must be finite")
    if not np.all((rate_array > 0) & (rate_array < 1)):
        raise ValueError("error rates must be strictly between zero and one")

    fields = np.array(syndrome_array, dtype=np.float32, order="C", copy=True)
    logits = np.log(rate_array / (1.0 - rate_array)).astype(np.float32)
    noise = np.broadcast_to(logits[:, None, None], (fields.shape[0], 1, fields.shape[2]))
    return np.array(np.concatenate((fields, noise), axis=1), dtype=np.float32, order="C", copy=True)
