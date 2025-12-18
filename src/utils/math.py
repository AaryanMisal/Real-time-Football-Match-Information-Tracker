from __future__ import annotations
import numpy as np

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    if len(arr) == 0:
        return arr
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")
