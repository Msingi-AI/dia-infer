"""Post-generation audio helpers (Dia has no native speaking-rate control)."""

from __future__ import annotations

import numpy as np


def apply_speed_factor(audio: np.ndarray, speed_factor: float) -> np.ndarray:
    """Time-stretch via linear interpolation (same approach as nari Gradio).

    ``speed_factor < 1`` slows playback (longer); ``> 1`` speeds up.
    Simple interp shifts pitch slightly — good enough for listen/sample loops.
    """
    factor = float(speed_factor)
    if factor <= 0:
        raise ValueError("speed_factor must be > 0")
    factor = max(0.1, min(factor, 5.0))
    if abs(factor - 1.0) < 1e-6 or audio.size == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)
    original_len = int(audio.shape[0])
    target_len = int(original_len / factor)
    if target_len <= 0 or target_len == original_len:
        return np.ascontiguousarray(audio, dtype=np.float32)
    x_original = np.arange(original_len, dtype=np.float64)
    x_resampled = np.linspace(0, original_len - 1, target_len, dtype=np.float64)
    out = np.interp(x_resampled, x_original, audio.astype(np.float64, copy=False))
    return np.ascontiguousarray(out, dtype=np.float32)
