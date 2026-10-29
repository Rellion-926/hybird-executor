from __future__ import annotations

import numpy as np


def _one(x: np.ndarray, gamma: np.ndarray, sin: np.ndarray, cos: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    scale = 1.0 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + 1.0e-6)
    norm = (x32 * scale * gamma.astype(np.float32)).astype(np.float16).astype(np.float32)
    first, second = norm[..., :64], norm[..., 64:]
    sin32 = sin.astype(np.float32)[None, :, None, :]
    cos32 = cos.astype(np.float32)[None, :, None, :]
    return np.concatenate((first * cos32 - second * sin32, second * cos32 + first * sin32), axis=-1).astype(np.float16)


def fused_qknorm_rope(q, k, gamma_q, gamma_k, sin, cos):
    return _one(q, gamma_q, sin, cos), _one(k, gamma_k, sin, cos)
