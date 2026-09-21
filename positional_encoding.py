"""
Sinusoidal and learned positional encodings from scratch (NumPy).

Formulas (Vaswani et al., 2017):
  PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Learned PE is a simple embedding table of shape (max_len, d_model).
"""

from __future__ import annotations

import numpy as np


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Fixed sinusoidal positional encoding matrix.

    Parameters
    ----------
    seq_len : sequence length T
    d_model : embedding / model dimension D

    Returns
    -------
    pe : (T, D) float64
    """
    if seq_len < 0 or d_model < 1:
        raise ValueError("seq_len >= 0 and d_model >= 1 required")
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    if seq_len == 0:
        return pe
    position = np.arange(seq_len, dtype=np.float64)[:, None]  # (T, 1)
    # div_term[i] = 10000^(2i / d_model) for even dims; pair with sin/cos
    i = np.arange(0, d_model, 2, dtype=np.float64)
    div_term = np.exp(i * -(np.log(10000.0) / d_model))  # (ceil(D/2),)
    pe[:, 0::2] = np.sin(position * div_term)
    # cos for odd dims; if d_model odd, last even-pair has no cos partner
    pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class LearnedPositionalEncoding:
    """
    Trainable position embedding table of shape (max_len, d_model).
    """

    def __init__(
        self,
        max_len: int,
        d_model: int,
        rng: np.random.Generator | None = None,
        scale: float = 0.02,
    ) -> None:
        if max_len < 1 or d_model < 1:
            raise ValueError("max_len and d_model must be >= 1")
        rng = rng or np.random.default_rng()
        self.max_len = max_len
        self.d_model = d_model
        self.weight = rng.normal(0.0, scale, size=(max_len, d_model)).astype(np.float64)

    def forward(self, seq_len: int | None = None) -> np.ndarray:
        """Return the first `seq_len` rows (default: full table)."""
        T = self.max_len if seq_len is None else seq_len
        if T > self.max_len:
            raise ValueError(f"seq_len {T} > max_len {self.max_len}")
        return self.weight[:T]

    def parameters(self) -> dict[str, np.ndarray]:
        return {"weight": self.weight}


def add_positional_encoding(x: np.ndarray, pe: np.ndarray) -> np.ndarray:
    """
    Add positional encoding to token embeddings.

    Parameters
    ----------
    x : (..., T, D) or (T, D)
    pe : (T, D) or (1, T, D) / broadcastable

    Returns
    -------
    x + pe with broadcasting over leading batch dims.
    """
    x = np.asarray(x, dtype=np.float64)
    pe = np.asarray(pe, dtype=np.float64)
    if pe.ndim == 2 and x.ndim >= 2:
        # broadcast pe over batch: (T, D) -> align to trailing dims
        return x + pe
    return x + pe


def relative_distances(seq_len: int) -> np.ndarray:
    """Absolute pairwise position distances |i - j|, shape (T, T)."""
    idx = np.arange(seq_len)
    return np.abs(idx[:, None] - idx[None, :])


def position_cosine_similarity(pe: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine similarity between PE vectors.

    Parameters
    ----------
    pe : (T, D)

    Returns
    -------
    sim : (T, T)
    """
    pe = np.asarray(pe, dtype=np.float64)
    norms = np.linalg.norm(pe, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = pe / norms
    return unit @ unit.T


def check_sinusoidal_properties(
    pe: np.ndarray,
    d_model: int | None = None,
    atol: float = 1e-8,
) -> dict[str, bool]:
    """
    Unit-checkable properties of a sinusoidal PE matrix.

    Returns a dict of named boolean checks (all should be True).
    """
    pe = np.asarray(pe, dtype=np.float64)
    T, D = pe.shape
    d_model = D if d_model is None else d_model
    expected = sinusoidal_positional_encoding(T, d_model)

    checks: dict[str, bool] = {}
    checks["shape_T_D"] = pe.shape == (T, d_model)
    checks["matches_vaswani_formula"] = np.allclose(pe, expected, atol=atol)
    checks["finite"] = bool(np.isfinite(pe).all())
    # Period structure: dim 0 is sin(pos) with period 2π; dim 1 is cos(pos)
    if T >= 4 and D >= 2:
        # PE[pos, 0] = sin(pos)  (since 10000^0 = 1)
        pos = np.arange(T, dtype=np.float64)
        checks["dim0_is_sin_pos"] = np.allclose(pe[:, 0], np.sin(pos), atol=atol)
        checks["dim1_is_cos_pos"] = np.allclose(pe[:, 1], np.cos(pos), atol=atol)
    else:
        checks["dim0_is_sin_pos"] = True
        checks["dim1_is_cos_pos"] = True
    # Nearby positions are more similar than far ones (on average)
    if T >= 8:
        sim = position_cosine_similarity(pe)
        near = np.mean([sim[i, i + 1] for i in range(T - 1)])
        far = float(sim[0, T - 1])
        checks["nearby_more_similar_than_far"] = bool(near > far)
    else:
        checks["nearby_more_similar_than_far"] = True
    return checks
