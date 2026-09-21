#!/usr/bin/env python3
"""
Smoke demo: marker-quarter classification — prove positional encoding matters.

Task
----
Sequences of length T=16; each position has a token id from vocab V=20.
Exactly one position holds the special marker token id=0.
Label = which of 4 quarters (slots) contains the marker (chance = 0.25).

Model (content-gated PE readout)
--------------------------------
  1. Embed tokens -> X (B, T, D)
  2. Soft-select the marker position by content: a = softmax(X @ w_query)
  3. Readout:
       - no PE:    h = sum_t a_t X_t          (~ marker embedding — position-blind)
       - with PE:  h = sum_t a_t PE_t         (~ PE[marker_pos] — position signal)
  4. Linear classifier: logits = h @ W + b   (4-way)

Three conditions: **no PE**, **sinusoidal PE**, **learned PE**.
No-PE stays near chance; sinusoidal + learned should crush it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from positional_encoding import (
    LearnedPositionalEncoding,
    add_positional_encoding,
    check_sinusoidal_properties,
    position_cosine_similarity,
    relative_distances,
    sinusoidal_positional_encoding,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
SEQ_LEN = 16
VOCAB = 20
MARKER = 0
N_CLASSES = 4
D_MODEL = 32
N_TRAIN = 1024
N_TEST = 256
BATCH = 64
EPOCHS = 80
LR = 0.2
PE_LR = 0.1


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def quarter_of(pos: np.ndarray, seq_len: int = SEQ_LEN, n_slots: int = N_CLASSES) -> np.ndarray:
    slot = seq_len // n_slots
    return np.clip(pos // slot, 0, n_slots - 1).astype(np.int64)


def make_batch(rng: np.random.Generator, n: int):
    tokens = rng.integers(1, VOCAB, size=(n, SEQ_LEN), dtype=np.int64)
    marker_pos = rng.integers(0, SEQ_LEN, size=n)
    tokens[np.arange(n), marker_pos] = MARKER
    labels = quarter_of(marker_pos)
    return tokens, labels, marker_pos


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    probs = softmax(logits, axis=-1)
    B = y.shape[0]
    return float(-np.mean(np.log(probs[np.arange(B), y] + 1e-12)))


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=-1) == y))


def forward(tokens, emb, w_query, W, b, mode, sin_pe, learned):
    """Returns logits (B, C), alpha (B, T), h (B, D), X (B, T, D)."""
    X = emb[tokens]
    scores = X @ w_query
    alpha = softmax(scores, axis=-1)

    if mode == "none":
        h = np.einsum("bt,btd->bd", alpha, X)
    else:
        if mode == "sinusoidal":
            pe = sin_pe
        elif mode == "learned":
            pe = learned.forward(SEQ_LEN)
        else:
            raise ValueError(mode)
        h = alpha @ pe

    logits = h @ W + b
    return logits, alpha, h, X


def train_condition(mode, tokens_tr, y_tr, tokens_te, y_te, rng: np.random.Generator):
    emb = rng.normal(0, 0.4, size=(VOCAB, D_MODEL))
    emb[MARKER] = rng.normal(0, 0.4, size=(D_MODEL,)) + 1.5
    w_query = rng.normal(0, 0.3, size=(D_MODEL,))
    W = rng.normal(0, 0.3, size=(D_MODEL, N_CLASSES))
    b = np.zeros(N_CLASSES, dtype=np.float64)
    sin_pe = sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)
    learned = (
        LearnedPositionalEncoding(SEQ_LEN, D_MODEL, rng=rng, scale=0.05)
        if mode == "learned"
        else None
    )

    history = {"loss": [], "train_acc": [], "test_acc": []}
    n_train = tokens_tr.shape[0]

    for epoch in range(EPOCHS):
        perm = rng.permutation(n_train)
        losses = []
        lr = LR * (0.96 ** (epoch // 8))
        for start in range(0, n_train, BATCH):
            idx = perm[start : start + BATCH]
            tok = tokens_tr[idx]
            y = y_tr[idx]
            logits, alpha, h, X = forward(tok, emb, w_query, W, b, mode, sin_pe, learned)
            losses.append(cross_entropy(logits, y))

            Bsz = tok.shape[0]
            probs = softmax(logits, axis=-1)
            dlogits = probs.copy()
            dlogits[np.arange(Bsz), y] -= 1.0
            dlogits /= Bsz

            dW = h.T @ dlogits
            db = dlogits.sum(axis=0)
            dh = dlogits @ W.T

            if mode == "none":
                dX = alpha[:, :, None] * dh[:, None, :]
                dalpha = np.einsum("bd,btd->bt", dh, X)
            else:
                pe = sin_pe if mode == "sinusoidal" else learned.forward(SEQ_LEN)
                dalpha = dh @ pe.T
                dX = np.zeros_like(X)
                if mode == "learned":
                    dpe = alpha.T @ dh
                    learned.weight -= PE_LR * (0.96 ** (epoch // 8)) * dpe

            sum_da = np.sum(dalpha * alpha, axis=-1, keepdims=True)
            dscores = alpha * (dalpha - sum_da)

            dw_query = np.einsum("bt,btd->d", dscores, X)
            dX = dX + dscores[:, :, None] * w_query[None, None, :]

            demb = np.zeros_like(emb)
            np.add.at(demb, tok, dX)

            W -= lr * dW
            b -= lr * db
            w_query -= lr * dw_query
            emb -= lr * demb

        logits_tr, _, _, _ = forward(tokens_tr, emb, w_query, W, b, mode, sin_pe, learned)
        logits_te, alpha_te, _, _ = forward(tokens_te, emb, w_query, W, b, mode, sin_pe, learned)
        tr_acc = accuracy(logits_tr, y_tr)
        te_acc = accuracy(logits_te, y_te)
        mean_loss = float(np.mean(losses))
        history["loss"].append(mean_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"  [{mode:11s}] epoch {epoch:02d}  loss={mean_loss:.4f}  "
                f"train={tr_acc:.3f}  test={te_acc:.3f}"
            )

    marker_pos_te = np.argmax(tokens_te == MARKER, axis=1)
    _, alpha_te, _, _ = forward(tokens_te, emb, w_query, W, b, mode, sin_pe, learned)
    sel_acc = float(np.mean(np.argmax(alpha_te, axis=1) == marker_pos_te))

    return {
        "history": history,
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "final_train_loss": history["loss"][-1],
        "marker_selection_accuracy": sel_acc,
        "emb": emb,
        "W": W,
        "b": b,
        "w_query": w_query,
        "learned": learned,
        "sin_pe": sin_pe,
    }


def run_unit_checks() -> dict:
    pe = sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)
    checks = check_sinusoidal_properties(pe, d_model=D_MODEL)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, SEQ_LEN, D_MODEL))
    Y = add_positional_encoding(X, pe)
    checks["add_pe_broadcast"] = bool(np.allclose(Y - X, pe))

    lp = LearnedPositionalEncoding(SEQ_LEN, D_MODEL, rng=rng)
    checks["learned_shape"] = lp.forward().shape == (SEQ_LEN, D_MODEL)

    rd = relative_distances(SEQ_LEN)
    checks["relative_dist_diag_zero"] = bool(np.all(np.diag(rd) == 0))

    checks["all_passed"] = all(checks.values())
    return checks


from pe_smoke_plots import (
    plot_pe_heatmap,
    plot_position_similarity,
    plot_accuracy_comparison,
)
