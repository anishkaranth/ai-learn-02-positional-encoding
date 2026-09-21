from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from positional_encoding import position_cosine_similarity


def plot_pe_heatmap(pe: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pe.T, aspect="auto", cmap="RdBu", interpolation="nearest")
    ax.set_xlabel("position")
    ax.set_ylabel("dimension")
    ax.set_title("Sinusoidal positional encoding (PE^T heatmap)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_position_similarity(pe: np.ndarray, path: Path) -> None:
    sim = position_cosine_similarity(pe)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_xlabel("position j")
    ax.set_ylabel("position i")
    ax.set_title("Cosine similarity between PE rows")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_accuracy_comparison(results: dict, path: Path) -> None:
    modes = ["none", "sinusoidal", "learned"]
    labels = ["no PE", "sinusoidal", "learned"]
    test_accs = [results[m]["final_test_acc"] for m in modes]
    train_accs = [results[m]["final_train_acc"] for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(modes))
    width = 0.35
    axes[0].bar(x - width / 2, train_accs, width, label="train", color="#4C78A8")
    axes[0].bar(x + width / 2, test_accs, width, label="test", color="#F58518")
    axes[0].axhline(0.25, color="gray", ls="--", lw=1, label="chance (0.25)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Final accuracy by PE type")
    axes[0].legend(fontsize=8)

    for m, lab, color in zip(modes, labels, ["#54A24B", "#4C78A8", "#E45756"]):
        axes[1].plot(results[m]["history"]["test_acc"], label=lab, color=color)
    axes[1].axhline(0.25, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Test accuracy curves")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
