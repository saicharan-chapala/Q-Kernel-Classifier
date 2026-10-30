import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw_feature_map(n_qubits, reps, entanglement, x=None):
    from .kernels import FidelityKernel
    pairs = FidelityKernel(n_qubits, 1, entanglement).pairs
    cols_per_rep = 2 + len(pairs)
    fig, ax = plt.subplots(figsize=(min(2 + 1.0 * cols_per_rep * reps, 16), 0.9 * n_qubits + 0.6))
    for q in range(n_qubits):
        ax.plot([0, cols_per_rep * reps + 1], [-q, -q], color="#888", lw=1, zorder=0)
        ax.text(-0.2, -q, f"q{q}", ha="right", va="center")
    c = 1
    for _ in range(reps):
        for q in range(n_qubits):
            ax.text(c, -q, "H", ha="center", va="center", bbox=dict(boxstyle="round", fc="#cfe8ff"))
        c += 1
        for q in range(n_qubits):
            lab = "P(2x%d)" % q if x is None else "P(%.1f)" % (2 * x[q])
            ax.text(c, -q, lab, ha="center", va="center", fontsize=7, bbox=dict(boxstyle="round", fc="#ffe3b3"))
        c += 1
        for i, j in pairs:
            ax.plot([c, c], [-i, -j], color="#7a3fbf", lw=2)
            ax.scatter([c, c], [-i, -j], color="#7a3fbf", s=25, zorder=3)
            ax.text(c + 0.12, -(i + j) / 2, "ZZ", fontsize=6, color="#7a3fbf", va="center")
            c += 1
    ax.set_xlim(-0.8, c + 0.5); ax.set_ylim(-n_qubits + 0.4, 0.6); ax.axis("off")
    fig.tight_layout()
    return fig


def kernel_heatmap(K, labels, title="Fidelity kernel |<phi(x)|phi(z)>|^2"):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(K, vmin=0, vmax=1, cmap="viridis")
    labels = np.asarray(labels)
    edges = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    for e in edges:
        ax.axhline(e - 0.5, color="w", lw=0.6); ax.axvline(e - 0.5, color="w", lw=0.6)
    starts = [0] + edges + [len(labels)]
    ticks = [(a + b - 1) / 2 for a, b in zip(starts[:-1], starts[1:])]
    ax.set_xticks(ticks); ax.set_xticklabels([labels[int(t)] for t in ticks], rotation=60, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels([labels[int(t)] for t in ticks], fontsize=7)
    ax.set_title(title, fontsize=9); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def attribution_bar(df, title="Top contributing original features"):
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5, 0.45 * len(d) + 0.8))
    ax.barh(d["feature"], d["delta_score"], color=["#d9534f" if v > 0 else "#5bc0de" for v in d["delta_score"]])
    ax.axvline(0, color="k", lw=0.6); ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def score_gauge(score, tau, lo=0.0, hi=None):
    hi = hi or max(1.0, score * 1.1, tau * 1.5)
    fig, ax = plt.subplots(figsize=(5, 1.0))
    ax.barh([0], [score], color="#d9534f" if score > tau else "#5cb85c")
    ax.axvline(tau, color="k", ls="--"); ax.text(tau, 0.45, " threshold", fontsize=8)
    ax.set_xlim(lo, hi); ax.set_yticks([]); ax.set_title("Calibrated novelty score g(x)", fontsize=9)
    fig.tight_layout()
    return fig
