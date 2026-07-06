"""
benchmarks/vectorial_embeddings/benchmark.py
--------------------------------------------
Shared functions for vectorial embeddings benchmark.
Entry points: run.py (training + vectors) and plot.py (figures).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "benchmarks" / "gene_programme_recovery"))

from harissa_to_anndata import (
    harissa_to_anndata,
    intra_inter_cosine,
)
from sklearn.preprocessing import normalize as sk_normalize
from core.scoring import kl_vec, ot_residual_vec
from core.train import train_gat as _train_gat_fn, get_embedding
from core.graph import build_graph
from core.model import GATEncoder


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sim(sim_dir: Path):
    """Load all runs. Returns list of (X0, XT) tuples, cluster_ids, meta.

    X0/XT include a zero stimulus column prepended (to match the original
    harissa_benchmark format: n_genes+1 columns, stimulus at index 0).
    cluster_ids is for the n_genes real genes (no stimulus).
    """
    meta        = json.loads((sim_dir / "metadata.json").read_text())
    cluster_ids = np.load(sim_dir / "cluster_ids.npy")
    tps         = sorted(meta["timepoints"])
    t0, tT      = tps[0], tps[-1]

    # Load all runs
    data_files = sorted(f for f in (sim_dir / "data").iterdir()
                        if f.name.startswith("data_"))
    runs = []
    for fpath in data_files:
        raw   = np.loadtxt(fpath, delimiter="\t").T   # (n_cells, 1+n_genes+1)
        times = raw[:, 0]
        # HARISSA format: col 0 = time, col 1 = stimulus, cols 2.. = genes
        # Take stimulus + all genes (n_genes+1 columns total, stimulus first)
        expr  = raw[:, 1:]                             # (n_cells, n_genes+1) incl. stimulus
        X0_r  = expr[times == t0].astype(float)
        XT_r  = expr[times == tT].astype(float)
        runs.append((X0_r, XT_r))

    return runs, cluster_ids, meta


# ── GAT embedding + vectors ───────────────────────────────────────────────────

def train_gat(runs: list, n_genes_with_stim: int,
              n_epochs: int = 200, seed: int = 42):
    """Train GATEncoder on all runs."""
    import torch
    torch.manual_seed(seed)
    X0, XT = runs[0]
    data, *_ = build_graph(X0)
    model = GATEncoder(n_genes=data.x.shape[0], d_hid=64, d_out=16, n_heads=4)
    X0, XT = runs[0]
    _train_gat_fn(model, [X0, XT], n_epochs=n_epochs)
    return model


def compute_vectors(model, X0: np.ndarray, XT: np.ndarray):
    """Compute per-gene vectors, skipping stimulus (index 0)."""
    E0 = get_embedding(model, X0)
    ET = get_embedding(model, XT)
    return {
        "Naive":       (ET - E0)[1:],
        "OT residual": ot_residual_vec(E0, ET)[1:],
        "KL vector":   kl_vec(E0, ET)[1:],
    }, E0, ET


def apply_dropout(X: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    X_d = X.copy()
    X_d[rng.random(X.shape) < p] = 0.0
    return X_d


# ── Plotting ──────────────────────────────────────────────────────────────────

CLR = {"OT residual": "#e63946", "KL vector": "#2a9d8f", "Naive": "#aaa"}


def plot_cosine_matrices(vectors: dict, cluster_ids: np.ndarray,
                         scenario: str, out_path: Path):
    order   = np.argsort(cluster_ids)
    methods = list(vectors.keys())

    fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 4.5))
    if len(methods) == 1:
        axes = [axes]

    for ax, name in zip(axes, methods):
        V = vectors[name][order]
        Vn = sk_normalize(V, norm="l2", axis=1)
        S = Vn @ Vn.T
        vabs = max(abs(float(S.min())), abs(float(S.max())))
        im = ax.imshow(S, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        ax.set_title(name, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Cosine similarity matrices — {scenario}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def plot_dropout_sweep(model, X0: np.ndarray, XT: np.ndarray,
                       cluster_ids: np.ndarray, scenario: str, out_path: Path,
                       dropouts=None, n_reps: int = 5, seed: int = 42):
    if dropouts is None:
        dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    rng          = np.random.default_rng(seed)
    method_names = ["Naive", "OT residual", "KL vector"]
    results      = {m: {"intra": [], "inter": []} for m in method_names}

    for p in dropouts:
        rep_intra = {m: [] for m in method_names}
        rep_inter = {m: [] for m in method_names}
        for _ in range(n_reps):
            X0d = apply_dropout(X0, p, rng)
            XTd = apply_dropout(XT, p, rng)
            E0d = get_embedding(model, X0d)
            ETd = get_embedding(model, XTd)
            vecs = {
                "Naive":       (ETd - E0d)[1:],
                "OT residual": ot_residual_vec(E0d, ETd)[1:],
                "KL vector":   kl_vec(E0d, ETd)[1:],
            }
            for m, V in vecs.items():
                intra, inter = intra_inter_cosine(V, cluster_ids, exclude_sentinel=False)
                rep_intra[m].append(intra)
                rep_inter[m].append(inter)
        for m in method_names:
            results[m]["intra"].append(np.mean(rep_intra[m]))
            results[m]["inter"].append(np.mean(rep_inter[m]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name in method_names:
        axes[0].plot(dropouts, results[name]["intra"], "o-",  color=CLR[name], label=name, lw=2)
        axes[1].plot(dropouts, results[name]["inter"], "s--", color=CLR[name], label=name, lw=2)
    axes[0].set_title("Intra-cluster cosine similarity")
    axes[1].set_title("Inter-cluster cosine similarity")
    for ax in axes:
        ax.set_xlabel("Dropout rate")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
    axes[0].set_ylabel("Mean cosine similarity")
    axes[0].legend(frameon=False, fontsize=8)
    plt.suptitle(f"Intra vs inter-cluster cosine similarity ({scenario})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cascade_dir", default="simulation/block2_cascade")
    p.add_argument("--switch_dir",  default="simulation/block2_switch")
    p.add_argument("--gcn_epochs",  type=int, default=300)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args    = parse_args()
    fig_dir = _ROOT / "figures" / "block2_vectorial_embeddings"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Cascade ───────────────────────────────────────────────────────────────
    print("\n[cascade] Loading simulation...")
    runs, cluster_ids, meta = load_sim(_ROOT / args.cascade_dir)
    print(f"  {len(runs)} runs, {len(cluster_ids)} genes")

    print("[cascade] Training GAT...")
    model = train_gat(runs, n_genes_with_stim=runs[0][0].shape[1],
                      n_epochs=args.gcn_epochs, seed=args.seed)

    print("[cascade] Computing vectors (last run)...")
    X0, XT = runs[-1]
    vectors, E0, ET = compute_vectors(model, X0, XT)

    print("[cascade] Cosine similarity matrices...")
    plot_cosine_matrices(vectors, cluster_ids, meta["scenario"],
                         fig_dir / "vectorial_cascade.png")

    print("[cascade] Dropout sweep...")
    plot_dropout_sweep(model, X0, XT, cluster_ids, meta["scenario"],
                       fig_dir / "vectorial_dropout.png", seed=args.seed)

    # ── Switch ────────────────────────────────────────────────────────────────
    print("\n[switch] Loading simulation...")
    runs, cluster_ids, meta = load_sim(_ROOT / args.switch_dir)
    print(f"  {len(runs)} runs, {len(cluster_ids)} genes")

    print("[switch] Training GAT...")
    model = train_gat(runs, n_genes_with_stim=runs[0][0].shape[1],
                      n_epochs=args.gcn_epochs, seed=args.seed)

    print("[switch] Computing vectors (last run)...")
    X0, XT = runs[-1]
    vectors, E0, ET = compute_vectors(model, X0, XT)

    print("[switch] Cosine similarity matrices...")
    plot_cosine_matrices(vectors, cluster_ids, meta["scenario"],
                         fig_dir / "vectorial_switch.png")

    print("[switch] Dropout sweep...")
    plot_dropout_sweep(model, X0, XT, cluster_ids, meta["scenario"],
                       fig_dir / "vectorial_switch_dropout.png", seed=args.seed)

    print("\n✓ Done")


if __name__ == "__main__":
    main()
