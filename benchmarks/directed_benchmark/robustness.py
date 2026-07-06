"""
benchmarks/dropout/run.py
--------------------------
Robustness sweep: apply increasing dropout to expression data and measure
how well the directed graph recovers inter-cluster signal.

Usage
-----
  cd /Users/zeev/Documents/GSET_2
  python benchmarks/dropout/run.py --sim_dir simulation/output_cascade_48
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.directed_benchmark.gset_utils import load_scenario, compute_kl_vecs, compute_ot_vecs
from core import train_gcn
from benchmarks.directed_benchmark.graph import build_directed_graph, build_directed_graph_ot, symmetric_cosine_graph
from benchmarks.directed_benchmark.expand import inter_cluster_edge_fraction, edge_cluster_matrix


def add_dropout(runs, rate, rng=None):
    """Zero out each entry independently with probability rate."""
    if rate == 0.0:
        return runs
    if rng is None:
        rng = np.random.default_rng(0)
    noisy = []
    for run in runs:
        noisy_run = {}
        for tp, X in run.items():
            mask = rng.binomial(1, 1 - rate, X.shape).astype(X.dtype)
            noisy_run[tp] = X * mask
        noisy.append(noisy_run)
    return noisy


def count_directed_edges(G, cluster_ids, a, b):
    """Count edges from cluster a to cluster b in graph G."""
    count = 0
    for i, j in G.edges():
        if cluster_ids[i] == a and cluster_ids[j] == b:
            count += 1
    return count


def direction_accuracy(G, cluster_ids, a, b):
    """Fraction of edges between clusters a and b that go in the correct direction a->b."""
    forward = 0
    reverse = 0
    for i, j in G.edges():
        ci = cluster_ids[i]
        cj = cluster_ids[j]
        if ci == a and cj == b:
            forward += 1
        if ci == b and cj == a:
            reverse += 1
    total = forward + reverse
    if total == 0:
        return 0.5
    return forward / total


def run_dropout_sweep(runs, cluster_ids, tp_list, gcn_model, threshold, out_dir):
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rng = np.random.default_rng(42)

    K = len(np.unique(cluster_ids))
    pairs = [(a, a + 1) for a in range(K - 1)]

    kl_inter_fracs = []
    ot_inter_fracs = []
    sym_inter_fracs = []

    kl_direction = []
    ot_direction = []
    sym_direction = []

    kl_edge_counts = []
    ot_edge_counts = []

    fixed_budget = None

    for rate in dropout_rates:
        print(f"  dropout={rate:.1f} ...", end=" ", flush=True)

        noisy_runs = add_dropout(runs, rate, rng)

        kl_vecs = compute_kl_vecs(noisy_runs, tp_list, model=gcn_model)
        ot_vecs = compute_ot_vecs(noisy_runs, tp_list, model=gcn_model)

        G_kl, _ = build_directed_graph(kl_vecs, threshold=threshold)
        G_ot, _ = build_directed_graph_ot(ot_vecs, threshold=threshold)

        C_sym = symmetric_cosine_graph(kl_vecs)

        if fixed_budget is None:
            fixed_budget = G_kl.number_of_edges()

        import networkx as nx
        flat = C_sym.copy()
        np.fill_diagonal(flat, -1)
        top_indices = np.unravel_index(
            np.argsort(flat.ravel())[::-1][:fixed_budget], flat.shape)
        G_sym = nx.DiGraph()
        G_sym.add_nodes_from(range(len(cluster_ids)))
        for i, j in zip(*top_indices):
            G_sym.add_edge(int(i), int(j))

        kl_inter_fracs.append(inter_cluster_edge_fraction(G_kl, cluster_ids))
        ot_inter_fracs.append(inter_cluster_edge_fraction(G_ot, cluster_ids))
        sym_inter_fracs.append(inter_cluster_edge_fraction(G_sym, cluster_ids))

        kl_dir_per_pair = {}
        ot_dir_per_pair = {}
        sym_dir_per_pair = {}
        kl_count_per_pair = {}
        ot_count_per_pair = {}

        for a, b in pairs:
            kl_dir_per_pair[(a, b)] = direction_accuracy(G_kl, cluster_ids, a, b)
            ot_dir_per_pair[(a, b)] = direction_accuracy(G_ot, cluster_ids, a, b)
            sym_dir_per_pair[(a, b)] = direction_accuracy(G_sym, cluster_ids, a, b)
            kl_count_per_pair[(a, b)] = count_directed_edges(G_kl, cluster_ids, a, b)
            ot_count_per_pair[(a, b)] = count_directed_edges(G_ot, cluster_ids, a, b)

        kl_direction.append(kl_dir_per_pair)
        ot_direction.append(ot_dir_per_pair)
        sym_direction.append(sym_dir_per_pair)
        kl_edge_counts.append(kl_count_per_pair)
        ot_edge_counts.append(ot_count_per_pair)

        print(f"KL={kl_inter_fracs[-1]:.2f}  OT={ot_inter_fracs[-1]:.2f}  Sym={sym_inter_fracs[-1]:.2f}")

    n_panels = 1 + len(pairs) * 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    ax = axes[0]
    ax.plot(dropout_rates, kl_inter_fracs, "o-", color="#d62728", lw=2, label="KL")
    ax.plot(dropout_rates, ot_inter_fracs, "s-", color="#1f77b4", lw=2, label="OT")
    ax.plot(dropout_rates, sym_inter_fracs, "^--", color="#7f7f7f", lw=2, label="Symmetric cosine")
    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Inter-cluster edge fraction")
    ax.set_title("Inter-cluster edge fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for idx, (a, b) in enumerate(pairs):
        ax = axes[1 + idx * 2]
        kl_vals = [kl_direction[i][(a, b)] for i in range(len(dropout_rates))]
        ot_vals = [ot_direction[i][(a, b)] for i in range(len(dropout_rates))]
        sym_vals = [sym_direction[i][(a, b)] for i in range(len(dropout_rates))]
        ax.plot(dropout_rates, kl_vals, "o-", color="#d62728", lw=2, label="KL")
        ax.plot(dropout_rates, ot_vals, "s-", color="#1f77b4", lw=2, label="OT")
        ax.plot(dropout_rates, sym_vals, "^--", color="#7f7f7f", lw=2, label="Symmetric cosine")
        ax.axhline(0.5, color="black", lw=1, linestyle=":", label="random")
        ax.set_xlabel("Dropout rate")
        ax.set_ylabel("Fraction correct")
        ax.set_title(f"Direction accuracy C{a}->C{b}")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2 + idx * 2]
        kl_counts = [kl_edge_counts[i][(a, b)] for i in range(len(dropout_rates))]
        ot_counts = [ot_edge_counts[i][(a, b)] for i in range(len(dropout_rates))]
        ax.plot(dropout_rates, kl_counts, "o-", color="#d62728", lw=2, label="KL")
        ax.plot(dropout_rates, ot_counts, "s-", color="#1f77b4", lw=2, label="OT")
        ax.set_xlabel("Dropout rate")
        ax.set_ylabel("Number of edges")
        ax.set_title(f"Edge count C{a}->C{b}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Robustness to dropout", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "robustness.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_dir", default="simulation/block4_cascade")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--n_epochs", type=int, default=300)
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    sim_dir = _ROOT / args.sim_dir
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {sim_dir} ...")
    runs, cluster_ids, tp_list = load_scenario(str(sim_dir))
    print(f"  {len(runs)} runs, timepoints: {tp_list}")

    print("Training GCN ...")
    n_genes = len(cluster_ids)
    active = [tp for tp in tp_list if np.concatenate([run[tp] for run in runs]).mean() > 1.0]
    X_list = [run[tp] for run in runs for tp in active]
    gcn_model = train_gcn(X_list, n_genes=n_genes, n_epochs=args.n_epochs, d_out=args.embed_dim)

    print("Running dropout sweep ...")
    run_dropout_sweep(runs, cluster_ids, tp_list, gcn_model, args.threshold, out_dir)


if __name__ == "__main__":
    main()
