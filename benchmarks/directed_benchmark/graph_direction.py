"""
graph_direction.py
------------------
Directed benchmark: edge cluster matrix showing how well KL, OT, and
symmetric cosine graphs recover the causal direction of cluster activation.

Usage
-----
  cd /Users/zeev/Documents/GSET_2
  python benchmarks/directed_benchmark/graph_direction.py --sim_dir simulation/block4_cascade --threshold 0.4 --seed 10
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import sys
import numpy as np
import networkx as nx
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.directed_benchmark.gset_utils import load_scenario, compute_kl_vecs, compute_ot_vecs
from core.train import train_gcn
from benchmarks.directed_benchmark.graph import (
    build_directed_graph, build_directed_graph_ot, symmetric_cosine_graph)
from benchmarks.directed_benchmark.expand import edge_cluster_matrix, inter_cluster_edge_fraction
from benchmarks.directed_benchmark.plot import plot_edge_cluster_matrix


def _active_tps(runs, tp_list):
    means = {tp: np.concatenate([run[tp] for run in runs]).mean() for tp in tp_list}
    active = [tp for tp in tp_list if means[tp] > 1.0]
    return active if len(active) >= 2 else tp_list


def top_k_graph(C, n_genes, k):
    """Build directed graph from top-k edges by cosine value."""
    flat = C.copy()
    np.fill_diagonal(flat, -np.inf)
    top_idx = np.unravel_index(np.argsort(flat.ravel())[::-1][:k], flat.shape)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_genes))
    for i, j in zip(*top_idx):
        G.add_edge(int(i), int(j))
    return G


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_dir", default="simulation/output_cascade_150")
    p.add_argument("--threshold", type=float, default=None,
                   help="Cosine threshold for adding an edge")
    p.add_argument("--budget", type=int, default=400,
                   help="Top-K edges (used if --threshold not set)")
    p.add_argument("--n_epochs", type=int, default=300)
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    sim_dir = _ROOT / args.sim_dir
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────
    meta = json.loads((sim_dir / "metadata.json").read_text())
    print(f"Scenario : {meta['scenario']}")
    print(f"Genes    : {meta['n_genes']},  Clusters: {meta['n_clusters']}")
    print(f"Timepoints: {meta['timepoints']}")

    runs, cluster_ids, tp_list = load_scenario(str(sim_dir))
    n_genes = len(cluster_ids)
    n_trans = len(tp_list) - 1
    print(f"\nLoaded {len(runs)} runs,  {n_trans} transitions")
    for k in np.unique(cluster_ids):
        print(f"  Cluster {k}: {(cluster_ids==k).sum()} genes")

    # ── Train shared GCN ───────────────────────────────────────────────────
    print("\nTraining shared GCN...")
    active = _active_tps(runs, tp_list)
    print(f"  Active timepoints: {active}")
    X_list = [run[tp] for run in runs for tp in active]
    gcn_model = train_gcn(X_list, n_genes=n_genes, n_epochs=args.n_epochs, d_out=args.embed_dim, seed=args.seed)

    # ── KL vectors ────────────────────────────────────────────────────────
    print("\nComputing KL vectors...")
    kl_vecs = compute_kl_vecs(runs, tp_list, model=gcn_model)

    # ── OT residual vectors ───────────────────────────────────────────────
    print("\nComputing OT residual vectors...")
    ot_vecs = compute_ot_vecs(runs, tp_list, model=gcn_model)

    # ── Cosine matrices ────────────────────────────────────────────────────
    _, C_kl = build_directed_graph(kl_vecs,   threshold=0.0)
    _, C_ot = build_directed_graph_ot(ot_vecs, threshold=0.0)
    C_sym   = symmetric_cosine_graph(kl_vecs)

    if args.threshold is not None:
        G_kl, _ = build_directed_graph(kl_vecs,   threshold=args.threshold)
        G_ot, _ = build_directed_graph_ot(ot_vecs, threshold=args.threshold)
        G_sym   = top_k_graph(C_sym, n_genes, G_kl.number_of_edges())
        print(f"\nThreshold={args.threshold}")
    else:
        G_kl  = top_k_graph(C_kl,  n_genes, args.budget)
        G_ot  = top_k_graph(C_ot,  n_genes, args.budget)
        G_sym = top_k_graph(C_sym, n_genes, args.budget)
        print(f"\nBudget={args.budget}")

    print(f"  KL  edges={G_kl.number_of_edges()}  range [{C_kl.min():.3f}, {C_kl.max():.3f}]")
    print(f"  OT  edges={G_ot.number_of_edges()}  range [{C_ot.min():.3f}, {C_ot.max():.3f}]")

    # ── Edge cluster matrices ───────────────────────────────────────────────
    M_kl  = edge_cluster_matrix(G_kl,  cluster_ids)
    M_ot  = edge_cluster_matrix(G_ot,  cluster_ids)
    M_sym = edge_cluster_matrix(G_sym, cluster_ids)

    inter_frac_kl  = inter_cluster_edge_fraction(G_kl,  cluster_ids)
    inter_frac_ot  = inter_cluster_edge_fraction(G_ot,  cluster_ids)
    inter_frac_sym = inter_cluster_edge_fraction(G_sym, cluster_ids)
    print(f"\nInter-cluster edges:")
    print(f"  KL       : {inter_frac_kl:.1%}")
    print(f"  OT       : {inter_frac_ot:.1%}")
    print(f"  Symmetric: {inter_frac_sym:.1%}")

    print(f"\nKL edge cluster matrix:\n{M_kl}")
    print(f"\nOT edge cluster matrix:\n{M_ot}")

    print(f"\nSaving figures to {out_dir}/")
    plot_edge_cluster_matrix(M_kl.astype(float), M_sym.astype(float),
                             str(out_dir / "edge_cluster_matrix.png"),
                             M_ot=M_ot.astype(float))
    print("\nDone.")


if __name__ == "__main__":
    main()
