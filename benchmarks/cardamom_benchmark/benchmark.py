"""
benchmarks/cardamom_benchmark/benchmark.py
------------------------------------------
CardamomOT causal subnetwork selection benchmark.

Idea (Elias Ventre, June 2026):
  The real validation target is not "did we recover the global cluster flow"
  but "can GSET select the right query-centred causal subnetwork to feed
  CardamomOT". Pick a query gene, expand outward along the directed graph
  (upstream regulators, downstream targets, bridges) up to a budget of ~100
  genes. Validate by EDGE COHERENCE of the CardamomOT-inferred network vs
  the ground-truth HARISSA theta restricted to those genes — not by fit
  quality (which always looks good because CardamomOT compensates spurious
  edges for missing drivers).

Pipeline
--------
  1. Load HARISSA cascade simulation (ground truth theta known)
  2. Build directed KL graph (lagged cosine, no GCN)
  3. Pick query gene (a TF in the last cluster: most downstream)
  4. Expand subnetwork (GSET directed closure) — budget 100
  5. Prepare three CardamomOT projects:
       (A) GSET-selected genes

       (C) Full gene set (baseline)
  6. Run CardamomOT inference on each (subprocess)
  7. Evaluate edge coherence vs inter_opt.npy
  8. Plot and report

Usage
-----
  cd /Users/zeev/Documents/GSET_2
  python benchmarks/cardamom_benchmark/benchmark.py \\
      --sim_dir simulation/block4_cascade \\
      --budget 50 \\
      --threshold 0.3 \\
      --seed 42 \\
      --out_dir figures/cardamom_benchmark
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import argparse
import shutil
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.cardamom_benchmark.subnetwork import (
    build_directed_kl_graph,
    expand_from_seed,
    subgraph_recall,
)
from benchmarks.cardamom_benchmark.prepare_harissa import prepare_cardamom_project
from benchmarks.cardamom_benchmark.cardamom_runner import run_cardamom
from benchmarks.cardamom_benchmark.evaluate import edge_coherence, restrict_inter, compare_selections
from benchmarks.directed_benchmark.data import (
    load_scenario, train_gcn, compute_kl_vecs
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim_dir",   default="simulation/block4_cascade",
                   help="HARISSA simulation directory")
    p.add_argument("--budget",    type=int,   default=None,
                   help="Max genes in subnetwork (default: full gene set size)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Cosine threshold for directed KL graph (default: auto = 90th percentile of cosine matrix)")
    p.add_argument("--max_depth", type=int,   default=6,
                   help="BFS depth for subnetwork expansion")
    p.add_argument("--query_gene", type=int,  default=None,
                   help="Query gene index (0-indexed). If None, picks last-cluster TF.")
    p.add_argument("--skip_cardamom", action="store_true",
                   help="Skip CardamomOT inference (only run expansion + recall)")
    p.add_argument("--gcn_epochs", type=int, default=300,
                   help="GCN training epochs")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    sim_dir       = _ROOT / args.sim_dir
    fig_dir       = _ROOT / "figures" / "block5_cardamom"
    cardamom_work = _ROOT / "benchmarks" / "cardamom_benchmark" / "output" / "cardamom_projects"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ───────────────────────────────────────────────────────
    meta        = json.loads((sim_dir / "metadata.json").read_text())
    cluster_ids = np.load(sim_dir / "cluster_ids.npy")
    inter_mask  = np.load(sim_dir / "inter_mask.npy")    # (G+1, G+1)
    inter_opt   = np.load(sim_dir / "inter_opt.npy")     # (G+1, G+1)
    tf_mask     = np.load(sim_dir / "tf_mask.npy")       # (G,) bool

    n_genes    = len(cluster_ids)
    budget     = args.budget or n_genes
    tp_list    = sorted(meta["timepoints"])

    print("=" * 60)
    print(f"CardamomOT benchmark  |  {meta['scenario']}  G={n_genes}")
    print(f"Budget={budget}  Timepoints={tp_list}")
    print("=" * 60)

    # ── Load expression data ────────────────────────────────────────────────
    print("\n[1] Loading simulation data...")
    runs, cluster_ids_b4, tp_list = load_scenario(str(sim_dir))
    print(f"  {len(runs)} runs, {len(tp_list)} timepoints")

    # ── Train GCN (shared weights, same as Block 4) ─────────────────────────
    print("\n[2] Training GCN...")
    gcn = train_gcn(runs, tp_list, n_epochs=args.gcn_epochs, seed=args.seed)

    # ── Compute KL vectors from GCN embeddings (same as Block 4) ────────────
    print("\n[3] Computing KL vectors from GCN embeddings...")
    emb_kl_vecs = compute_kl_vecs(runs, tp_list, model=gcn)

    # ── Build directed graph ─────────────────────────────────────────────────
    print("\n[4] Building directed graph from GCN embeddings...")
    _, C_full = build_directed_kl_graph(emb_kl_vecs, threshold=0.0)
    threshold = args.threshold if args.threshold is not None else float(np.percentile(C_full[C_full > 0], 90))
    G_dir, C_dir = build_directed_kl_graph(emb_kl_vecs, threshold=threshold)
    print(f"  {G_dir.number_of_edges()} directed edges (threshold={threshold:.4f})")


    # ── Pick query gene ─────────────────────────────────────────────────────
    if args.query_gene is not None:
        query = args.query_gene
    else:
        # Default: pick a TF in the last (most downstream) cluster
        K = cluster_ids.max()
        last_cluster_tfs = [g for g in range(n_genes)
                            if cluster_ids[g] == K and tf_mask[g]]
        if last_cluster_tfs:
            query = max(last_cluster_tfs, key=lambda g: G_dir.in_degree(g))
        else:
            last_cluster = np.where(cluster_ids == K)[0]
            query = int(last_cluster[0])

    print(f"\n[5] Query gene: {query}  (cluster {cluster_ids[query]})")
    print(f"  In-degree={G_dir.in_degree(query)}  Out-degree={G_dir.out_degree(query)}")

    # Diagnose: path from stim-direct genes to query at threshold=0
    stim_direct_genes = list(np.where(inter_mask[0, 1:])[0])
    print(f"  Stim-direct genes: {stim_direct_genes}")
    G_dense, _ = build_directed_kl_graph(emb_kl_vecs, threshold=0.0)
    for sd in stim_direct_genes:
        if sd in G_dense and query in G_dense and nx.has_path(G_dense, sd, query):
            path = nx.shortest_path(G_dense, sd, query)
            min_w = min(G_dense[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            print(f"  Gene {sd}→{query}: path len={len(path)-1}, bottleneck_weight={min_w:.4f}")
        else:
            print(f"  Gene {sd}→{query}: NO PATH even at threshold=0")

    # ── Expand subnetworks ──────────────────────────────────────────────────
    print(f"\n[6] Expanding subnetwork (budget={budget})...")

    gset_genes = expand_from_seed(
        G_dir,
        seed=query,
        budget=budget,
        max_depth=args.max_depth,
        cluster_ids=cluster_ids,
    )
    full_genes = list(range(n_genes))

    print(f"  GSET selected: {len(gset_genes)} genes")
    print(f"  Full gene set: {len(full_genes)} genes")
    for k in range(int(cluster_ids.max()) + 1):
        n = sum(1 for g in gset_genes if cluster_ids[g] == k)
        print(f"    Cluster {k}: {n}/{(cluster_ids == k).sum()} genes")

    # ── Subgraph recall (intrinsic validation, Elias Ventre design) ────────────
    print("\n[7] Subgraph recall vs ground truth GRN...")
    recall_results = {}
    for label, genes in [("GSET", gset_genes), ("Full", full_genes)]:
        recall = subgraph_recall(genes, inter_mask, inter_opt, query=query)
        recall_results[label] = recall
        path_str = f"{recall['recall_path']:.1%}" if recall['recall_path'] is not None else "N/A"
        print(f"  {label:<8}: stim_direct={recall['recall_stim_direct']:.1%}"
              f"  bridge={recall['recall_bridge']:.1%}"
              f"  path={path_str}"
              f"  stim_connected={recall['stim_connected']}")

    # Subgraph recall bar chart (Elias design: stim_direct, bridge, path)
    recall_metrics = ["recall_stim_direct", "recall_bridge", "recall_downstream", "recall_reachable"]
    recall_labels  = ["Stim-direct", "Bridge", "Downstream", "All affected"]
    colors_r       = {"GSET": "#2196F3", "Full": "#90CAF9"}
    x = np.arange(len(recall_metrics))
    width = 0.35
    fig_r, axes_r = plt.subplots(1, 2, figsize=(12, 4))

    # Left: recall bars
    ax_r = axes_r[0]
    for i, (cond, color) in enumerate(colors_r.items()):
        vals = [recall_results[cond].get(m, 0) or 0 for m in recall_metrics]
        bars = ax_r.bar(x + (i - 0.5) * width, vals, width,
                        label=cond, color=color, edgecolor="k", linewidth=0.7)
        for bar, val in zip(bars, vals):
            ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                      f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(recall_labels, fontsize=9)
    ax_r.set_ylim(0, 1.2)
    ax_r.set_ylabel("Recall")
    ax_r.legend()
    ax_r.set_title("Stimulus-rooted recall by gene category", fontsize=10)

    # Right: path recall + connectivity
    ax_p = axes_r[1]
    path_vals = {
        label: recall_results[label].get("recall_path") or 0
        for label in ["GSET", "Full"]
    }
    conn_vals = {
        label: int(recall_results[label].get("stim_connected", False))
        for label in ["GSET", "Full"]
    }
    conds = ["GSET", "Full"]
    x2 = np.arange(2)
    bars1 = ax_p.bar(x2 - 0.2, [path_vals[c] for c in conds], 0.35,
                     color=["#2196F3", "#90CAF9"], edgecolor="k", linewidth=0.7,
                     label="Path recall (stim→query)")
    bars2 = ax_p.bar(x2 + 0.2, [conn_vals[c] for c in conds], 0.35,
                     color=["#F44336", "#EF9A9A"], edgecolor="k", linewidth=0.7,
                     label="Stim connected")
    for bars in [bars1, bars2]:
        for bar, val in zip(bars, [path_vals["GSET"], path_vals["Full"]] if bars is bars1 else [conn_vals["GSET"], conn_vals["Full"]]):
            ax_p.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                      f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax_p.set_xticks(x2)
    ax_p.set_xticklabels(conds)
    ax_p.set_ylim(0, 1.3)
    ax_p.set_ylabel("Score")
    ax_p.legend(fontsize=8)
    ax_p.set_title(f"Path recall & connectivity\n(stim → query gene {query})", fontsize=10)

    plt.suptitle(
        f"Subgraph recall vs ground truth GRN  |  {meta['scenario']}  G={n_genes}  budget={budget}",
        fontsize=11)
    plt.tight_layout()
    fig_r.savefig(fig_dir / "subgraph_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig_r)
    print(f"  → {fig_dir / 'subgraph_recall.png'}")

    # ── CardamomOT inference ────────────────────────────────────────────────
    # Design (Elias Ventre):
    #   GSET:      CardamomOT on GSET genes only  → vs theta[GSET genes]
    #   Full→GSET: CardamomOT on ALL genes         → restrict output to GSET genes → vs theta[GSET genes]
    #
    # Same reference theta, different inference context.
    # GSET ≈ Full→GSET → selection is causally closed (good)
    # GSET << Full→GSET → selection misses upstream drivers (bad)
    results = {}
    if not args.skip_cardamom:
        print("\n[8] Running CardamomOT inference...")
        cardamom_work.mkdir(parents=True, exist_ok=True)


        # Step A: run on GSET subset
        proj_gset = cardamom_work / "gset"
        if proj_gset.exists():
            shutil.rmtree(proj_gset)
        prepare_cardamom_project(sim_dir, proj_gset, gene_subset=gset_genes)
        print(f"\n  Running CardamomOT on GSET subset ({len(gset_genes)} genes)...")
        inter_gset, _ = run_cardamom(proj_gset)
        if inter_gset is not None:
            inter_true_gset = restrict_inter(inter_opt, gset_genes)
            results["GSET"] = edge_coherence(inter_gset, inter_true_gset)
            m = results["GSET"]
            print(f"  GSET: AUROC={m['auroc']:.3f}  sign_agree={m['sign_agreement']:.3f}")
        else:
            print("  GSET: CardamomOT FAILED")
            results["GSET"] = {}

        # Step B: run on full gene set
        proj_full = cardamom_work / "full"
        if proj_full.exists():
            shutil.rmtree(proj_full)
        prepare_cardamom_project(sim_dir, proj_full, gene_subset=None)
        print(f"\n  Running CardamomOT on Full gene set ({len(full_genes)} genes)...")
        inter_full, _ = run_cardamom(proj_full)

        # Step C: restrict full output to GSET genes (Elias control)
        if inter_full is not None:
            inter_full_gset = restrict_inter(inter_full, gset_genes)
            inter_true_gset = restrict_inter(inter_opt, gset_genes)
            results["Full→GSET"] = edge_coherence(inter_full_gset, inter_true_gset)
            m = results["Full→GSET"]
            print(f"  Full→GSET: AUROC={m['auroc']:.3f}  sign_agree={m['sign_agreement']:.3f}")
        else:
            print("  Full→GSET: CardamomOT FAILED")
            results["Full→GSET"] = {}

        compare_selections(results, ["GSET", "Full→GSET"])

    print("\n[9] Saving plots...")

    # ── Edge coherence bar chart ─────────────────────────────────────────────
    if results:
        conditions = [l for l in ["GSET", "Full→GSET"] if l in results and results[l]]
        metrics    = ["auroc", "sign_agreement"]
        labels_m   = ["AUROC", "Sign agreement"]
        colors     = {"GSET": "#2196F3", "Full→GSET": "#90CAF9"}

        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
        for ax, metric, mlabel in zip(axes2, metrics, labels_m):
            vals = [results[c].get(metric, float("nan")) for c in conditions]
            bars = ax.bar(conditions, vals,
                          color=[colors[c] for c in conditions],
                          edgecolor="k", linewidth=0.7, width=0.5)
            ax.set_title(mlabel, fontsize=11, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)
            ax.set_ylabel(mlabel)
            for bar, val in zip(bars, vals):
                if val == val:
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.suptitle(
            f"CardamomOT edge coherence vs ground truth\n"
            f"{meta['scenario']}  G={n_genes}  budget={budget}  query=C{cluster_ids[query]}",
            fontsize=10)
        plt.tight_layout()
        fig2.savefig(fig_dir / "edge_coherence.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  → {fig_dir / 'edge_coherence.png'}")


    print("\n✓ Done")


if __name__ == "__main__":
    main()
