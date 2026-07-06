"""
diagnostics.py
--------------
Diagnostic visualisations for GSET simulation outputs:

  plot_umap(data, grn, out_path)
      UMAP of cells coloured by timepoint (and optionally cluster / gene expression)

  plot_marginals(data, grn, genes, timepoints, out_path)
      Evolution of marginal mRNA distributions for selected genes across timepoints

  check_scalability(G, K, n_cells, seed)
      Quick feasibility check for large networks (G=200+)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(data_path):
    """Load HARISSA data file. Returns (X, times, stim) where X shape (C, G)."""
    raw   = np.loadtxt(data_path, delimiter="\t").T.astype(int)
    times = raw[1:, 0].astype(float)
    X     = raw[1:, 2:].astype(float)   # mRNA counts, genes 1..G
    stim  = raw[1:, 1].astype(float)
    return X, times, stim


def _timepoint_colors(times):
    """Map timepoints to colours on a sequential colormap."""
    tps     = np.unique(times)
    cmap    = cm.plasma
    tp_to_c = {t: cmap(i / max(len(tps) - 1, 1)) for i, t in enumerate(tps)}
    colors  = np.array([tp_to_c[t] for t in times])
    return colors, tp_to_c, tps


# ─────────────────────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap(
    data_path: str,
    grn,
    out_path:  str,
    n_neighbors: int  = 15,
    min_dist:    float= 0.3,
    log1p:       bool = True,
    seed:        int  = 42,
    extra_title: str  = "",
):
    """
    UMAP of all cells coloured by timepoint.
    Special genes (shared / hubs) are highlighted with a second panel
    showing their expression overlaid on the UMAP.
    """
    import umap as umap_lib

    X, times, _ = _load_data(data_path)
    if log1p:
        X_emb = np.log1p(X)
    else:
        X_emb = X.copy()

    # Fit UMAP
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             random_state=seed, n_components=2)
    emb = reducer.fit_transform(X_emb)   # (C, 2)

    colors, tp_to_c, tps = _timepoint_colors(times)
    special = {}
    for kind in ("shared", "hubs"):
        ids = grn.special_ids.get(kind, [])
        if ids:
            special[kind] = ids

    n_panels = 1 + len(special)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # Panel 0: coloured by timepoint
    ax = axes[0]
    for t in tps:
        mask = times == t
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[tp_to_c[t]], s=8, alpha=0.7, label=f"t={int(t)}")
    ax.set_title(f"UMAP — coloured by timepoint{' ('+extra_title+')' if extra_title else ''}")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=6, markerscale=2, loc="best", ncol=2)
    ax.set_xticks([]); ax.set_yticks([])

    # Extra panels: mean expression of special genes
    for ax, (kind, ids) in zip(axes[1:], special.items()):
        # Mean normalised expression of special genes
        expr = np.log1p(X[:, ids]).mean(axis=1)
        sc   = ax.scatter(emb[:, 0], emb[:, 1], c=expr,
                          cmap="YlOrRd", s=8, alpha=0.8)
        plt.colorbar(sc, ax=ax, fraction=0.04)
        ax.set_title(f"UMAP — mean {kind} expression")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  UMAP              → {out_path}")
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Marginal distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_marginals(
    data_path:  str,
    grn,
    out_path:   str,
    n_per_cluster: int = 2,
    n_special:     int = 2,
    max_count:     int = None,
):
    """
    For each selected gene, show the marginal mRNA count distribution at each
    timepoint as overlapping histograms (one colour per timepoint).

    Genes shown:
      - n_per_cluster genes from each cluster (TFs preferred)
      - n_special genes from shared / hub list
    """
    X, times, _ = _load_data(data_path)
    tps          = np.unique(times)
    colors, tp_to_c, _ = _timepoint_colors(times)

    # Select genes to show
    gene_ids   = []   # 0-indexed into X (= HARISSA gene -1)
    gene_labels = []

    for k in range(grn.n_clusters):
        tfs = grn.tfs_per_cluster[k]
        cands = list(tfs[:n_per_cluster])
        if len(cands) < n_per_cluster:
            others = [g for g in grn.genes_per_cluster[k] if g not in tfs]
            cands += others[:n_per_cluster - len(cands)]
        for g in cands:
            gene_ids.append(int(g))
            gene_labels.append(f"G{g+1} (C{k}{'·TF' if grn.tf_mask[g] else ''})")

    for kind in ("shared", "hubs"):
        ids = grn.special_ids.get(kind, [])
        for g in ids[:n_special]:
            gene_ids.append(int(g))
            gene_labels.append(f"G{g+1} ({kind[:3]})")

    n_genes = len(gene_ids)
    if n_genes == 0:
        return

    fig, axes = plt.subplots(
        1, n_genes,
        figsize=(max(3, 3.2 * n_genes), 3.5),
        squeeze=False
    )
    axes = axes[0]

    for ax, g_idx, label in zip(axes, gene_ids, gene_labels):
        counts = X[:, g_idx]
        xmax   = max_count or int(np.percentile(counts[counts > 0], 98)) if (counts > 0).any() else 10
        bins   = np.arange(0, xmax + 2)

        for t in tps:
            mask = times == t
            ax.hist(counts[mask], bins=bins, density=True,
                    color=tp_to_c[t], alpha=0.45, histtype="stepfilled")
            ax.hist(counts[mask], bins=bins, density=True,
                    color=tp_to_c[t], alpha=0.9, histtype="step", lw=1.2)

        ax.set_title(label, fontsize=8)
        ax.set_xlabel("mRNA count", fontsize=7)
        ax.set_ylabel("Density" if ax is axes[0] else "", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.5, xmax + 0.5)

    # Shared colorbar (timepoint)
    sm = plt.cm.ScalarMappable(cmap=cm.plasma,
                                norm=plt.Normalize(tps.min(), tps.max()))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=axes.tolist(), fraction=0.015, pad=0.01)
    cb.set_label("Timepoint", fontsize=8)
    cb.set_ticks(tps); cb.set_ticklabels([str(int(t)) for t in tps], fontsize=6)

    plt.suptitle("Marginal mRNA distributions across timepoints", fontsize=9, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Marginals         → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Scalability check
# ─────────────────────────────────────────────────────────────────────────────

def check_scalability(
    n_genes:   int = 200,
    n_clusters: int = 4,
    n_cells:   int = 500,
    scenario:  str = "switch",
    seed:      int = 0,
) -> Dict:
    """
    Quick feasibility check for large networks.
    Tests: GRN generation, parameter optimisation (10 epochs), one simulation run.
    Returns timing dict.
    """
    import time, sys
    sys.path.insert(0, ".")
    from grn_generator import generate_grn
    from harissa_optimizer import HarissaParameterOptimizer, make_scenario
    from simulator import SimulationScenario, build_harissa_model, simulate_dataset

    print(f"\n{'='*55}")
    print(f"Scalability check: G={n_genes}, K={n_clusters}, C={n_cells}")
    print(f"{'='*55}")
    results = {}

    # GRN generation
    t0 = time.time()
    inter_sign = "negative" if scenario in ("switch",) else "none"
    grn = generate_grn(n_genes=n_genes, n_clusters=n_clusters,
                       n_tfs_per_cluster=2, rho_in=0.4, rho_out=0.05,
                       inter_cluster_sign=inter_sign, seed=seed)
    grn.summary()
    results["grn_sec"] = time.time() - t0
    print(f"  GRN generation:   {results['grn_sec']:.2f}s")

    # Scenario + optimisation (short)
    t0 = time.time()
    grn, signs, states, transitions, kp = make_scenario(grn, scenario, seed=seed)
    opt = HarissaParameterOptimizer(grn=grn, sign_matrix=signs,
        cell_states=states, transitions=transitions,
        kinetic_params=kp, scenario=scenario)
    result = opt.optimise(n_epochs=10, lr=0.05, verbose=False, log_every=10)
    results["opt_sec"] = time.time() - t0
    print(f"  Optimisation (10e): {results['opt_sec']:.2f}s")

    # Simulation (one timepoint, n_cells cells)
    t0 = time.time()
    model = build_harissa_model(n_genes, result)
    scen  = SimulationScenario(timepoints=[0, 48], n_cells_per_tp=n_cells // 2,
                               burnin=5.0)
    data  = simulate_dataset(model, scen, seed=seed, verbose=False)
    results["sim_sec"] = time.time() - t0
    print(f"  Simulation ({n_cells} cells): {results['sim_sec']:.2f}s")
    print(f"  Data shape: {data[1:, 2:].shape}")
    print(f"  → {results['sim_sec']/n_cells*1000:.1f} ms/cell")

    return results


def load_grn_from_dir(out_dir: str):
    """
    Reconstruct a minimal GRNStructure from saved numpy arrays in out_dir.
    Also loads special_ids from special_ids.json if present.
    """
    import json
    from grn_generator import GRNStructure
    d = Path(out_dir)
    inter_mask  = np.load(d / "inter_mask.npy")
    cluster_ids = np.load(d / "cluster_ids.npy")
    tf_mask     = np.load(d / "tf_mask.npy")
    n_genes     = len(cluster_ids)
    n_clusters  = int(cluster_ids[cluster_ids < cluster_ids.max()].max() + 1)                   if (cluster_ids < cluster_ids.max()).any()                   else int(cluster_ids.max() + 1)
    # Count TFs per cluster
    n_tfs = max(
        int(((cluster_ids == k) & tf_mask).sum()) for k in range(n_clusters)
    )
    special_ids = {}
    sp_path = d / "special_ids.json"
    if sp_path.exists():
        with open(sp_path) as f:
            raw = json.load(f)
        special_ids = {k: [int(v) for v in vs] for k, vs in raw.items()}

    return GRNStructure(
        inter_mask=inter_mask, cluster_ids=cluster_ids, tf_mask=tf_mask,
        n_genes=n_genes, n_clusters=n_clusters,
        n_tfs_per_cluster=n_tfs, rho_in=0.5, rho_out=0.0,
        special_ids=special_ids,
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")

    data_path = "out_shared2/data/data_1.txt"
    out_dir   = "out_shared2"
    if os.path.exists(data_path):
        grn = load_grn_from_dir(out_dir)
        print("Loaded GRN:", grn.n_genes, "genes, special_ids:", grn.special_ids)
        print("=== UMAP ===")
        plot_umap(data_path, grn, "/tmp/test_umap.png", extra_title="shared_targets")
        print("=== Marginals ===")
        plot_marginals(data_path, grn, "/tmp/test_marginals.png",
                       n_per_cluster=1, n_special=2)

    print("\n=== Scalability: G=100, K=3 ===")
    check_scalability(n_genes=100, n_clusters=3, n_cells=200, scenario="switch")
    print("\n=== Scalability: G=300, K=4 ===")
    check_scalability(n_genes=300, n_clusters=4, n_cells=500, scenario="switch")
