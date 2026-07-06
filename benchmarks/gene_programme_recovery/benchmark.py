"""
benchmark.py
------------
End-to-end benchmark connecting HARISSA simulations to gene programme recovery.

For each simulation scenario:
  1. Load simulated data -> AnnData
  4. Compute multi-timepoint ablation vectors (new):
       (a) z_global    — single embedding pooling ALL cells/timepoints (no temporal info)
       (b) z_concat    — concatenation of per-timepoint embeddings [z_1 ‖ … ‖ z_T]
       (c) delta_concat — concatenation of pairwise deltas [δ_(1,2) ‖ … ‖ δ_(T-1,T)]
                          for both OT and KL variants
       (d) full_concat — z_concat ‖ delta_concat (embeddings + deltas together)
  5. Compute classic gene_ot W matrix  (no temporal info)
  6. Compute temporal gene_ot W matrix (cells connected only to adjacent timepoints)
  7. Evaluate ALL methods with UNIFIED METRICS:
       - ARI / NMI from distance-matrix-based clustering
       - Intra vs inter-cluster distance (lower = better grouping)
       - Intra vs inter-cluster cosine similarity (for vector methods)
  8. Plot summary: cosine similarity matrices + intra/inter bars

Multi-timepoint vector construction
------------------------------------
Given T timepoints t_1 < ... < t_T, per-timepoint embeddings z_t in R^(GxD) are
produced by one of two backends, controlled by the `use_gcn` flag:

  use_gcn=False (default, fast):
    z_t = Pearson correlation profile at time t (row i of the GxG corr matrix).
    D = G.  No training required.

  use_gcn=True:
    A shared-weight 2-layer GCN (gcn_embedding.py) is trained jointly on all
    timepoints. Input features H^(0)_t are the same correlation profiles; the
    GCN compresses them to D=embed_dim (default 16) while learning to reconstruct
    WGCNA edge weights via a dot-product decoder.  Weights W^(0), W^(1) are
    shared across ALL timepoints.

In both cases the six ablation vectors are built identically:
  z_global        : pooled embedding (all cells, no time split), GxD
  z_concat        : [z_1 || ... || z_T] (L2-norm per block), Gx(T*D)
  delta_OT_concat : [d_OT_(1,2) || ... || d_OT_(T-1,T)], Gx((T-1)*D)
  delta_KL_concat : [vec_KL_(1,2) || ... || vec_KL_(T-1,T)], Gx((T-1)*G)
  full_OT_concat  : z_concat || delta_OT_concat
  full_KL_concat  : z_concat || delta_KL_concat

All vectors are compared via cosine distance (1 - cosine_sim) to produce a GxG
distance matrix, fed into evaluate_W_clustering (ARI, NMI, intra/inter).

Unified evaluation
------------------
All methods (GSET 2-tp, multi-tp ablations, gene_ot) are evaluated with the same
ARI/NMI pipeline. GSET/ablation methods use cosine-distance W; gene_ot uses its W.

Usage
-----
  python benchmark.py --scenarios switch cascade shared_targets --out_dir bench_out
  python benchmark.py --scenarios all --out_dir bench_out
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import sys
import os
_GSET_ROOT = Path(__file__).resolve().parents[2]
if str(_GSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_GSET_ROOT))

from harissa_to_anndata import (
    harissa_to_anndata, compute_gene_embeddings,
    cosine_similarity_matrix, intra_inter_cosine, evaluate_W_clustering,
)
from core import GCNEncoder, train_gcn, get_embedding
from core.scoring import kl_vec, ot_residual_vec


class GCNEmbedder:
    """Wrapper around gset GCNEncoder to match the GCNEmbedder interface."""

    def __init__(self, hidden_dim=64, embed_dim=16, beta=6, tau=0.01,
                 lr=1e-3, n_epochs=300, verbose=True, seed=1):
        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim
        self.n_epochs   = n_epochs
        self.verbose    = verbose
        self.seed       = seed
        self.model      = None

    def fit(self, adata, active_tps):
        times  = adata.obs["time"].values
        n_genes = adata.shape[1]
        X_list = [adata.X[times == t].astype(float) for t in active_tps
                  if (times == t).sum() > 0]
        self.model = train_gcn(
            X_list,
            n_genes  = n_genes,
            n_epochs = self.n_epochs,
            d_out    = self.embed_dim,
            seed     = self.seed,
        )

    def transform(self, adata, t):
        times = adata.obs["time"].values
        X = adata.X[times == t].astype(float)
        return get_embedding(self.model, X)

    def transform_pooled(self, adata):
        X = adata.X.astype(float)
        return get_embedding(self.model, X)

ALL_SCENARIOS = ["switch", "cascade", "shared_targets", "parallel_sync", "parallel_async", "early_response"]

SCENARIO_DIRS = {
    "switch":          "block3_switch",
    "cascade":         "block3_5_cascade",
    "shared_targets":  "block3_shared_targets",
    "parallel_sync":   "block3_parallel_sync",
    "parallel_async":  "block3_parallel_async",
    "early_response":  "block3_early_response",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: convert GSET vectors to distance matrices for unified evaluation
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance_matrix(V: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix (1 - cosine_similarity) for rows of V.
    Returns a symmetric matrix in [0, 2] with zeros on the diagonal.
    """
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    V_norm = V / norms
    cos_sim = V_norm @ V_norm.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim   # cosine distance in [0, 2]


def gset_vector_to_W(V: np.ndarray) -> np.ndarray:
    """
    Convert a GSET gene-vector matrix (G x d) to a W-style distance matrix (G x G)
    using cosine distance, so it can be fed into evaluate_W_clustering.
    """
    return cosine_distance_matrix(V).astype(np.float32)


def auroc_from_W(W: np.ndarray, cluster_ids: np.ndarray,
                 exclude_sentinel: bool = True) -> float:
    """
    AUROC: same-cluster pairs should have LOWER distance than inter-cluster pairs.
    Uses upper triangle only (no self-pairs).
    """
    from sklearn.metrics import roc_auc_score
    ids = cluster_ids.copy()
    if exclude_sentinel:
        K = ids.max()
        mask = ids < K
        W = W[np.ix_(mask, mask)]
        ids = ids[mask]
    G = len(ids)
    rows, cols = np.triu_indices(G, k=1)
    scores = -W[rows, cols]          # higher similarity = lower distance
    labels = (ids[rows] == ids[cols]).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return roc_auc_score(labels, scores)


def max_rank_distance(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Element-wise max of normalised rank matrices from two distance matrices."""
    G = W1.shape[0]
    def rank_mat(W):
        flat = W.flatten()
        return np.argsort(np.argsort(flat)).reshape(W.shape) / (G * G)
    return np.maximum(rank_mat(W1), rank_mat(W2)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timepoint vector construction
# ─────────────────────────────────────────────────────────────────────────────

def _l2_normalize_blocks(blocks: list[np.ndarray]) -> list[np.ndarray]:
    """
    L2-normalise each block (G×d_k) row-wise so that each block contributes
    equally to the cosine similarity of the concatenated vector regardless of
    its dimension d_k.  A gene with zero norm gets a zero vector.
    """
    normed = []
    for B in blocks:
        norms = np.linalg.norm(B, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        normed.append(B / norms)
    return normed


def compute_global_embedding(adata, gcn) -> np.ndarray:
    """(a) z_global -- GCN embedding pooling ALL cells across ALL timepoints."""
    return gcn.transform_pooled(adata)   # G x embed_dim


def compute_multitimepoint_vectors(
    adata,
    active_tps,
    gcn,
    verbose: bool = True,
) -> dict:
    """
    Compute the six ablation vectors for all active timepoints using GCN embeddings.
    """
    T = len(active_tps)
    assert T >= 2, "Need at least 2 active timepoints."

    # -- (a) z_global ---------------------------------------------------------
    if verbose:
        print(f"  [multi-tp] (a) z_global [GCN] -- pooled embedding...")
    z_global = compute_global_embedding(adata, gcn)   # G x D

    # -- per-timepoint embeddings and pairwise deltas -------------------------
    z_blocks        = []
    dOT_blocks      = []
    dKL_blocks      = []

    for i in range(T - 1):
        t_i  = active_tps[i]
        t_i1 = active_tps[i + 1]
        if verbose:
            print(f"  [multi-tp] pair ({t_i:.2g} -> {t_i1:.2g}) [GCN]...")

        # Correlation profiles: needed for KL vectors
        E_i, E_i1 = compute_gene_embeddings(
            adata, t0=t_i, tT=t_i1, method="correlation"
        )

        Z_i  = gcn.transform(adata, t_i)    # G x embed_dim
        Z_i1 = gcn.transform(adata, t_i1)

        if i == 0:
            z_blocks.append(Z_i)   # z_1
        z_blocks.append(Z_i1)      # z_{i+2}

        dOT_blocks.append(ot_residual_vec(Z_i, Z_i1))
        dKL_blocks.append(kl_vec(Z_i, Z_i1))

    # -- (b) z_concat --------------------------------------------------------
    z_normed        = _l2_normalize_blocks(z_blocks)
    z_concat        = np.concatenate(z_normed, axis=1)

    # -- (c) delta_concat -----------------------------------------------------
    dOT_normed        = _l2_normalize_blocks(dOT_blocks)
    dKL_normed        = _l2_normalize_blocks(dKL_blocks)
    delta_OT_concat   = np.concatenate(dOT_normed, axis=1)
    delta_KL_concat   = np.concatenate(dKL_normed, axis=1)

    # -- (d) full_concat ------------------------------------------------------
    full_OT_concat  = np.concatenate([z_concat, delta_OT_concat], axis=1)
    full_KL_concat  = np.concatenate([z_concat, delta_KL_concat], axis=1)

    if verbose:
        G_n = z_global.shape[0]
        print(f"  [multi-tp] shapes (G={G_n}, T={T}, backend=GCN):")
        for name, V in [("z_global", z_global), ("z_concat", z_concat),
                        ("delta_OT", delta_OT_concat), ("delta_KL", delta_KL_concat),
                        ("full_OT", full_OT_concat), ("full_KL", full_KL_concat)]:
            print(f"    {name:16s}: {V.shape}")

    return {
        "z_global":              z_global,
        "z_concat":              z_concat,
        "delta_OT_concat":       delta_OT_concat,
        "delta_KL_concat":       delta_KL_concat,
        "full_OT_concat":        full_OT_concat,
        "full_KL_concat":        full_KL_concat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gene OT distances (adapted from gene_ot_distances.py, no scanpy preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

import ot as pot

def _sinkhorn_divergence(a, b, C, reg, numItermax=100000, stopThr=1e-9):

    Wab = pot.sinkhorn2(a, b, C, reg=reg, numItermax=numItermax, stopThr=stopThr, warn=False)
    Waa = pot.sinkhorn2(a, a, C, reg=reg, numItermax=numItermax, stopThr=stopThr, warn=False)
    Wbb = pot.sinkhorn2(b, b, C, reg=reg, numItermax=numItermax, stopThr=stopThr, warn=False)

    Wab = float(Wab[0]) if hasattr(Wab, "__len__") else float(Wab)
    Waa = float(Waa[0]) if hasattr(Waa, "__len__") else float(Waa)
    Wbb = float(Wbb[0]) if hasattr(Wbb, "__len__") else float(Wbb)

    return max(Wab - 0.5 * Waa - 0.5 * Wbb, 0.0)


def compute_gene_ot_W(
    adata,
    mode:              str   = "classic",
    k_knn:             int   = 5,
    n_meta:            int   = 50,
    n_meta_per_tp:     int   = 20,
    n_neighbors_gene:  int   = 10,
    sinkhorn_reg:      float = 0.05,
    dense:             bool  = True,    # NEW: compute all gene pairs, no sparsification
    verbose:           bool  = True,
) -> np.ndarray:
    """
    Compute gene×gene Wasserstein distance matrix from AnnData.
    Minimal preprocessing: log1p normalisation, no HVG filtering.
    Returns W of shape (G, G).

    Parameters
    ----------
    dense : bool, default True
        If True, compute OT distances for ALL gene pairs (O(G²) OT solves).
        No L1 pre-screening, no kNN sparsification, no shortest-path completion.
        This gives an exact dense distance matrix.
        If False, use the sparse approximation: L1 pre-screening to select
        `n_neighbors_gene * 5` candidate pairs per gene, then kNN graph
        shortest-path completion for missing pairs.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA

    X_raw   = adata.X.astype(float)
    X_log   = np.log1p(X_raw)
    times   = adata.obs["time"].values
    G       = X_raw.shape[1]
    n_cells = X_raw.shape[0]

    # Gene distributions: rho[g, c] = normalised expression of gene g in cell c
    col_sums = X_log.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    rho = (X_log / col_sums).T   # (G, n_cells)

    # PCA for cell graph
    n_pcs = min(15, n_cells - 1, G - 1)
    pca   = PCA(n_components=n_pcs, random_state=42)
    X_pca = pca.fit_transform(X_log)   # (n_cells, n_pcs)

    # Build cell graph and get cost matrix C + aggregation matrix M
    if mode == "classic":
        C_cost, M = _build_classic(X_pca, k_knn, n_meta, verbose)
    else:
        C_cost, M = _build_temporal(X_pca, times, k_knn, n_meta_per_tp, verbose)

    # Aggregate gene distributions to metacells
    rho_meta = rho @ M   # (G, n_meta)
    rho_meta += 1e-16
    rho_meta /= rho_meta.sum(axis=1, keepdims=True)

    # Cost matrix for OT (normalised)
    C_ot = (C_cost.astype(np.float64)) ** 1
    C_ot /= (C_ot.max() + 1e-10)

    W = np.zeros((G, G), dtype=np.float32)

    if dense:
        # ── Dense mode: compute ALL gene pairs ──────────────────────────────
        n_pairs = G * (G - 1) // 2
        if verbose:
            print(f"    [{mode}] dense mode: computing all {n_pairs} gene-pair "
                  f"OT distances (G={G})…")

        for i in range(G):
            for j in range(i + 1, G):
                a = np.asarray(rho_meta[i], dtype=np.float64) + 1e-12
                b = np.asarray(rho_meta[j], dtype=np.float64) + 1e-12
                a /= a.sum()
                b /= b.sum()
                if sinkhorn_reg is not None and sinkhorn_reg > 0:
                    w = _sinkhorn_divergence(a, b, C_ot, reg=sinkhorn_reg)
                else:
                    w = float(pot.emd2(a, b, C_ot, numItermax=1000))
                W[i, j] = w
                W[j, i] = w

    else:
        # ── Sparse mode: L1 pre-screening + kNN graph completion ────────────
        k_sparse = min(n_neighbors_gene * 5, G - 1)
        L1 = np.abs(rho_meta[:, None, :] - rho_meta[None, :, :]).sum(axis=-1)
        gene_neighbors = np.argsort(L1, axis=1)[:, 1:k_sparse + 1]

        pairs = [(i, j)
                 for i in range(G)
                 for j in gene_neighbors[i] if j > i]

        if verbose:
            print(f"    [{mode}] sparse mode: computing {len(pairs)} gene-pair "
                  f"OT distances (out of {G*(G-1)//2} total)…")

        for i, j in pairs:
            a = np.asarray(rho_meta[i], dtype=np.float64) + 1e-12
            b = np.asarray(rho_meta[j], dtype=np.float64) + 1e-12
            a /= a.sum()
            b /= b.sum()
            if sinkhorn_reg is not None and sinkhorn_reg > 0:
                w = _sinkhorn_divergence(a, b, C_ot, reg=sinkhorn_reg)
            else:
                w = float(pot.emd2(a, b, C_ot, numItermax=100000))
            W[i, j] = w
            W[j, i] = w

        # Shortest-path completion: fill missing gene pairs via graph distances
        W_sp = shortest_path(csr_matrix(W), method="D", directed=False)
        finite_max = W_sp[np.isfinite(W_sp)].max() if np.isfinite(W_sp).any() else 1.0
        W_sp[~np.isfinite(W_sp)] = finite_max * 2.0
        W = W_sp.astype(np.float32)

    return W


def _build_classic(X_pca, k, n_meta, verbose):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    n_cells = X_pca.shape[0]
    n_meta  = min(n_meta, n_cells)
    km      = MiniBatchKMeans(n_clusters=n_meta, random_state=42,
                               batch_size=min(2048, n_cells), n_init=3).fit(X_pca)
    centers = km.cluster_centers_
    labels  = km.labels_

    M = np.zeros((n_cells, n_meta), dtype=np.float32)
    for i, l in enumerate(labels): M[i, l] = 1.0
    M /= np.clip(M.sum(axis=0, keepdims=True), 1, None)

    k_m  = min(k * 2, n_meta - 1)
    nbrs = NearestNeighbors(n_neighbors=k_m + 1, metric="euclidean",
                             n_jobs=-1).fit(centers)
    d, idx = nbrs.kneighbors(centers)
    rows = np.repeat(np.arange(n_meta), k_m)
    cols = idx[:, 1:].ravel(); vals = d[:, 1:].ravel()
    G_sp = csr_matrix((vals, (rows, cols)), shape=(n_meta, n_meta))
    G_sp = (G_sp + G_sp.T) / 2
    C    = shortest_path(G_sp, method="D", directed=False).astype(np.float32)
    fm   = C[np.isfinite(C)].max()
    C[~np.isfinite(C)] = fm * 10
    return C, M


def _build_temporal(X_pca, times, k, n_meta_per_tp, verbose,
                    max_time_jump=None,
                    temporal_penalty=0.5,
                    penalty_mode="exp"):
    from sklearn.cluster import MiniBatchKMeans
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import NearestNeighbors

    k_intra = k
    k_inter = int(k / 2)
    unique_times = np.sort(np.unique(times))
    time_to_rank = {t: i for i, t in enumerate(unique_times)}
    tp_idx = {t: np.where(times == t)[0] for t in unique_times}

    all_centers, all_tp = [], []
    cell_to_meta = np.full(len(times), -1, dtype=int)
    offset = 0

    for t in unique_times:
        idx = tp_idx[t]
        nm = min(n_meta_per_tp, len(idx))
        km = MiniBatchKMeans(
            n_clusters=nm,
            random_state=42,
            batch_size=min(2048, len(idx)),
            n_init=10
        ).fit(X_pca[idx])

        all_centers.append(km.cluster_centers_)
        all_tp.extend([t] * nm)
        cell_to_meta[idx] = offset + km.labels_
        offset += nm

    centers = np.vstack(all_centers)
    meta_tp = np.asarray(all_tp)
    n_meta = len(meta_tp)

    M = np.zeros((len(times), n_meta), dtype=np.float32)
    M[np.arange(len(times)), cell_to_meta] = 1.0
    M /= np.clip(M.sum(axis=0, keepdims=True), 1, None)

    dist_mat = np.full((n_meta, n_meta), np.inf, dtype=np.float64)
    np.fill_diagonal(dist_mat, 0.0)

    tp_meta_idx = {t: np.where(meta_tp == t)[0] for t in unique_times}

    def time_penalty(dt):
        if penalty_mode == "linear":
            return 1.0 + temporal_penalty * dt
        elif penalty_mode == "exp":
            return float(np.exp(temporal_penalty * dt))
        else:
            raise ValueError("penalty_mode must be 'linear' or 'exp'")

    # intra-timepoint
    for t in unique_times:
        idxm = tp_meta_idx[t]
        if len(idxm) < 2:
            continue
        k_eff = min(k_intra * 2, len(idxm) - 1)
        nbrs = NearestNeighbors(
            n_neighbors=k_eff + 1, metric="euclidean", n_jobs=-1
        ).fit(centers[idxm])
        d, nb = nbrs.kneighbors(centers[idxm])

        for i in range(len(idxm)):
            for jpos in range(1, k_eff + 1):
                gi, gj = idxm[i], idxm[nb[i, jpos]]
                w = float(d[i, jpos])
                if w < dist_mat[gi, gj]:
                    dist_mat[gi, gj] = w
                    dist_mat[gj, gi] = w

    # inter-timepoint, avec pénalité selon |Δt|
    for t0 in unique_times:
        r0 = time_to_rank[t0]
        idx0 = tp_meta_idx[t0]

        for t1 in unique_times:
            if t1 == t0:
                continue
            r1 = time_to_rank[t1]
            dt = max(0, abs(r1 - r0) - .5)

            if max_time_jump is not None and dt > max_time_jump:
                continue

            idx1 = tp_meta_idx[t1]
            if len(idx1) == 0:
                continue

            D = np.linalg.norm(
                centers[idx0][:, None, :] - centers[idx1][None, :, :],
                axis=-1
            )

            k_eff = min(k_inter, len(idx1))
            pen = time_penalty(dt)

            for i in range(len(idx0)):
                nn = np.argsort(D[i])[:k_eff]
                local_scale = np.median(D[i, nn]) + 1e-12
                for j in nn:
                    gi, gj = idx0[i], idx1[j]
                    w = float((D[i, j] / local_scale) * pen)
                    if w < dist_mat[gi, gj]:
                        dist_mat[gi, gj] = w
                        dist_mat[gj, gi] = w

    rows, cols = np.where(np.isfinite(dist_mat) & (dist_mat > 0))
    vals = dist_mat[rows, cols]
    G = csr_matrix((vals, (rows, cols)), shape=(n_meta, n_meta))

    C_cost = shortest_path(G, method="D", directed=False).astype(np.float64)
    finite = np.isfinite(C_cost)
    max_finite = C_cost[finite].max()
    C_cost[~finite] = max_finite * 10.0

    return C_cost, M


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_cosine_matrices(
    vectors:     dict,
    cluster_ids: np.ndarray,
    scenario:    str,
    out_path:    str,
):
    """
    Plot pairwise cosine similarity matrices sorted by cluster, one panel per method.
    """
    methods = list(vectors.keys())
    K = cluster_ids.max()
    mask = cluster_ids < K   # exclude sentinel special genes

    # Sort by cluster
    order    = np.argsort(cluster_ids[mask])
    ids_sort = cluster_ids[mask][order]

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1: axes = [axes]

    for ax, name in zip(axes, methods):
        V  = vectors[name]
        Vs = V[mask][order]
        C  = cosine_similarity_matrix(Vs)

        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Gene (sorted by cluster)")
        ax.set_ylabel("Gene (sorted by cluster)")
        ax.set_xticks([]); ax.set_yticks([])

        # Cluster boundary lines
        cum = 0
        for k in range(cluster_ids.max()):
            cum += int((ids_sort == k).sum())
            if cum < len(ids_sort):
                ax.axhline(cum - 0.5, color="k", lw=1.0)
                ax.axvline(cum - 0.5, color="k", lw=1.0)

        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.suptitle(f"Cosine similarity — {scenario}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()


def plot_W_heatmaps(
    W_dict:      dict,
    cluster_ids: np.ndarray,
    scenario:    str,
    out_path:    str,
):
    """
    Plot distance/W matrices sorted by cluster, one panel per method.
    Used for gene_ot W matrices and GSET cosine-distance matrices side by side.
    """
    methods = list(W_dict.keys())
    K = cluster_ids.max()
    mask = cluster_ids < K

    order    = np.argsort(cluster_ids[mask])
    ids_sort = cluster_ids[mask][order]

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1: axes = [axes]

    for ax, name in zip(axes, methods):
        W  = W_dict[name]
        Ws = W[np.ix_(np.where(mask)[0][order], np.where(mask)[0][order])]

        # Normalise for display
        vmax = np.percentile(Ws[Ws > 0], 95) if (Ws > 0).any() else 1.0
        im = ax.imshow(Ws, cmap="viridis_r", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Gene (sorted by cluster)")
        ax.set_ylabel("Gene (sorted by cluster)")
        ax.set_xticks([]); ax.set_yticks([])

        cum = 0
        for k in range(cluster_ids.max()):
            cum += int((ids_sort == k).sum())
            if cum < len(ids_sort):
                ax.axhline(cum - 0.5, color="w", lw=1.0)
                ax.axvline(cum - 0.5, color="w", lw=1.0)

        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.suptitle(f"Distance matrices (lower = more similar) — {scenario}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()


def plot_intra_inter_bars(
    results:  dict,
    scenario: str,
    out_path: str,
):
    """
    Bar chart: intra-cluster vs inter-cluster cosine similarity per GSET method.
    """
    methods = list(results.keys())
    intra_vals = [results[m]["intra"] for m in methods]
    inter_vals = [results[m]["inter"] for m in methods]
    gaps       = [i - e for i, e in zip(intra_vals, inter_vals)]

    x     = np.arange(len(methods))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.bar(x - width/2, intra_vals, width, label="Intra-cluster", color="steelblue", alpha=0.85)
    ax.bar(x + width/2, inter_vals, width, label="Inter-cluster", color="salmon",    alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title(f"Intra vs Inter — {scenario}")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=0.5)

    ax2 = axes[1]
    colors = ["steelblue" if g > 0 else "salmon" for g in gaps]
    ax2.bar(x, gaps, color=colors, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=8, rotation=20, ha="right")
    ax2.set_ylabel("Gap (intra − inter)")
    ax2.set_title(f"Separation gap — {scenario}")
    ax2.axhline(0, color="gray", lw=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()


def plot_unified_bars(
    unified_results: dict,
    scenario:        str,
    out_path:        str,
):
    """
    Unified bar chart comparing ALL methods (GSET + gene_ot) with shared metrics:
    ARI, NMI, and intra_W / inter_W distance ratio.

    unified_results: {method_name: {"ARI": ..., "NMI": ..., "intra_W": ..., "inter_W": ...}}
    """
    methods    = list(unified_results.keys())
    ari_vals   = [unified_results[m].get("ARI",     np.nan) for m in methods]
    nmi_vals   = [unified_results[m].get("NMI",     np.nan) for m in methods]
    intra_vals = [unified_results[m].get("intra_W", np.nan) for m in methods]
    inter_vals = [unified_results[m].get("inter_W", np.nan) for m in methods]

    x     = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: ARI
    ax = axes[0]
    colors = ["steelblue" if not np.isnan(v) else "lightgray" for v in ari_vals]
    ax.bar(x, ari_vals, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("ARI"); ax.set_ylim(-0.2, 1.05)
    ax.set_title(f"Adjusted Rand Index — {scenario}")
    ax.axhline(0, color="gray", lw=0.5)

    # Panel 2: NMI
    ax = axes[1]
    colors = ["darkorange" if not np.isnan(v) else "lightgray" for v in nmi_vals]
    ax.bar(x, nmi_vals, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("NMI"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Normalised Mutual Info — {scenario}")

    # Panel 3: intra vs inter distance
    ax = axes[2]
    ax.bar(x - width/2, intra_vals, width, label="Intra-cluster dist", color="steelblue", alpha=0.85)
    ax.bar(x + width/2, inter_vals, width, label="Inter-cluster dist", color="salmon",    alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Mean pairwise distance")
    ax.set_title(f"Intra vs inter distance — {scenario}\n(lower intra = better)")
    ax.legend(fontsize=7)

    plt.suptitle(f"Unified benchmark — {scenario}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()


def plot_summary_table(all_results: dict, out_path: str):
    """
    Summary table across all scenarios and methods.
    Shows ARI for all methods using the unified evaluation.
    """
    scenarios = list(all_results.keys())
    # Collect the union of all method names across scenarios
    all_methods = []
    for sc in scenarios:
        for m in all_results[sc].get("unified", {}).keys():
            if m not in all_methods:
                all_methods.append(m)

    ari_data = np.array([
        [all_results[sc]["unified"].get(m, {}).get("ARI", np.nan)
         for m in all_methods]
        for sc in scenarios
    ])

    nmi_data = np.array([
        [all_results[sc]["unified"].get(m, {}).get("NMI", np.nan)
         for m in all_methods]
        for sc in scenarios
    ])

    auroc_data = np.array([
        [all_results[sc]["unified"].get(m, {}).get("AUROC", np.nan)
         for m in all_methods]
        for sc in scenarios
    ])

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(all_methods) * 1.5),
                                             max(3, len(scenarios) * 0.7 + 1.5)))

    for ax, data, title, cmap, vmin, vmax in [
        (axes[0], ari_data,   "ARI (higher = better cluster recovery)", "RdYlGn", -0.2, 1.0),
        (axes[1], nmi_data,   "NMI (higher = better)",                  "RdYlGn",  0.0, 1.0),
        (axes[2], auroc_data, "AUROC (higher = better)",                "RdYlGn",  0.4, 1.0),
    ]:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(all_methods)))
        ax.set_xticklabels(all_methods, fontsize=8, rotation=35, ha="right")
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios, fontsize=8)
        ax.set_title(title, fontsize=9)
        for i in range(len(scenarios)):
            for j in range(len(all_methods)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if abs(v) > 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.suptitle("Unified benchmark summary: gene programme recovery\n"
                 "(all methods evaluated with same ARI/NMI metrics)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Summary table → {out_path}")




# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_scenario(
    sim_dir:          str,
    cluster_ids:      np.ndarray,
    n_clusters:       int,
    has_sentinel:     bool  = False,
    pool_runs:        bool  = True,
    k_knn:            int   = 5,
    n_meta:           int   = 30,
    n_meta_per_tp:    int   = 15,
    n_neighbors_gene: int   = 10,
    sinkhorn_reg:     float = 0.05,
    dense_gene_ot:    bool  = True,
    gcn_hidden_dim:   int   = 32,
    gcn_embed_dim:    int   = 16,
    gcn_epochs:       int   = 200,
    gcn_seed:         int   = 1,
    gcn_lr:           float = 1e-3,
    gcn_beta:         int   = 6,
    gcn_tau:          float = 0.01,
    verbose:          bool  = True,
) -> dict:
    """
    Run the full benchmark for one simulation directory.

    Returns nested dict with:
      - gset       : {method_name: {intra, inter, gap}}   (cosine similarity, 2-tp GSET)
      - gene_ot    : {mode: {ARI, NMI, intra_W, inter_W}} (W-based clustering)
      - unified    : {method_name: {ARI, NMI, intra_W, inter_W}}
                     for ALL methods using the same evaluate_W_clustering metrics:
                       · multi-tp ablations  (z_global, z_concat, delta_OT_concat,
                                              delta_KL_concat, full_OT_concat,
                                              full_KL_concat)
                       · gene_ot methods     (classic, temporal)
      - vectors    : {method_name: np.ndarray (G × d)}  — all gene vectors
      - W_matrices : {method_name: np.ndarray (G × G)}  — all distance matrices
      - cluster_ids, gene_names, adata
    """
    sim_dir  = Path(sim_dir)
    data_dir = sim_dir / "data"

    # ── Load data ──────────────────────────────────────────────────────────
    adata      = harissa_to_anndata(str(data_dir), pool_runs=pool_runs)
    G          = adata.n_vars
    gene_names = np.array(adata.var_names)
    tps        = np.sort(np.unique(adata.obs["time"].values))

    if verbose:
        print(f"  Data: {adata.n_obs} cells, {G} genes, timepoints {tps}")

    # Smart timepoint selection: skip early flat t=0 (HARISSA resting state)
    X_by_tp    = {t: adata.X[adata.obs["time"].values == t] for t in tps}
    mean_by_tp = {t: X_by_tp[t].mean() for t in tps}
    active_tps = np.array([t for t in tps if mean_by_tp[t] > 1.0])

    if len(active_tps) < 2:
        active_tps = tps   # fallback: use all timepoints

    t0_use = active_tps[0]
    tT_use = active_tps[-1]

    if verbose:
        print(f"  Active timepoints: {active_tps}  "
              f"(mean expr range: {mean_by_tp[t0_use]:.1f} → {mean_by_tp[tT_use]:.1f})")



    # -- GCN training ----------------------------------------------------------
    if verbose:
        print(f"\n  -- GCN training (shared weights, {len(active_tps)} timepoints) --")
    gcn = GCNEmbedder(
        hidden_dim = gcn_hidden_dim,
        embed_dim  = gcn_embed_dim,
        beta       = gcn_beta,
        tau        = gcn_tau,
        lr         = gcn_lr,
        n_epochs   = gcn_epochs,
        verbose    = verbose,
        seed       = gcn_seed,
    )
    gcn.fit(adata, active_tps)

    # -- Multi-timepoint ablations ------------------------------------------
    if verbose:
        print(f"\n  -- Multi-tp ablations ({len(active_tps)} timepoints, GCN) --")
    ablations = compute_multitimepoint_vectors(adata, active_tps, gcn, verbose=verbose)

    # δ_OT_0T: OT residual between first and last active timepoint only
    Z0_use = gcn.transform(adata, t0_use)
    ZT_use = gcn.transform(adata, tT_use)
    delta_OT_0T = ot_residual_vec(Z0_use, ZT_use)

    # Rename for display clarity
    ablation_display = {
        "(a) z_global":           ablations["z_global"],
        "(b) z_concat":           ablations["z_concat"],
        "(c) δ_OT_concat":        ablations["delta_OT_concat"],
        "(c) δ_OT_0T":            delta_OT_0T,
        "(c) δ_KL_concat":        ablations["delta_KL_concat"],
        "(d) full_OT_concat":     ablations["full_OT_concat"],
        "(d) full_KL_concat":     ablations["full_KL_concat"],
    }

    # MaxRank(δ_OT_0T, δ_OT_concat) — computed after W matrices are built
    _W_OT_concat = gset_vector_to_W(ablations["delta_OT_concat"])
    _W_OT_0T     = gset_vector_to_W(delta_OT_0T)
    _W_max_ot    = max_rank_distance(_W_OT_0T, _W_OT_concat)


    # ── gene_ot W matrices ─────────────────────────────────────────────────
    if verbose:
        print(f"\n  ── gene_ot ──")
    gene_ot_results = {}
    W_matrices      = {}

    for mode in ["classic", "temporal"]:
        dense_str = "dense" if dense_gene_ot else "sparse"
        if verbose:
            print(f"  [gene_ot/{mode}/{dense_str}] W matrix…")
        W = compute_gene_ot_W(
            adata, mode=mode, k_knn=k_knn, n_meta=n_meta,
            n_meta_per_tp=n_meta_per_tp, n_neighbors_gene=n_neighbors_gene,
            sinkhorn_reg=sinkhorn_reg, dense=dense_gene_ot, verbose=verbose)
        np.save(sim_dir / f"W_{mode}.npy", W)
        W_matrices[f"gene_ot_{mode}"] = W

        res = evaluate_W_clustering(W, cluster_ids, gene_names,
                                    n_clusters=n_clusters, exclude_sentinel=has_sentinel,
                                    dense=False)
        res["AUROC"] = auroc_from_W(W, cluster_ids, exclude_sentinel=has_sentinel)
        gene_ot_results[mode] = res
        if verbose:
            print(f"    ARI={res['ARI']:.3f}  NMI={res['NMI']:.3f}  "
                  f"AUROC={res['AUROC']:.3f}")

    # ── Unified evaluation: ALL methods with ARI/NMI ──────────────────────
    if verbose:
        print(f"\n  ── Unified evaluation ──")
    unified_results = {}

    # All vector-based methods → cosine-distance W
    all_named_vectors = ablation_display
    for name, V in all_named_vectors.items():
        W_cos = gset_vector_to_W(V)
        W_matrices[name] = W_cos
        res = evaluate_W_clustering(W_cos, cluster_ids, gene_names,
                                    n_clusters=n_clusters, exclude_sentinel=has_sentinel,
                                    dense=True)
        res["AUROC"] = auroc_from_W(W_cos, cluster_ids, exclude_sentinel=has_sentinel)
        unified_results[name] = res
        if verbose:
            print(f"    {name:26s}: ARI={res['ARI']:.3f}  NMI={res['NMI']:.3f}  "
                  f"AUROC={res['AUROC']:.3f}")

    # MaxRank(δ_OT_0T, δ_OT_concat)
    res_max = evaluate_W_clustering(_W_max_ot, cluster_ids, gene_names,
                                    n_clusters=n_clusters, exclude_sentinel=has_sentinel,
                                    dense=True)
    res_max["AUROC"] = auroc_from_W(_W_max_ot, cluster_ids, exclude_sentinel=has_sentinel)
    unified_results["(c) MaxRank(OT_0T,concat)"] = res_max
    W_matrices["(c) MaxRank(OT_0T,concat)"]      = _W_max_ot
    if verbose:
        print(f"    {'(c) MaxRank(OT_0T,concat)':26s}: ARI={res_max['ARI']:.3f}  AUROC={res_max['AUROC']:.3f}")

    # MaxRank(δ_OT_concat, δ_KL_concat)
    _W_KL        = gset_vector_to_W(ablations["delta_KL_concat"])
    _W_max_otc_kl = max_rank_distance(_W_OT_concat, _W_KL)
    res_max_otc_kl = evaluate_W_clustering(_W_max_otc_kl, cluster_ids, gene_names,
                                           n_clusters=n_clusters, exclude_sentinel=has_sentinel,
                                           dense=True)
    res_max_otc_kl["AUROC"] = auroc_from_W(_W_max_otc_kl, cluster_ids, exclude_sentinel=has_sentinel)
    unified_results["(c) MaxRank(OT,KL)"] = res_max_otc_kl
    W_matrices["(c) MaxRank(OT,KL)"]      = _W_max_otc_kl
    if verbose:
        print(f"    {'(c) MaxRank(OT,KL)':26s}: ARI={res_max_otc_kl['ARI']:.3f}  AUROC={res_max_otc_kl['AUROC']:.3f}")

    # MaxRank(δ_OT_0T, δ_KL_concat)
    _W_KL        = gset_vector_to_W(ablations["delta_KL_concat"])
    _W_max_ot_kl = max_rank_distance(_W_OT_0T, _W_KL)
    res_max_kl   = evaluate_W_clustering(_W_max_ot_kl, cluster_ids, gene_names,
                                         n_clusters=n_clusters, exclude_sentinel=has_sentinel,
                                         dense=True)
    res_max_kl["AUROC"] = auroc_from_W(_W_max_ot_kl, cluster_ids, exclude_sentinel=has_sentinel)
    unified_results["(c) MaxRank(OT_0T,KL)"] = res_max_kl
    W_matrices["(c) MaxRank(OT_0T,KL)"]      = _W_max_ot_kl
    if verbose:
        print(f"    {'(c) MaxRank(OT_0T,KL)':26s}: ARI={res_max_kl['ARI']:.3f}  AUROC={res_max_kl['AUROC']:.3f}")

    # gene_ot: reuse already-computed results
    for mode, res in gene_ot_results.items():
        unified_results[f"gene_ot_{mode}"] = res

    return {
        "gene_ot":         gene_ot_results,
        "unified":         unified_results,
        "vectors":         all_named_vectors,
        "W_matrices":      W_matrices,
        "cluster_ids":     cluster_ids,
        "gene_names":      gene_names,
        "adata":           adata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    sim_base_dir:     str,
    scenarios:        list,
    pool_runs:        bool  = True,
    k_knn:            int   = 5,
    n_meta:           int   = 500,
    n_meta_per_tp:    int   = 50,
    n_neighbors_gene: int   = 10,
    sinkhorn_reg:     float = 0.05,
    dense_gene_ot:    bool  = True,
    gcn_hidden_dim:   int   = 32,
    gcn_embed_dim:    int   = 16,
    gcn_epochs:       int   = 200,
    gcn_seed:         int   = 1,
    gcn_lr:           float = 1e-3,
    gcn_beta:         int   = 6,
    gcn_tau:          float = 0.01,
    verbose:          bool  = True,
    out_dir:          str   = None,
    fig_name:         str   = "gene_programme_recovery.png",
):
    sim_base_dir = Path(sim_base_dir)

    print("=" * 62)
    print("GSET Benchmark: gene programme recovery")
    print(f"Scenarios: {scenarios}")
    print(f"gene_ot mode: {'DENSE (all pairs)' if dense_gene_ot else 'SPARSE (L1 kNN)'}")
    print(f"Embeddings:   GCN (shared weights, d={gcn_embed_dim})")
    print("=" * 62)

    all_results = {}

    for scenario in scenarios:
        scen_dir = sim_base_dir / SCENARIO_DIRS.get(scenario, scenario)
        if not scen_dir.exists():
            print(f"\n[{scenario}] Directory not found: {scen_dir} — skipping")
            continue

        print(f"\n{'─'*55}")
        print(f"  Scenario: {scenario.upper()}")
        print(f"{'─'*55}")

        # Load ground truth cluster ids
        cluster_ids = np.load(scen_dir / "cluster_ids.npy")
        meta = json.load(open(scen_dir / "metadata.json"))
        n_raw_clusters = meta["n_clusters"]

        # Load special_ids if present
        sp_path = scen_dir / "special_ids.json"
        special_ids = {}
        if sp_path.exists():
            with open(sp_path) as f:
                special_ids = json.load(f)

        print(f"  n_clusters={n_raw_clusters}, "
              f"special genes: {special_ids if special_ids else 'none'}")

        try:
            res = benchmark_scenario(
                sim_dir=str(scen_dir),
                cluster_ids=cluster_ids,
                n_clusters=n_raw_clusters,
                has_sentinel=bool(special_ids),
                pool_runs=pool_runs, k_knn=k_knn, n_meta=n_meta,
                n_meta_per_tp=n_meta_per_tp,
                n_neighbors_gene=n_neighbors_gene,
                sinkhorn_reg=sinkhorn_reg,
                dense_gene_ot=dense_gene_ot,
                gcn_hidden_dim=gcn_hidden_dim,
                gcn_embed_dim=gcn_embed_dim,
                gcn_epochs=gcn_epochs,
                gcn_seed=gcn_seed,
                gcn_lr=gcn_lr,
                gcn_beta=gcn_beta,
                gcn_tau=gcn_tau,
                verbose=verbose,
            )
            all_results[scenario] = res


        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    # Global summary
    if all_results:
        figures_dir = Path(out_dir) if out_dir else (_GSET_ROOT / "figures" / "block3_gene_programme_recovery")
        figures_dir.mkdir(parents=True, exist_ok=True)
        summary_path = figures_dir / fig_name
        plot_summary_table(all_results, str(summary_path))
        print(f"  Summary table → {summary_path}")

    figures_dir = _GSET_ROOT / "figures" / "block3_gene_programme_recovery"
    print(f"\n✓ Benchmark complete — {figures_dir}/")
    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="GSET benchmark pipeline")
    p.add_argument("--sim_dir",          type=str,   default="simulations",
                   help="Base dir containing one subdir per scenario")
    p.add_argument("--scenarios",        type=str,   nargs="+", default=["all"],
                   help="Scenarios to benchmark (or 'all')")
    p.add_argument("--pool_runs",        type=int,   default=1)
    p.add_argument("--k_knn",            type=int,   default=5)
    p.add_argument("--n_meta",           type=int,   default=30)
    p.add_argument("--n_meta_per_tp",    type=int,   default=15)
    p.add_argument("--n_neighbors_gene", type=int,   default=5)
    p.add_argument("--sinkhorn_reg",     type=float, default=0)
    p.add_argument("--sparse_gene_ot",   action="store_true",
                   help="Use sparse L1 + kNN approximation for gene_ot distances "
                        "(default: dense, compute all gene pairs)")
    # GCN options
    p.add_argument("--gcn_hidden_dim", type=int,   default=32,
                   help="GCN first-layer hidden dimension (default 32)")
    p.add_argument("--gcn_embed_dim",  type=int,   default=16,
                   help="GCN output embedding dimension d (default 16)")
    p.add_argument("--gcn_epochs",     type=int,   default=200,
                   help="GCN training epochs (default 200)")
    p.add_argument("--gcn_seed",       type=int,   default=1,
                   help="GCN random seed (default 1)")
    p.add_argument("--gcn_lr",         type=float, default=1e-3,
                   help="GCN Adam learning rate (default 1e-3)")
    p.add_argument("--gcn_beta",       type=int,   default=6,
                   help="WGCNA soft-threshold power beta (default 6)")
    p.add_argument("--gcn_tau",        type=float, default=0.01,
                   help="WGCNA edge weight cutoff tau (default 0.01)")
    p.add_argument("--out_dir",        type=str,   default=None,
                   help="Output directory for figures and JSON (default: <gset_root>/figures)")
    p.add_argument("--fig_name",       type=str,   default="gene_programme_recovery.png",
                   help="Filename for the summary heatmap (default: gene_programme_recovery.png)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenarios = (ALL_SCENARIOS if args.scenarios == ["all"] else args.scenarios)
    run_benchmark(
        sim_base_dir=args.sim_dir,
        scenarios=scenarios,
        pool_runs=bool(args.pool_runs),
        k_knn=args.k_knn,
        n_meta=args.n_meta,
        n_meta_per_tp=args.n_meta_per_tp,
        n_neighbors_gene=args.n_neighbors_gene,
        sinkhorn_reg=args.sinkhorn_reg,
        dense_gene_ot=not args.sparse_gene_ot,
        gcn_hidden_dim=args.gcn_hidden_dim,
        gcn_embed_dim=args.gcn_embed_dim,
        gcn_epochs=args.gcn_epochs,
        gcn_seed=args.gcn_seed,
        gcn_lr=args.gcn_lr,
        gcn_beta=args.gcn_beta,
        gcn_tau=args.gcn_tau,
        out_dir=args.out_dir,
        fig_name=args.fig_name,
    )