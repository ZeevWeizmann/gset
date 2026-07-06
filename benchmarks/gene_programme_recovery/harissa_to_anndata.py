"""
harissa_to_anndata.py
---------------------
Convert HARISSA simulation output (one or multiple data_*.txt files) to an
AnnData object compatible with gene_ot_distances.py.

Also provides the GSET-style evaluation function: given gene embeddings at
two timepoints, compute OT residual, KL vector, and naive displacement, then
measure intra- vs inter-cluster cosine similarity.
"""

import numpy as np
import anndata as ad
from pathlib import Path
from typing import List, Optional, Union
from scipy.special import softmax


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_harissa_run(data_path: str) -> tuple:
    """
    Load one HARISSA data file.
    Returns (X, times, gene_names) where X shape (C, G).
    """
    raw   = np.loadtxt(data_path, delimiter="\t").T.astype(float)
    times = raw[1:, 0]
    X     = raw[1:, 2:]   # mRNA counts, genes 1..G (skip time + stim)
    G     = X.shape[1]
    gene_names = np.array([f"G{i+1}" for i in range(G)])
    return X, times, gene_names


def harissa_to_anndata(
    data_dir: str,
    pattern:  str = "data_*.txt",
    pool_runs: bool = True,
    seed:     int  = 0,
) -> ad.AnnData:
    """
    Load all HARISSA runs matching pattern and build a single AnnData.
    
    Parameters
    ----------
    data_dir  : directory containing data_*.txt files
    pattern   : glob pattern for data files
    pool_runs : if True, concatenate all runs; if False, use only the first run
    
    Returns
    -------
    AnnData with:
        X         : raw mRNA counts (C, G)
        obs['time']: timepoint per cell (float)
        obs['run'] : run index per cell
        var_names  : gene names G1..GG
    """
    data_dir = Path(data_dir)
    files    = sorted(data_dir.glob(pattern))
    assert files, f"No files matching {pattern} in {data_dir}"

    if not pool_runs:
        files = files[:1]

    all_X, all_times, all_runs = [], [], []
    gene_names = None

    for r, fp in enumerate(files):
        X, times, gn = load_harissa_run(str(fp))
        all_X.append(X)
        all_times.append(times)
        all_runs.extend([r] * len(times))
        if gene_names is None:
            gene_names = gn

    X_all     = np.vstack(all_X)
    times_all = np.concatenate(all_times)

    adata           = ad.AnnData(X=X_all.astype(np.float32))
    adata.var_names = gene_names
    adata.obs["time"] = times_all.astype(float)
    adata.obs["run"]  = np.array(all_runs).astype(int)
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# GSET-style gene embeddings from expression data
# ─────────────────────────────────────────────────────────────────────────────

def compute_gene_embeddings(
    adata: ad.AnnData,
    t0:   float = None,
    tT:   float = None,
    method: str = "correlation",
) -> tuple:
    """
    Compute per-gene embedding vectors at t=t0 and t=tT.
    
    method='correlation': gene embedding = row of the Pearson correlation matrix
        (same as WGCNA node features in the paper — no GAT needed)
    method='mean': gene embedding = mean expression across cells at that timepoint
    
    Returns
    -------
    E0, ET : (G, D) float arrays — gene embeddings at t0 and tT
    """
    times = adata.obs["time"].values
    tps   = np.sort(np.unique(times))

    if t0 is None:
        t0 = tps[0]
    if tT is None:
        tT = tps[-1]

    X0 = adata.X[times == t0].astype(float)
    XT = adata.X[times == tT].astype(float)

    if method == "correlation":
        # Pearson correlation profile (G, G) — each gene's row is its embedding
        def safe_corr(X):
            X_log   = np.log1p(X)
            mu      = X_log.mean(axis=0, keepdims=True)
            std     = X_log.std(axis=0, keepdims=True) + 1e-8
            X_norm  = (X_log - mu) / std
            corr    = X_norm.T @ X_norm / X_log.shape[0]
            return corr   # (G, G)
        E0 = safe_corr(X0)
        ET = safe_corr(XT)

    elif method == "mean":
        E0 = np.log1p(X0).mean(axis=0, keepdims=True).T  # (G, 1)
        ET = np.log1p(XT).mean(axis=0, keepdims=True).T
        # Augment with variance
        V0 = np.log1p(X0).var(axis=0, keepdims=True).T
        VT = np.log1p(XT).var(axis=0, keepdims=True).T
        E0 = np.hstack([E0, V0])
        ET = np.hstack([ET, VT])

    return E0, ET


# ─────────────────────────────────────────────────────────────────────────────
# GSET scores
# ─────────────────────────────────────────────────────────────────────────────

def compute_ot_residual(E0: np.ndarray, ET: np.ndarray) -> np.ndarray:
    """
    OT displacement residual δ_OT_i = E_T(i) - E_T_opt(i).
    E_T_opt(i) = barycentric projection of E0(i) under the OT plan.
    Returns delta_OT of shape (G, D).
    """
    import ot as pot
    G = E0.shape[0]
    a = np.ones(G) / G
    b = np.ones(G) / G

    # Cost matrix: squared Euclidean distances between gene embeddings
    M = np.sum((E0[:, None, :] - ET[None, :, :]) ** 2, axis=-1)
    M = M / (M.max() + 1e-10)

    # OT plan
    gamma = pot.emd(a, b, M, numItermax=500_000)   # (G, G)

    # Barycentric projection: predicted position of each gene
    ET_opt = G * (gamma @ ET)   # (G, D)

    delta_OT = ET - ET_opt
    return delta_OT


def compute_kl_vector(E0: np.ndarray, ET: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    KL vector δ_KL_i(j) = p0_i(j) * log(p0_i(j) / pT_i(j)).
    p^t_i(j) = softmax of -||E_t(i) - E_t(j)||² / σ²  (all pairs, dense).

    Returns vec_delta_KL of shape (G, G).
    """
    G = E0.shape[0]

    def neighbourhood_probs(E, sig):
        D2 = np.sum((E[:, None, :] - E[None, :, :]) ** 2, axis=-1)   # (G, G)
        np.fill_diagonal(D2, np.inf)
        logits = -D2 / (sig ** 2)
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        np.fill_diagonal(P, 0.0)
        P = P / (P.sum(axis=1, keepdims=True) + 1e-10)
        return P

    if sigma is None:
        D2_0 = np.sum((E0[:, None, :] - E0[None, :, :]) ** 2, axis=-1)
        sigma = np.sqrt(np.median(D2_0[D2_0 > 0]))
        sigma = max(sigma, 1e-6)

    P0 = neighbourhood_probs(E0, sigma)
    PT = neighbourhood_probs(ET, sigma)

    eps = 1e-10
    vec_KL = P0 * np.log((P0 + eps) / (PT + eps))
    np.fill_diagonal(vec_KL, 0.0)
    return vec_KL


def compute_kl_vector_graph(E0: np.ndarray, ET: np.ndarray,
                            beta: float = 6.0, thresh: float = 0.01) -> np.ndarray:
    """
    Graph-based KL vector: neighbourhood defined by WGCNA graph edges only.
    p^t_i(j) = A^t_ij / sum_k A^t_ik  where A = |corr|^beta thresholded.

    Returns vec_delta_KL of shape (G, G).
    """
    G = E0.shape[0]

    def wgcna_graph_probs(E):
        Xl = np.log1p(E)
        mu = Xl.mean(1, keepdims=True)
        std = Xl.std(1, keepdims=True) + 1e-8
        Z = (Xl - mu) / std
        C = Z @ Z.T / Xl.shape[1]
        np.fill_diagonal(C, 0.0)
        A = np.clip(np.abs(C) ** beta, 0.0, 1.0)
        A[A < thresh] = 0.0
        np.fill_diagonal(A, 0.0)
        row_sum = A.sum(1, keepdims=True)
        no_edge = (row_sum < 1e-10).flatten()
        P = A / (row_sum + 1e-10)
        if no_edge.any():
            uniform = np.ones((no_edge.sum(), G)) / G
            np.fill_diagonal(uniform, 0.0)
            P[no_edge] = uniform
        return P

    P0 = wgcna_graph_probs(E0)
    PT = wgcna_graph_probs(ET)

    eps = 1e-10
    vec_KL = P0 * np.log((P0 + eps) / (PT + eps))
    np.fill_diagonal(vec_KL, 0.0)
    return vec_KL


def compute_naive_displacement(E0: np.ndarray, ET: np.ndarray) -> np.ndarray:
    """Naive displacement: E_T - E_0 (no OT correction)."""
    return ET - E0


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity evaluation (as in paper Figures 4,6)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_matrix(V: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity for rows of V (G, D).
    Returns (G, G) matrix.
    """
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-10
    V_norm = V / norms
    return V_norm @ V_norm.T


def intra_inter_cosine(
    V:          np.ndarray,
    cluster_ids: np.ndarray,
    exclude_sentinel: bool = True,
) -> tuple:
    """
    Compute mean intra-cluster and inter-cluster cosine similarity.
    
    Parameters
    ----------
    V            : (G, D) gene vectors
    cluster_ids  : (G,) int cluster labels
    exclude_sentinel : if True, exclude genes with cluster_id == max(cluster_ids)
                       (sentinel for special genes with ambiguous cluster membership)
    
    Returns
    -------
    intra_mean, inter_mean : float
    """
    K = cluster_ids.max()
    if exclude_sentinel:
        mask = cluster_ids < K
        V    = V[mask]
        ids  = cluster_ids[mask]
    else:
        ids  = cluster_ids

    C = cosine_similarity_matrix(V)
    G = len(ids)
    intra, inter = [], []

    for i in range(G):
        for j in range(i + 1, G):
            if ids[i] == ids[j]:
                intra.append(C[i, j])
            else:
                inter.append(C[i, j])

    return (np.mean(intra) if intra else 0.0,
            np.mean(inter) if inter else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Gene-space evaluation from W matrix (gene_ot_distances output)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_W_clustering(
    W:           np.ndarray,
    cluster_ids: np.ndarray,
    gene_names:  np.ndarray,
    n_clusters:  int,
    exclude_sentinel: bool = True,
    dense:       bool = False,
) -> dict:
    """
    Given a W (gene×gene distance) matrix from gene_ot_distances, evaluate
    how well agglomerative clustering recovers the ground-truth clusters.

    Parameters
    ----------
    dense : bool, default False
        If True, skip the shortest-path completion step and use W directly for
        MDS. Use this for dense cosine-distance matrices where zero entries mean
        "identical" (not "no edge") — shortest-path on a sparse matrix would
        drop zero-distance pairs and corrupt the geodesic.

    Returns dict with: ARI, NMI, intra_W, inter_W, labels
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix

    K = cluster_ids.max()
    if exclude_sentinel:
        mask = cluster_ids < K
    else:
        mask = np.ones(len(cluster_ids), dtype=bool)

    W_sub   = W[np.ix_(mask, mask)]
    ids_sub = cluster_ids[mask]
    gn_sub  = gene_names[mask]
    G_sub   = mask.sum()

    if dense:
        # Dense matrix: zero entries mean "identical genes", not "no edge".
        # Shortest-path via csr_matrix would drop zero entries and corrupt distances.
        W_geo = W_sub.copy().astype(float)
        W_geo = (W_geo + W_geo.T) / 2  # symmetrize (fixes float noise from rank ops)
        np.fill_diagonal(W_geo, 0.0)
    else:
        # Sparse matrix: complete missing distances via shortest path
        graph   = csr_matrix(W_sub)
        W_geo   = shortest_path(graph, method="D", directed=False)
        fm      = W_geo[np.isfinite(W_geo)].max() if np.isfinite(W_geo).any() else 1.0
        W_geo[~np.isfinite(W_geo)] = fm * 1.5
        np.fill_diagonal(W_geo, 0.0)

    # Clustering (MDS embedding first, then ward on Euclidean)
    from sklearn.manifold import MDS
    n_cl  = min(n_clusters, G_sub - 1)
    n_mds = min(10, G_sub - 1)
    mds   = MDS(n_components=n_mds, dissimilarity="precomputed",
                random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(W_geo)
    cl     = AgglomerativeClustering(n_clusters=n_cl, linkage="ward")
    labels = cl.fit_predict(coords)

    ari = adjusted_rand_score(ids_sub, labels)
    nmi = normalized_mutual_info_score(ids_sub, labels)

    # Intra vs inter W distance
    intra_w, inter_w = [], []
    for i in range(G_sub):
        for j in range(i + 1, G_sub):
            if ids_sub[i] == ids_sub[j]:
                intra_w.append(W_geo[i, j])
            else:
                inter_w.append(W_geo[i, j])

    return {
        "ARI":       ari,
        "NMI":       nmi,
        "intra_W":   np.mean(intra_w) if intra_w else 0.0,
        "inter_W":   np.mean(inter_w) if inter_w else 0.0,
        "labels":    labels,
        "gene_names": gn_sub,
        "n_genes":   G_sub,
    }
