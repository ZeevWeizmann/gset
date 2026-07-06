"""
directed_graph.py
-----------------
Build a directed gene-gene graph from lagged KL vectors.

Edge i→j: cosine(kl_vec_i[t], kl_vec_j[t+1]) > threshold,
averaged over all consecutive transition pairs (t, t+1).

Uses core.scoring.kl_vec() which takes GCN/correlation embeddings
at two timepoints and returns (n_genes, n_genes) KL vectors.
"""

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from typing import List


def lagged_cosine_matrix(kl_vecs: List[np.ndarray]) -> np.ndarray:
    """
    Compute the directed lagged cosine similarity matrix.

    C[i, j] = mean_t  cosine(kl_vec_i[t], kl_vec_j[t+1])

    Parameters
    ----------
    kl_vecs : list of (n_genes, n_genes) arrays, one per transition t→t+1

    Returns
    -------
    C : (n_genes, n_genes) — directed similarity; C[i,j] ≠ C[j,i]
    """
    if len(kl_vecs) < 2:
        raise ValueError("Need at least 2 transitions for lagged cosine.")

    norm_vecs = [normalize(v, norm="l2", axis=1) for v in kl_vecs]
    n_genes = kl_vecs[0].shape[0]
    n_lags  = len(kl_vecs) - 1

    C = np.zeros((n_genes, n_genes), dtype=np.float64)
    for t in range(n_lags):
        # C_t[i,j] = dot(norm_vecs[t][i,:], norm_vecs[t+1][j,:])
        C += norm_vecs[t] @ norm_vecs[t + 1].T
    C /= n_lags
    return C


def build_directed_graph(kl_vecs: List[np.ndarray],
                         threshold: float = 0.35) -> tuple:
    """
    Build a directed NetworkX DiGraph from lagged KL vectors.

    Parameters
    ----------
    kl_vecs   : list of (n_genes, n_genes) KL vector matrices
    threshold : minimum lagged cosine for an edge to be added

    Returns
    -------
    G : nx.DiGraph
    C : (n_genes, n_genes) lagged cosine matrix
    """
    C = lagged_cosine_matrix(kl_vecs)
    n_genes = C.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(n_genes))

    ii, jj = np.where(C > threshold)
    for i, j in zip(ii, jj):
        if i != j:
            G.add_edge(int(i), int(j), weight=float(C[i, j]))

    return G, C


def build_directed_graph_ot(ot_vecs: List[np.ndarray],
                            threshold: float = 0.35) -> tuple:
    """
    Build directed graph from lagged OT residual vectors.

    C_OT[i,j] = mean_t cosine(r_i[t], r_j[t+1])
    where r_i[t] = OT residual vector of gene i at transition t (absolute displacement).

    Catches driven targets that move as a block — the KL blind spot.
    """
    if len(ot_vecs) < 2:
        raise ValueError("Need at least 2 transitions for lagged OT.")

    # row-normalise each residual matrix
    def row_norm(R):
        norms = np.linalg.norm(R, axis=1, keepdims=True)
        return R / (norms + 1e-10)

    norm_vecs = [row_norm(R) for R in ot_vecs]
    n_genes = ot_vecs[0].shape[0]
    n_lags  = len(ot_vecs) - 1
    C = np.zeros((n_genes, n_genes))
    for t in range(n_lags):
        # C[i,j] += cosine(r_i[t], r_j[t+1])
        C += norm_vecs[t] @ norm_vecs[t + 1].T
    C /= n_lags

    G = nx.DiGraph()
    G.add_nodes_from(range(n_genes))
    ii, jj = np.where(C > threshold)
    for i, j in zip(ii, jj):
        if i != j:
            G.add_edge(int(i), int(j), weight=float(C[i, j]))
    return G, C


def symmetric_cosine_graph(kl_vecs: List[np.ndarray]) -> np.ndarray:
    """
    Baseline: symmetric cosine similarity on concatenated KL vectors.
    This is the current GSET approach (delta_KL_concat).

    Returns (n_genes, n_genes) symmetric similarity matrix.
    """
    kl_cat  = np.concatenate(kl_vecs, axis=1)          # (n_genes, n_genes*n_trans)
    kl_norm = normalize(kl_cat, norm="l2", axis=1)
    return kl_norm @ kl_norm.T
