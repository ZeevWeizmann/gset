"""
expand.py
---------
Program retrieval via directed neighbourhood expansion vs cosine k-NN.

directed_program(G, seed)  → reachable set in DiGraph  (causal cascade)
cosine_program(C, seed, k) → top-k cosine neighbours   (current GSET)
"""

import numpy as np
import networkx as nx
from typing import Set


def directed_program(G: nx.DiGraph, seed: int, max_depth: int = 4) -> Set[int]:
    """
    Directed closure: all genes reachable from seed via directed edges.
    = causal cascade triggered by the seed gene.
    """
    return set(nx.single_source_shortest_path_length(
        G, seed, cutoff=max_depth).keys())


def cosine_program(C_sym: np.ndarray, seed: int, k: int = 20) -> Set[int]:
    """
    Top-k cosine neighbours of seed gene (current GSET approach).
    C_sym : (n_genes, n_genes) symmetric cosine similarity matrix
    """
    s = C_sym[seed].copy()
    s[seed] = -1.0
    return set(np.argsort(s)[::-1][:k])


def cluster_breakdown(program: Set[int], cluster_ids: np.ndarray) -> dict:
    """Fraction of program genes in each cluster."""
    n = max(len(program), 1)
    return {k: sum(1 for g in program if cluster_ids[g] == k) / n
            for k in np.unique(cluster_ids)}


def edge_cluster_matrix(G: nx.DiGraph, cluster_ids: np.ndarray) -> np.ndarray:
    """
    Count directed edges between each pair of clusters.
    Returns (n_clusters, n_clusters) matrix.
    Edge [k, l] = number of edges from cluster-k gene to cluster-l gene.
    """
    n_cls = len(np.unique(cluster_ids))
    M = np.zeros((n_cls, n_cls), dtype=int)
    for i, j in G.edges():
        M[cluster_ids[i], cluster_ids[j]] += 1
    return M


def inter_cluster_edge_fraction(G: nx.DiGraph, cluster_ids: np.ndarray) -> float:
    """Fraction of edges that cross cluster boundaries."""
    total = G.number_of_edges()
    if total == 0:
        return 0.0
    inter = sum(1 for i, j in G.edges() if cluster_ids[i] != cluster_ids[j])
    return inter / total


def compare_programs(G_dir: nx.DiGraph, C_sym: np.ndarray,
                     cluster_ids: np.ndarray,
                     seeds: np.ndarray,
                     max_depth: int = 4) -> list:
    """
    For each seed gene, compute directed and cosine programs and compare
    their cluster composition.

    Returns list of dicts with keys:
        seed, dir_size, cos_size, dir_C{k}, cos_C{k} for each cluster k
    """
    results = []
    for seed in seeds:
        if G_dir.out_degree(seed) == 0:
            continue
        prog_d = directed_program(G_dir, seed, max_depth=max_depth)
        prog_c = cosine_program(C_sym, seed, k=max(len(prog_d), 5))

        row = {"seed": int(seed),
               "dir_size": len(prog_d),
               "cos_size": len(prog_c)}
        bd = cluster_breakdown(prog_d, cluster_ids)
        bc = cluster_breakdown(prog_c, cluster_ids)
        for k in np.unique(cluster_ids):
            row[f"dir_C{k}"] = bd.get(k, 0.0)
            row[f"cos_C{k}"] = bc.get(k, 0.0)
        results.append(row)
    return results
