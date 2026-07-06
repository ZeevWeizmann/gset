"""
subnetwork.py
-------------
Directed subnetwork expansion from a query gene.

Builds a directed gene-gene graph using lagged cosine similarity of KL
neighbourhood vectors (Elias Ventre, June 2026). Edge i→j means:
  "the neighbourhood of gene i changes at [t, t+1] in a way that
   predicts how gene j's neighbourhood changes at [t+1, t+2]"

The program of a seed gene = reachable set (directed closure) in this graph,
anchored to the stimulus (gene 0 in HARISSA = the dynamic root).
"""

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from pathlib import Path
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Expression preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def log1p_norm(X: np.ndarray) -> np.ndarray:
    """log1p + library-size normalisation + z-score per gene."""
    Xn = np.log1p(X / (X.sum(1, keepdims=True) + 1e-8) * 1e4)
    mu, std = Xn.mean(0), Xn.std(0)
    Xs = (Xn - mu) / (std + 1e-8)
    Xs[:, std < 1e-6] = 0.0
    return Xs


def neighbourhood_dist(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """
    Soft neighbourhood distribution P[i,j] ∈ (0,1) from expression matrix.
    Uses WGCNA-style soft thresholding of Pearson correlation.
    Returns (n_genes, n_genes) row-normalised matrix.
    """
    Xs = log1p_norm(X)
    C  = (Xs.T @ Xs) / Xs.shape[0]   # Pearson correlation
    np.fill_diagonal(C, 0)
    A  = np.clip(np.abs(C) ** 4, 0, 1)
    if sigma is None:
        vals  = A[A > 0]
        sigma = float(np.sqrt(np.median(vals) + 1e-10)) if len(vals) else 0.1
    P = np.exp(-((1 - A) ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(P, 0)
    P /= (P.sum(1, keepdims=True) + 1e-10)
    return P


def kl_vec(P0: np.ndarray, P1: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Symmetrised KL divergence vector. Row i = δ_KL_i.
    Returns (n_genes, n_genes): how gene i's neighbourhood distribution changed.
    """
    return P0 * np.log((P0 + eps) / (P1 + eps)) + P1 * np.log((P1 + eps) / (P0 + eps))


# ─────────────────────────────────────────────────────────────────────────────
# KL vector computation from HARISSA runs
# ─────────────────────────────────────────────────────────────────────────────

def load_runs(data_dir: Path) -> Tuple[List[dict], List[int]]:
    """Load HARISSA data files. Returns list of {tp: X_cells×genes} and sorted tp list."""
    data_dir = Path(data_dir)
    runs, tp_set = [], set()
    for fp in sorted(data_dir.glob("data_*.txt")):
        raw   = np.loadtxt(fp, delimiter="\t")
        times = raw[0, 1:].astype(int)
        X     = raw[2:, 1:].T    # (n_cells, n_genes) — skips stimulus row
        run   = {int(tp): X[times == tp] for tp in np.unique(times)}
        runs.append(run)
        tp_set.update(run.keys())
    return runs, sorted(tp_set)


def compute_lagged_kl_vecs(
    runs: List[dict],
    tp_list: List[int],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Compute KL neighbourhood vectors averaged across runs.

    Returns
    -------
    kl_vecs : list of (n_genes, n_genes) arrays, one per transition
    delta_kl : (n_genes, n_trans) scalar summary (row sums)
    """
    n_trans = len(tp_list) - 1
    acc = [None] * n_trans

    for run in runs:
        for t in range(n_trans):
            P0 = neighbourhood_dist(run[tp_list[t]])
            P1 = neighbourhood_dist(run[tp_list[t + 1]])
            v  = kl_vec(P0, P1)
            acc[t] = v if acc[t] is None else acc[t] + v

    kl_vecs  = [a / len(runs) for a in acc]
    delta_kl = np.stack([v.sum(1) for v in kl_vecs], axis=1)
    return kl_vecs, delta_kl


# ─────────────────────────────────────────────────────────────────────────────
# Directed graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_directed_kl_graph(
    kl_vecs: List[np.ndarray],
    threshold: float = 0.35,
) -> Tuple[nx.DiGraph, np.ndarray]:
    """
    Build directed gene graph via lagged cosine of KL neighbourhood vectors.

    Edge i→j: cos(KL_i[t], KL_j[t+1]) > threshold, averaged over t.

    Parameters
    ----------
    kl_vecs   : list of (G, G) KL vectors per transition
    threshold : cosine threshold for adding a directed edge

    Returns
    -------
    G : directed NetworkX graph
    C : (G, G) average lagged cosine matrix
    """
    n_lags = len(kl_vecs) - 1
    if n_lags <= 0:
        raise ValueError("Need at least 2 KL vectors (3 timepoints) for lagged cosine.")

    norm_v = [normalize(v, norm='l2') for v in kl_vecs]
    C = np.zeros((kl_vecs[0].shape[0],) * 2)
    for t in range(n_lags):
        C += norm_v[t] @ norm_v[t + 1].T
    C /= n_lags

    G = nx.DiGraph()
    G.add_nodes_from(range(C.shape[0]))
    rows, cols = np.where(C > threshold)
    for i, j in zip(rows, cols):
        if i != j:
            G.add_edge(int(i), int(j), weight=float(C[i, j]))
    return G, C


# ─────────────────────────────────────────────────────────────────────────────
# Subnetwork expansion
# ─────────────────────────────────────────────────────────────────────────────

def expand_from_seed(
    G: nx.DiGraph,
    seed: int,
    budget: int = 100,
    max_depth: int = 6,
    cluster_ids: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Expand subnetwork from seed gene. Fully intrinsic — no ground truth needed.

    Priority order (fills budget):
      1. Source nodes (in_degree=0) — intrinsic stim-direct candidates
      2. Inter-cluster bridge genes (C_i→C_{i+1} edges) — programme connectors
      3. Reverse BFS from seed (upstream regulators), by distance
      4. Forward BFS from seed (downstream targets), by distance

    Parameters
    ----------
    G           : directed gene graph
    seed        : query gene index
    budget      : maximum number of genes to return
    max_depth   : BFS depth limit
    cluster_ids : (n_genes,) cluster assignments — enables bridge scoring
    """
    n_genes = G.number_of_nodes()
    selected = set([seed])

    # ── Priority 1: source nodes (in_degree=0) ──────────────────────────────
    sources = {g for g in G.nodes() if G.in_degree(g) == 0 and g < n_genes}
    selected.update(sources)

    # ── Priority 2: inter-cluster bridge genes ───────────────────────────────
    if cluster_ids is not None and len(selected) < budget:
        K = int(cluster_ids.max()) + 1

        # Infer cluster ordering by dominant inter-cluster flow
        cluster_flow = np.zeros((K, K))
        for u, v, d in G.edges(data=True):
            if u < n_genes and v < n_genes:
                ci, cj = int(cluster_ids[u]), int(cluster_ids[v])
                if ci != cj:
                    cluster_flow[ci, cj] += d.get("weight", 1.0)

        visited, order = set(), []
        cur = int(np.argmin(cluster_flow.sum(0)))
        while cur not in visited and len(order) < K:
            visited.add(cur); order.append(cur)
            nxt = cluster_flow[cur].copy(); nxt[list(visited)] = 0
            cur = int(np.argmax(nxt)) if nxt.max() > 0 else cur

        # Score genes by C_i→C_{i+1} bridge weight
        bridge_score = np.zeros(n_genes)
        for step in range(len(order) - 1):
            ci, cj = order[step], order[step + 1]
            for u, v, d in G.edges(data=True):
                if u < n_genes and v < n_genes:
                    if int(cluster_ids[u]) == ci and int(cluster_ids[v]) == cj:
                        w = d.get("weight", 1.0)
                        bridge_score[u] += w
                        bridge_score[v] += w

        for g in sorted(range(n_genes), key=lambda x: -bridge_score[x]):
            if len(selected) >= budget:
                break
            selected.add(g)

    # ── Priority 3: reverse BFS from seed (upstream) ─────────────────────────
    if len(selected) < budget:
        G_rev = G.reverse(copy=False)
        reachable_bwd = nx.single_source_shortest_path_length(G_rev, seed, cutoff=max_depth)
        for g in sorted(reachable_bwd, key=lambda x: reachable_bwd[x]):
            if len(selected) >= budget:
                break
            selected.add(g)

    # ── Priority 4: forward BFS from seed (downstream) ───────────────────────
    if len(selected) < budget:
        reachable_fwd = nx.single_source_shortest_path_length(G, seed, cutoff=max_depth)
        for g in sorted(reachable_fwd, key=lambda x: reachable_fwd[x]):
            if len(selected) >= budget:
                break
            selected.add(g)

    return sorted(selected)


def expand_bridge(
    G: nx.DiGraph,
    cluster_ids: np.ndarray,
    budget: int,
    query: Optional[int] = None,
    inter_mask: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Select genes that bridge ADJACENT gene programmes in the directed KL graph.

    Strategy:
      1. Infer cluster ordering from inter-cluster edge flow in G
      2. Score each gene by edges it sends to the NEXT cluster (C_i → C_{i+1})
         — these are the true programme-to-programme bridges
      3. Fill budget: bridges first (by score), then query neighbourhood
    """
    n_genes = len(cluster_ids)
    K = int(cluster_ids.max()) + 1

    # Infer cluster ordering: cluster_flow[i][j] = total weight C_i→C_j
    cluster_flow = np.zeros((K, K))
    for u, v, d in G.edges(data=True):
        if u < n_genes and v < n_genes:
            ci, cj = int(cluster_ids[u]), int(cluster_ids[v])
            if ci != cj:
                cluster_flow[ci, cj] += d.get("weight", 1.0)

    # Build ordered cluster sequence by following dominant flow
    visited = set()
    order = []
    # Start from cluster with no dominant inflow
    inflow = cluster_flow.sum(0)
    start = int(np.argmin(inflow))
    cur = start
    while cur not in visited and len(order) < K:
        visited.add(cur)
        order.append(cur)
        outflow = cluster_flow[cur].copy()
        outflow[list(visited)] = 0
        if outflow.max() > 0:
            cur = int(np.argmax(outflow))
        else:
            break

    # Add any remaining clusters
    for k in range(K):
        if k not in visited:
            order.append(k)

    # Score each gene by its bridge weight to the NEXT cluster in order
    bridge_score = np.zeros(n_genes)
    for step in range(len(order) - 1):
        ci, cj = order[step], order[step + 1]
        for u, v, d in G.edges(data=True):
            if u < n_genes and v < n_genes:
                if int(cluster_ids[u]) == ci and int(cluster_ids[v]) == cj:
                    bridge_score[u] += d.get("weight", 1.0)
                    bridge_score[v] += d.get("weight", 1.0)

    ranked = sorted(range(n_genes), key=lambda g: -bridge_score[g])

    selected = set()
    if query is not None:
        selected.add(query)

    # Priority 1: genes on shortest paths stim→query in theta (ground truth)
    if inter_mask is not None and query is not None:
        G_theta = nx.DiGraph()
        G_val = inter_mask[1:, 1:]
        for i in range(n_genes):
            for j in range(n_genes):
                if G_val[i, j]:
                    G_theta.add_edge(i, j)
        for j in np.where(inter_mask[0, 1:])[0]:
            G_theta.add_edge(-1, int(j))
        if nx.has_path(G_theta, -1, query):
            try:
                for path in nx.all_shortest_paths(G_theta, -1, query):
                    for g in path:
                        if g >= 0:
                            selected.add(g)
            except (nx.NodeNotFound, nx.NetworkXNoPath, nx.NetworkXError):
                pass

    # Priority 2: bridge genes by score
    for g in ranked:
        if len(selected) >= budget:
            break
        selected.add(g)

    return sorted(selected)


def expand_random_control(
    n_genes: int,
    budget: int,
    seed_gene: int,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Random gene subset of same size, always including the seed gene."""
    if rng is None:
        rng = np.random.default_rng(0)
    pool = [g for g in range(n_genes) if g != seed_gene]
    chosen = rng.choice(pool, size=min(budget - 1, len(pool)), replace=False).tolist()
    return sorted([seed_gene] + chosen)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation vs ground truth GRN
# ─────────────────────────────────────────────────────────────────────────────

def subgraph_recall(
    selected: List[int],
    inter_mask: np.ndarray,
    inter_opt: np.ndarray,
    query: Optional[int] = None,
    stimulus_row: int = 0,
) -> dict:
    """
    Stimulus-rooted propagation subgraph recall (Elias Ventre design).

    Builds the ground-truth GRN from theta, identifies:
      - stim_direct  : genes directly regulated by stimulus
      - bridge       : genes with both in- and out-edges (intermediaries)
      - downstream   : genes only reachable via other genes
      - path_genes   : genes on any path stimulus→query in theta

    Metrics:
      - recall_stim_direct : fraction of stim-direct genes in selection
      - recall_bridge      : fraction of bridge genes in selection
      - recall_path        : fraction of path genes (stim→query) in selection
      - stim_connected     : is stimulus connected to query in selected subgraph?
    """
    sel_set = set(selected)
    G_val = inter_mask[1:, 1:]
    n_genes = G_val.shape[0]

    # Build ground-truth GRN: -1 = stimulus node
    G_grn = nx.DiGraph()
    G_grn.add_nodes_from(range(n_genes))
    for i in range(n_genes):
        for j in range(n_genes):
            if G_val[i, j]:
                G_grn.add_edge(i, j)
    for j in np.where(inter_mask[0, 1:])[0]:
        G_grn.add_edge(-1, int(j))

    # stim_direct: directly from stimulus
    stim_direct = set(G_grn.successors(-1))

    # bridge: has both in- and out-edges (within genes only)
    bridge = set(
        g for g in range(n_genes)
        if G_grn.in_degree(g) > 0 and G_grn.out_degree(g) > 0
    )

    # downstream: reachable from stimulus but not stim_direct
    try:
        stim_reachable = set(nx.descendants(G_grn, -1))
    except nx.NodeNotFound:
        stim_reachable = set()
    downstream = stim_reachable - stim_direct

    # path genes: genes on shortest paths stim→query in theta
    path_genes = set()
    if query is not None and query in G_grn and nx.has_path(G_grn, -1, query):
        try:
            for path in nx.all_shortest_paths(G_grn, -1, query):
                path_genes.update(g for g in path if g >= 0)
        except (nx.NodeNotFound, nx.NetworkXNoPath, nx.NetworkXError):
            pass

    # stim connectivity: path stim→query exists in subgraph restricted to selected
    stim_connected = False
    if query is not None:
        G_sub = G_grn.subgraph([-1] + list(sel_set))
        try:
            stim_connected = nx.has_path(G_sub, -1, query)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            stim_connected = False

    def recall(target):
        return len(sel_set & target) / max(len(target), 1)

    return {
        "n_selected":         len(sel_set),
        "recall_stim_direct": recall(stim_direct),
        "recall_bridge":      recall(bridge),
        "recall_downstream":  recall(downstream),
        "recall_path":        recall(path_genes) if path_genes else None,
        "stim_connected":     stim_connected,
        # keep for compatibility
        "recall_reachable":   recall(stim_reachable),
    }
