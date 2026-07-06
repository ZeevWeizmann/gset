"""
gene_trajectory_temporal.py
===========================

End-to-end pipeline that adapts GeneTrajectory to REAL time-resolved data and
produces a DIRECTED gene-gene score matrix, to be benchmarked alongside
delta_KL, delta_OT (our embedding method) and OTVelo-Corr.

Idea
----
GeneTrajectory represents a gene as a distribution over cells and measures
gene-gene relationships by the optimal-transport (OT) distance between those
distributions, using the cell-graph geometry as ground metric. We adapt this
in three ways:

  1. Cells are pooled into per-timepoint metacells, and the ground metric is a
     geodesic distance on a metacell graph that is CONNECTED ACROSS TIMEPOINTS
     (your construction: a classic, time-agnostic metacell backbone projected
     onto the per-timepoint metacells, plus an explicit cross-timepoint kNN
     safeguard). Connectivity is mandatory: without it, a gene's distribution at
     t and at t+1 live on disjoint supports and their OT cost is meaningless.

  2. Each gene becomes, at each timepoint, a MASS-NORMALISED distribution over
     that timepoint's metacells. Mass normalisation is what puts a lowly
     expressed TF on equal footing with a highly expressed gene -- the property
     that rescues the drivers a correlation embedding tends to miss.

  3. The directed score is
         S[i, j] = sum_t  W( gene_i^t , gene_j^{t+1} )
     where W is the OT cost with ground metric = C_cost restricted to the
     (t, t+1) metacell blocks. Direction comes purely from the lag (gene i at t,
     gene j at t+1), exactly as in the lagged-KL / lagged-OT graphs.

Output
------
`S`  : (n_genes x n_genes) DIRECTED OT distance. SMALL S[i,j] => gene i's
       territory at t maps cheaply onto gene j's territory at t+1 => strong i->j.
`A`  : affinity version, A = exp(-S / scale), diagonal zeroed. LARGE A[i,j] =>
       strong directed edge i->j. Use `A` to slot next to delta_KL / delta_OT /
       OTVelo-Corr in the flow figure and the query-centred expansion.

Caveats worth keeping in mind
-----------------------------
* The ground metric is symmetric, so ALL directionality comes from the lag.
  S is asymmetric only because i is taken at t and j at t+1.
* Raw lagged OT conflates "i leads j" with "i and j merely co-occupy cells".
  `same_time_ot_baseline` (optional, off by default) lets you subtract the
  static co-location term to isolate directed flow (co-location != causation).
* Cost scales as O(n_genes^2 * (T-1)) OT solves. Fine for the simulated
  benchmark (tens of genes); for thousands of genes restrict to a candidate set
  or switch to `sinkhorn_reg` (entropic OT).

Dependencies: numpy, scipy, scikit-learn, POT (`pip install pot`).
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

try:
    import ot  # POT
except ImportError as _e:  # pragma: no cover
    raise ImportError("This module requires POT: `pip install pot`.") from _e


# ---------------------------------------------------------------------------
# 1. Temporal metacell graph  ->  geodesic cost matrix C_cost
# ---------------------------------------------------------------------------
def _backbone_tp_graph(X_pca, cell_to_meta_tp, n_meta_tp, k, random_state):
    """
    'backbone' base graph: build a classic, TIME-AGNOSTIC metacell backbone
    (KMeans over all cells, so clusters pool cells across timepoints), take its
    local kNN graph, and project it onto the tp-metacells via the cell overlap
    O:   D = O @ G @ O.T. Two tp-metacells from t and t+1 become linked when
    they fall in the same / neighbouring classic metacell.
    """
    n_cells = X_pca.shape[0]
    km_c = MiniBatchKMeans(n_clusters=n_meta_tp, random_state=random_state,
                           batch_size=min(4096, n_cells), n_init=3).fit(X_pca)
    centers_c, labels_c = km_c.cluster_centers_, km_c.labels_

    k_m = min(k * 2, n_meta_tp - 1)
    nbrs = NearestNeighbors(n_neighbors=k_m + 1).fit(centers_c)
    d, idx = nbrs.kneighbors(centers_c)
    rows = np.repeat(np.arange(n_meta_tp), k_m)
    cols = idx[:, 1:].ravel()
    vals = d[:, 1:].ravel()
    G = csr_matrix((vals, (rows, cols)), shape=(n_meta_tp, n_meta_tp))
    G = (G + G.T) * 0.5

    O = np.zeros((n_meta_tp, n_meta_tp))
    np.add.at(O, (cell_to_meta_tp, labels_c), 1.0)
    O /= np.clip(O.sum(1, keepdims=True), 1.0, None)     # row-normalised

    D = np.asarray(O @ (G @ O.T))
    np.fill_diagonal(D, 0.0)
    return D


def _direct_tp_graph(centers_tp, k, T):
    """
    'direct' base graph (GeneTrajectory-style, no classic backbone): a kNN graph
    built DIRECTLY on the tp-metacell centroids. The neighbour budget k is split:
    k*(T-1)/T neighbours here for the general geometry (within + across time),
    leaving ~k/T of the budget to be guaranteed as explicit temporal edges by the
    caller (so the two parts sum back to ~k).
    """
    n = centers_tp.shape[0]
    k_geom = max(1, min(int(round(k * (T - 1) / T)), n - 1))
    nbrs = NearestNeighbors(n_neighbors=k_geom + 1).fit(centers_tp)
    d, idx = nbrs.kneighbors(centers_tp)
    rows = np.repeat(np.arange(n), k_geom)
    cols = idx[:, 1:].ravel()
    vals = d[:, 1:].ravel()
    G = csr_matrix((vals, (rows, cols)), shape=(n, n))
    G = (G + G.T) * 0.5

    D = np.asarray(G.todense())
    np.fill_diagonal(D, 0.0)
    return D


def build_temporal_cost(X_pca, times, k=15, n_meta=200, temporal_knn=3,
                        mode="direct", random_state=42):
    """
    Build the geodesic cost matrix between per-timepoint metacells.

    mode : "backbone" -> classic time-agnostic metacell bridge, projected onto
                         tp-metacells (D = O @ G @ O.T).
           "direct"   -> kNN graph built directly on tp-metacell centroids with
                         k*(T-1)/T neighbours, and the temporal safeguard floored
                         to temporal_knn = max(temporal_knn, round(k/T)) so the
                         cross-timepoint budget is guaranteed. No classic backbone.

    Returns a dict with:
      C_cost          : (n_meta_tp x n_meta_tp) float32 geodesic distances
      M               : (n_cells x n_meta_tp) column-normalised assignment (mean pooling)
      meta_tp_rank    : (n_meta_tp,) time rank (0..T-1) of each tp-metacell
      cell_to_meta_tp : (n_cells,) tp-metacell index of each cell
      centers_tp      : (n_meta_tp x n_pcs) PCA centroid of each tp-metacell
      connectivity    : dict {(rank_t, rank_t1): fraction_of_finite_distances}
                        computed BEFORE the infinity->sentinel fill; should be 1.0.
    """
    X_pca = np.ascontiguousarray(X_pca, dtype=np.float64)
    times = np.asarray(times)
    n_cells, n_pcs = X_pca.shape
    unique_times = np.sort(np.unique(times))
    T = len(unique_times)
    time_rank = {t: r for r, t in enumerate(unique_times)}

    # -- per-timepoint metacells (NEVER mix timepoints) ----------------------
    # Budget split proportionally to the number of cells at each timepoint.
    cell_to_meta_tp = np.full(n_cells, -1, dtype=int)
    meta_tp_rank = []
    offset = 0
    for t in unique_times:
        idx_t = np.where(times == t)[0]
        nm = max(1, min(int(round(n_meta * len(idx_t) / n_cells)), len(idx_t)))
        bs = max(1, min(int(round(4096 * len(idx_t) / n_cells)), len(idx_t)))
        km = MiniBatchKMeans(n_clusters=nm, random_state=random_state,
                             batch_size=bs, n_init=3).fit(X_pca[idx_t])
        cell_to_meta_tp[idx_t] = offset + km.labels_
        meta_tp_rank.extend([time_rank[t]] * nm)
        offset += nm
    n_meta_tp = offset
    meta_tp_rank = np.asarray(meta_tp_rank, dtype=int)

    # tp-metacell centroids in PCA space (used by the temporal-edge safeguard)
    counts = np.bincount(cell_to_meta_tp, minlength=n_meta_tp).astype(float)
    centers_tp = np.zeros((n_meta_tp, n_pcs))
    np.add.at(centers_tp, cell_to_meta_tp, X_pca)
    centers_tp /= np.clip(counts[:, None], 1.0, None)

    # -- base tp-metacell graph D (depends on mode) --------------------------
    if mode == "backbone":
        D = _backbone_tp_graph(X_pca, cell_to_meta_tp, n_meta_tp, k, random_state)
    elif mode == "direct":
        D = _direct_tp_graph(centers_tp, k, T)
        # split of the k-budget: floor the temporal edges to ~k/T so that the
        # geometry part (k*(T-1)/T) and the temporal part (k/T) sum back to ~k.
        temporal_knn = max(temporal_knn, int(round(k / T)))
    else:
        raise ValueError(f"mode must be 'backbone' or 'direct', got {mode!r}")

    # -- SAFEGUARD: explicit cross-timepoint edges ---------------------------
    # Guarantees that consecutive timepoints are connected. In 'direct' mode this
    # carries the reserved k/T temporal budget; in 'backbone' mode it patches any
    # bridge the backbone failed to provide (e.g. timepoints far apart in PCA).
    if temporal_knn and temporal_knn > 0:
        ranks_sorted = np.unique(meta_tp_rank)
        for r_t, r_t1 in zip(ranks_sorted[:-1], ranks_sorted[1:]):
            src = np.where(meta_tp_rank == r_t)[0]
            dst = np.where(meta_tp_rank == r_t1)[0]
            kk = min(temporal_knn, len(dst))
            nn = NearestNeighbors(n_neighbors=kk).fit(centers_tp[dst])
            dd, ii = nn.kneighbors(centers_tp[src])
            for a_local, a in enumerate(src):
                for b_local in range(kk):
                    b = dst[ii[a_local, b_local]]
                    w = float(dd[a_local, b_local])
                    cur = D[a, b]
                    new = w if cur == 0.0 else min(cur, w)
                    D[a, b] = new
                    D[b, a] = new

    # -- geodesic distances on the tp-metacell graph -------------------------
    G_tp = csr_matrix(D)
    G_tp = (G_tp + G_tp.T) * 0.5
    C = shortest_path(G_tp, method="D", directed=False)

    # connectivity diagnostic on consecutive-timepoint blocks (before inf-fill)
    connectivity = {}
    ranks_sorted = np.unique(meta_tp_rank)
    for r_t, r_t1 in zip(ranks_sorted[:-1], ranks_sorted[1:]):
        src = np.where(meta_tp_rank == r_t)[0]
        dst = np.where(meta_tp_rank == r_t1)[0]
        block = C[np.ix_(src, dst)]
        connectivity[(int(r_t), int(r_t1))] = float(np.isfinite(block).mean())

    # replace unreachable pairs (inf) by a large finite sentinel so OT stays finite
    finite_max = C[np.isfinite(C)].max() if np.isfinite(C).any() else 1.0
    C[~np.isfinite(C)] = finite_max * 10.0
    C = C.astype(np.float32)

    # column-normalised assignment (mean pooling); kept for compatibility
    M = np.zeros((n_cells, n_meta_tp), dtype=np.float32)
    M[np.arange(n_cells), cell_to_meta_tp] = 1.0
    M /= np.clip(M.sum(0, keepdims=True), 1.0, None)

    return dict(C_cost=C, M=M, meta_tp_rank=meta_tp_rank,
                cell_to_meta_tp=cell_to_meta_tp, centers_tp=centers_tp,
                connectivity=connectivity)


# ---------------------------------------------------------------------------
# 2. Gene distributions over metacells, per timepoint
# ---------------------------------------------------------------------------
def gene_distributions_by_time(expression, cell_to_meta_tp, meta_tp_rank,
                               aggregate="sum", eps=1e-8):
    """
    Turn each gene into a mass-normalised distribution over the metacells of
    each timepoint.

    Parameters
    ----------
    expression : (n_cells x n_genes) non-negative expression (counts or normalised).
    aggregate  : "sum"  -> total gene mass per metacell (faithful coarse-graining;
                           bigger/more-expressing metacells carry more mass),
                 "mean" -> average expression per metacell (treats metacells as
                           equal-mass graph nodes).

    Returns
    -------
    dists    : dict {rank: (n_genes x n_meta_rank) array}, each row sums to 1.
    meta_idx : dict {rank: indices of that rank's metacells in the global order}.
    """
    expression = np.clip(np.asarray(expression, dtype=np.float64), 0.0, None)
    n_cells, n_genes = expression.shape
    n_meta_tp = meta_tp_rank.shape[0]

    # total expression mass of each gene in each tp-metacell
    meta_mass = np.zeros((n_meta_tp, n_genes))
    np.add.at(meta_mass, cell_to_meta_tp, expression)
    if aggregate == "mean":
        counts = np.bincount(cell_to_meta_tp, minlength=n_meta_tp).astype(float)
        meta_mass /= np.clip(counts[:, None], 1.0, None)

    dists, meta_idx = {}, {}
    for r in np.unique(meta_tp_rank):
        cols = np.where(meta_tp_rank == r)[0]
        block = meta_mass[cols, :].T + eps            # (n_genes, n_meta_r)
        block /= block.sum(1, keepdims=True)          # mass-normalise per gene
        dists[int(r)] = block.astype(np.float64)
        meta_idx[int(r)] = cols
    return dists, meta_idx


# ---------------------------------------------------------------------------
# 3. Lagged OT gene-gene scores  S[i,j] = sum_t W(gene_i^t, gene_j^{t+1})
# ---------------------------------------------------------------------------
def lagged_ot_scores(dists, meta_idx, C_cost, sinkhorn_reg=None, verbose=True):
    """Directed OT distance S[i,j] = sum_t W(gene_i^t, gene_j^{t+1})."""
    ranks = sorted(dists.keys())
    n_genes = dists[ranks[0]].shape[0]
    S = np.zeros((n_genes, n_genes), dtype=np.float64)

    for r_t, r_t1 in zip(ranks[:-1], ranks[1:]):
        A = dists[r_t]        # (n_genes, n_meta_t)     sources at time t
        B = dists[r_t1]       # (n_genes, n_meta_{t+1}) targets at time t+1
        cols_t, cols_t1 = meta_idx[r_t], meta_idx[r_t1]
        # ground cost between t-metacells (rows) and (t+1)-metacells (cols)
        Msub = np.ascontiguousarray(C_cost[np.ix_(cols_t, cols_t1)], dtype=np.float64)
        # POT's emd_c requires C-contiguous float64 histograms
        A = np.ascontiguousarray(A / A.sum(1, keepdims=True))
        B = np.ascontiguousarray(B / B.sum(1, keepdims=True))
        for i in range(n_genes):
            a = np.ascontiguousarray(A[i])
            for j in range(n_genes):
                b = np.ascontiguousarray(B[j])
                if sinkhorn_reg is None:
                    S[i, j] += ot.emd2(a, b, Msub)
                else:
                    S[i, j] += ot.sinkhorn2(a, b, Msub, reg=sinkhorn_reg)
        if verbose:
            print(f"  transition t{r_t} -> t{r_t1} done")
    return S


def same_time_ot_baseline(dists, meta_idx, C_cost, sinkhorn_reg=None):
    """
    Optional static co-location baseline B0[i,j] = sum_t W(gene_i^t, gene_j^t).

    Subtracting B0 from S isolates DIRECTED FLOW from mere co-location
    (co-location != causation). Not used by default. Note that S - B0 is a
    signed flow score (higher = more directed flow); rank it rather than
    feeding it to `scores_to_affinity`, which assumes a non-negative distance.
    """
    ranks = sorted(dists.keys())
    n_genes = dists[ranks[0]].shape[0]
    B0 = np.zeros((n_genes, n_genes), dtype=np.float64)
    for r in ranks:
        A = dists[r]
        cols = meta_idx[r]
        Msub = np.ascontiguousarray(C_cost[np.ix_(cols, cols)], dtype=np.float64)
        A = np.ascontiguousarray(A / A.sum(1, keepdims=True))
        for i in range(n_genes):
            a = np.ascontiguousarray(A[i])
            for j in range(n_genes):
                b = np.ascontiguousarray(A[j])
                if sinkhorn_reg is None:
                    B0[i, j] += ot.emd2(a, b, Msub)
                else:
                    B0[i, j] += ot.sinkhorn2(a, b, Msub, reg=sinkhorn_reg)
    return B0


# ---------------------------------------------------------------------------
# 4. Distance -> affinity (edge weight), to match the other benchmark scores
# ---------------------------------------------------------------------------
def scores_to_affinity(S):
    """A = exp(-S / scale), scale = median off-diagonal distance. Large A => strong i->j."""
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    scale = np.nanmedian(off)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    A = np.exp(-S / scale)
    np.fill_diagonal(A, 0.0)
    return A, float(scale)


# ---------------------------------------------------------------------------
# 5. End-to-end wrapper
# ---------------------------------------------------------------------------
def run_gene_trajectory_temporal(expression, times, X_pca, k=15, n_meta=200,
                                 temporal_knn=3, mode="backbone", aggregate="sum",
                                 sinkhorn_reg=None, verbose=True):
    """
    Full pipeline: (expression, times, X_pca) -> directed gene-gene scores.

    mode : "backbone" (classic metacell bridge) or "direct" (kNN on tp-metacells
           with k*(T-1)/T neighbours + floored temporal edges). See build_temporal_cost.

    Returns dict: S (directed distance), A (directed affinity), scale,
    C_cost, meta_tp_rank, connectivity.
    """
    geo = build_temporal_cost(X_pca, times, k=k, n_meta=n_meta,
                              temporal_knn=temporal_knn, mode=mode)

    if verbose:
        print("Temporal connectivity between consecutive timepoints "
              "(fraction finite, should be 1.0):")
        for (r0, r1), frac in geo["connectivity"].items():
            flag = "" if frac >= 0.999 else "   <-- WARNING: DISCONNECTED"
            print(f"  t{r0} -> t{r1} : {frac:.3f}{flag}")

    dists, meta_idx = gene_distributions_by_time(
        expression, geo["cell_to_meta_tp"], geo["meta_tp_rank"],
        aggregate=aggregate)

    if verbose:
        print("Computing lagged OT gene-gene scores ...")
    S = lagged_ot_scores(dists, meta_idx, geo["C_cost"],
                         sinkhorn_reg=sinkhorn_reg, verbose=verbose)
    A, scale = scores_to_affinity(S)

    return dict(S=S, A=A, scale=scale, C_cost=geo["C_cost"],
                meta_tp_rank=geo["meta_tp_rank"],
                connectivity=geo["connectivity"])


# ---------------------------------------------------------------------------
# Smoke test on tiny synthetic data (runs the whole pipeline end-to-end)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # toy: 3 timepoints, a tiny 3-cluster cascade over 12 genes
    T, cells_per_t, n_genes = 3, 300, 12
    times = np.repeat(np.arange(T), cells_per_t)
    n_cells = times.size
    clusters = [slice(0, 4), slice(4, 8), slice(8, 12)]  # C0, C1, C2

    E = 0.1 * rng.random((n_cells, n_genes))
    for t in range(T):                       # sequential activation C0->C1->C2
        m = times == t
        E[m, clusters[t]] += 3.0 + rng.random((m.sum(), 4))

    # cheap "PCA": just use expression (or PCA-reduce it in real use)
    X_pca = E - E.mean(0)

    for mode in ("backbone", "direct"):
        print(f"\n===== mode = {mode} =====")
        out = run_gene_trajectory_temporal(E, times, X_pca, k=8, n_meta=45,
                                           temporal_knn=3, mode=mode, verbose=True)
        print("S shape:", out["S"].shape, "| affinity scale:", round(out["scale"], 4))
