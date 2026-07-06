"""
utils_Velo.py — OTVelo utility functions.
Source: https://github.com/Sandstede-Lab/OT-Velocity
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph


def compute_graph_distances(data, n_neighbors=5, mode="distance", metric="euclidean"):
    graph = kneighbors_graph(data, n_neighbors=n_neighbors, mode=mode,
                             metric=metric, include_self=True)
    shortestPath = dijkstra(csgraph=csr_matrix(graph), directed=False,
                            return_predecessors=False)
    max_dist = np.nanmax(shortestPath[shortestPath != np.inf])
    shortestPath[shortestPath > max_dist] = max_dist
    return np.asarray(shortestPath)


def solve_prior(counts, counts_pca, Nt, labels, eps_samp, alpha=0.5):
    """
    Compute FGW transport plans between consecutive timepoints.

    counts     : (n_genes, n_cells_total)
    counts_pca : (n_genes, n_cells_total)  — used for OT cost matrix
    Nt         : number of timepoints
    labels     : (1, n_cells_total) with integer timepoint indices 0..Nt-1
    """
    import ot
    from ot.gromov import entropic_fused_gromov_wasserstein

    Ts  = [[0]] * (Nt - 1)
    log = [[0]] * (Nt - 1)

    for t in range(Nt - 1):
        idx_t  = np.where(labels == t)[1]
        idx_t1 = np.where(labels == t + 1)[1]
        X1 = counts[:, idx_t].T
        X2 = counts[:, idx_t1].T

        M = ot.dist(counts_pca[:, idx_t].T, counts_pca[:, idx_t1].T)
        M = M / M.max()

        n_neighbors = max(1, min(int(0.2 * X1.shape[0]),
                                 int(0.2 * X2.shape[0]), 50))
        D1 = compute_graph_distances(X1, n_neighbors=n_neighbors)
        D2 = compute_graph_distances(X2, n_neighbors=n_neighbors)

        alpha_t = alpha
        if D1.max() == 0 or D2.max() == 0:
            alpha_t = 0.0
        else:
            D1 = D1 / D1.max()
            D2 = D2 / D2.max()

        Ts[t], log_t = entropic_fused_gromov_wasserstein(
            M, D1, D2, epsilon=eps_samp, alpha=alpha_t, log=True)
        if np.isnan(Ts[t][0, 0]) or abs(Ts[t].sum() - 1) > 1e-3:
            Ts[t] = np.ones(Ts[t].shape) / Ts[t].size
        log[t] = log_t['fgw_dist']

    return Ts, log


def solve_velocities(counts_all, Ts_prior, order=1,
                     idx_marker=None, dt=None, stimulation=False):
    """
    Compute gene velocities from OT couplings.

    counts_all : list of Nt arrays (n_genes, n_cells_per_tp)
    Ts_prior   : list of Nt-1 coupling matrices
    """
    Nt = len(counts_all)
    if dt is None:
        dt = [1] * Nt
    n = counts_all[0].shape[0]
    if idx_marker is None:
        idx_marker = list(range(n))

    nums = [counts_all[i].shape[1] for i in range(Nt)]
    velocities_all        = [[0]] * Nt
    velocities_all_signed = [[0]] * Nt

    # t = 0: forward only
    i = 0
    T_norm = Ts_prior[i].T / (np.sum(Ts_prior[i].T, axis=0, keepdims=True) + 1e-10)
    count_t_mapped = counts_all[i + 1][idx_marker, :] @ T_norm
    velocities_all[i] = (count_t_mapped - counts_all[i][idx_marker, :]) / dt[0]

    if order > 1 and Nt > 2:
        coupling2 = Ts_prior[i] @ Ts_prior[i + 1]
        T2_norm = coupling2.T / (np.sum(coupling2.T, axis=0, keepdims=True) + 1e-10)
        count_mapped2 = counts_all[i + 2][idx_marker, :] @ T2_norm
        velocities2 = (count_mapped2 - counts_all[i][idx_marker, :]) / (dt[0] + dt[1])
        velocities_all[i] = (dt[0] + dt[1]) / dt[1] * velocities_all[i] - dt[0] / dt[1] * velocities2

    if stimulation:
        velocities_all[0] = np.zeros((n, nums[0]))
        velocities_all[0][0, :] = 1

    velocities_all_signed[i] = velocities_all[i].copy()
    velocities_all[i] = np.abs(velocities_all[i])

    # middle timepoints
    for i in range(1, Nt - 1):
        T_fwd = Ts_prior[i].T / (np.sum(Ts_prior[i].T, axis=0, keepdims=True) + 1e-10)
        count_fwd = counts_all[i + 1][idx_marker, :] @ T_fwd

        T_bwd = Ts_prior[i - 1] / (np.sum(Ts_prior[i - 1], axis=0, keepdims=True) + 1e-10)
        count_bwd = counts_all[i - 1][idx_marker, :] @ T_bwd

        w_fwd = dt[i - 1] / (dt[i] + dt[i - 1])
        w_bwd = dt[i] / (dt[i] + dt[i - 1])
        v = (count_fwd - counts_all[i][idx_marker, :]) / dt[i] * w_fwd \
          + (counts_all[i][idx_marker, :] - count_bwd) / dt[i - 1] * w_bwd

        if stimulation:
            v[0, :] = 0.0

        velocities_all_signed[i] = v.copy()
        velocities_all[i] = np.abs(v)

    # t = Nt-1: backward only
    i = Nt - 1
    T_bwd = Ts_prior[i - 1] / (np.sum(Ts_prior[i - 1], axis=0, keepdims=True) + 1e-10)
    count_bwd = counts_all[i - 1][idx_marker, :] @ T_bwd
    velocities_all[i] = (counts_all[i][idx_marker, :] - count_bwd) / dt[i - 1]

    if order > 1 and i >= 2:
        coupling2 = Ts_prior[i - 1] @ Ts_prior[i - 2]
        T2_norm = coupling2 / (np.sum(coupling2, axis=0, keepdims=True) + 1e-10)
        count_mapped2 = counts_all[i - 2][idx_marker, :] @ T2_norm
        velocities2 = (count_mapped2 - counts_all[i - 1][idx_marker, :]) / (dt[i - 1] + dt[i - 2])
        velocities_all[i] = (dt[i - 1] + dt[i - 2]) / dt[i - 1] * velocities_all[i] \
                           - dt[i - 2] / dt[i - 1] * velocities2

    if stimulation:
        velocities_all[i][0, :] = 0.0
    velocities_all_signed[i] = velocities_all[i].copy()
    velocities_all[i] = np.abs(velocities_all[i])

    return velocities_all, velocities_all_signed


def OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior,
                          normalization=True, stimulation=False,
                          elastic_Net=False, alpha_opt=1.0, l1_opt=0.5,
                          tune=False, g_s=None, g_t=None,
                          return_slice=False, signed=False, lags=False):
    positive = not signed

    Nt = len(velocities_all_signed)
    n  = velocities_all_signed[0].shape[0]

    if g_s is None:
        g_s = range(n)
    if g_t is None:
        g_t = range(1, n) if stimulation else range(n)

    vel_norm = copy.deepcopy(velocities_all_signed)

    if normalization:
        for i in range(n):
            var_i = np.var(velocities_signed[i, :])
            if var_i > 0:
                for j in range(Nt):
                    vel_norm[j][i, :] = velocities_all_signed[j][i, :] / np.sqrt(var_i)

    corr_OT    = np.zeros((n, n))
    corr_slice = [np.zeros((n, n))] * (Nt - 1)

    if not elastic_Net:
        corr_slice = _correlation_varied_lags(vel_norm, Ts_prior, n, g_s, g_t, lag=1)
        for t in range(Nt - 1):
            corr_OT += corr_slice[t]
    else:
        for t in range(Nt - 1):
            T_norm = Ts_prior[t] / (np.sum(Ts_prior[t], axis=0, keepdims=True) + 1e-10)
            V_src  = np.dot(vel_norm[t][list(g_s), :], T_norm).T  # (n_cells, g_s)
            V_tgt  = vel_norm[t + 1][list(g_t), :].T              # (n_cells, g_t)

            from sklearn.linear_model import ElasticNet, Ridge
            model = (Ridge(alpha=alpha_opt, fit_intercept=False, positive=positive)
                     if l1_opt == 0
                     else ElasticNet(alpha=alpha_opt, fit_intercept=False,
                                     positive=positive, l1_ratio=l1_opt))
            if signed:
                model.fit(V_src, V_tgt)
            else:
                model.fit(np.abs(V_src), np.abs(V_tgt))

            cs = np.zeros((n, n))
            cs[np.ix_(list(g_s), list(g_t))] = model.coef_.T
            corr_slice[t] = cs
            corr_OT += cs

    if return_slice:
        return corr_OT, corr_slice
    return corr_OT


def _correlation_varied_lags(vel_norm, Ts_prior, n, g_s, g_t, lag=1):
    Nt = len(Ts_prior) + 1
    corr_slice = [np.zeros((n, n))] * (Nt - 1)

    for t in range(Nt - lag):
        T_lag = Ts_prior[t].copy()
        for tt in range(1, lag):
            T_lag = T_lag @ Ts_prior[t + tt]
        T_lag = T_lag / (T_lag.sum() + 1e-10)

        cs = np.zeros((n, n))
        cs[np.ix_(list(g_s), list(g_t))] = (
            np.dot(vel_norm[t][list(g_s), :], T_lag)
            .dot(vel_norm[t + lag][list(g_t), :].T)
            / (Nt - lag)
        )
        corr_slice[t] = cs

    return corr_slice
