import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import ot


def kl_score(E0: np.ndarray, E1: np.ndarray) -> np.ndarray:
    """Symmetrised KL divergence of Gaussian neighbourhood distributions. Returns (n_genes,)."""
    def pairwise_sq(E):
        return ((E[:, None, :] - E[None, :, :]) ** 2).sum(-1)
    D0 = pairwise_sq(E0); D1 = pairwise_sq(E1)
    s = np.sqrt(np.median(D0[D0 > 0]) + 1e-10)
    def nbr(D):
        L = -D / (2 * s ** 2); np.fill_diagonal(L, -np.inf)
        L -= L.max(1, keepdims=True)
        P = np.exp(L); P /= P.sum(1, keepdims=True)
        return P
    P0, P1 = nbr(D0), nbr(D1); eps = 1e-10
    return ((P0 * np.log((P0 + eps) / (P1 + eps))).sum(1) +
            (P1 * np.log((P1 + eps) / (P0 + eps))).sum(1))


def kl_vec(E0: np.ndarray, E1: np.ndarray) -> np.ndarray:
    """KL vector per gene: p_i^0(j) * log(p_i^0(j)/p_i^1(j)). Returns (n_genes, n_genes)."""
    def pairwise_sq(E):
        return ((E[:, None, :] - E[None, :, :]) ** 2).sum(-1)
    D0 = pairwise_sq(E0); D1 = pairwise_sq(E1)
    s = np.sqrt(np.median(D0[D0 > 0]) + 1e-10)
    def nbr(D):
        L = -D / (2 * s ** 2); np.fill_diagonal(L, -np.inf)
        L -= L.max(1, keepdims=True)
        P = np.exp(L); P /= P.sum(1, keepdims=True)
        return P
    eps = 1e-10
    P0 = nbr(D0); P1 = nbr(D1)
    return P0 * np.log((P0 + eps) / (P1 + eps))


def ot_score(E0: np.ndarray, E1: np.ndarray) -> np.ndarray:
    """OT residual norm per gene. Returns (n_genes,)."""
    C = cdist(E0, E1); C /= C.max() + 1e-10
    n = len(E0)
    T = ot.sinkhorn(np.ones(n) / n, np.ones(n) / n, C, reg=0.05, warn=False)
    return np.linalg.norm(E1 - (T * n) @ E1, axis=1)


def ot_residual_vec(E0: np.ndarray, E1: np.ndarray) -> np.ndarray:
    """OT residual vector per gene: E1_i - [T @ E1]_i. Returns (n_genes, d)."""
    C = cdist(E0, E1); C /= C.max() + 1e-10
    n = len(E0)
    T = ot.sinkhorn(np.ones(n) / n, np.ones(n) / n, C, reg=0.05, warn=False)
    return E1 - (T * n) @ E1


def gw_score(E0: np.ndarray, E1: np.ndarray) -> np.ndarray:
    """Gromov-Wasserstein residual norm per gene (rotation-invariant). Returns (n_genes,)."""
    n = len(E0)
    C0 = cdist(E0, E0).astype(np.float64); C0 /= C0.max() + 1e-10
    C1 = cdist(E1, E1).astype(np.float64); C1 /= C1.max() + 1e-10
    p = np.ones(n) / n
    res = ot.gromov_wasserstein(C0, C1, p, p, "square_loss", verbose=False)
    T_gw = res[0] if isinstance(res, tuple) else res
    return np.linalg.norm(E1 - (T_gw * n) @ E1, axis=1)


def max_rank(*scores) -> np.ndarray:
    """Element-wise maximum of rankdata applied to each score array."""
    return np.maximum.reduce([rankdata(s) for s in scores])


def cosine_sim_matrix(V: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix."""
    V_norm = normalize(V, norm="l2", axis=1)
    return V_norm @ V_norm.T


def auroc_from_sim(S: np.ndarray, cluster_ids: np.ndarray) -> float:
    """AUROC: same-cluster pairs ranked higher than different-cluster pairs."""
    scores, labels = [], []
    G = len(cluster_ids)
    for i in range(G):
        for j in range(i + 1, G):
            scores.append(S[i, j])
            labels.append(int(cluster_ids[i] == cluster_ids[j]))
    return roc_auc_score(labels, scores)


def intra_inter_cosine(S: np.ndarray, cluster_ids: np.ndarray) -> tuple:
    """Mean intra- and inter-cluster cosine similarity."""
    n = len(cluster_ids)
    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (intra if cluster_ids[i] == cluster_ids[j] else inter).append(S[i, j])
    return np.mean(intra), np.mean(inter)
