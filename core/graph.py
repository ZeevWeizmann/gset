import numpy as np
import torch
from torch_geometric.data import Data


def build_graph(X: np.ndarray, beta: float = 6.0, thresh: float = 0.01) -> tuple:
    """WGCNA-style gene co-expression graph from expression matrix X (n_cells, n_genes)."""
    X_norm = np.log1p(X / (X.sum(1, keepdims=True) + 1e-8) * 1e4)
    mu_g = X_norm.mean(0); std_g = X_norm.std(0)
    X_std = (X_norm - mu_g) / (std_g + 1e-8)
    X_std[:, std_g < 1e-6] = 0.0
    n_genes = X_std.shape[1]
    C = (X_std.T @ X_std) / X_std.shape[0]
    np.fill_diagonal(C, 0)
    C = np.nan_to_num(C, nan=0.0)
    A = np.clip(np.abs(C) ** beta, 0.0, 1.0)
    rows, cols = np.where(A >= thresh)
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    weights = A[rows, cols].astype(np.float32)
    if len(rows) == 0:
        rows = np.arange(n_genes - 1, dtype=np.int64)
        cols = np.arange(1, n_genes, dtype=np.int64)
        weights = np.full(len(rows), thresh, dtype=np.float32)
    ei = torch.tensor(np.stack([np.concatenate([rows, cols]),
                                 np.concatenate([cols, rows])]), dtype=torch.long)
    ew = torch.tensor(np.concatenate([weights, weights]), dtype=torch.float32)
    data = Data(x=torch.tensor(C, dtype=torch.float32),
                edge_index=ei, edge_attr=ew.unsqueeze(1))
    return data, ew, rows, cols
