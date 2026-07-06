import torch
import torch.nn.functional as F
import numpy as np

from .graph import build_graph
from .model import GCNEncoder, GATEncoder


def train_gcn(
    X_list:   list,
    n_genes:  int,
    n_epochs: int   = 300,
    d_out:    int   = 16,
    lr:       float = 1e-3,
    seed:     int   = None,
) -> GCNEncoder:
    """
    Train shared-weight GCN on a list of expression matrices.

    Loss: MSE( z_i · z_j,  w_ij )  over WGCNA edges.
    Weights are shared across all timepoints and runs.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = GCNEncoder(n_genes=n_genes, d_out=d_out)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"  Training GCN on {len(X_list)} matrices ({n_epochs} epochs)...")
    model.train()
    for epoch in range(n_epochs):
        L_ep = 0.0
        for X in X_list:
            data, ew, rows, cols = build_graph(X)
            if len(rows) == 0:
                continue
            opt.zero_grad()
            z    = model(data)
            src  = torch.tensor(rows, dtype=torch.long)
            dst  = torch.tensor(cols, dtype=torch.long)
            pred = model.decode(z, src, dst).reshape(-1)
            loss = F.mse_loss(pred, ew[:len(rows)].reshape(-1))
            loss.backward()
            opt.step()
            L_ep += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch+1}  loss={L_ep/len(X_list):.5f}")

    model.eval()
    return model


def train_gat(
    model:     GATEncoder,
    X_list:    list,
    n_epochs:  int   = 300,
    n_train:   int   = 1,
    sample_fn  = None,
    beta:      float = 6.0,
    thresh:    float = 0.01,
    lr:        float = 1e-3,
) -> list:
    torch.manual_seed(42)
    np.random.seed(42)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for epoch in range(n_epochs):
        L_ep = 0.0
        iters = n_train if sample_fn is not None else (n_train if n_train > 1 else len(X_list))
        for i in range(iters):
            if sample_fn is not None:
                Xb = sample_fn()
            elif n_train > 1:
                Xb = X_list[np.random.randint(len(X_list))]
            else:
                Xb = X_list[i]
            data, ew, rows, cols = build_graph(Xb, beta=beta, thresh=thresh)
            if len(rows) == 0:
                continue
            opt.zero_grad()
            z    = model(data)
            src  = torch.tensor(rows, dtype=torch.long)
            dst  = torch.tensor(cols, dtype=torch.long)
            pred = model.decode(z, src, dst)
            loss = F.mse_loss(pred, ew[:len(rows)])
            loss.backward()
            opt.step()
            L_ep += loss.item()
        losses.append(L_ep / iters)
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:3d}  loss={losses[-1]:.6f}")
    model.eval()
    return losses


def get_embedding(model: GATEncoder, X: np.ndarray, beta: float = 6.0, thresh: float = 0.01) -> np.ndarray:
    data, *_ = build_graph(X, beta=beta, thresh=thresh)
    with torch.no_grad():
        z = model(data).cpu().numpy()
    return np.nan_to_num(z, nan=0.0)
