"""
data.py — load HARISSA simulation data, train GCN, compute KL/OT vectors.
Adapted for GSET_2 (imports from core instead of core).
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple

_GSET2_ROOT = Path(__file__).resolve().parents[2]
if str(_GSET2_ROOT) not in sys.path:
    sys.path.insert(0, str(_GSET2_ROOT))

import torch
import torch.nn.functional as F
from core.scoring import kl_vec, ot_residual_vec
from core.model import GCNEncoder
from core.graph import build_graph


def load_run(path: str) -> Tuple[np.ndarray, np.ndarray]:
    raw   = np.loadtxt(path, delimiter="\t")
    times = raw[0, 1:].astype(int)
    X     = raw[2:, 1:].T
    return X, times


def load_scenario(sim_dir: str) -> Tuple[List[dict], np.ndarray, list]:
    sim_dir = Path(sim_dir)
    cluster_ids = np.load(sim_dir / "cluster_ids.npy")
    data_files = sorted((sim_dir / "data").glob("data_*.txt"))
    assert data_files, f"No data files found in {sim_dir / 'data'}"
    runs, tp_set = [], set()
    for fp in data_files:
        X, times = load_run(str(fp))
        run = {}
        for tp in np.unique(times):
            run[int(tp)] = X[times == tp]
            tp_set.add(int(tp))
        runs.append(run)
    return runs, cluster_ids, sorted(tp_set)


def _active_tps(runs: List[dict], tp_list: list) -> list:
    means = {tp: np.concatenate([run[tp] for run in runs]).mean() for tp in tp_list}
    active = [tp for tp in tp_list if means[tp] > 1.0]
    return active if len(active) >= 2 else tp_list


def train_gcn(runs: List[dict], tp_list: list,
              n_epochs: int = 300, d_out: int = 16, seed: int = 42) -> GCNEncoder:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(seed)
    n_genes = next(iter(runs[0].values())).shape[1]
    active  = _active_tps(runs, tp_list)
    print(f"  Active timepoints: {active}")

    model = GCNEncoder(n_genes=n_genes, d_out=d_out)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X_list = [run[tp] for run in runs for tp in active]

    print(f"  Training shared GCN on {len(X_list)} matrices ({n_epochs} epochs)...")
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


def gcn_embedding(model: GCNEncoder, X: np.ndarray) -> np.ndarray:
    data, *_ = build_graph(X)
    with torch.no_grad():
        z = model(data).cpu().numpy()
    return np.nan_to_num(z, nan=0.0)


def compute_kl_vecs(runs: List[dict], tp_list: list,
                    model: GCNEncoder = None, **kwargs) -> List[np.ndarray]:
    n_genes  = next(iter(runs[0].values())).shape[1]
    n_trans  = len(tp_list) - 1
    acc = [np.zeros((n_genes, n_genes)) for _ in range(n_trans)]
    for run in runs:
        for t in range(n_trans):
            E0 = gcn_embedding(model, run[tp_list[t]])
            E1 = gcn_embedding(model, run[tp_list[t + 1]])
            acc[t] += kl_vec(E0, E1)
    return [a / len(runs) for a in acc]


def compute_ot_vecs(runs: List[dict], tp_list: list,
                    model: GCNEncoder = None, **kwargs) -> List[np.ndarray]:
    n_genes  = next(iter(runs[0].values())).shape[1]
    n_trans  = len(tp_list) - 1
    sample_X = next(iter(runs[0].values()))
    d_out    = gcn_embedding(model, sample_X).shape[1]
    acc = [np.zeros((n_genes, d_out)) for _ in range(n_trans)]
    for run in runs:
        for t in range(n_trans):
            E0 = gcn_embedding(model, run[tp_list[t]])
            E1 = gcn_embedding(model, run[tp_list[t + 1]])
            acc[t] += ot_residual_vec(E0, E1)
    return [a / len(runs) for a in acc]
