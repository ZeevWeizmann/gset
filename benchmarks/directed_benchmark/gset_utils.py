"""
data.py
-------
Load HARISSA simulation data and compute KL / OT vectors from GCN embeddings.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from core.scoring import kl_vec, ot_residual_vec
from core.model import GCNEncoder
from core.graph import build_graph


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def gcn_embedding(model: GCNEncoder, X: np.ndarray) -> np.ndarray:
    data, *_ = build_graph(X)
    with torch.no_grad():
        z = model(data).cpu().numpy()
    return np.nan_to_num(z, nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# KL vectors
# ─────────────────────────────────────────────────────────────────────────────

def compute_kl_vecs(runs: List[dict], tp_list: list,
                    model: GCNEncoder = None, **kwargs) -> List[np.ndarray]:
    n_genes = next(iter(runs[0].values())).shape[1]
    n_trans = len(tp_list) - 1
    acc = [np.zeros((n_genes, n_genes)) for _ in range(n_trans)]
    for run in runs:
        for t in range(n_trans):
            E0 = gcn_embedding(model, run[tp_list[t]])
            E1 = gcn_embedding(model, run[tp_list[t + 1]])
            acc[t] += kl_vec(E0, E1)
    return [a / len(runs) for a in acc]


# ─────────────────────────────────────────────────────────────────────────────
# OT residual vectors
# ─────────────────────────────────────────────────────────────────────────────

def compute_ot_vecs(runs: List[dict], tp_list: list,
                    model: GCNEncoder = None, **kwargs) -> List[np.ndarray]:
    n_genes = next(iter(runs[0].values())).shape[1]
    n_trans = len(tp_list) - 1
    sample_X = next(iter(runs[0].values()))
    d_out = gcn_embedding(model, sample_X).shape[1]
    acc = [np.zeros((n_genes, d_out)) for _ in range(n_trans)]
    for run in runs:
        for t in range(n_trans):
            E0 = gcn_embedding(model, run[tp_list[t]])
            E1 = gcn_embedding(model, run[tp_list[t + 1]])
            acc[t] += ot_residual_vec(E0, E1)
    return [a / len(runs) for a in acc]
