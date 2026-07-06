"""
genetraj_worker.py — runs Lagged GeneTrajectory in an isolated process (no torch).

Usage (called by benchmark.py via subprocess):
  python genetraj_worker.py --sim_dir <path> --out <path.npy>
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_scenario(sim_dir):
    sim_dir = Path(sim_dir)
    cluster_ids = np.load(sim_dir / "cluster_ids.npy")
    data_files = sorted((sim_dir / "data").glob("data_*.txt"))
    assert data_files, f"No data files in {sim_dir / 'data'}"
    runs, tp_set = [], set()
    for fp in data_files:
        raw   = np.loadtxt(fp, delimiter="\t")
        times = raw[0, 1:].astype(int)
        X     = raw[2:, 1:].T
        run   = {}
        for tp in np.unique(times):
            run[int(tp)] = X[times == tp]
            tp_set.add(int(tp))
        runs.append(run)
    return runs, cluster_ids, sorted(tp_set)


def compute_lagged_genetraj(runs, tp_list):
    from gene_trajectory_temporal import run_gene_trajectory_temporal

    n_genes = next(iter(runs[0].values())).shape[1]
    A_acc   = np.zeros((n_genes, n_genes))

    for idx, run in enumerate(runs):
        print(f"  GeneTrajectory run {idx+1}/{len(runs)} ...", flush=True)
        Xs, time_labels = [], []
        for rank, tp in enumerate(tp_list):
            X = run[tp].astype(np.float64)
            Xs.append(X)
            time_labels.extend([rank] * len(X))
        expression  = np.vstack(Xs)
        times       = np.array(time_labels)
        X_pca       = np.log1p(expression)
        out = run_gene_trajectory_temporal(
            expression, times, X_pca,
            k=10, n_meta=100, temporal_knn=3, mode="direct",
            sinkhorn_reg=0.01, verbose=False)
        A_acc += out["A"]

    return A_acc / len(runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_dir", required=True)
    p.add_argument("--out",     required=True)
    args = p.parse_args()

    runs, cluster_ids, tp_list = load_scenario(args.sim_dir)
    tp_active = [tp for tp in tp_list if tp > 0]
    print(f"GeneTrajectory worker: {len(runs)} runs, {len(tp_active)} active timepoints", flush=True)

    A = compute_lagged_genetraj(runs, tp_active)
    np.save(args.out, A)
    print(f"Saved GeneTrajectory result to {args.out}", flush=True)


if __name__ == "__main__":
    main()
