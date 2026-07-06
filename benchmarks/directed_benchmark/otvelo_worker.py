"""
otvelo_worker.py — runs OTVelo in an isolated process (no torch) to avoid
OpenMP conflict between pot and torch on macOS.

Usage (called by benchmark.py via subprocess):
  python otvelo_worker.py --sim_dir <path> --out <path.npy>
               [--eps_samp 0.01] [--alpha 0.5] [--stimulation]
"""

import sys
import argparse
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
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


def compute_otvelo(runs, tp_list, eps_samp=1e-2, alpha=0.5, stimulation=True):
    import copy
    from utils_Velo import solve_prior, solve_velocities, OT_lagged_correlation

    n_genes = next(iter(runs[0].values())).shape[1]
    C_acc   = np.zeros((n_genes, n_genes))

    for idx, run in enumerate(runs):
        print(f"  OTVelo run {idx+1}/{len(runs)} ...", flush=True)
        counts_all = [run[tp].astype(np.float64).T for tp in tp_list]
        n_per_tp   = [ca.shape[1] for ca in counts_all]
        counts     = np.hstack(counts_all)
        labels     = np.zeros((1, sum(n_per_tp)))
        offset = 0
        for t, n in enumerate(n_per_tp):
            labels[0, offset:offset + n] = t
            offset += n

        Nt = len(tp_list)
        Ts, _ = solve_prior(counts, counts, Nt, labels,
                            eps_samp=eps_samp, alpha=alpha)
        _, velocities_all_signed = solve_velocities(
            counts_all, Ts, order=1, stimulation=stimulation)

        velocities_signed = np.hstack(velocities_all_signed)
        Tv, _ = OT_lagged_correlation(
            copy.deepcopy(velocities_all_signed),
            velocities_signed.copy(),
            Ts,
            stimulation=stimulation,
            elastic_Net=False,
            signed=True,
            return_slice=True,
        )
        C_acc += Tv

    return C_acc / len(runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_dir",    required=True)
    p.add_argument("--out",        required=True)
    p.add_argument("--eps_samp",   type=float, default=1e-2)
    p.add_argument("--alpha",      type=float, default=0.5)
    p.add_argument("--stimulation", action="store_true", default=True)
    args = p.parse_args()

    runs, cluster_ids, tp_list = load_scenario(args.sim_dir)
    tp_active = [tp for tp in tp_list if tp > 0]
    print(f"OTVelo worker: {len(runs)} runs, {len(tp_active)} active timepoints", flush=True)

    C = compute_otvelo(runs, tp_active,
                       eps_samp=args.eps_samp,
                       alpha=args.alpha,
                       stimulation=args.stimulation)
    np.save(args.out, C)
    print(f"Saved OTVelo result to {args.out}", flush=True)


if __name__ == "__main__":
    main()
