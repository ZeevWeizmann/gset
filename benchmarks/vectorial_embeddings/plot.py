"""
benchmarks/vectorial_embeddings/plot.py
----------------------------------------
Load saved vectors and generate figures.

Usage
-----
  cd /Users/zeev/Documents/GSET_2
  python benchmarks/vectorial_embeddings/plot.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from benchmarks.vectorial_embeddings.utils import (
    plot_cosine_matrices,
    plot_dropout_sweep,
    train_gat,
    load_sim,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="benchmarks/vectorial_embeddings/results")
    p.add_argument("--cascade_dir", default="simulation/block2_cascade")
    p.add_argument("--switch_dir",  default="simulation/block2_switch")
    p.add_argument("--gcn_epochs",  type=int, default=300)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--skip_dropout", action="store_true",
                   help="Skip dropout sweep (requires retraining GAT)")
    return p.parse_args()


def load_vectors(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    scenario = str(d["scenario"])
    cluster_ids = d["cluster_ids"]
    vectors = {
        "Naive":       d["Naive"],
        "OT residual": d["OT_residual"],
        "KL vector":   d["KL_vector"],
    }
    X0 = d["X0"]
    XT = d["XT"]
    return vectors, cluster_ids, scenario, X0, XT


def main():
    args    = parse_args()
    res_dir = _ROOT / args.results_dir
    fig_dir = _ROOT / "figures" / "block2_vectorial_embeddings"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for tag, sim_dir_arg in [("cascade", args.cascade_dir), ("switch", args.switch_dir)]:
        npz = res_dir / f"vectors_{tag}.npz"
        if not npz.exists():
            print(f"  [!] {npz} not found — run run.py first")
            continue

        vectors, cluster_ids, scenario, X0, XT = load_vectors(npz)
        print(f"\n[{tag}] Cosine similarity matrices...")
        plot_cosine_matrices(vectors, cluster_ids, scenario,
                             fig_dir / f"vectorial_{tag}.png")

        if not args.skip_dropout:
            print(f"[{tag}] Dropout sweep (retraining GAT)...")
            runs, _, _ = load_sim(_ROOT / sim_dir_arg)
            model = train_gat(runs, n_genes_with_stim=runs[0][0].shape[1],
                              n_epochs=args.gcn_epochs, seed=args.seed)
            plot_dropout_sweep(model, X0, XT, cluster_ids, scenario,
                               fig_dir / f"vectorial_{tag}_dropout.png",
                               seed=args.seed)

    print("\n✓ Done")


if __name__ == "__main__":
    main()
