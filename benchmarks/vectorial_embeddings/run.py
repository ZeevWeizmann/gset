"""
benchmarks/vectorial_embeddings/run.py
---------------------------------------
Train GAT, compute OT/KL/Naive vectors, save to results/.

Usage
-----
  cd /Users/zeev/Documents/GSET_2
  python benchmarks/vectorial_embeddings/run.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import argparse
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from benchmarks.vectorial_embeddings.utils import load_sim, train_gat, compute_vectors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cascade_dir", default="simulation/block2_cascade")
    p.add_argument("--switch_dir",  default="simulation/block2_switch")
    p.add_argument("--gcn_epochs",  type=int, default=300)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def run_scenario(sim_dir: Path, n_epochs: int, seed: int, out_dir: Path):
    print(f"\n[{sim_dir.name}] Loading simulation...")
    runs, cluster_ids, meta = load_sim(sim_dir)
    print(f"  {len(runs)} runs, {len(cluster_ids)} genes")

    print(f"[{sim_dir.name}] Training GAT ({n_epochs} epochs)...")
    model = train_gat(runs, n_genes_with_stim=runs[0][0].shape[1],
                      n_epochs=n_epochs, seed=seed)

    print(f"[{sim_dir.name}] Computing vectors...")
    X0, XT = runs[-1]
    vectors, E0, ET = compute_vectors(model, X0, XT)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = meta["scenario"]
    np.savez(
        out_dir / f"vectors_{tag}.npz",
        cluster_ids=cluster_ids,
        scenario=np.array(tag),
        **{k.replace(" ", "_"): v for k, v in vectors.items()},
        X0=X0, XT=XT,
    )
    print(f"  → {out_dir / f'vectors_{tag}.npz'}")
    return model, runs, cluster_ids, meta, X0, XT


def main():
    args    = parse_args()
    out_dir = _ROOT / "benchmarks" / "vectorial_embeddings" / "results"

    run_scenario(_ROOT / args.cascade_dir, args.gcn_epochs, args.seed, out_dir)
    run_scenario(_ROOT / args.switch_dir,  args.gcn_epochs, args.seed, out_dir)

    print("\n✓ Done — results saved to", out_dir)


if __name__ == "__main__":
    main()
