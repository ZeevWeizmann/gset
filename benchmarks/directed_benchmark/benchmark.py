"""
benchmark.py — directed (causal direction) benchmark for GSET_2.

Compares methods on how well they recover the causal order of cluster activation
(e.g. C0→C1→C2 in cascade).

Methods:
  1. Lagged KL  (GCN) — directed lagged cosine on KL vectors  [GSET]
  2. Lagged OT  (GCN) — directed lagged cosine on OT residuals [GSET]
  3. OTVelo           — FGW-based gene velocity baseline
  4. Lagged GeneTrajectory — temporal metacell OT baseline
  5. Symmetric KL     — undirected baseline (existing GSET delta_KL)

Metric: fraction of inter-cluster pairs (i∈Ca, j∈Ca+1) where
        score(i→j) > score(j→i)  (sign-based, all pairs).

Usage
-----
  cd /Users/zeev/Documents/GSET_2
  KMP_DUPLICATE_LIB_OK=TRUE python benchmarks/directed_benchmark/benchmark.py \
      --sim_dir simulation/cascade
"""

import sys
import argparse
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.directed_benchmark.data import (
    load_scenario, train_gcn, compute_kl_vecs, compute_ot_vecs)
from benchmarks.directed_benchmark.directed_graph import (
    lagged_cosine_matrix, symmetric_cosine_graph)


# ─────────────────────────────────────────────────────────────────────────────
# OTVelo — runs in a subprocess to avoid OpenMP conflict with torch on macOS
# ─────────────────────────────────────────────────────────────────────────────

def compute_otvelo_score(sim_dir, eps_samp=1e-2, alpha=0.5, stimulation=True):
    """
    Runs otvelo_worker.py as a subprocess (no torch imported there),
    then loads the result. Avoids OpenMP segfault between pot and torch.
    """
    worker = Path(__file__).parent / "otvelo_worker.py"
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        out_path = f.name

    cmd = [
        sys.executable, str(worker),
        "--sim_dir", str(sim_dir),
        "--out",     out_path,
        "--eps_samp", str(eps_samp),
        "--alpha",    str(alpha),
    ]
    if stimulation:
        cmd.append("--stimulation")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"otvelo_worker.py failed with code {result.returncode}")

    C = np.load(out_path)
    Path(out_path).unlink(missing_ok=True)
    return C


# ─────────────────────────────────────────────────────────────────────────────
# Lagged GeneTrajectory
# ─────────────────────────────────────────────────────────────────────────────

def compute_lagged_genetraj(sim_dir):
    """Runs in a subprocess to avoid OpenMP conflict with torch on macOS."""
    worker = Path(__file__).parent / "genetraj_worker.py"
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        out_path = f.name

    result = subprocess.run(
        [sys.executable, str(worker), "--sim_dir", str(sim_dir), "--out", out_path],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"genetraj_worker.py failed with code {result.returncode}")

    A = np.load(out_path)
    Path(out_path).unlink(missing_ok=True)
    return A


# ─────────────────────────────────────────────────────────────────────────────
# Lagged OT cosine
# ─────────────────────────────────────────────────────────────────────────────

def lagged_cosine_ot(ot_vecs):
    def row_norm(R):
        return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-10)
    norm_vecs = [row_norm(R) for R in ot_vecs]
    n_genes   = ot_vecs[0].shape[0]
    C = np.zeros((n_genes, n_genes))
    for t in range(len(ot_vecs) - 1):
        C += norm_vecs[t] @ norm_vecs[t + 1].T
    return C / (len(ot_vecs) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def direction_accuracy_sign(C, cluster_ids):
    """
    For each consecutive cluster pair (a, a+1):
      fraction of (i∈Ca, j∈Ca+1) pairs where C[i,j] > C[j,i].
    """
    K = len(np.unique(cluster_ids))
    results = {}
    for a in range(K - 1):
        b = a + 1
        idx_a = np.where(cluster_ids == a)[0]
        idx_b = np.where(cluster_ids == b)[0]
        fwd = rev = 0
        for i in idx_a:
            for j in idx_b:
                if   C[i, j] > C[j, i]: fwd += 1
                elif C[j, i] > C[i, j]: rev += 1
        total = fwd + rev
        results[(a, b)] = dict(fwd=fwd, rev=rev,
                               acc=fwd / total if total > 0 else 0.5,
                               total=total)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results, methods, pairs, out_path):
    colors = {
        "Lagged KL (GCN)":      "#d62728",
        "Lagged OT (GCN)":      "#1f77b4",
        "OTVelo":               "#2ca02c",
        "Lagged GeneTrajectory":"#ff7f0e",
        "Symmetric KL":         "#7f7f7f",
    }

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]

    for idx, (a, b) in enumerate(pairs):
        ax    = axes[idx]
        names = list(methods.keys())
        accs  = [all_results[n][(a, b)]["acc"] for n in names]
        bars  = ax.bar(names, accs,
                       color=[colors.get(n, "#aaaaaa") for n in names],
                       alpha=0.85)
        ax.axhline(0.5, color="black", lw=1, linestyle=":", label="random")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Direction accuracy C{a}→C{b}")
        ax.set_ylabel("Fraction correct (sign-based, all pairs)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                    f"{acc:.2f}", ha="center", va="bottom", fontsize=9)

    K = max(b for _, b in pairs) + 1
    transitions = "→".join(f"C{k}" for k in range(K))
    fig.suptitle(f"Causal direction recovery — {transitions}", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_directed_benchmark(sim_dir: str, out_dir: str,
                           gcn_epochs: int = 500, gcn_d_out: int = 16,
                           skip_otvelo: bool = False,
                           skip_genetraj: bool = False):
    sim_dir = Path(sim_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading simulation from {sim_dir} ...")
    runs, cluster_ids, tp_list = load_scenario(str(sim_dir))
    tp_active = [tp for tp in tp_list if tp > 0]
    K      = len(np.unique(cluster_ids))
    pairs  = [(a, a + 1) for a in range(K - 1)]
    print(f"  {len(runs)} runs, {cluster_ids.shape[0]} genes, "
          f"K={K}, timepoints={tp_list}")

    # ── GCN ──────────────────────────────────────────────────────────────────
    print("\nTraining GCN ...")
    gcn = train_gcn(runs, tp_active, n_epochs=gcn_epochs, d_out=gcn_d_out)

    # ── KL / OT vectors ──────────────────────────────────────────────────────
    print("\nComputing KL vectors ...")
    kl_vecs = compute_kl_vecs(runs, tp_active, model=gcn)

    print("Computing OT residual vectors ...")
    ot_vecs = compute_ot_vecs(runs, tp_active, model=gcn)

    # ── Methods ──────────────────────────────────────────────────────────────
    methods = {}
    methods["Lagged KL (GCN)"] = lagged_cosine_matrix(kl_vecs)
    methods["Lagged OT (GCN)"] = lagged_cosine_ot(ot_vecs)

    if not skip_otvelo:
        print("Computing OTVelo ...")
        try:
            methods["OTVelo"] = compute_otvelo_score(sim_dir)
        except Exception as e:
            print(f"  OTVelo failed: {e} — skipping")

    if not skip_genetraj:
        print("Computing Lagged GeneTrajectory ...")
        try:
            methods["Lagged GeneTrajectory"] = compute_lagged_genetraj(sim_dir)
        except Exception as e:
            print(f"  GeneTrajectory failed: {e} — skipping")

    methods["Symmetric KL"] = symmetric_cosine_graph(kl_vecs)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n=== Direction accuracy (sign-based, all inter-cluster pairs) ===")
    all_results = {}
    for name, C in methods.items():
        res = direction_accuracy_sign(C, cluster_ids)
        all_results[name] = res
        parts = "  ".join(f"C{a}→C{b}: {r['acc']:.2f} ({r['total']} pairs)"
                          for (a, b), r in res.items())
        print(f"  {name:25s}  {parts}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    scenario_name = sim_dir.name
    out_path = out_dir / f"directed_benchmark_{scenario_name}.png"
    plot_results(all_results, methods, pairs, out_path)
    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="Directed benchmark for GSET_2")
    p.add_argument("--sim_dir",       type=str, default="simulation/cascade",
                   help="Simulation directory (e.g. simulation/cascade)")
    p.add_argument("--out_dir",       type=str,
                   default="figures",
                   help="Output directory for figures")
    p.add_argument("--gcn_epochs",    type=int,   default=500)
    p.add_argument("--gcn_d_out",     type=int,   default=16)
    p.add_argument("--skip_otvelo",   action="store_true")
    p.add_argument("--skip_genetraj", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_directed_benchmark(
        sim_dir=args.sim_dir,
        out_dir=args.out_dir,
        gcn_epochs=args.gcn_epochs,
        gcn_d_out=args.gcn_d_out,
        skip_otvelo=args.skip_otvelo,
        skip_genetraj=args.skip_genetraj,
    )
