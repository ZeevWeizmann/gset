"""
pipeline.py — GSET biological simulation pipeline (HARISSA-based)

Scenarios
---------
activation    : single cluster activates from rest (stimulus → C0)
switch        : bistable toggle; stimulus drives C1, mutual inhibition with C0
cascade       : feedforward relay with timescale separation (C0→C1→C2, co-activation)
switch_cascade: relay + inhibitory feedback (sequential exclusive activation)

Usage
-----
python pipeline.py --n_genes 20 --n_clusters 2 --scenario switch
python pipeline.py --n_genes 24 --n_clusters 3 --scenario cascade --rho_out 0.0
python pipeline.py --n_genes 24 --n_clusters 3 --scenario switch_cascade
python pipeline.py --help
"""

import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from grn_generator import generate_grn
from harissa_optimizer import HarissaParameterOptimizer, make_scenario
from simulator import (
    SimulationScenario, build_harissa_model,
    simulate_multiple_runs, extract_expression_matrix,
    compute_correlation_by_timepoint,
)

TIMEPOINTS = {
    "activation":    [0, 12, 24, 48, 72, 96],
    "switch":        [0, 12, 24, 48, 72, 96],
    "cascade":       [0, 12, 24, 48, 72, 96, 120, 144],
    "switch_cascade":[0, 12, 24, 48, 72, 96, 120, 144],
}

# Default optimiser settings per scenario (tuned for convergence speed)
SCENARIO_OPT = {
    "activation":    dict(T_cascade=200.0, n_steps_cascade=600),
    "switch":        dict(T_cascade=200.0, n_steps_cascade=600),
    "cascade":       dict(T_cascade=250.0, n_steps_cascade=750, cascade_weight=3.0),
    "switch_cascade":dict(T_cascade=250.0, n_steps_cascade=750, cascade_weight=4.0),
}


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_grn_matrix(grn, result, out_path):
    """
    Signed interaction matrix sorted by cluster.
    Red = activation (+), Blue = inhibition (−), White = absent.
    Black lines delimit cluster boundaries.
    """
    inter = result["inter"]          # (G+1, G+1)
    G     = grn.n_genes
    vmax  = np.percentile(np.abs(inter[inter != 0]), 95) if (inter != 0).any() else 1.0
    inter_vis = np.clip(inter, -vmax, vmax)

    # Reorder: stimulus row 0, then genes sorted by cluster
    gene_order = np.argsort(grn.cluster_ids)           # 0-indexed
    idx = np.concatenate([[0], gene_order + 1])        # HARISSA indices
    M   = inter_vis[np.ix_(idx, idx)]
    n   = len(idx)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.46)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    labels = ["Stim"] + [
        f"G{gene_order[g]+1}\n(C{grn.cluster_ids[gene_order[g]]})"
        for g in range(G)
    ]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=7,
                                                 rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Target", fontsize=9)
    ax.set_ylabel("Source", fontsize=9)
    ax.set_title("GRN interaction matrix  (red=activation, blue=inhibition)", fontsize=9)

    # Cluster boundary lines (offset by 1 for stimulus row)
    cum = 1
    for k in range(grn.n_clusters - 1):
        cum += int((grn.cluster_ids == k).sum())
        ax.axhline(cum - 0.5, color="k", lw=1.2)
        ax.axvline(cum - 0.5, color="k", lw=1.2)

    plt.colorbar(im, ax=ax, fraction=0.03, label="Strength")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  GRN matrix        → {out_path}")


def plot_ode_trajectory(optimizer, result, out_path):
    """
    Single ODE trajectory: resting state + stimulus ON → dynamics unfold.
    One panel only (no 'ODE from B' etc.).
    """
    times, traj = optimizer.simulate_from_resting(result, T=400.0, n_save=200)
    K      = optimizer.grn.n_clusters
    colors = plt.cm.tab10(np.linspace(0, 0.9, K))

    fig, ax = plt.subplots(figsize=(7, 4))
    for k in range(K):
        for i, g in enumerate(optimizer.grn.genes_per_cluster[k]):
            lbl = f"Cluster {k}" if i == 0 else None
            ax.plot(times, traj[:, g + 1], color=colors[k], alpha=0.7,
                    lw=1.5, label=lbl)

    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Time"); ax.set_ylabel("Protein level P ∈ [0,1]")
    ax.set_title("Deterministic ODE: resting state → stimulus ON")
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  ODE trajectory    → {out_path}")


def plot_loss_curve(loss_history, out_path):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.semilogy(loss_history, color="steelblue", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)"); ax.set_title("Optimisation loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Loss curve        → {out_path}")


def plot_expression_heatmap(data, grn, timepoints, out_path):
    X, times = extract_expression_matrix(data)
    G  = X.shape[1]
    tps = sorted(np.unique(times))
    mean_expr  = np.array([X[times == t].mean(axis=0) for t in tps])
    gene_order = np.argsort(grn.cluster_ids)

    fig, ax = plt.subplots(figsize=(max(6, G * 0.48), 4))
    im = ax.imshow(mean_expr[:, gene_order].T, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(G))
    ax.set_yticklabels(
        [f"G{gene_order[g]+1}(C{grn.cluster_ids[gene_order[g]]})" for g in range(G)],
        fontsize=7)
    ax.set_xticks(range(len(tps)))
    ax.set_xticklabels([f"t={int(t)}" for t in tps], fontsize=7, rotation=45)
    ax.set_title("Mean mRNA expression (genes × timepoints)")

    cum = 0
    for k in range(grn.n_clusters - 1):
        cum += int((grn.cluster_ids[gene_order] == k).sum())
        ax.axhline(cum - 0.5, color="k", lw=1.0)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Expression heatmap→ {out_path}")


def plot_correlation_structure(data, grn, out_path):
    corr  = compute_correlation_by_timepoint(data, grn.cluster_ids)
    tps   = sorted(corr.keys())
    intra = [corr[t]["intra_mean"] for t in tps]
    inter = [corr[t]["inter_mean"] for t in tps]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(tps, intra, "o-", label="Intra-cluster", color="steelblue", lw=2)
    ax.plot(tps, inter, "s--", label="Inter-cluster", color="salmon",    lw=2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Timepoint"); ax.set_ylabel("Mean Pearson r")
    ax.set_title("Co-expression structure over time"); ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Correlation       → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"GSET Pipeline  |  scenario={args.scenario}  "
          f"G={args.n_genes}  K={args.n_clusters}  seed={args.seed}")
    print(f"rho_in={args.rho_in}  rho_out={args.rho_out}")
    print("=" * 64)

    # [1] GRN ─────────────────────────────────────────────────────────────────
    print("\n[1] Generating GRN...")
    inter_sign = "negative" if args.rho_out > 0 else "none"
    grn = generate_grn(
        n_genes=args.n_genes, n_clusters=args.n_clusters,
        n_tfs_per_cluster=args.n_tfs_per_cluster,
        rho_in=args.rho_in, rho_out=args.rho_out,
        self_activation=True, inter_cluster_sign=inter_sign,
        seed=args.seed,
    )
    grn.summary()

    # [2] Scenario ─────────────────────────────────────────────────────────────
    print(f"\n[2] Configuring scenario '{args.scenario}'...")
    grn, signs, cell_states, transitions, kp = make_scenario(
        grn, args.scenario, seed=args.seed)

    np.save(out_dir / "inter_mask.npy",  grn.inter_mask)
    np.save(out_dir / "sign_matrix.npy", signs)
    np.save(out_dir / "cluster_ids.npy", grn.cluster_ids)
    np.save(out_dir / "tf_mask.npy",     grn.tf_mask)

    for s in cell_states:
        print(f"  Target state '{s.name}': clusters={s.active_clusters}")
    for tr in transitions:
        print(f"  Transition: '{tr.source.name}' → '{tr.target.name}'")

    # [3] Optimise ─────────────────────────────────────────────────────────────
    print(f"\n[3] Optimising ({args.n_epochs} epochs)...")
    scen_kw = SCENARIO_OPT.get(args.scenario, {})
    optimizer = HarissaParameterOptimizer(
        grn=grn, sign_matrix=signs,
        cell_states=cell_states, transitions=transitions,
        kinetic_params=kp, scenario=args.scenario,
    )
    result = optimizer.optimise(
        n_epochs=args.n_epochs, lr=args.lr,
        fp_weight=args.fp_weight, ctr_weight=args.ctr_weight,
        traj_weight=args.traj_weight, resting_weight=args.resting_weight,
        n_perturb=args.n_perturb, eps_perturb=args.eps_perturb,
        T_short=args.T_short, n_steps_short=args.n_steps_short,
        T_transit=args.T_transit, n_steps_traj=args.n_steps_traj,
        verbose=True, log_every=max(1, args.n_epochs // 10),
        **scen_kw,
    )
    optimizer.verify_fixed_points(result)

    np.save(out_dir / "basal_opt.npy",    result["basal"])
    np.save(out_dir / "inter_opt.npy",    result["inter"])
    np.save(out_dir / "kinetics_a.npy",   result["a"])
    np.save(out_dir / "kinetics_d.npy",   result["d"])
    np.save(out_dir / "loss_history.npy", np.array(result["loss_history"]))

    print("\n[3b] Diagnostic plots...")
    plot_grn_matrix(grn, result, out_dir / "grn_matrix.png")
    plot_ode_trajectory(optimizer, result, out_dir / "ode_trajectory.png")
    plot_loss_curve(result["loss_history"], out_dir / "loss_curve.png")

    # [4] Simulate ─────────────────────────────────────────────────────────────
    timepoints = TIMEPOINTS.get(args.scenario, [0, 24, 48, 96])
    print(f"\n[4] Simulating {args.n_runs}×{args.n_cells_per_tp} cells/tp "
          f"at t={timepoints}...")
    datasets = simulate_multiple_runs(
        G=grn.n_genes, result=result,
        scenario=SimulationScenario(
            timepoints=timepoints,
            n_cells_per_tp=args.n_cells_per_tp,
            burnin=args.burnin),
        n_runs=args.n_runs, seed=args.seed, verbose=True,
    )
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for r, data in enumerate(datasets):
        np.savetxt(data_dir / f"data_{r+1}.txt",
                   data.T, fmt="%d", delimiter="\t")
    print(f"  Saved {args.n_runs} runs → {data_dir}")

    # [5] Expression diagnostics ───────────────────────────────────────────────
    print("\n[5] Expression diagnostics...")
    data0 = datasets[0]
    plot_expression_heatmap(data0, grn, timepoints,
                            out_dir / "expression_heatmap.png")
    plot_correlation_structure(data0, grn,
                               out_dir / "correlation_structure.png")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"scenario": args.scenario, "n_genes": args.n_genes,
                   "n_clusters": args.n_clusters, "rho_in": args.rho_in,
                   "rho_out": args.rho_out, "timepoints": timepoints,
                   "seed": args.seed}, f, indent=2)

    print(f"\n✓ Done — all outputs in: {out_dir}/")
    return datasets, result, grn, optimizer


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="GSET biological simulation pipeline (HARISSA)")
    # GRN
    p.add_argument("--n_genes",            type=int,   default=20)
    p.add_argument("--n_clusters",         type=int,   default=2)
    p.add_argument("--n_tfs_per_cluster",  type=int,   default=2)
    p.add_argument("--rho_in",             type=float, default=0.5)
    p.add_argument("--rho_out",            type=float, default=0.1)
    # Scenario
    p.add_argument("--scenario", type=str, default="switch",
                   choices=["activation","switch","cascade","switch_cascade"])
    # Optimiser
    p.add_argument("--n_epochs",       type=int,   default=500)
    p.add_argument("--lr",             type=float, default=0.05)
    p.add_argument("--fp_weight",      type=float, default=2.0)
    p.add_argument("--ctr_weight",     type=float, default=3.0)
    p.add_argument("--traj_weight",    type=float, default=1.0)
    p.add_argument("--resting_weight", type=float, default=2.0)
    p.add_argument("--n_perturb",      type=int,   default=4)
    p.add_argument("--eps_perturb",    type=float, default=0.15)
    p.add_argument("--T_short",        type=float, default=12.0)
    p.add_argument("--n_steps_short",  type=int,   default=100)
    p.add_argument("--T_transit",      type=float, default=100.0)
    p.add_argument("--n_steps_traj",   type=int,   default=500)
    # Simulation
    p.add_argument("--n_cells_per_tp", type=int,   default=100)
    p.add_argument("--n_runs",         type=int,   default=5)
    p.add_argument("--burnin",         type=float, default=5.0)
    # General
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out_dir", type=str, default="output")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
