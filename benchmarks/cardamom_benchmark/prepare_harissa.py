"""
prepare_harissa.py
------------------
Convert HARISSA simulation output to CardamomOT project format.

Creates:
  <project_dir>/Data/data_train.h5ad   — AnnData with obs['time'], var['d0','d1']
  <project_dir>/Data/data_full.h5ad    — same (full = train for simulated data)

The AnnData uses gene-subset indexing if `gene_subset` is provided.
Kinetic rates d0 (mRNA degradation) and d1 (protein degradation) are taken
directly from HARISSA's kinetics_d.npy, bypassing get_kinetic_rates.py.

Usage
-----
  from prepare_harissa import prepare_cardamom_project
  prepare_cardamom_project(sim_dir, project_dir)
"""

import numpy as np
import anndata as ad
import pandas as pd
from pathlib import Path
from typing import List, Optional


def load_harissa_runs(sim_dir: Path):
    """
    Load all HARISSA data files from sim_dir/data/.

    Returns
    -------
    X      : (n_cells_total, n_genes) expression matrix (mRNA)
    times  : (n_cells_total,) timepoint labels (float for CardamomOT)
    run_ids: (n_cells_total,) run index per cell
    n_genes: int
    """
    data_dir = sim_dir / "data"
    all_X, all_t, all_r = [], [], []
    run_idx = 0
    for fp in sorted(data_dir.glob("data_*.txt")):
        raw   = np.loadtxt(fp, delimiter="\t")
        times = raw[0, 1:].astype(float)
        X     = raw[2:, 1:].T   # (n_cells, n_genes) — skip stimulus row + index col
        all_X.append(X)
        all_t.append(times)
        all_r.append(np.full(len(X), run_idx, dtype=int))
        run_idx += 1
    return (np.vstack(all_X), np.concatenate(all_t),
            np.concatenate(all_r), all_X[0].shape[1])


def prepare_cardamom_project(
    sim_dir: str,
    project_dir: str,
    gene_subset: Optional[List[int]] = None,
    split: str = "train",
) -> Path:
    """
    Convert HARISSA simulation output to a CardamomOT project directory.

    Parameters
    ----------
    sim_dir      : path to HARISSA simulation output (contains data/, inter_opt.npy, ...)
    project_dir  : path to create CardamomOT project
    gene_subset  : if given, only include these gene indices (0-indexed)
    split        : filename suffix for h5ad (default: 'train')

    Returns
    -------
    Path to project_dir
    """
    sim_dir     = Path(sim_dir)
    project_dir = Path(project_dir)
    data_out    = project_dir / "Data"
    data_out.mkdir(parents=True, exist_ok=True)

    # Load expression data
    X, times, run_ids, n_genes = load_harissa_runs(sim_dir)

    # Apply gene subset if requested
    if gene_subset is not None:
        X = X[:, gene_subset]
        subset_harissa = [g + 1 for g in gene_subset]   # HARISSA index (0 = stimulus)
    else:
        gene_subset     = list(range(n_genes))
        subset_harissa  = list(range(1, n_genes + 1))

    n_sel = len(gene_subset)

    # Load HARISSA kinetic parameters (kinetics_d.npy: shape (2, G+1))
    # Row 0 = mRNA degradation (d0), Row 1 = protein degradation (d1)
    kinetics_d = np.load(sim_dir / "kinetics_d.npy")   # (2, n_genes+1)
    d0 = kinetics_d[0, subset_harissa]   # per selected gene
    d1 = kinetics_d[1, subset_harissa]

    # Build AnnData
    gene_names = [f"Gene{g+1:03d}" for g in gene_subset]
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs["time"]       = times.astype(float)
    adata.obs["dataset_id"] = run_ids.astype(int)
    adata.obs_names         = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.var_names         = gene_names
    adata.var["d0"] = d0
    adata.var["d1"] = d1

    # Save
    h5ad_path = data_out / f"data_{split}.h5ad"
    h5ad_full = data_out / "data_full.h5ad"
    adata.write_h5ad(h5ad_path)
    adata.write_h5ad(h5ad_full)
    print(f"[prepare_harissa] Saved {adata.n_obs} cells × {n_sel} genes → {h5ad_path}")

    # Save gene mapping for later reconstruction
    np.save(project_dir / "gene_subset.npy", np.array(gene_subset))
    return project_dir
