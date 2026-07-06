"""
simulator.py
------------
Generate single-cell RNA-seq data using HARISSA's stochastic PDMP simulator,
given optimised parameters from HarissaParameterOptimizer.

The simulator:
1. Creates a HARISSA NetworkModel with the optimised basal and inter values
2. Assigns cells to timepoints along specified trajectories
3. Runs the PDMP simulation per cell
4. Returns a data matrix in HARISSA format + metadata

Output format (HARISSA convention)
-----------------------------------
data matrix: (C+1, G+2)
  - Row 0: header [_, 0, 1, 2, ..., G]
  - Rows 1..: [timepoint, stimulus_count, gene1_count, ..., geneG_count]
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from harissa import NetworkModel


@dataclass
class SimulationScenario:
    """
    Defines which cells are generated at which timepoints,
    and what the stimulus level is at each timepoint.

    timepoints : list of (float) simulation times
    n_cells_per_tp : number of cells per timepoint
    stimulus_at_t0 : stimulus expression level injected at t=0
                     (HARISSA convention: stimulus is gene 0 with
                      high count = 100 when ON)
    burnin : burnin time passed to model.simulate()
    """
    timepoints: List[float]
    n_cells_per_tp: int = 100
    stimulus_at_t0: float = 100.0
    burnin: float = 5.0


def build_harissa_model(
    G: int,
    result: Dict,
) -> NetworkModel:
    """
    Build a HARISSA NetworkModel from optimised parameters.

    Parameters
    ----------
    G : number of genes (excluding stimulus)
    result : dict from HarissaParameterOptimizer.optimise()
             keys: 'basal', 'inter', 'a', 'd'

    Returns
    -------
    model : NetworkModel ready to simulate
    """
    model = NetworkModel(G)

    # Set kinetic parameters
    a = result["a"]   # (3, G+1)
    d = result["d"]   # (2, G+1)
    model.a[:] = a
    model.d[:] = d

    # Set network parameters
    model.basal[:] = result["basal"]    # (G+1,)
    model.inter[:] = result["inter"]    # (G+1, G+1)

    return model


def simulate_dataset(
    model: NetworkModel,
    scenario: SimulationScenario,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run HARISSA PDMP simulation for all cells.

    Returns
    -------
    data : (C+1, G+2) int array in HARISSA format
    """
    rng = np.random.default_rng(seed)
    G = model.inter.shape[0] - 1  # number of real genes
    T = scenario.timepoints
    n_tp = len(T)
    n_cells_per_tp = scenario.n_cells_per_tp
    C = n_tp * n_cells_per_tp

    # Build time array for all cells
    cell_times = np.repeat(T, n_cells_per_tp).astype(float)

    # Header row
    data = np.zeros((C + 1, G + 2), dtype=int)
    data[0, 1:] = np.arange(G + 1)  # gene indices: 0=stimulus, 1..G=genes

    # Stimulus: ON whenever timepoint > 0
    data[1:, 0] = cell_times.astype(int)
    data[1:, 1] = (scenario.stimulus_at_t0 * (cell_times > 0)).astype(int)

    # Simulate each cell
    for c in range(C):
        t = cell_times[c]
        if verbose and (c % max(1, C // 10) == 0):
            print(f"  Simulating cell {c+1}/{C} (t={t:.1f})")

        sim = model.simulate(t, burnin=scenario.burnin)
        # sim.m[-1] shape: (G,) — mRNA counts for genes 1..G
        data[c + 1, 2:] = rng.poisson(sim.m[-1])

    return data


def simulate_multiple_runs(
    G: int,
    result: Dict,
    scenario: SimulationScenario,
    n_runs: int = 5,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[np.ndarray]:
    """
    Run multiple independent simulation replicates.

    Returns
    -------
    list of data matrices, one per run
    """
    datasets = []
    base_seed = seed if seed is not None else 0

    for r in range(n_runs):
        if verbose:
            print(f"\n--- Run {r+1}/{n_runs} ---")
        model = build_harissa_model(G, result)
        data = simulate_dataset(model, scenario, seed=base_seed + r, verbose=verbose)
        datasets.append(data)

    return datasets


def extract_expression_matrix(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract clean expression matrix from HARISSA data format.

    Returns
    -------
    X : (C, G) float array — mRNA counts, genes 1..G
    times : (C,) float array — timepoint per cell
    """
    times = data[1:, 0].astype(float)
    X = data[1:, 2:].astype(float)
    return X, times


def compute_correlation_by_timepoint(
    data: np.ndarray, grn_cluster_ids: np.ndarray
) -> Dict:
    """
    Compute intra-cluster and inter-cluster correlation at each timepoint.
    Useful for verifying that simulated data respects the GRN cluster structure.

    Returns
    -------
    dict with keys = timepoints, values = dict with 'intra' and 'inter' correlations
    """
    X, times = extract_expression_matrix(data)
    G = X.shape[1]
    unique_times = np.unique(times)
    n_clusters = len(np.unique(grn_cluster_ids))
    results = {}

    for t in unique_times:
        mask = times == t
        Xt = X[mask]
        if Xt.shape[0] < 3:
            continue

        # Pearson correlation matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(Xt.T)  # (G, G)
            corr = np.nan_to_num(corr)

        intra, inter = [], []
        for i in range(G):
            for j in range(i + 1, G):
                if grn_cluster_ids[i] == grn_cluster_ids[j]:
                    intra.append(corr[i, j])
                else:
                    inter.append(corr[i, j])

        results[t] = {
            "intra_mean": np.mean(intra) if intra else 0.0,
            "inter_mean": np.mean(inter) if inter else 0.0,
            "intra_std": np.std(intra) if intra else 0.0,
            "inter_std": np.std(inter) if inter else 0.0,
        }

    return results


if __name__ == "__main__":
    # Quick test using manually defined parameters (no optimizer needed)
    from grn_generator import generate_grn, make_sign_matrix
    import numpy as np

    np.random.seed(42)
    G = 8

    # Fake result dict (would normally come from optimizer)
    a = np.zeros((3, G + 1), dtype=np.float32)
    a[1] = 2.0; a[2] = 0.02
    d = np.zeros((2, G + 1), dtype=np.float32)
    d[0] = 0.2; d[1] = 0.04

    result = {
        "basal": np.array([-5.0] * (G + 1)),
        "inter": np.zeros((G + 1, G + 1)),
        "a": a,
        "d": d,
    }
    result["basal"][0] = 0.0
    result["inter"][0, 1] = 10.0   # stimulus -> gene 1
    result["inter"][1, 1] = 10.0   # gene 1 self-activates

    scenario = SimulationScenario(
        timepoints=[0, 10, 20, 40],
        n_cells_per_tp=50,
        burnin=5.0,
    )

    model = build_harissa_model(G, result)
    data = simulate_dataset(model, scenario, seed=0, verbose=True)
    X, times = extract_expression_matrix(data)
    print(f"\nData shape: {X.shape}")
    print(f"Mean expression per gene: {X.mean(axis=0).round(2)}")
