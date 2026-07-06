"""
evaluate.py
-----------
Edge coherence metrics between CardamomOT-inferred and true HARISSA GRN.

Key insight (Elias Ventre): validate by edge coherence, NOT fit quality.
A badly-chosen gene subset still reproduces training data but with spurious edges
compensating for missing drivers. Edge coherence separates "good selection"
from "CardamomOT's own error".

Metrics
-------
- AUROC on edge recovery (binary sign classification)
- Sign agreement (fraction of same-sign edges)
- Pearson correlation of edge weights
- Stimulus-connectivity score (are stimulus-driven edges recovered?)
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Optional


def restrict_inter(
    inter: np.ndarray,
    gene_subset: List[int],
    stimulus_row: int = 0,
) -> np.ndarray:
    """
    Restrict a (G+1, G+1) interaction matrix to a gene subset.

    Parameters
    ----------
    inter        : full interaction matrix (HARISSA format: row/col 0 = stimulus)
    gene_subset  : list of gene indices (0-indexed, NOT including stimulus)
    stimulus_row : row/col index of stimulus in inter (default 0)

    Returns
    -------
    (|subset|+1, |subset|+1) restricted matrix with stimulus as first row/col
    """
    # HARISSA indices: gene g → row/col (g+1)
    harissa_idx = [stimulus_row] + [g + 1 for g in gene_subset]
    return inter[np.ix_(harissa_idx, harissa_idx)]


def edge_coherence(
    inter_inferred: np.ndarray,
    inter_true: np.ndarray,
    gene_subset: Optional[List[int]] = None,
) -> dict:
    """
    Compute edge coherence between inferred and true interaction matrices.

    Compares only the gene→gene block (rows/cols 1:), excluding stimulus row 0
    since CardamomOT doesn't model the stimulus node.

    Parameters
    ----------
    inter_inferred : CardamomOT output (G+1, G+1) — restricted to gene subset
    inter_true     : HARISSA inter_opt (G+1, G+1) — restricted to same subset
    gene_subset    : if given, restrict both matrices using restrict_inter first

    Returns
    -------
    dict with auroc, sign_agreement, pearson_r, n_genes, n_true_edges
    """
    if gene_subset is not None:
        inter_inferred = restrict_inter(inter_inferred, gene_subset)
        inter_true     = restrict_inter(inter_true, gene_subset)

    G_plus1 = inter_true.shape[0]

    # Compare only gene→gene block (rows/cols 1:) — exclude stimulus row 0
    # because CardamomOT doesn't model the stimulus node
    w_true = inter_true[1:, 1:].ravel()
    w_inf  = inter_inferred[1:, 1:].ravel()

    # Remove diagonal (self-loops)
    G = G_plus1 - 1
    diag_mask = ~np.eye(G, dtype=bool).ravel()
    w_true = w_true[diag_mask]
    w_inf  = w_inf[diag_mask]

    # AUROC: binary "is this a true non-zero interaction?" vs inferred weight
    labels = (np.abs(w_true) > 1e-6).astype(int)
    scores = np.abs(w_inf)
    n_pos  = labels.sum()
    auroc  = roc_auc_score(labels, scores) if 0 < n_pos < len(labels) else 0.5

    # Sign agreement (only on edges that exist in ground truth)
    true_edges = np.abs(w_true) > 1e-6
    if true_edges.sum() > 0:
        sign_agree = (np.sign(w_true[true_edges]) == np.sign(w_inf[true_edges])).mean()
    else:
        sign_agree = float("nan")

    return {
        "auroc":          auroc,
        "sign_agreement": sign_agree,
        "n_genes":        G,
        "n_true_edges":   int(n_pos),
    }


def compare_selections(
    results_by_condition: dict,
    condition_labels: Optional[List[str]] = None,
) -> None:
    """Pretty-print comparison table across conditions."""
    if condition_labels is None:
        condition_labels = list(results_by_condition.keys())
    print("\n" + "=" * 62)
    print(f"{'Condition':<25} {'AUROC':>7} {'SignAgr':>8} {'nGenes':>7}")
    print("-" * 52)
    for label in condition_labels:
        m = results_by_condition.get(label, {})
        auroc = f"{m.get('auroc', float('nan')):.3f}"
        sign  = f"{m.get('sign_agreement', float('nan')):.3f}"
        ng    = f"{m.get('n_genes', '?')}"
        print(f"{label:<25} {auroc:>7} {sign:>8} {ng:>7}")
    print("=" * 52)
