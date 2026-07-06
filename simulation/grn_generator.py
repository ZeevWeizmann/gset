"""
grn_generator.py
----------------
Generate modular Gene Regulatory Networks (GRNs) using a signed directed
Stochastic Block Model (SBM).

Structure
---------
- K clusters (gene programmes)
- Each cluster has n_tfs_per_cluster TF genes (sources) and target genes
- Intra-cluster density rho_in  (strong co-regulation)
- Inter-cluster density rho_out (weak / absent cross-regulation)
- Signs: intra-cluster edges mostly positive (co-activation),
  inter-cluster edges can be positive or negative (parameterisable)
- The stimulus (gene 0 in HARISSA) drives TFs in the "active at t=0" cluster

Output
------
A GRNStructure dataclass containing:
  - inter_mask : (G+1, G+1) bool array  - which interactions are non-zero
  - cluster_ids : (G,) int array         - cluster membership per gene (1-indexed)
  - tf_mask : (G,) bool array            - which genes are TFs
  - n_genes : int                         - number of genes (excluding stimulus)
  - n_clusters : int
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GRNStructure:
    inter_mask: np.ndarray       # (G+1, G+1) bool, row=source, col=target
    cluster_ids: np.ndarray      # (G,) int, 0-indexed cluster per gene
    tf_mask: np.ndarray          # (G,) bool, True if gene is a TF
    n_genes: int
    n_clusters: int
    n_tfs_per_cluster: int
    rho_in: float
    rho_out: float

    @property
    def genes_per_cluster(self):
        return [np.where(self.cluster_ids == k)[0] for k in range(self.n_clusters)]

    @property
    def tfs_per_cluster(self):
        return [np.where((self.cluster_ids == k) & self.tf_mask)[0]
                for k in range(self.n_clusters)]

    def summary(self):
        print(f"GRN: {self.n_genes} genes, {self.n_clusters} clusters")
        for k in range(self.n_clusters):
            tfs = self.tfs_per_cluster[k]
            genes = self.genes_per_cluster[k]
            print(f"  Cluster {k}: {len(genes)} genes "
                  f"(TFs: {[int(g)+1 for g in tfs]}, "  # +1 for HARISSA indexing
                  f"targets: {[int(g)+1 for g in genes if not self.tf_mask[g]]})")
        n_edges = self.inter_mask[1:, 1:].sum()
        print(f"  Edges (gene-gene): {n_edges} "
              f"(intra: {self._count_intra()}, inter: {self._count_inter()})")

    def _count_intra(self):
        count = 0
        for k in range(self.n_clusters):
            idx = self.genes_per_cluster[k]
            count += self.inter_mask[np.ix_(idx+1, idx+1)].sum()
        return count

    def _count_inter(self):
        return self.inter_mask[1:, 1:].sum() - self._count_intra()


def generate_grn(
    n_genes: int = 20,
    n_clusters: int = 3,
    n_tfs_per_cluster: int = 2,
    rho_in: float = 0.7,
    rho_out: float = 0.05,
    self_activation: bool = True,
    inter_cluster_sign: str = "random",  # "random", "negative", "none"
    seed: Optional[int] = None,
) -> GRNStructure:
    """
    Generate a modular signed directed GRN.

    Parameters
    ----------
    n_genes : total number of genes (excluding HARISSA stimulus)
    n_clusters : number of gene programmes / clusters
    n_tfs_per_cluster : number of TF master regulators per cluster
    rho_in : edge probability within a cluster
    rho_out : edge probability between clusters
    self_activation : if True, TFs have self-activation edges
    inter_cluster_sign : sign of inter-cluster edges
        "none"     -> no inter-cluster edges (rho_out ignored)
        "negative" -> inter-cluster edges are always inhibitory
        "random"   -> inter-cluster edges are ±1 with equal probability
    seed : random seed

    Returns
    -------
    GRNStructure
    """
    rng = np.random.default_rng(seed)

    assert n_genes >= n_clusters * n_tfs_per_cluster, \
        "Not enough genes to assign TFs to all clusters"

    # --- Assign genes to clusters ---
    # First n_tfs_per_cluster genes in each cluster are TFs
    genes_per_cluster = n_genes // n_clusters
    remainder = n_genes % n_clusters

    cluster_ids = np.zeros(n_genes, dtype=int)
    tf_mask = np.zeros(n_genes, dtype=bool)
    idx = 0
    for k in range(n_clusters):
        size = genes_per_cluster + (1 if k < remainder else 0)
        cluster_ids[idx:idx + size] = k
        tf_mask[idx:idx + n_tfs_per_cluster] = True
        idx += size

    # --- Build (G+1)x(G+1) interaction mask (HARISSA convention) ---
    # Gene 0 = stimulus; genes 1..G = real genes
    G = n_genes
    inter_mask = np.zeros((G + 1, G + 1), dtype=bool)

    # Stimulus -> TFs of cluster 0 (these are the "initial" TFs)
    for g in np.where((cluster_ids == 0) & tf_mask)[0]:
        inter_mask[0, g + 1] = True  # +1 for HARISSA indexing

    # Gene-gene edges
    for i in range(G):
        for j in range(G):
            if i == j:
                # Self-activation for TFs
                if self_activation and tf_mask[i]:
                    inter_mask[i + 1, i + 1] = True
                continue
            same_cluster = cluster_ids[i] == cluster_ids[j]
            p = rho_in if same_cluster else rho_out
            if inter_cluster_sign == "none" and not same_cluster:
                continue
            if rng.random() < p:
                # Directed: TFs preferentially regulate targets
                # Within cluster: TF -> target (preferred), target -> target (ok)
                # We only restrict: targets don't regulate TFs of other clusters
                if not same_cluster and tf_mask[j] and not tf_mask[i]:
                    continue  # Non-TF doesn't regulate TF of another cluster
                inter_mask[i + 1, j + 1] = True

    return GRNStructure(
        inter_mask=inter_mask,
        cluster_ids=cluster_ids,
        tf_mask=tf_mask,
        n_genes=n_genes,
        n_clusters=n_clusters,
        n_tfs_per_cluster=n_tfs_per_cluster,
        rho_in=rho_in,
        rho_out=rho_out,
    )


def make_sign_matrix(
    grn: GRNStructure,
    intra_positive_fraction: float = 0.8,
    inter_sign: str = "negative",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Assign +1 / -1 signs to each edge in the GRN mask.

    Parameters
    ----------
    intra_positive_fraction : fraction of intra-cluster edges that are activating
    inter_sign : "negative" (all inter-cluster edges are inhibitory),
                 "random"   (random signs for inter-cluster edges)

    Returns
    -------
    sign_matrix : (G+1, G+1) int array with values in {-1, 0, +1}
    """
    rng = np.random.default_rng(seed)
    G = grn.n_genes
    signs = np.zeros((G + 1, G + 1), dtype=int)

    for i in range(G + 1):
        for j in range(G + 1):
            if not grn.inter_mask[i, j]:
                continue
            # Stimulus always activates
            if i == 0:
                signs[i, j] = 1
                continue
            gi, gj = i - 1, j - 1  # gene indices (0-indexed)
            # Self-activation
            if gi == gj:
                signs[i, j] = 1
                continue
            same_cluster = grn.cluster_ids[gi] == grn.cluster_ids[gj]
            if same_cluster:
                signs[i, j] = 1 if rng.random() < intra_positive_fraction else -1
            else:
                if inter_sign == "negative":
                    signs[i, j] = -1
                else:
                    signs[i, j] = rng.choice([-1, 1])

    return signs


if __name__ == "__main__":
    grn = generate_grn(
        n_genes=20, n_clusters=3, n_tfs_per_cluster=2,
        rho_in=0.6, rho_out=0.05, seed=42
    )
    grn.summary()
    signs = make_sign_matrix(grn, seed=42)
    print("Sign matrix (gene-gene block):\n", signs[1:, 1:])
