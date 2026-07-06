"""
grn_generator.py
----------------
Generate modular Gene Regulatory Networks (GRNs) using a signed directed
Stochastic Block Model (SBM), with support for two complex scenarios:

Standard scenarios
------------------
- K clusters of genes with dense intra-cluster edges (co-activation)
- Sparse inter-cluster edges (parameterised by rho_out)
- TF master regulators per cluster

Complex scenarios (requiring temporal info to recover gene programmes)
----------------------------------------------------------------------
shared_targets : Some genes receive inputs from TFs of TWO different clusters.
    When cluster C0 is active, shared genes correlate with C0.
    When cluster C1 takes over (stimulus-driven switch), shared genes correlate with C1.
    SAME GRN throughout — only the regulatory CONTEXT changes.
    → static embedding sees shared genes in an ambiguous intermediate position
    → δ_KL high (neighbourhood probability shift), δ_OT high (positional shift)

parallel_sync  : Two programmes activate in PARALLEL with different kinetics.
    Hub genes receive inputs from TFs of BOTH programmes.
    Early: fast programme dominates → hubs correlate with fast programme.
    Late:  fast programme near saturation (low variance), slow programme climbing
           (high variance) → hubs now correlate with slow programme.
    → static embedding: hubs appear ambiguous (receive both inputs)
    → δ_KL high (neighbourhood changes as kinetics diverge)
    Mechanism is variance-driven, not mean-driven. GRN is FIXED.

Output
------
GRNStructure dataclass with:
  inter_mask  : (G+1, G+1) bool — non-zero interactions (row=source, col=target)
  cluster_ids : (G,) int — cluster membership (0-indexed, excluding stimulus)
  tf_mask     : (G,) bool — True if gene is a TF
  special_ids : dict — {'shared': [...], 'hubs': [...]} gene indices (0-indexed)
  n_genes, n_clusters, n_tfs_per_cluster, rho_in, rho_out
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class GRNStructure:
    inter_mask:        np.ndarray    # (G+1, G+1) bool
    cluster_ids:       np.ndarray    # (G,) int, 0-indexed
    tf_mask:           np.ndarray    # (G,) bool
    n_genes:           int
    n_clusters:        int
    n_tfs_per_cluster: int
    rho_in:            float
    rho_out:           float
    special_ids:       Dict[str, List[int]] = field(default_factory=dict)

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
            tfs  = self.tfs_per_cluster[k]
            tgts = [g for g in self.genes_per_cluster[k] if not self.tf_mask[g]]
            print(f"  Cluster {k}: {len(self.genes_per_cluster[k])} genes "
                  f"(TFs: {[int(g)+1 for g in tfs]}, "
                  f"targets: {[int(g)+1 for g in tgts]})")
        if self.special_ids:
            for kind, ids in self.special_ids.items():
                print(f"  Special [{kind}]: HARISSA genes {[i+1 for i in ids]}")
        n_edges = self.inter_mask[1:, 1:].sum()
        print(f"  Edges (gene-gene): {n_edges} "
              f"(intra: {self._count_intra()}, inter: {self._count_inter()})")

    def _count_intra(self):
        return sum(
            self.inter_mask[np.ix_(g+1, g+1)].sum()
            for g in [self.genes_per_cluster[k] for k in range(self.n_clusters)]
        )

    def _count_inter(self):
        return int(self.inter_mask[1:, 1:].sum()) - self._count_intra()


# ═════════════════════════════════════════════════════════════════════════════
# Standard GRN generator (SBM)
# ═════════════════════════════════════════════════════════════════════════════

def generate_grn(
    n_genes:           int   = 20,
    n_clusters:        int   = 3,
    n_tfs_per_cluster: int   = 2,
    rho_in:            float = 0.6,
    rho_out:           float = 0.05,
    self_activation:   bool  = True,
    inter_cluster_sign: str  = "none",   # "none" | "negative" | "random"
    seed: Optional[int]      = None,
) -> GRNStructure:
    rng = np.random.default_rng(seed)
    assert n_genes >= n_clusters * n_tfs_per_cluster

    # Assign genes to clusters
    base, rem = divmod(n_genes, n_clusters)
    cluster_ids = np.zeros(n_genes, dtype=int)
    tf_mask     = np.zeros(n_genes, dtype=bool)
    idx = 0
    for k in range(n_clusters):
        sz = base + (1 if k < rem else 0)
        cluster_ids[idx:idx+sz] = k
        tf_mask[idx:idx+n_tfs_per_cluster] = True
        idx += sz

    G = n_genes
    inter_mask = np.zeros((G+1, G+1), dtype=bool)

    # Stimulus -> TFs of cluster 0 (default; make_scenario will redirect)
    for g in np.where((cluster_ids == 0) & tf_mask)[0]:
        inter_mask[0, g+1] = True

    for i in range(G):
        for j in range(G):
            if i == j:
                if self_activation and tf_mask[i]:
                    inter_mask[i+1, i+1] = True
                continue
            same = cluster_ids[i] == cluster_ids[j]
            p = rho_in if same else rho_out
            if inter_cluster_sign == "none" and not same:
                continue
            if rng.random() < p:
                if not same and tf_mask[j] and not tf_mask[i]:
                    continue
                inter_mask[i+1, j+1] = True

    return GRNStructure(
        inter_mask=inter_mask, cluster_ids=cluster_ids, tf_mask=tf_mask,
        n_genes=n_genes, n_clusters=n_clusters,
        n_tfs_per_cluster=n_tfs_per_cluster,
        rho_in=rho_in, rho_out=rho_out,
    )


def make_sign_matrix(
    grn:                     GRNStructure,
    intra_positive_fraction: float = 1.0,
    inter_sign:              str   = "negative",
    seed: Optional[int]            = None,
) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    G     = grn.n_genes
    signs = np.zeros((G+1, G+1), dtype=int)
    for i in range(G+1):
        for j in range(G+1):
            if not grn.inter_mask[i, j]:
                continue
            if i == 0:             # stimulus always activates
                signs[i, j] = 1; continue
            gi, gj = i-1, j-1
            if gi == gj:           # self-loop always activating
                signs[i, j] = 1; continue
            same = grn.cluster_ids[gi] == grn.cluster_ids[gj]
            if same:
                signs[i, j] = 1 if rng.random() < intra_positive_fraction else -1
            else:
                signs[i, j] = -1 if inter_sign == "negative" else rng.choice([-1,1])
    return signs


# ═════════════════════════════════════════════════════════════════════════════
# Shared-targets GRN  (complex scenario 4 — rewiring contextuel)
# ═════════════════════════════════════════════════════════════════════════════

def generate_shared_targets_grn(
    n_genes_per_cluster: int   = 6,
    n_clusters:          int   = 2,
    n_tfs_per_cluster:   int   = 2,
    n_shared:            int   = 3,
    rho_in:              float = 0.5,
    self_activation:     bool  = True,
    seed: Optional[int]        = None,
) -> GRNStructure:
    """
    Build a GRN with n_shared 'shared target' genes that receive activating
    inputs from TFs of BOTH cluster 0 and cluster 1.

    Gene layout (all 0-indexed, HARISSA adds +1):
      [0 .. n_per_clust-1]          : cluster 0 (first n_tfs = TFs)
      [n_per_clust .. 2*n-1]        : cluster 1
      [2*n .. 2*n + n_shared - 1]   : shared targets (cluster_id = -1, shown as K)

    The shared targets are assigned cluster_id = n_clusters (a sentinel value)
    so they appear as a separate group in visualisations.

    Switch scenario wiring (handled by make_scenario 'shared_targets'):
      - Stimulus -> TF_C1 (drives the switch from C0 to C1)
      - C0 self-sustains without stimulus (via self-activation)
      - C1 activates with stimulus, inhibits C0 TFs
      - Shared genes: input from TF_C0 (+) AND TF_C1 (+)
        -> correlate with C0 when C0 dominant, with C1 when C1 dominant
    """
    rng = np.random.default_rng(seed)
    n_cluster_genes = n_genes_per_cluster * n_clusters
    G = n_cluster_genes + n_shared

    cluster_ids = np.full(G, n_clusters, dtype=int)  # sentinel for shared
    tf_mask     = np.zeros(G, dtype=bool)

    idx = 0
    for k in range(n_clusters):
        sz = n_genes_per_cluster
        cluster_ids[idx:idx+sz] = k
        tf_mask[idx:idx+n_tfs_per_cluster] = True
        idx += sz
    # shared genes: indices [n_cluster_genes .. G-1], cluster_id = n_clusters

    shared_indices = list(range(n_cluster_genes, G))  # 0-indexed

    inter_mask = np.zeros((G+1, G+1), dtype=bool)

    # Intra-cluster edges (SBM)
    for i in range(n_cluster_genes):
        for j in range(n_cluster_genes):
            if cluster_ids[i] != cluster_ids[j]:
                continue
            if i == j:
                if self_activation and tf_mask[i]:
                    inter_mask[i+1, i+1] = True
                continue
            if rng.random() < rho_in:
                inter_mask[i+1, j+1] = True

    # Shared targets: receive input from ALL TFs of C0 and C1
    for k in range(n_clusters):
        for tf_g in np.where((cluster_ids == k) & tf_mask)[0]:
            for sh_g in shared_indices:
                inter_mask[tf_g+1, sh_g+1] = True

    # Shared targets have NO self-loop and NO output edges
    # (pure sensors of the regulatory context)

    return GRNStructure(
        inter_mask=inter_mask, cluster_ids=cluster_ids, tf_mask=tf_mask,
        n_genes=G, n_clusters=n_clusters,
        n_tfs_per_cluster=n_tfs_per_cluster,
        rho_in=rho_in, rho_out=0.0,
        special_ids={"shared": shared_indices},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Parallel-sync GRN  (complex scenario 3 — synchronisation)
# ═════════════════════════════════════════════════════════════════════════════

def generate_parallel_sync_grn(
    n_genes_per_cluster: int   = 6,
    n_clusters:          int   = 2,
    n_tfs_per_cluster:   int   = 2,
    n_hubs:              int   = 3,
    rho_in:              float = 0.5,
    self_activation:     bool  = True,
    seed: Optional[int]        = None,
) -> GRNStructure:
    """
    Build a GRN with n_hubs 'hub' genes that receive activating inputs from
    TFs of ALL clusters simultaneously.

    Both programmes are driven by the SAME stimulus (no switch).
    Kinetic timescale separation (D1 per cluster) makes C0 fast and C1 slow.

    Hub co-expression pattern:
      - Early (C0 climbing, C1 near zero): hub correlates with C0
      - Late  (C0 saturated, C1 climbing): hub correlates with C1
      GRN is FIXED; only variance structure changes with time.

    Gene layout:
      [0 .. n_per_clust-1]          : cluster 0
      [n_per_clust .. 2*n-1]        : cluster 1
      [2*n .. 2*n + n_hubs - 1]     : hub genes (cluster_id = n_clusters)
    """
    rng = np.random.default_rng(seed)
    n_cluster_genes = n_genes_per_cluster * n_clusters
    G = n_cluster_genes + n_hubs

    cluster_ids = np.full(G, n_clusters, dtype=int)
    tf_mask     = np.zeros(G, dtype=bool)

    idx = 0
    for k in range(n_clusters):
        sz = n_genes_per_cluster
        cluster_ids[idx:idx+sz] = k
        tf_mask[idx:idx+n_tfs_per_cluster] = True
        idx += sz

    hub_indices = list(range(n_cluster_genes, G))

    inter_mask = np.zeros((G+1, G+1), dtype=bool)

    # Intra-cluster SBM edges
    for i in range(n_cluster_genes):
        for j in range(n_cluster_genes):
            if cluster_ids[i] != cluster_ids[j]:
                continue
            if i == j:
                if self_activation and tf_mask[i]:
                    inter_mask[i+1, i+1] = True
                continue
            if rng.random() < rho_in:
                inter_mask[i+1, j+1] = True

    # Hub genes: receive input from ALL TFs of ALL clusters
    for k in range(n_clusters):
        for tf_g in np.where((cluster_ids == k) & tf_mask)[0]:
            for h_g in hub_indices:
                inter_mask[tf_g+1, h_g+1] = True

    # Hubs have no output (pure sensors)

    return GRNStructure(
        inter_mask=inter_mask, cluster_ids=cluster_ids, tf_mask=tf_mask,
        n_genes=G, n_clusters=n_clusters,
        n_tfs_per_cluster=n_tfs_per_cluster,
        rho_in=rho_in, rho_out=0.0,
        special_ids={"hubs": hub_indices},
    )


if __name__ == "__main__":
    print("=== Standard GRN ===")
    grn = generate_grn(n_genes=12, n_clusters=2, n_tfs_per_cluster=2,
                       rho_in=0.5, rho_out=0.1, seed=42)
    grn.summary()

    print("\n=== Shared-targets GRN ===")
    grn_sh = generate_shared_targets_grn(
        n_genes_per_cluster=5, n_clusters=2, n_tfs_per_cluster=2,
        n_shared=3, rho_in=0.5, seed=42)
    grn_sh.summary()

    print("\n=== Parallel-sync GRN ===")
    grn_ps = generate_parallel_sync_grn(
        n_genes_per_cluster=5, n_clusters=2, n_tfs_per_cluster=2,
        n_hubs=3, rho_in=0.5, seed=42)
    grn_ps.summary()
