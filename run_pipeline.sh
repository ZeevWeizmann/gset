#!/bin/bash
set -e
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$(dirname "$0")"


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1: NB SIMULATIONS
# Naive negative-binomial baselines to understand how GSET scores behave
# under controlled statistical conditions (binary and three-mode expressions).
# ═══════════════════════════════════════════════════════════════════════════════

python benchmarks/nb_binary/run.py
python benchmarks/nb_three_modes/run.py


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2: VECTORIAL EMBEDDINGS
# Validates that OT residual and KL vectors encode gene programme identity
# via cosine similarity, without supervision.
# Two scenarios: cascade (C0→C1→C2, positional signal) and switch (rewiring).
# Three vectors per gene: naive displacement, delta_OT, delta_KL.
# ═══════════════════════════════════════════════════════════════════════════════

# Step 1: generate simulations (run once)
# python simulation/pipeline.py --scenario cascade --n_genes 24 --n_clusters 3 --n_runs 5 --seed 42 --rho_in 0.5 --rho_out 0.1 --out_dir simulation/block2_cascade
# python simulation/pipeline.py --scenario switch  --n_genes 24 --n_clusters 3 --n_runs 5 --seed 42 --rho_in 0.5 --rho_out 0.1 --out_dir simulation/block2_switch

# Step 2: run benchmark
python benchmarks/vectorial_embeddings/run.py \
    --cascade_dir simulation/block2_cascade \
    --switch_dir  simulation/block2_switch
python benchmarks/vectorial_embeddings/plot.py \
    --cascade_dir simulation/block2_cascade \
    --switch_dir  simulation/block2_switch


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3: GENE PROGRAMME RECOVERY
# Evaluates how well GSET vectors (KL, OT, z_global, z_concat) identify
# gene programmes across biologically realistic HARISSA scenarios.
# ═══════════════════════════════════════════════════════════════════════════════

# Step 1: generate simulations (run once)
# python benchmarks/gene_programme_recovery/pipeline.py --scenario switch         --n_genes 50  --n_clusters 2 --n_runs 3 --seed 42 --skip-umap --rho_out 0.1 --out_dir simulation/block3_switch
# python benchmarks/gene_programme_recovery/pipeline.py --scenario cascade        --n_genes 150 --n_clusters 3 --n_runs 3 --seed 42 --skip-umap --rho_out 0.0 --out_dir simulation/block3_5_cascade
# python benchmarks/gene_programme_recovery/pipeline.py --scenario shared_targets --n_genes 50  --n_clusters 2 --n_runs 3 --seed 42 --skip-umap --rho_out 0.1 --out_dir simulation/block3_shared_targets
# python benchmarks/gene_programme_recovery/pipeline.py --scenario parallel_sync  --n_genes 50  --n_clusters 2 --n_runs 3 --seed 42 --skip-umap --rho_out 0.1 --out_dir simulation/block3_parallel_async
# PARALLEL_SYNC_SYM=1 python benchmarks/gene_programme_recovery/pipeline.py --scenario parallel_sync --n_genes 50 --n_clusters 2 --n_runs 3 --seed 42 --skip-umap --rho_out 0.1 --out_dir simulation/block3_parallel_sync
# python benchmarks/gene_programme_recovery/pipeline.py --scenario early_response --n_genes 50  --n_clusters 2 --n_runs 3 --seed 42 --skip-umap --rho_out 0.1 --out_dir simulation/block3_early_response

# Step 2: run benchmark
python benchmarks/gene_programme_recovery/benchmark.py \
    --sim_dir simulation \
    --scenarios switch cascade shared_targets parallel_async parallel_sync early_response \
    --gcn_epochs 300 --gcn_seed 1


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4: DIRECTED BENCHMARK
# Evaluates how well GSET recovers the causal direction of programme activation
# (e.g. C0→C1→C2 in cascade) compared to OTVelo and symmetric baselines.
# Also tests robustness of direction recovery under dropout noise.
# ═══════════════════════════════════════════════════════════════════════════════

# Generate cascade simulation (run once)
# python simulation/pipeline.py --n_genes 24 --n_clusters 3 --scenario cascade --rho_out 0.0 --seed 42 --out_dir simulation/block4_cascade

python benchmarks/directed_benchmark/graph_direction.py --sim_dir simulation/block4_cascade --threshold 0.4 --seed 10
python benchmarks/directed_benchmark/robustness.py --sim_dir simulation/block4_cascade
python benchmarks/directed_benchmark/benchmark.py --sim_dir simulation/block4_cascade


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5: CARDAMOM SUBNETWORK BENCHMARK
# Validates GSET's causal subnetwork selection using CardamomOT on a cascade
# simulation (G=150). Two validations:
#   (1) Intrinsic: recall of selected genes vs stimulus-rooted propagation
#       subgraph from ground truth theta → figures/subgraph_recall.png
#   (2) CardamomOT: edge coherence of inferred GRN vs theta, two conditions:
#       (A) GSET-selected genes only
#       (B) Full gene set → restrict to same genes (control)
#       → figures/edge_coherence.png
# ═══════════════════════════════════════════════════════════════════════════════

python benchmarks/cardamom_benchmark/benchmark.py \
    --sim_dir simulation/block3_5_cascade \
    --budget 50 \
    --threshold 0.15 \
    --seed 5
