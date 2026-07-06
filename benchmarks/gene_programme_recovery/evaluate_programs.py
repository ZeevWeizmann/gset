"""
evaluate_programs.py
--------------------
Évalue la récupération des programmes géniques à partir des matrices W
produites par gene_ot_distances.py, et génère des figures.

Prend en entrée le dossier d'un scénario de simulation qui contient :
  - W_classic.npy        matrice de distances gène×gène (mode classique)
  - W_temporal.npy       idem (mode temporel)
  - gene_names.npy       noms des gènes (produits par gene_ot_distances.py)
  - cluster_ids.npy      vérité terrain (produit par pipeline.py)
  - special_ids.json     gènes spéciaux (optionnel)

Figures produites :
  - umap_classic.png / umap_temporal.png
      UMAP sur la matrice W, coloré par cluster prédit et par cluster réel
  - ari_comparison.png
      ARI et NMI classic vs temporal pour tous les scénarios
  - cosine_similarity.png
      Matrice de similarité cosinus sur l'embedding MDS, triée par cluster réel

Usage :
  # Évaluer un seul scénario
  python evaluate_programs.py --sim_dir simulations/switch

  # Évaluer tous les scénarios et produire le résumé comparatif
  python evaluate_programs.py --sim_dirs simulations/switch simulations/cascade \\
      simulations/shared_targets --out_dir figures/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def complete_W(W: np.ndarray) -> np.ndarray:
    """Complétion géodésique de la matrice de distances sparse."""
    graph = csr_matrix(W)
    W_geo = shortest_path(graph, method="D", directed=False)
    fm = W_geo[np.isfinite(W_geo)].max()
    W_geo[~np.isfinite(W_geo)] = fm * 1.5
    np.fill_diagonal(W_geo, 0.0)
    return W_geo.astype(np.float32)


def mds_embed(W_geo: np.ndarray, n_components: int = 10) -> np.ndarray:
    n = min(n_components, W_geo.shape[0] - 1)
    mds = MDS(n_components=n, dissimilarity="precomputed",
              random_state=42, normalized_stress="auto")
    return mds.fit_transform(W_geo)


def umap_embed(W_geo: np.ndarray) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_components=2, metric="precomputed",
                         random_state=42, n_neighbors=min(15, W_geo.shape[0]-1),
                         min_dist=0.1, verbose=False)
    return reducer.fit_transform(W_geo)


def cluster_from_W(W_geo: np.ndarray, n_clusters: int) -> np.ndarray:
    """Clustering agglomératif (ward) sur embedding MDS."""
    coords = mds_embed(W_geo, n_components=min(10, W_geo.shape[0]-1))
    cl = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return cl.fit_predict(coords)


def align_labels(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Ré-étiquette pred pour maximiser l'accord avec true
    (résolution hongroise simple par vote majoritaire).
    """
    from scipy.optimize import linear_sum_assignment
    K_pred = pred.max() + 1
    K_true = true.max() + 1
    K = max(K_pred, K_true)
    cost = np.zeros((K, K), dtype=int)
    for p, t in zip(pred, true):
        cost[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(p, p) for p in pred])


def cosine_sim_matrix(coords: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(coords, axis=1, keepdims=True) + 1e-10
    V = coords / norms
    return V @ V.T


# ─────────────────────────────────────────────────────────────────────────────
# Chargement d'un scénario
# ─────────────────────────────────────────────────────────────────────────────

def load_scenario(sim_dir: str) -> dict:
    sim_dir = Path(sim_dir)
    out = {"name": sim_dir.name, "sim_dir": sim_dir}

    # Matrices W
    for mode in ("classic", "temporal"):
        p = sim_dir / f"W_{mode}.npy"
        if p.exists():
            out[f"W_{mode}"] = np.load(str(p))

    # Noms de gènes (produits par gene_ot_distances.py dans le dossier de sortie OT)
    # Cherche dans sim_dir ou dans sim_dir/gene_ot/
    for candidate in [sim_dir / "gene_names.npy",
                       sim_dir / "gene_ot" / "gene_names.npy"]:
        if candidate.exists():
            out["gene_names"] = np.load(str(candidate), allow_pickle=True)
            break

    # Vérité terrain
    p = sim_dir / "cluster_ids.npy"
    if p.exists():
        out["cluster_ids"] = np.load(str(p))

    # Gènes spéciaux
    p = sim_dir / "special_ids.json"
    if p.exists():
        with open(p) as f:
            out["special_ids"] = json.load(f)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figures par scénario
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_comparison(sc: dict, out_dir: Path):
    """
    UMAP sur W_classic et W_temporal, coloré par :
      - cluster prédit (agglomératif)
      - cluster réel (ground truth)
    """
    cluster_ids = sc.get("cluster_ids")
    gene_names  = sc.get("gene_names")
    modes       = [m for m in ("classic", "temporal") if f"W_{m}" in sc]
    if not modes:
        return

    # Exclure les gènes sentinelles (cluster_id == max = gènes spéciaux)
    if cluster_ids is not None:
        K_real = int(cluster_ids.max())
        mask   = cluster_ids < K_real      # gènes réguliers
        n_clusters = K_real
    else:
        mask = np.ones(sc[f"W_{modes[0]}"].shape[0], dtype=bool)
        n_clusters = 3

    n_cols  = len(modes)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    colors_pred = plt.cm.tab10(np.linspace(0, 0.9, n_clusters))
    colors_true = plt.cm.Set2(np.linspace(0, 0.9, n_clusters))

    for col, mode in enumerate(modes):
        W      = sc[f"W_{mode}"]
        W_sub  = W[np.ix_(mask, mask)]
        W_geo  = complete_W(W_sub)
        coords = umap_embed(W_geo)

        pred_labels = cluster_from_W(W_geo, n_clusters)
        true_labels = cluster_ids[mask] if cluster_ids is not None else None

        # Row 0 : cluster prédit
        ax = axes[0, col]
        for k in range(n_clusters):
            m = pred_labels == k
            ax.scatter(coords[m, 0], coords[m, 1],
                       c=[colors_pred[k]], s=40, alpha=0.8, label=f"Pred {k}")
            if gene_names is not None:
                for gi in np.where(m)[0]:
                    ax.annotate(gene_names[mask][gi],
                                (coords[gi, 0], coords[gi, 1]),
                                fontsize=5, alpha=0.7)
        ax.set_title(f"{mode.capitalize()} — cluster prédit", fontsize=9)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 1 : cluster réel
        ax = axes[1, col]
        if true_labels is not None:
            for k in range(n_clusters):
                m = true_labels == k
                ax.scatter(coords[m, 0], coords[m, 1],
                           c=[colors_true[k]], s=40, alpha=0.8, label=f"True C{k}")
                if gene_names is not None:
                    for gi in np.where(m)[0]:
                        ax.annotate(gene_names[mask][gi],
                                    (coords[gi, 0], coords[gi, 1]),
                                    fontsize=5, alpha=0.7)
        ax.set_title(f"{mode.capitalize()} — cluster réel", fontsize=9)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"UMAP sur W — {sc['name']}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"umap_{sc['name']}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  UMAP           → {out_path}")


def plot_cosine_similarity(sc: dict, out_dir: Path):
    """
    Matrice de similarité cosinus sur l'embedding MDS,
    gènes triés par cluster réel, pour classic et temporal.
    """
    cluster_ids = sc.get("cluster_ids")
    modes       = [m for m in ("classic", "temporal") if f"W_{m}" in sc]
    if not modes or cluster_ids is None:
        return

    K_real = int(cluster_ids.max())
    mask   = cluster_ids < K_real
    ids_m  = cluster_ids[mask]
    order  = np.argsort(ids_m)

    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4.5))
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        W     = sc[f"W_{mode}"]
        W_sub = W[np.ix_(mask, mask)]
        W_geo = complete_W(W_sub)
        coords = mds_embed(W_geo, n_components=min(10, W_geo.shape[0]-1))
        C      = cosine_sim_matrix(coords)
        C_sort = C[np.ix_(order, order)]

        im = ax.imshow(C_sort, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"{mode.capitalize()}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Gène (trié par cluster réel)")
        ax.set_ylabel("Gène (trié par cluster réel)")
        ax.set_xticks([]); ax.set_yticks([])

        # Frontières de clusters
        ids_sorted = ids_m[order]
        cum = 0
        for k in range(K_real - 1):
            cum += int((ids_sorted == k).sum())
            ax.axhline(cum - 0.5, color="k", lw=1.0)
            ax.axvline(cum - 0.5, color="k", lw=1.0)

        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.suptitle(f"Cosine similarity (MDS embedding) — {sc['name']}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"cosine_{sc['name']}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Cosine sim     → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure de résumé multi-scénarios
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(sc: dict) -> dict:
    """ARI et NMI pour classic et temporal sur un scénario."""
    cluster_ids = sc.get("cluster_ids")
    if cluster_ids is None:
        return {}

    K_real = int(cluster_ids.max())
    mask   = cluster_ids < K_real
    true   = cluster_ids[mask]
    out    = {}

    for mode in ("classic", "temporal"):
        if f"W_{mode}" not in sc:
            continue
        W      = sc[f"W_{mode}"]
        W_sub  = W[np.ix_(mask, mask)]
        W_geo  = complete_W(W_sub)
        pred   = cluster_from_W(W_geo, n_clusters=K_real)
        out[mode] = {
            "ARI": adjusted_rand_score(true, pred),
            "NMI": normalized_mutual_info_score(true, pred),
        }
    return out


def plot_ari_summary(all_scenarios: list, out_dir: Path):
    """
    Barplot ARI et NMI pour tous les scénarios × {classic, temporal}.
    """
    names   = [sc["name"] for sc in all_scenarios]
    metrics = [compute_metrics(sc) for sc in all_scenarios]
    modes   = ["classic", "temporal"]
    colors  = {"classic": "steelblue", "temporal": "darkorange"}

    fig, axes = plt.subplots(1, 2, figsize=(max(7, len(names) * 1.8), 4.5))

    for ax, metric in zip(axes, ["ARI", "NMI"]):
        x = np.arange(len(names))
        width = 0.35
        for i, mode in enumerate(modes):
            vals = [m.get(mode, {}).get(metric, np.nan) for m in metrics]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=mode.capitalize(), color=colors[mode], alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — Classic vs Temporal")
        ax.axhline(1.0, color="gray", lw=0.6, ls="--")
        ax.legend(fontsize=8)

    plt.suptitle("Récupération des programmes géniques — Résumé",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "ari_summary.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ARI summary    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(sim_dirs: list, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [load_scenario(d) for d in sim_dirs]

    for sc in scenarios:
        has = [m for m in ("classic", "temporal") if f"W_{m}" in sc]
        if not has:
            print(f"[{sc['name']}] Pas de matrice W trouvée — skipping")
            print(f"  (Lance d'abord gene_ot_distances.py sur {sc['sim_dir']}/data.h5ad)")
            continue
        print(f"\n── {sc['name'].upper()} ──")
        plot_umap_comparison(sc, out_dir)
        plot_cosine_similarity(sc, out_dir)

    print("\n── Résumé global ──")
    plot_ari_summary(scenarios, out_dir)
    print(f"\n✓ Évaluation terminée → {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sim_dirs", type=str, nargs="+", required=True,
                   help="Dossiers de scénarios (contenant W_classic.npy etc.)")
    p.add_argument("--out_dir",  type=str, default="figures",
                   help="Dossier de sortie des figures")
    args = p.parse_args()
    run_evaluation(args.sim_dirs, args.out_dir)
