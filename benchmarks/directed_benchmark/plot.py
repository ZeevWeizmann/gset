import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_edge_cluster_matrix(M_dir, M_sym, out_path, M_ot=None):
    """Edge count heatmap per cluster pair: KL directed, OT directed, symmetric k-NN."""
    panels = [
        (M_dir, "Lagged KL divergence"),
        (M_sym, "Symmetric cosine"),
    ]
    if M_ot is not None:
        panels.insert(1, (M_ot, "Lagged OT residual"))

    n_cls = M_dir.shape[0]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (M, title) in zip(axes, panels):
        im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Target cluster j")
        ax.set_ylabel("Source cluster i")
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=9)

    plt.suptitle("Causal flow between gene clusters", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")
