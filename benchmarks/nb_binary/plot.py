import matplotlib.pyplot as plt


def plot_results(results, scenarios, dr_list, out_path):
    clr = {"KL": "#2a9d8f", "OT": "#e63946", "Max rank": "#e76f51"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, (tag, title, gt, mu_t) in zip(axes, scenarios):
        for name, vals in results[tag].items():
            ax.plot(dr_list, vals,
                    "o-" if name != "Max rank" else "D--",
                    lw=1.5 if name != "Max rank" else 2.2,
                    label=name, color=clr[name])
        ax.axhline(0.5, ls=":", color="#aaa")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Dropout rate")
        ax.set_ylim(0.1, 1.05)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(frameon=False, fontsize=8)
    plt.suptitle("NB binary: KL wins on rewiring, OT wins on positional deviation", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
