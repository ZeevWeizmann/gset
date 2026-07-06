import matplotlib.pyplot as plt

CLR = {"KL": "#2a9d8f", "OT": "#e63946", "GW": "#6a4c93",
       "MaxRank(KL,GW)": "#e76f51", "MaxRank(KL,GW,OT)": "#f4a261"}
STYLES = {"KL": ("o-", 1.5), "OT": ("s-", 1.5), "GW": ("^-", 1.5),
          "MaxRank(KL,GW)": ("D--", 2.0), "MaxRank(KL,GW,OT)": ("P--", 2.4)}


def plot_results(results, scenarios, dr_list, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for ax, (tag, title, *_) in zip(axes, scenarios):
        for name, vals in results[tag].items():
            mk, lw = STYLES[name]
            ax.plot(dr_list, vals, mk, lw=lw, label=name, color=CLR[name])
        ax.axhline(0.5, ls=":", color="#aaa")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Dropout rate")
        ax.set_ylim(0.1, 1.05)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(frameon=False, fontsize=7)
    plt.suptitle("Three-mode NB benchmark", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
