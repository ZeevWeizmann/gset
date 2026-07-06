"""
NB binary benchmark: two-mode Negative Binomial simulation.
Usage: python run.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'gset'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

from core import GATEncoder, train_gat, get_embedding, kl_score, ot_score, max_rank
from sim import sample_nb, add_dropout, mu_base, make_scenarios, N_GENES, N_CELLS
from plot import plot_results

BETA     = 6.0
THRESH   = 0.01
D_HID    = 64
D_OUT    = 16
N_EPOCHS = 300
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def run_scores(model, X0, X1):
    E0 = get_embedding(model, X0, beta=BETA, thresh=THRESH)
    E1 = get_embedding(model, X1, beta=BETA, thresh=THRESH)
    return kl_score(E0, E1), ot_score(E0, E1), max_rank(kl_score(E0, E1), ot_score(E0, E1))


def sweep(model, scenarios, dr_list, n_trials=10):
    results = {}
    for tag, title, gt, mu_t in scenarios:
        kl_a, ot_a, mx_a = [], [], []
        for dr in dr_list:
            kl_v, ot_v, mx_v = [], [], []
            for seed in range(n_trials):
                np.random.seed(seed * 100)
                X0 = add_dropout(sample_nb(mu_base, N_CELLS), dr)
                X1 = add_dropout(sample_nb(mu_t, N_CELLS), dr)
                skl, sot, smx = run_scores(model, X0, X1)
                kl_v.append(roc_auc_score(gt, skl))
                ot_v.append(roc_auc_score(gt, sot))
                mx_v.append(roc_auc_score(gt, smx))
            kl_a.append(np.mean(kl_v))
            ot_a.append(np.mean(ot_v))
            mx_a.append(np.mean(mx_v))
        results[tag] = {"KL": kl_a, "OT": ot_a, "Max rank": mx_a}
        print(f"{tag} done")
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios = make_scenarios()

    model = GATEncoder(n_genes=N_GENES, d_hid=D_HID, d_out=D_OUT)
    print("Training GAT...")
    train_gat(model, X_list=[], n_epochs=N_EPOCHS, n_train=30,
              sample_fn=lambda: sample_nb(mu_base, N_CELLS), beta=BETA, thresh=THRESH)

    dr_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = sweep(model, scenarios, dr_list)
    plot_results(results, scenarios, dr_list, f"{OUT_DIR}/nb_binary.png")


if __name__ == "__main__":
    main()
