"""
NB three-modes benchmark: three-mode Negative Binomial simulation.
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
import torch
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

from core import GATEncoder, train_gat, get_embedding, kl_score, ot_score, gw_score, max_rank
from sim import sample_nb, add_dropout, mu_t0, mu_t1, make_scenarios, N_GENES, N_CELLS
from plot import plot_results

BETA     = 6
THRESH   = 0.01
D_HID    = 64
D_OUT    = 16
N_EPOCHS = 300
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def sweep(model, scenarios, dr_list, n_trials=10):
    results = {}
    for tag, title, gt, mu0, mu1_, rot in scenarios:
        kl_a, ot_a, gw_a, mx_a, mx3_a = [], [], [], [], []
        for dr in dr_list:
            kl_v, ot_v, gw_v, mx_v, mx3_v = [], [], [], [], []
            for seed in range(n_trials):
                np.random.seed(seed * 100)
                X0 = add_dropout(sample_nb(mu0,  N_CELLS), dr)
                X1 = add_dropout(sample_nb(mu1_, N_CELLS), dr)
                E0 = get_embedding(model, X0, beta=BETA, thresh=THRESH)
                E1 = get_embedding(model, X1, beta=BETA, thresh=THRESH)
                if rot is not False:
                    E1 = E1 @ rot
                skl = kl_score(E0, E1)
                sot = ot_score(E0, E1)
                sgw = gw_score(E0, E1)
                kl_v.append(roc_auc_score(gt, skl))
                ot_v.append(roc_auc_score(gt, sot))
                gw_v.append(roc_auc_score(gt, sgw))
                mx_v.append(roc_auc_score(gt, max_rank(skl, sgw)))
                mx3_v.append(roc_auc_score(gt, max_rank(skl, sgw, sot)))
            kl_a.append(np.mean(kl_v)); ot_a.append(np.mean(ot_v))
            gw_a.append(np.mean(gw_v)); mx_a.append(np.mean(mx_v))
            mx3_a.append(np.mean(mx3_v))
        results[tag] = {"KL": kl_a, "OT": ot_a, "GW": gw_a,
                        "MaxRank(KL,GW)": mx_a, "MaxRank(KL,GW,OT)": mx3_a}
        print(f"{tag} done")
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios = make_scenarios()

    np.random.seed(42); torch.manual_seed(42)
    model = GATEncoder(n_genes=N_GENES, d_hid=D_HID, d_out=D_OUT)
    print("Training GAT...")
    def sample_fn():
        mu = [mu_t0, mu_t1][np.random.randint(2)]
        return sample_nb(mu, N_CELLS)

    train_gat(model, X_list=[], n_epochs=N_EPOCHS, n_train=30,
              sample_fn=sample_fn, beta=BETA, thresh=THRESH)

    dr_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = sweep(model, scenarios, dr_list)
    plot_results(results, scenarios, dr_list, f"{OUT_DIR}/nb_three_modes.png")


if __name__ == "__main__":
    main()
