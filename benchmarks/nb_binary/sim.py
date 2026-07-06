import numpy as np

N_GENES = 40
N_CELLS = 5000
K       = 4

MU_HIGH = 8.0
MU_LOW  = 0.5

mu_base = np.ones((K, N_GENES)) * MU_LOW
mu_base[0,  0: 7] = MU_HIGH   # comp 0a → genes 0-6
mu_base[1,  7:14] = MU_HIGH   # comp 0b → genes 7-13
mu_base[2, 14:27] = MU_HIGH   # comp 1  → genes 14-26
mu_base[3, 27:40] = MU_HIGH   # comp 2  → genes 27-39


def sample_nb(mu, n_cells, r=2.0):
    K_, G = mu.shape
    asgn = np.random.choice(K_, size=n_cells, p=np.ones(K_) / K_)
    X = np.zeros((n_cells, G))
    for k in range(K_):
        m = asgn == k
        if m.sum() == 0: continue
        p = r / (r + mu[k])
        X[m] = np.random.negative_binomial(r, p, size=(m.sum(), G)).astype(float)
    return X


def add_dropout(X, rate):
    return X * np.random.binomial(1, 1 - rate, X.shape)


def make_scenarios():
    """Return list of (tag, title, gt, mu_t) for each scenario."""
    gt1 = np.zeros(N_GENES); gt1[list(range(0, 14))] = 1
    mu1 = mu_base.copy()
    mu1[0, 0:7] = MU_LOW;  mu1[0, 7:14] = MU_HIGH
    mu1[1, 7:14] = MU_LOW; mu1[1, 0:7] = MU_HIGH

    gt2 = np.zeros(N_GENES); gt2[list(range(14, 22))] = 1
    mu2 = mu_base.copy() + 1.0
    for g in range(14, 22):
        mu2[2, g] = mu_base[2, g] + 6.0

    gt3 = np.zeros(N_GENES)
    gt3[list(range(0, 4)) + list(range(7, 11)) + list(range(14, 18))] = 1
    mu3 = mu_base.copy() + 0.5
    mu3[0, 0:4] = MU_LOW;  mu3[0, 7:11] = MU_HIGH
    mu3[1, 7:11] = MU_LOW; mu3[1, 0:4] = MU_HIGH
    for g in range(14, 18):
        mu3[2, g] = mu_base[2, g] + 5.5

    return [
        ("T1", "(a) Rotational rewiring", gt1, mu1),
        ("T2", "(b) Positional deviation", gt2, mu2),
        ("T3", "(c) Mixed", gt3, mu3),
    ]
