import numpy as np

N_GENES  = 40
N_CELLS  = 5000
K        = 4
D_OUT    = 16

MU_LOW  = 0.5
MU_MED  = 4.0
MU_HIGH = 8.0

NET_A = {0: list(range(0,  10)),
         1: list(range(10, 20)),
         2: list(range(20, 30)),
         3: list(range(30, 40))}

NET_B = {0: list(range(0,  5)) + list(range(20, 25)),
         1: list(range(5, 10)) + list(range(25, 30)),
         2: list(range(10,15)) + list(range(30, 35)),
         3: list(range(15,20)) + list(range(35, 40))}


def make_mu(network, mode_active, mode_inactive=MU_LOW):
    mu = np.full((K, N_GENES), mode_inactive)
    for k, genes in network.items():
        mu[k, genes] = mode_active
    return mu


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


mu_t0 = make_mu(NET_A, mode_active=MU_MED, mode_inactive=MU_LOW)
mu_t1 = make_mu(NET_B, mode_active=MU_HIGH, mode_inactive=MU_MED)


def make_scenarios():
    mu_partial = np.full((K, N_GENES), MU_LOW)
    mu_partial[0, list(range(0,  5)) + list(range(10, 15))] = MU_MED
    mu_partial[1, list(range(5, 10)) + list(range(15, 20))] = MU_MED
    mu_partial[2, list(range(20, 30))] = MU_MED
    mu_partial[3, list(range(30, 40))] = MU_MED

    mu_dev = mu_t0.copy()
    mu_dev[0, list(range(0, 10))] = MU_HIGH

    mu_mix = mu_partial.copy()
    mu_mix[2, list(range(20, 30))] = MU_HIGH

    np.random.seed(7)
    ROT = np.linalg.qr(np.random.randn(D_OUT, D_OUT))[0].astype(np.float32)

    gt1 = np.zeros(N_GENES); gt1[list(range(0, 20))] = 1
    gt2 = np.zeros(N_GENES); gt2[list(range(0, 10))] = 1
    gt4 = np.zeros(N_GENES); gt4[list(range(0, 30))] = 1

    return [
        ("T1", "(a) Network-switch rewiring", gt1,        mu_t0, mu_t1,     False),
        ("T2", "(b) Positional deviation",    gt2,        mu_t0, mu_dev,    False),
        ("T3", "(c) Embedding rotation",      gt1.copy(), mu_t0, mu_t1,     ROT),
        ("T4", "(d) Mixed",                   gt4,        mu_t0, mu_mix,    False),
    ]
