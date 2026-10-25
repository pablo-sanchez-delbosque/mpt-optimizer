import numpy as np


def simulate_portfolios(mu, cov, n_sim=10000, seed=42):
    """Genera n_sim portafolios aleatorios (long-only, suman 1).

    Devuelve retornos, volatilidades, sharpes (con rf=0) y pesos.
    """
    n = len(mu)
    rng = np.random.default_rng(seed)

    rets = np.zeros(n_sim)
    vols = np.zeros(n_sim)
    pesos = np.zeros((n_sim, n))

    for i in range(n_sim):
        w = rng.random(n)
        w = w / w.sum()
        pesos[i] = w
        rets[i] = w @ mu
        vols[i] = np.sqrt(w @ cov @ w)

    return rets, vols, rets / vols, pesos