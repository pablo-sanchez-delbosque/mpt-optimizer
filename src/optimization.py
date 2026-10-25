import numpy as np
from scipy.optimize import minimize


def _neg_sharpe(w, mu, cov, rf):
    return -(w @ mu - rf) / np.sqrt(w @ cov @ w)


def _port_vol(w, cov):
    return np.sqrt(w @ cov @ w)


def _setup(n):
    w0 = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    return w0, bounds, cons


def max_sharpe(mu, cov, rf=0.0):
    """Pesos que maximizan el Sharpe (long-only, suman 1)."""
    w0, bounds, cons = _setup(len(mu))
    res = minimize(_neg_sharpe, w0, args=(mu, cov, rf),
                   method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"La optimización no convergió: {res.message}")
    return res.x


def min_volatility(mu, cov):
    """Pesos de mínima volatilidad (long-only, suman 1)."""
    w0, bounds, cons = _setup(len(mu))
    res = minimize(_port_vol, w0, args=(cov,),
                   method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"La optimización no convergió: {res.message}")
    return res.x


def efficient_frontier(mu, cov, n_points=50):
    """Devuelve (vols, rets) de la frontera eficiente."""
    n = len(mu)
    w0, bounds, _ = _setup(n)
    ret_min = min_volatility(mu, cov) @ mu
    vols_f, rets_f = [], []
    for target in np.linspace(ret_min, mu.max(), n_points):
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        )
        r = minimize(_port_vol, w0, args=(cov,),
                     method="SLSQP", bounds=bounds, constraints=cons)
        if r.success:
            vols_f.append(r.fun)
            rets_f.append(target)
    return np.array(vols_f), np.array(rets_f)