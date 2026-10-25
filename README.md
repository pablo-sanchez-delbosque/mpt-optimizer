# Markowitz Portfolio Optimizer

Mean-variance portfolio optimization on a multi-asset ETF universe, built in Python.
The project compares a Monte Carlo simulation of random portfolios against the exact
solution obtained with constrained optimization (`scipy.optimize`), and visualizes
the efficient frontier interactively with Plotly.

![Efficient Frontier](results/efficient_frontier.png)

## Theory

### Log returns

Daily log returns are computed from adjusted closing prices:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Log returns are additive over time, which makes annualization straightforward.
The portfolio return is approximated as a weighted sum of asset log returns,
an approximation that is negligible at daily frequency.

### Annualized estimators

With 252 trading days per year:

$$\mu = 252 \cdot \bar{r}_{\text{daily}}, \qquad \Sigma = 252 \cdot \text{Cov}(r_{\text{daily}})$$

### Portfolio return, risk and Sharpe ratio

For a weight vector $w$ with $\sum_i w_i = 1$:

$$\mu_p = w^\top \mu, \qquad \sigma_p = \sqrt{w^\top \Sigma w}, \qquad
\text{Sharpe} = \frac{\mu_p - r_f}{\sigma_p}$$

### Optimization problem

The maximum Sharpe portfolio solves

$$\max_{w} \; \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
\quad \text{s.t.} \quad \sum_i w_i = 1, \;\; w_i \ge 0$$

and the efficient frontier is obtained by minimizing $\sigma_p$ for each target return.
Since `scipy.optimize.minimize` only minimizes, the negative Sharpe ratio is used as objective (SLSQP method).