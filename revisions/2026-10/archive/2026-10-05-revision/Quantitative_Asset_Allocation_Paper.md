# Quantitative Asset Allocation: Corrected Demonstrations

Original project: Spark Hu, November 2025. Assisted technical revision: 5 October 2026. Draft for author review.

Two separate demonstrations are presented; no matched HRP-versus-MVO superiority claim is made.

## In-sample maximum-Sharpe allocation

Common prices: 2021-01-04 to 2024-12-31; 1004 return observations.

| Asset | Weight |
|---|---:|
| MSFT | 22.7169% |
| JPM | 38.5299% |
| TLT | 0.0000% |
| GLD | 26.9277% |
| BTC-USD | 11.8255% |

Estimated annualized arithmetic mean: 20.8298%; volatility: 17.0279%; daily-excess Sharpe: 0.9929. Historical daily fifth percentile: -1.6536%.

These statistics assume daily restoration of the fitted weights in the same sample and no costs. The quantile is not a future loss guarantee.

![Random feasible portfolios and correlations](results/Portfolio_Optimization_Model.png)

## Rolling HRP experiment

Investable assets: MSFT, GOOG, JPM, XLE and GLD. SPY is excluded from fitting. Equal weighting uses the same assets and schedule. Settings: 252-observation lookback, one-observation signal delay, single linkage, and 21-observation holding blocks with drift.

| Strategy | Cost bps | Compound growth (252/N) | Annual volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| HRP | 0 | 12.4102% | 12.2933% | 0.6942 | -17.8800% |
| Equal weight (same assets) | 0 | 21.6128% | 16.0147% | 1.0573 | -20.0583% |
| SPY buy and hold | 0 | 13.9531% | 16.4767% | 0.6373 | -24.4964% |
| HRP | 5 | 12.3354% | 12.2930% | 0.6888 | -17.9021% |
| Equal weight (same assets) | 5 | 21.5648% | 16.0142% | 1.0549 | -20.0708% |
| SPY buy and hold | 5 | 13.9388% | 16.4762% | 0.6365 | -24.4964% |

![Walk-forward wealth and drawdowns](results/Walk_Forward_Backtest.png)

The 5-bps case charges purchases and sales, including initial investment. Costs are illustrative. Growth uses 252/N annualization; daily excess returns use a constant effective annual cash return of 4%. Drawdown includes initial capital.

## Interpretation and limits

These selected assets and dates are an illustrative case. Historical data are freshly retrieved adjusted-price observations, not an original point-in-time execution feed. MVO's universe and fitting procedure differ from HRP's. No general superiority, future loss bound, statistical significance or completed covariance-shrinkage extension is claimed. See CHANGES.md and WALKTHROUGH.md for corrections and full explanations.

## Reproducibility

Retrieval: 2026-10-05T08:58:25.056430+00:00. Data hash: `0c386ae03a56b097c63930a040b43a49b8851948887a742274e32325dbf5a105`. Original commit: `ce87a9d13131d03699dd1b0834acd945c54bbb6d`. Random seed: 20261005. Exact versions and numerical outputs are recorded in results/summary.json and requirements-lock.txt.

## Method sources

- [López de Prado (2016), HRP](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [PyPortfolioOpt implementation](https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/hierarchical_portfolio.html)
- [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
- [yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)

The October correction and explanatory draft were prepared with Codex assistance for author review. The author should reproduce the calculations and supply their own interpretation before presenting this as evidence of their understanding.
