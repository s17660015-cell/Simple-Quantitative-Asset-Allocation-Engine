# Maximum-Sharpe allocation and walk-forward HRP

Spark Hu. Original project: November 2025. Historical calculations corrected on
5 October 2026; project documentation revised on 18 October 2026.

This project contains two historical allocation experiments. A long-only
maximum-Sharpe portfolio is fitted on its full sample. A separate rolling
hierarchical risk parity (HRP) allocation is evaluated against equal weighting
on the same assets, with SPY as a buy-and-hold market reference. The experiments
use different asset universes and fitting procedures, so they do not establish
a matched comparison of MVO and HRP.

The October code corrections, tests and technical documentation were prepared
with Codex assistance.

## Models and results

The maximum-Sharpe fit uses MSFT, JPM, TLT, GLD and BTC-USD, with adjusted
prices from 2021 through 2024. Complete common prices are aligned before
simple returns are calculated. The fitted allocation has an annualized
arithmetic mean of 20.83%, volatility of 17.03% and Sharpe ratio of 0.993.
These are in-sample measurements under daily rebalancing without costs.

HRP invests in MSFT, GOOG, JPM, XLE and GLD. Its targets use 252 past return
observations, a one-observation signal delay and 21-observation holding blocks.
Holdings drift between updates. Equal weighting uses the same assets and
schedule; SPY is excluded from fitting. The 2020 data supply the initial
training window, followed by evaluation through 2024.

| Strategy | Annualized compound growth | Annualized volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| HRP | 12.41% | 12.29% | 0.694 | -17.88% |
| Equal weighting | 21.61% | 16.01% | 1.057 | -20.06% |
| SPY buy-and-hold | 13.95% | 16.48% | 0.637 | -24.50% |

The table shows the zero-cost case. A separate sensitivity case charges
5 basis points per dollar traded, including both purchases and sales and the
initial investment. The constant effective annual cash-return assumption is 4%;
annualization uses 252 observations. HRP reduced measured volatility and
drawdown in this sample, while equal weighting achieved higher growth and Sharpe.

![In-sample portfolio allocations](results/Portfolio_Optimization_Model.png)

![Walk-forward wealth and drawdowns](results/Walk_Forward_Backtest.png)

## Project contents

| Artifact | Content |
|---|---|
| [Technical report](Quantitative_Asset_Allocation_Paper.pdf) and [Markdown source](Quantitative_Asset_Allocation_Paper.md) | Methods, historical results, cost sensitivity and limitations |
| [MVO notebook](MVO.ipynb) and [HRP notebook](HRP.ipynb) | Calculations and saved executed outputs |
| [Mathematical note](COVARIANCE_DERIVATIONS.md) | GMV allocation, shrinkage, conditioning and allocation sensitivity |
| [Correction record](CHANGES.md) | Technical differences from the original project |
| `allocation.py`, `run_analysis.py` | Portfolio accounting, optimization and historical analysis |
| `covariance_lab.py`, `covariance_reference.py` | Covariance simulation engine and GMV/shrinkage estimators |
| `fetch_data.py` | Adjusted-price retrieval and snapshot hash verification |
| `tests/` | Allocation, timing, accounting and covariance checks |
| `build_notebooks.py`, `execute_notebooks.py`, `build_report.py` | Notebook and report generation |

The covariance extension separates numerical conditioning from estimation error.
It contains mathematical derivations, estimators and a simulation engine with
paired comparisons against sample GMV, equal weighting and inverse-variance
weighting. The full empirical covariance study has not been run; no covariance
study findings are reported here.

## Data and reproducibility

The historical results use a frozen local adjusted-price snapshot retrieved on
5 October 2026. Its dates, source and SHA-256 hash are recorded in
[data-manifest.json](results/data-manifest.json). The vendor price CSV is excluded
from the repository. A fresh download may differ from this snapshot and from
the unarchived original 2025 download.

`run_analysis.py` verifies the local snapshot before analysis. Saved settings,
the random seed, dependency versions and metrics are in
[summary.json](results/summary.json). Python 3.12 was used for the revision;
dependencies are recorded in `requirements.txt` and `requirements-lock.txt`.

Numerical outputs include [MVO weights](results/mvo-weights.csv),
[walk-forward metrics](results/walk-forward-metrics.csv), daily returns at
[zero costs](results/walk-forward-returns-0bps.csv) and
[5 bps](results/walk-forward-returns-5bps.csv), and
[HRP execution records](results/hrp-rebalances-0bps.csv).
The [release manifest](results/release-manifest.json) records artifact hashes.

## Limitations

The asset universes are small and selected. The experiments cover one historical
period, with no untouched holdout or statistical superiority test. Daily adjusted
prices approximate total returns and execution values; they do not synchronize
intraday market timestamps. The signal delay specifies information timing but
does not eliminate every source of bias. Costs are illustrative and do not fully
model taxes, impact or slippage. The empirical fifth percentile is a historical
return quantile, not a bound on future losses.

HRP uses correlation-based clustering and covariance-based cluster risk. Its
recursive rule avoids a full covariance inverse but imposes no explicit sector
concentration cap. The measured historical differences do not establish general
risk superiority or future performance.

## Provenance and sources

The original eight project files remain unchanged at the repository root and
in `archive/2025-original`, whose manifest records file hashes. The source commit
is `ce87a9d13131d03699dd1b0834acd945c54bbb6d`. The original report is dated
November 2025; the available GitHub snapshot was last committed in December 2025.
This directory is a separate technical revision.

- [López de Prado (2016), HRP paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [PyPortfolioOpt HRP implementation](https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/hierarchical_portfolio.html)
- [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
- [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)

Executed dependency versions are recorded in the results; online documentation
may describe a different version. The original MIT license is retained.
