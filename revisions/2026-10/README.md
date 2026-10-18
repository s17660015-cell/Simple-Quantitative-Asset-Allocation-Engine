# Maximum-Sharpe Allocation and a Walk-Forward HRP Demonstration

Original project: Spark Hu, November 2025. Historical calculations corrected on
5 October 2026; report wording and study materials updated on 7 October 2026.

This repository contains two historical allocation demonstrations. The first fits
a long-only maximum-Sharpe portfolio on its full sample. The second evaluates a
rolling HRP allocation against equal weighting on the same assets and a SPY
buy-and-hold reference. The experiments have different asset universes and do not
establish that HRP outperforms MVO.

The October revision was prepared with Codex assistance for technical correction,
testing and explanation. The author should review the code, reproduce the
calculations and write their own interpretation before using it as evidence of
independent work. This is an educational prototype, not a trading recommendation.

## Start here

1. Read [WALKTHROUGH.md](WALKTHROUGH.md) for the ideas, equations and worked examples.
2. Open [MVO.ipynb](MVO.ipynb), then [HRP.ipynb]. They contain explanatory text and saved executed outputs.
3. Read [CHANGES.md](CHANGES.md) alongside the old files in [archive/2025-original](archive/2025-original).
4. Read the corrected technical note, [Quantitative_Asset_Allocation_Paper.pdf](Quantitative_Asset_Allocation_Paper.pdf).
5. See [START_HERE.md](START_HERE.md) for current status and remaining work.

## Guided study and the covariance experiment

- [Notebook checkpoints and worked checks](GUIDED_STUDY.md)
- [Portfolio derivations](COVARIANCE_DERIVATIONS.md)
- [Experiment starter](COVARIANCE_EXPERIMENT.ipynb)

The starter leaves two functions for you to implement and provides the simulation,
paired summaries and plotting infrastructure in `covariance_lab.py`. A five-trial
preview and a deliberate 300-trial run are distinct. In its supplied state, the
notebook skips simulation until you complete its tasks. Worked reference functions
are in `covariance_reference.py` for study and test verification. No completed
empirical covariance study is claimed.

The current report uses the same frozen historical results as the 5 October
revision. Its wording changes are recorded in [STYLE_REVISION.md](STYLE_REVISION.md),
and the earlier draft is archived under `archive/2026-10-05-revision`.

## What changed

- Asset weights are attached to asset labels throughout.
- Complete price observations are aligned before simple returns are calculated.
- Prices are explicitly adjusted; the downloaded local snapshot has a recorded hash.
- The MVO figure is described as sampled feasible portfolios, not a computed efficient frontier.
- HRP signals use a one-observation delay before execution, with drifting holdings between rebalances.
- Equal weighting is tested on the same investable assets and schedule as HRP.
- Zero-cost and illustrative 5-basis-point-per-traded-dollar cases are recorded.
- Solver convergence, timing, weight constraints and portfolio accounting are tested.
- Conclusions and risk descriptions are limited to what the experiments actually establish.

## Reproduce locally

Use Python 3.12, the version used for this revision. From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python fetch_data.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_analysis.py
.venv/bin/python execute_notebooks.py
.venv/bin/python build_report.py
```

The download is needed only once. Subsequent runs of `run_analysis.py` use and
verify the saved local snapshot. `fetch_data.py --refresh` intentionally retrieves
a replacement snapshot; that can change numerical results. A fresh historical
download is not necessarily identical to the original unarchived 2025 data.

The simple dependency list is in `requirements.txt`; exact installed versions are
in `requirements-lock.txt` and `results/summary.json`. The provided `.venv` is a
local convenience and should not be committed or treated as a portable environment.

## Results

![In-sample portfolio allocations](results/Portfolio_Optimization_Model.png)

![Walk-forward wealth and drawdowns](results/Walk_Forward_Backtest.png)

Numerical outputs:

- [Summary, settings and versions](results/summary.json)
- [Labelled MVO weights](results/mvo-weights.csv)
- [Walk-forward performance metrics](results/walk-forward-metrics.csv)
- [Daily returns, zero costs](results/walk-forward-returns-0bps.csv)
- [Daily returns, 5 bps](results/walk-forward-returns-5bps.csv)
- [HRP execution and training records](results/hrp-rebalances-0bps.csv)
- [Data provenance](results/data-manifest.json)

MVO's asset universe is MSFT, JPM, TLT, GLD and BTC-USD, with prices from 2021
through 2024. HRP's investable universe is MSFT, GOOG, JPM, XLE and GLD; SPY is
excluded from fitting. Its 2020 observations supply the initial training window,
with evaluation subsequently running through 2024. Actual observation dates are
recorded in the outputs.

## Assumptions and limitations

The historical universe was chosen in the original project. Daily adjusted
prices approximate total returns and execution values; they are not a synchronized
intraday feed. The HRP model updates targets every 21 retained observations,
uses a 252-observation lookback and holds allocations between those updates.
The lag specifies information timing; it does not eliminate all forms of bias.

Performance uses a constant effective annual cash return of 4% and 252-observation
annualization. Compound growth and arithmetic mean are reported separately.
Costs are a simple sensitivity assumption; taxes, impact, execution slippage and
fund-specific details are not comprehensively modeled. No statistical superiority
test or uncertainty interval for historical strategy performance is claimed.

HRP uses correlation-based clustering and covariance-based cluster risk. Its
recursive rule avoids a full covariance inverse but imposes no explicit sector
concentration cap. A single period and small selected universe do not demonstrate
general risk superiority, or the future performance of either approach.

## Earlier version

The eight original files are preserved unmodified in `archive/2025-original`,
with file hashes in `MANIFEST.json`; the original source commit is
`ce87a9d13131d03699dd1b0834acd945c54bbb6d`. The original PDF is dated November 2025;
the available GitHub snapshot's latest commit is December 2025. The archive is a
record of that available version, not a reconstruction of an earlier unsaved state.
The historical files at the repository root are also retained unchanged.
This folder contains a separate corrected revision.

## Method sources

- [PyPortfolioOpt HRP implementation](https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/hierarchical_portfolio.html)
- [López de Prado (2016), HRP paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
- [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)

The library version used for the executed revision is recorded in the results;
online documentation can describe a different version. The original MIT license
is retained.
