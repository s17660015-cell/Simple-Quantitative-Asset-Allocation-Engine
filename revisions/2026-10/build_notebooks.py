"""Generate the project notebooks; execution is handled separately."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def notebook(cells):
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

mvo = notebook([
    md("""# Maximum-Sharpe allocation

Original project: Spark Hu, November 2025. Technical corrections prepared with Codex assistance on 5 October 2026. Documentation revised on 18 October 2026.

This is an **in-sample allocation illustration** using MSFT, JPM, TLT, GLD and BTC-USD. We estimate mean and covariance on the full historical sample, then choose long-only weights. It is not a test of how those fitted weights would have performed on unseen future returns.

The complete earlier version is preserved in `archive/2025-original`. The local price snapshot was freshly retrieved for this revision; it is not the unarchived original download."""),
    md("""## 1. Price alignment

The combined download has weekend rows because it contains BTC. The analysis first retains dates with complete prices for this asset universe. Explicit ticker ordering and labelled weights prevent confusing one asset's allocation with another's. `load_snapshot` verifies the input file's recorded hash and makes no network request."""),
    code("""from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display, Image
from allocation import (prepare_prices, simple_returns, max_sharpe_weights,
                        fixed_weight_returns, performance_metrics, daily_risk_free)
from fetch_data import load_snapshot
from run_analysis import MVO_TICKERS, RESULTS, SEED, portfolio_cloud
RESULTS.mkdir(exist_ok=True)
raw = load_snapshot()
prices = prepare_prices(raw, MVO_TICKERS, start='2021-01-01', end='2025-01-01')
print('Price dates:', prices.index[0].date(), 'to', prices.index[-1].date())
print('Common price observations:', len(prices))
display(prices.head())"""),
    md(r"""## 2. Simple-return mean and covariance

An asset's simple return is $r_{j,t}=P_{j,t}/P_{j,t-1}-1$. For beginning-of-interval weights, $r_{p,t}=\sum_jw_jr_{j,t}$ is exact. The fitted portfolio mean is $w^T\hat\mu$ and its variance is $w^TSw$. The sample covariance below uses the usual $n-1$ denominator.

These quantities estimate the historical sample. Their accuracy for another period is a separate statistical question."""),
    code("""returns = simple_returns(prices)
display(pd.DataFrame({'daily_mean': returns.mean(), 'daily_std': returns.std()}))
display(returns.cov())
print('Return observations:', len(returns))"""),
    md(r"""## 3. Constrained maximum-Sharpe allocation

The objective is $\sqrt{252}(w^T\hat\mu-r_{f,d})/\sqrt{w^TSw}$, with $w_j\geq0$ and $\sum_jw_j=1$. We assume a constant effective annual cash return of 4%, so $r_{f,d}=1.04^{1/252}-1$.

`max_sharpe_weights` uses SLSQP and checks convergence and feasibility. The result is a **labelled Series**. A numerical solution is not a guarantee of future returns."""),
    code("""weights, solver_info = max_sharpe_weights(returns)
display(weights.to_frame('weight'))
print('Weight sum:', weights.sum())
print('Daily cash-return assumption:', daily_risk_free())
print('Solver:', solver_info)
weights.to_csv(RESULTS / 'mvo-weights.csv', index_label='asset')"""),
    md("""## 4. Sampled feasible allocations

Each dot represents a weight vector sampled from Dirichlet(1,...,1), evaluated using the same estimated mean/covariance. The seed is fixed. This cloud is **not a simulated distribution of future outcomes**, nor a separately computed efficient frontier."""),
    code("""portfolio_cloud(returns, weights, seed=SEED, draws=10000)
display(Image(filename=str(RESULTS / 'Portfolio_Optimization_Model.png')))
print('Random seed:', SEED)"""),
    md("""## 5. In-sample return series

The daily dot product describes restoring these fitted weights each day. We state that assumption explicitly here; the HRP notebook instead holds weights between updates. The historical fifth percentile is an estimated tail threshold, not a confidence interval or a guarantee about future losses. These in-sample measurements are not a matched comparison with the HRP rolling test."""),
    code("""in_sample = fixed_weight_returns(returns, weights)
metrics = performance_metrics(in_sample)
display(pd.Series(metrics, name='in_sample_value').to_frame())
in_sample.rename('daily_rebalanced_in_sample').to_csv(RESULTS / 'mvo-in-sample-returns.csv')"""),
    md("""## 6. Simple returns and weighted asset logs

With equal investments, one asset's +10% and the other's -10% produce zero portfolio return. Averaging their log returns does not produce zero. This is why the corrected accounting uses simple returns."""),
    code("""toy_asset_returns = np.array([0.10, -0.10])
toy_weights = np.array([0.5, 0.5])
print('Exact simple portfolio return:', toy_weights @ toy_asset_returns)
print('Exact portfolio log return:', np.log1p(toy_weights @ toy_asset_returns))
print('Weighted asset log returns:', toy_weights @ np.log1p(toy_asset_returns))"""),
])

hrp = notebook([
    md("""# HRP walk-forward allocation

Original project: Spark Hu, November 2025. Technical corrections prepared with Codex assistance on 5 October 2026. Documentation revised on 18 October 2026.

HRP invests in MSFT, GOOG, JPM, XLE and GLD. SPY is excluded from fitting and provides market context. Equal weighting uses the same assets and execution assumptions as HRP. The original MVO notebook uses a different universe, so this is not a matched MVO-versus-HRP test."""),
    md("""## 1. Common return observations

The 2020 data supply the first 252-observation training window. We keep complete common prices before computing simple returns. The first test block starts only after the lookback and a one-observation signal delay."""),
    code("""import numpy as np
import pandas as pd
from IPython.display import display, Image
from pypfopt import HRPOpt
from allocation import prepare_prices, simple_returns, hold_block, fixed_weight_returns
from fetch_data import load_snapshot
from run_analysis import HRP_TICKERS, RESULTS, run_hrp
RESULTS.mkdir(exist_ok=True)
raw = load_snapshot()
prices = prepare_prices(raw, HRP_TICKERS + ['SPY'], start='2020-01-01', end='2025-01-01')
all_returns = simple_returns(prices)
investable = all_returns.loc[:, HRP_TICKERS]
display(investable.head())
print('Common prices:', len(prices), 'returns:', len(all_returns))"""),
    md(r"""## 2. First allocation and information timing

HRP forms correlation distances $d_{i,j}=\sqrt{(1-\rho_{i,j})/2}$, clusters the assets, orders them and recursively allocates using cluster variances. It still uses historical covariance estimates.

Below, the last training return ends before the execution close. The first test return runs from that execution close to the following observation. The library's `single` linkage is specified explicitly."""),
    code("""window, lag = 252, 1
first_test = window + lag
train_end = first_test - lag
train = investable.iloc[train_end-window:train_end]
model = HRPOpt(train)
first_weights = pd.Series(model.optimize(linkage_method='single')).reindex(HRP_TICKERS)
display(first_weights.to_frame('first_target_weight'))
display(train.corr())
print('Latest training return:', train.index[-1].date())
print('Execution close:', investable.index[first_test-1].date())
print('First evaluated return:', investable.index[first_test].date())"""),
    md("""## 3. Holding-period accounting

Both assets start at half the capital. Asset A doubles on day 1 and B doubles on day 2. Holding the original positions ends at wealth 2; restoring 50/50 each day ends at 2.25. These are different trading rules. The corrected HRP test uses drifting holdings within its 21-observation blocks."""),
    code("""toy = pd.DataFrame({'A': [1.0, 0.0], 'B': [0.0, 1.0]})
toy_weights = pd.Series({'A': 0.5, 'B': 0.5})
held_returns, final_weights = hold_block(toy, toy_weights)
daily_restored = fixed_weight_returns(toy, toy_weights)
display(pd.DataFrame({'holding_wealth': (1+held_returns).cumprod(),
                      'daily_restored_wealth': (1+daily_restored).cumprod()}))"""),
    md("""## 4. Rolling experiment and cost sensitivity

Each target update uses 252 past return observations, a one-observation signal delay and a 21-observation holding block. Equal weighting uses the same schedule. SPY is bought and held over the same evaluation dates.

The second cost case charges 5 bps per dollar traded, including initial purchases. Buying and selling both count. This is an illustrative proportional charge; it does not claim full execution realism. Costs do not change fitted target weights in this experiment."""),
    code("""result = run_hrp(raw)
metrics = pd.DataFrame(result['metrics'])
selected = ['strategy', 'cost_bps_per_traded_dollar',
            'annualized_compound_growth_252', 'annualized_volatility',
            'sharpe_daily_excess', 'maximum_drawdown']
display(metrics[selected])
display(Image(filename=str(RESULTS / 'Walk_Forward_Backtest.png')))"""),
    md("""## 5. Rebalance chronology and interpretation

The saved log contains the training window, execution close, subsequent test dates and trade sizes. Drawdown calculations include initial wealth. Compound growth and arithmetic mean are kept separate.

Measured differences describe this historical sample under these assumptions. No statistical significance, guaranteed future risk reduction or general HRP superiority over MVO is established."""),
    code("""log = pd.read_csv(RESULTS / 'hrp-rebalances-0bps.csv')
display(log.head())
print('Allocation updates including initial investment:', len(log))
print('Total traded fraction of portfolio wealth:', log['traded_weight'].sum())"""),
])

if __name__ == "__main__":
    for name, document in [("MVO.ipynb", mvo), ("HRP.ipynb", hrp)]:
        nbf.write(document, ROOT / name)
        print("Created notebook:", name)
