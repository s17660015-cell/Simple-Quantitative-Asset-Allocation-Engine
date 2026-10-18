"""Reproduce the two corrected demonstrations from the verified local snapshot."""

from pathlib import Path
from importlib.metadata import version
import json
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from allocation import (
    TRADING_DAYS, ANNUAL_RISK_FREE, daily_risk_free, prepare_prices,
    simple_returns, max_sharpe_weights, fixed_weight_returns,
    performance_metrics, walk_forward, benchmark_returns,
)
from fetch_data import ROOT, load_snapshot

RESULTS = ROOT / "results"
MVO_TICKERS = ["MSFT", "JPM", "TLT", "GLD", "BTC-USD"]
HRP_TICKERS = ["MSFT", "GOOG", "JPM", "XLE", "GLD"]
SEED = 20261005


def portfolio_cloud(returns, weights, seed=SEED, draws=10000):
    rng = np.random.default_rng(seed)
    sampled = rng.dirichlet(np.ones(len(returns.columns)), size=draws)
    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()
    daily_mean = sampled @ mu
    daily_vol = np.sqrt(np.einsum("ij,jk,ik->i", sampled, cov, sampled))
    sharpe = np.sqrt(TRADING_DAYS) * (daily_mean - daily_risk_free()) / daily_vol
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    cloud = axes[0].scatter(daily_vol * np.sqrt(TRADING_DAYS),
                            daily_mean * TRADING_DAYS, c=sharpe,
                            cmap="viridis", s=7, alpha=0.45, rasterized=True)
    w = weights.reindex(returns.columns).to_numpy()
    axes[0].scatter(np.sqrt(w @ cov @ w * TRADING_DAYS), w @ mu * TRADING_DAYS,
                    color="#c43c39", edgecolors="black", s=65,
                    label="Fitted maximum Sharpe", zorder=3)
    axes[0].set(title="Random feasible portfolios (in-sample)",
                xlabel="Annualized volatility", ylabel="Annualized arithmetic mean")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    fig.colorbar(cloud, ax=axes[0], label="Estimated annualized Sharpe")
    corr = returns.corr().to_numpy()
    heat = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = returns.columns.tolist()
    axes[1].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_title("Sample asset correlations")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if abs(corr[i, j]) > 0.65 else "black")
    fig.colorbar(heat, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS / "Portfolio_Optimization_Model.png", dpi=200)
    plt.close(fig)


def run_mvo(raw):
    prices = prepare_prices(raw, MVO_TICKERS, start="2021-01-01", end="2025-01-01")
    returns = simple_returns(prices)
    weights, solver = max_sharpe_weights(returns)
    proxy = fixed_weight_returns(returns, weights)
    proxy.rename("daily_rebalanced_in_sample").to_csv(RESULTS / "mvo-in-sample-returns.csv")
    portfolio_cloud(returns, weights)
    return {
        "purpose": "In-sample allocation illustration; not out-of-sample validation.",
        "price_observations": len(prices),
        "price_start": prices.index[0].isoformat(),
        "price_end": prices.index[-1].isoformat(),
        "weights": weights.to_dict(),
        "solver": solver,
        "daily_rebalanced_in_sample_metrics": performance_metrics(proxy),
    }


def run_hrp(raw):
    prices = prepare_prices(raw, HRP_TICKERS + ["SPY"], start="2020-01-01", end="2025-01-01")
    all_returns = simple_returns(prices)
    investable = all_returns.loc[:, HRP_TICKERS]
    cases = {}
    metrics = []
    for cost in (0, 5):
        hrp = walk_forward(investable, method="hrp", cost_bps=cost)
        equal = walk_forward(investable, method="equal_weight", cost_bps=cost)
        spy = benchmark_returns(all_returns.loc[hrp.returns.index, "SPY"], cost_bps=cost)
        frame = pd.DataFrame({"HRP": hrp.returns,
                              "Equal weight (same assets)": equal.returns,
                              "SPY buy and hold": spy})
        assert not frame.isna().any().any()
        frame.to_csv(RESULTS / f"walk-forward-returns-{cost}bps.csv", index_label="Date")
        if cost == 0:
            hrp.target_weights.to_csv(RESULTS / "hrp-target-weights.csv",
                                      index_label="execution_close")
        hrp.rebalances.to_csv(RESULTS / f"hrp-rebalances-{cost}bps.csv", index=False)
        equal.rebalances.to_csv(RESULTS / f"equal-weight-rebalances-{cost}bps.csv", index=False)
        for strategy in frame:
            entry = {"strategy": strategy, "cost_bps_per_traded_dollar": cost,
                     **performance_metrics(frame[strategy])}
            if strategy == "HRP":
                entry["total_traded_weight"] = float(hrp.rebalances.traded_weight.sum())
                entry["rebalances_including_initial"] = len(hrp.rebalances)
            elif strategy.startswith("Equal"):
                entry["total_traded_weight"] = float(equal.rebalances.traded_weight.sum())
                entry["rebalances_including_initial"] = len(equal.rebalances)
            else:
                entry["total_traded_weight"] = 1.0
                entry["rebalances_including_initial"] = 1
            metrics.append(entry)
        cases[str(cost)] = frame

    base = cases["0"]
    wealth = (1 + base).cumprod()
    execution_date = pd.Timestamp(hrp.rebalances.iloc[0]["execution_close"])
    wealth = pd.concat([pd.DataFrame(1.0, index=[execution_date], columns=wealth.columns), wealth])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    styles = [("#005b91", "-"), ("#b86618", "-"), ("#66717a", "--")]
    for strategy, (color, linestyle) in zip(base.columns, styles):
        axes[0].plot(wealth.index, wealth[strategy], label=strategy,
                     color=color, linestyle=linestyle, linewidth=1.6)
        drawdown = wealth[strategy] / wealth[strategy].cummax() - 1
        axes[1].plot(wealth.index, drawdown, color=color, linestyle=linestyle, linewidth=1.1)
    axes[0].set(title="Walk-forward demonstration: HRP, equal weight and SPY",
                ylabel="Growth of initial wealth 1")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].set(ylabel="Drawdown", xlabel="Date")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.text(0.5, 0.015,
             "252-observation lookback; one-observation signal lag; rebalance every 21 observations; zero-cost case",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(RESULTS / "Walk_Forward_Backtest.png", dpi=200)
    plt.close(fig)
    return {
        "purpose": "Historical rolling HRP demonstration; no rolling MVO comparison.",
        "investable_assets": HRP_TICKERS,
        "benchmark": "SPY (excluded from the investable universe)",
        "window_observations": 252,
        "rebalance_observations": 21,
        "signal_lag_observations": 1,
        "hrp_linkage": "single",
        "cost_cases_bps_per_traded_dollar": [0, 5],
        "execution_assumption": "Lagged signals executed at the following observed adjusted close; "
                                "drifting sleeves inside each block. Approximate daily total-return accounting.",
        "metrics": metrics,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = load_snapshot()
    summary = {
        "revision_date": "2026-10-05",
        "original_source_commit": "ce87a9d13131d03699dd1b0834acd945c54bbb6d",
        "random_seed": SEED,
        "annualization_observations": TRADING_DAYS,
        "constant_effective_annual_risk_free_rate": ANNUAL_RISK_FREE,
        "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "matplotlib", "PyPortfolioOpt", "yfinance"]},
        "mvo": run_mvo(raw),
        "hrp": run_hrp(raw),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print("MVO labelled weights:")
    print(pd.Series(summary["mvo"]["weights"]).round(6).to_string())
    columns = ["strategy", "cost_bps_per_traded_dollar", "annualized_compound_growth_252",
               "annualized_volatility", "sharpe_daily_excess", "maximum_drawdown"]
    print("\nWalk-forward metrics:")
    print(pd.DataFrame(summary["hrp"]["metrics"])[columns].round(4).to_string(index=False))
    print("\nSaved reproducible results in", RESULTS)


if __name__ == "__main__":
    main()
