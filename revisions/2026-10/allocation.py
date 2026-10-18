"""Corrected allocation and portfolio accounting for the original demonstration.

Revision: 5 October 2026, prepared with Codex assistance for author review.
All return inputs are simple returns on a common observation calendar.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pypfopt import HRPOpt

TRADING_DAYS = 252
ANNUAL_RISK_FREE = 0.04


def daily_risk_free(annual_rate=ANNUAL_RISK_FREE):
    """Convert a constant effective annual cash return to a daily return."""
    if annual_rate <= -1:
        raise ValueError("Annual risk-free rate must exceed -100%.")
    return (1 + annual_rate) ** (1 / TRADING_DAYS) - 1


def prepare_prices(raw_prices, tickers, start=None, end=None):
    """Align complete prices BEFORE differencing; preserve explicit ticker order.

    The retained grid is the intersection of available observation dates. Crypto
    weekend movement is consequently included in the next common-date return.
    End is exclusive, matching the downloader's date convention.
    """
    missing = set(tickers) - set(raw_prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    prices = raw_prices.loc[:, list(tickers)].copy()
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing.")
    if start is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices.loc[prices.index < pd.Timestamp(end)]
    all_missing = prices.columns[prices.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"No observations for: {all_missing}")
    prices = prices.dropna(how="any")
    if len(prices) < 3:
        raise ValueError("Too few common price observations.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite and strictly positive.")
    return prices


def simple_returns(prices):
    """Fractional price changes; no implicit forward filling."""
    returns = prices.pct_change(fill_method=None).iloc[1:]
    validate_returns(returns)
    return returns


def validate_returns(returns):
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("Expected a nonempty return DataFrame.")
    if not returns.columns.is_unique or not returns.index.is_unique:
        raise ValueError("Duplicate assets or dates are not supported.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates must be increasing.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Return table contains missing or nonfinite values.")
    if (returns <= -1).any().any():
        raise ValueError("This positive-price demonstration requires returns > -100%.")


def align_weights(weights, columns):
    """Use labels, never a separate positional ticker list, to align allocations."""
    if not isinstance(weights, pd.Series) or not weights.index.is_unique:
        raise ValueError("Weights must be a Series with unique asset labels.")
    if set(weights.index) != set(columns):
        raise ValueError("Weight labels must match the investable asset columns.")
    aligned = weights.reindex(columns).astype(float)
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Nonfinite weights.")
    if not np.isclose(aligned.sum(), 1, atol=1e-7, rtol=0):
        raise ValueError("Weights do not sum to one.")
    if (aligned < -1e-8).any() or (aligned > 1 + 1e-8).any():
        raise ValueError("This demonstration uses long-only weights.")
    return aligned


def max_sharpe_weights(returns, annual_rf=ANNUAL_RISK_FREE):
    """Fit long-only maximum Sharpe weights; an IN-SAMPLE estimated optimum.

    Mean and covariance use simple returns. The annualized objective uses the
    same daily cash-return convention as reported performance. SLSQP supplies
    a numerical solution; convergence and feasibility are checked explicitly.
    """
    validate_returns(returns)
    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")
    cov = returns.cov().to_numpy()
    excess_mean = returns.mean().to_numpy() - daily_risk_free(annual_rf)
    p = len(returns.columns)
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("This optimizer requires a positive-definite covariance.")

    def objective(w):
        vol = np.sqrt(w @ cov @ w)
        return -np.sqrt(TRADING_DAYS) * (w @ excess_mean) / vol

    def gradient(w):
        vol = np.sqrt(w @ cov @ w)
        return np.sqrt(TRADING_DAYS) * (
            -excess_mean / vol + (w @ excess_mean) * (cov @ w) / vol**3
        )

    fitted = minimize(
        objective,
        np.full(p, 1 / p),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * p,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not fitted.success:
        raise RuntimeError(f"Maximum-Sharpe optimization failed: {fitted.message}")
    weights = pd.Series(fitted.x, index=returns.columns, name="weight")
    align_weights(weights, returns.columns)
    # Remove floating-point boundary noise only after validating the solution.
    weights = weights.clip(lower=0, upper=1)
    weights /= weights.sum()
    return weights, {
        "solver": "SLSQP",
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "iterations": int(fitted.nit),
        "covariance_condition_number": float(np.linalg.cond(cov)),
    }


def fixed_weight_returns(returns, weights):
    """Daily restored weights: appropriate only if that trading rule is stated."""
    validate_returns(returns)
    weights = align_weights(weights, returns.columns)
    return returns.mul(weights, axis="columns").sum(axis=1)


def hold_block(returns, weights):
    """Let holdings drift between rebalances; return daily gains and end weights.

    Each asset's gross returns compound independently. Initial block wealth is
    one. Adjusted-price returns represent a total-return approximation with
    distributions reinvested within each asset; this is not an execution feed.
    """
    validate_returns(returns)
    weights = align_weights(weights, returns.columns)
    asset_growth = (1 + returns).cumprod()
    sleeve_values = asset_growth.mul(weights, axis="columns")
    wealth = sleeve_values.sum(axis=1)
    previous = wealth.shift(1)
    previous.iloc[0] = 1.0
    portfolio_returns = wealth / previous - 1
    end_weights = sleeve_values.iloc[-1] / wealth.iloc[-1]
    return portfolio_returns, end_weights


@dataclass
class BacktestResult:
    returns: pd.Series
    target_weights: pd.DataFrame
    rebalances: pd.DataFrame


def walk_forward(returns, method="hrp", window=252, rebalance_period=21,
                 signal_lag=1, cost_bps=0):
    """Fit on past returns, execute after a lag, and hold within each block.

    With lag=1 the latest training return ends one observation before the
    execution close. The first test return runs from that execution close to
    the following close. New allocations occur every 21 retained observations,
    not at calendar month-end. Costs are an illustrative proportional model.
    """
    validate_returns(returns)
    if window < 2 or rebalance_period < 1 or signal_lag < 1:
        raise ValueError("Need window >= 2, rebalance period >= 1 and signal lag >= 1.")
    if cost_bps < 0 or not np.isfinite(cost_bps):
        raise ValueError("Costs must be finite and nonnegative.")
    first_test = window + signal_lag
    if len(returns) <= first_test:
        raise ValueError("Insufficient returns for training, lag and a test block.")
    if method not in {"hrp", "equal_weight"}:
        raise ValueError("Supported methods: hrp, equal_weight.")

    previous_weights = pd.Series(0.0, index=returns.columns)
    pieces, targets, records = [], [], []
    for i in range(first_test, len(returns), rebalance_period):
        train_end = i - signal_lag  # exclusive position
        train = returns.iloc[train_end - window:train_end]
        test = returns.iloc[i:i + rebalance_period]
        if method == "hrp":
            weights = pd.Series(HRPOpt(train).optimize(linkage_method="single"))
        else:
            weights = pd.Series(1 / len(returns.columns), index=returns.columns)
        weights = align_weights(weights, returns.columns)

        # Includes initial investment from cash; later trades use drifted weights.
        traded_weight = float((weights - previous_weights).abs().sum())
        cost_fraction = cost_bps / 10000 * traded_weight
        if cost_fraction >= 1:
            raise ValueError("Trade cost would exhaust the portfolio.")
        net_returns, end_weights = hold_block(test, weights)
        # Charge costs at execution before the first return in this block.
        net_returns.iloc[0] = (1 - cost_fraction) * (1 + net_returns.iloc[0]) - 1
        execution_close = returns.index[i - 1]
        assert train.index[-1] < execution_close < test.index[0]
        pieces.append(net_returns)
        weights.name = execution_close
        targets.append(weights)
        records.append({
            "training_start": train.index[0].isoformat(),
            "training_end": train.index[-1].isoformat(),
            "execution_close": execution_close.isoformat(),
            "first_test_date": test.index[0].isoformat(),
            "last_test_date": test.index[-1].isoformat(),
            "test_observations": len(test),
            "traded_weight": traded_weight,
            "cost_fraction": cost_fraction,
        })
        previous_weights = end_weights
    realized = pd.concat(pieces).rename(method)
    if not realized.index.is_unique:
        raise RuntimeError("Overlapping test blocks.")
    return BacktestResult(realized, pd.DataFrame(targets), pd.DataFrame(records))


def benchmark_returns(returns, cost_bps=0):
    """Buy-and-hold benchmark; charge the same model's initial purchase cost."""
    result = returns.copy()
    fee = cost_bps / 10000
    if fee < 0 or fee >= 1:
        raise ValueError("Invalid benchmark cost.")
    result.iloc[0] = (1 - fee) * (1 + result.iloc[0]) - 1
    return result


def performance_metrics(returns, annual_rf=ANNUAL_RISK_FREE):
    """Historical descriptive metrics, with 252-observation annualization.

    Sharpe uses daily excess returns. Compounded annual growth uses 252/N,
    rather than exact calendar elapsed time. Drawdown includes initial wealth.
    """
    x = returns.to_numpy(dtype=float)
    if len(x) < 2 or not np.isfinite(x).all() or (x <= -1).any():
        raise ValueError("Need at least two finite portfolio returns > -100%.")
    wealth = np.cumprod(1 + x)
    values_with_initial = np.r_[1.0, wealth]
    peaks = np.maximum.accumulate(values_with_initial)
    daily_vol = np.std(x, ddof=1)
    excess_mean = np.mean(x - daily_risk_free(annual_rf))
    return {
        "observations": len(x),
        "first_return_date": returns.index[0].isoformat(),
        "last_return_date": returns.index[-1].isoformat(),
        "terminal_wealth": float(wealth[-1]),
        "total_return": float(wealth[-1] - 1),
        "annualized_compound_growth_252": float(wealth[-1] ** (TRADING_DAYS / len(x)) - 1),
        "annualized_arithmetic_mean": float(np.mean(x) * TRADING_DAYS),
        "annualized_volatility": float(daily_vol * np.sqrt(TRADING_DAYS)),
        "sharpe_daily_excess": float(np.sqrt(TRADING_DAYS) * excess_mean / daily_vol)
            if daily_vol > 0 else None,
        "maximum_drawdown": float(np.min(values_with_initial / peaks - 1)),
        "daily_return_quantile_05": float(np.quantile(x, 0.05)),
        "historical_loss_var_95": float(-np.quantile(x, 0.05)),
    }
