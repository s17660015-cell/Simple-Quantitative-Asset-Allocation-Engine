# Understanding the project again

Read this with the two notebooks open. You do not need to remember your November
code: the aim is to rebuild the ideas in a clear order. These are teaching notes,
prepared with assistance, for you to work through and question.

## 1. What problem were you trying to solve?

An allocation rule decides how much of a portfolio to put into each asset. With
five assets, its output is a vector of five weights, not a prediction of five
future prices. A weight of 0.2 means 20% of the invested capital at the moment the
allocation is established. The code is long-only: weights are between zero and
one and sum to one.

Your first notebook chooses one allocation that looks attractive under estimated
historical return and risk. Your second repeatedly chooses HRP allocations using
past windows and records the subsequent returns. Those are different experiments.

For orientation: MSFT and GOOG are technology-company stocks; JPM is a bank stock;
TLT is a long-duration US Treasury ETF; GLD provides gold exposure; BTC-USD is a
Bitcoin price series; XLE is an energy-sector equity ETF; SPY is an S&P 500 ETF.
These descriptions organize the example, not their risk rankings.

## 2. From prices to returns

Let P_j,t be asset j's adjusted price at observation t. A simple return is

    r_j,t = P_j,t / P_j,t-1 - 1.

If the price moves from 100 to 102, the return is 0.02, or 2%. Adjusted prices
account for corporate actions such as splits and distributions under the vendor's
adjustment conventions. They are useful for an approximate total-return study,
but are not necessarily prices at which a trade could be executed.

Different assets have different observation calendars. BTC trades on weekends;
equities generally do not. If a combined table contains a Sunday crypto price
and a missing Sunday equity price, shifting that table first will make Monday's
equity return compare against a missing Sunday value. Dropping the resulting row
loses Monday. We instead keep complete common price rows first, then calculate
returns. Friday to Monday becomes one interval for every asset.

The method is an explicit daily approximation. It does not synchronize the exact
time of each market's daily observation. Crypto's intervening weekend movement
is folded into the next common-date return. No missing return is replaced by zero.

Functions to inspect: `prepare_prices` and `simple_returns` in `allocation.py`.

## 3. Portfolio returns: why labels and weights matter

For beginning-of-interval weights w_j, the exact simple portfolio return is

    r_portfolio,t = sum_j w_j r_j,t.

If 20% is in an asset returning +2%, and 80% is in an asset returning -1%, the
portfolio return is 0.2*0.02 + 0.8*(-0.01) = -0.004, or -0.4%.

This calculation is only correct when each weight is attached to the right asset.
Your old output printed an array against a different ticker order from the return
table. The saved results were numerically interpreted as approximately 47% GLD,
30.7% JPM and 22.29% MSFT; they were printed as a very different portfolio. A
labelled pandas Series prevents relying on a separate positional list.

Functions to inspect: `align_weights` and `fixed_weight_returns`.

The 22.29% figure above translates the old saved vector correctly; it is not the
new fitted result. The corrected rerun uses simple returns, proper calendar
alignment, a consistent cash-return convention and a fresh historical snapshot.
Its allocations are approximately MSFT 22.72%, JPM 38.53%, TLT 0%, GLD 26.93%
and BTC 11.83%. Several inputs changed, so you cannot attribute the difference
from the old allocation to only one correction.

## 4. Why the old log-return calculation was different

An asset log return is ell_j = log(1+r_j). It adds over time for that single asset.
It does not generally average over assets to give the portfolio log return.

    Exact portfolio log return = log(sum_j w_j exp(ell_j)).
    Weighted asset log return = sum_j w_j ell_j.

Equal investments in assets returning +10% and -10% leave total wealth unchanged:
the portfolio simple return is zero, so its log return is zero. However, the
weighted asset logs equal 0.5*log(1.1) + 0.5*log(0.9), approximately -0.005025.

The correction uses simple returns for the mean/covariance fit and actual
portfolio accounting. Log returns remain a useful concept, but their aggregation
rules must be respected.

## 5. Mean, covariance and risk

The sample mean mu_j estimates an asset's average return per observation. It is
a summary of this sample, not knowledge of its future expected return. The
covariance matrix S has each asset's variance on the diagonal and pairwise
covariances off the diagonal.

For weights w, the fitted daily portfolio mean and variance are

    fitted mean = w^T mu,
    fitted variance = w^T S w,
    fitted volatility = sqrt(w^T S w).

Covariance matters because assets do not move independently. An allocation is
evaluated using their joint movements as well as their individual volatilities.
Correlation is a normalized covariance, lying between -1 and 1; it is not simply
the assertion that one asset rises whenever another falls.

The old report suggested that matrix instability explained MVO's behaviour.
The correction records S's condition number so that this can be examined. The
condition number concerns sensitivity of a matrix problem; it does not alone
measure how accurately historical data estimate the future covariance.

## 6. What maximum Sharpe optimization does

The Sharpe ratio compares average excess return with volatility. This demonstration
assumes a constant effective annual cash return of 4%. Its consistent daily cash
return is (1.04)^(1/252)-1; this is an assumption, not a retrieved history of cash yields.

The fitted annualized objective is

    sqrt(252) * (w^T mu - daily_cash_return) / sqrt(w^T S w).

The code searches for weights maximizing that expression, subject to long-only
bounds and a total weight of one. SciPy minimizes functions, so the objective in
`max_sharpe_weights` is the negative of the ratio. SLSQP is a numerical solver
for this constrained problem. The helper checks its success flag, message,
weight bounds and budget constraint. An explicit derivative is supplied to improve
the numerical calculation; understanding that derivative can be a later exercise.

The returned weights were fitted using the entire 2021-2024 sample. Their reported
in-sample statistics are descriptive. Choosing weights on these observations and
then measuring them on the same observations does not validate future performance.

## 7. What the 10,000 dots mean

Each dot represents one randomly sampled long-only weight vector. Its horizontal
coordinate is the estimated annualized volatility; its vertical coordinate is the
estimated annualized arithmetic mean; its colour is the fitted Sharpe ratio.

The correction uses a seeded Dirichlet(1,...,1) distribution to sample uniformly
over the weight simplex. The older procedure normalized independent uniform
draws, which is a different weight distribution. Neither procedure simulates
future asset returns. Both evaluate weights using the same fitted mean/covariance.

The efficient frontier would require identifying the efficient boundary, for
example by solving minimum-variance problems for target returns. We have not done
that here, so the revised chart accurately describes random feasible portfolios.
The highlighted allocation comes from the numerical optimizer.

Inspect `portfolio_cloud` in `run_analysis.py`.

## 8. What HRP does instead

HRP is a particular allocation procedure. The notebook uses the existing
`HRPOpt` implementation from PyPortfolioOpt; integrating a library method is
different from inventing or independently implementing the algorithm.

The implementation computes historical correlations and maps them to distances:

    d_i,j = sqrt((1-rho_i,j)/2).

Perfect positive correlation gives distance zero; zero correlation gives about
0.707; perfect negative correlation gives distance one. Single-linkage hierarchical
clustering uses these distances to construct an asset tree. It groups statistical
similarities, without receiving sector labels or guaranteeing sector diversity.

The implementation uses the tree to order assets, then recursively bisects that
ordered list. For a candidate group, it calculates inverse-variance weights within
the group and evaluates that group's covariance-based variance. If the two groups
have variances v_L and v_R, its split gives a fraction v_R/(v_L+v_R) to the left
group. A riskier left group receives less capital. Recursion produces individual
asset allocations.

The split formula does not explicitly include the covariance between the two
group portfolios. This is one useful mathematical question for your later extension.
The algorithm avoids a full covariance inverse, but still depends on historical
covariance estimates. It does not generally solve the same optimization problem
as your maximum-Sharpe allocation.

Sources: [PyPortfolioOpt code](https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/hierarchical_portfolio.html)
and [López de Prado's method paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678).

## 9. What a walk-forward test means

At each allocation update, the model estimates weights from 252 past return
observations. The revised code leaves a one-observation information delay, executes
at the next observed close, and then measures the subsequent 21 returns using the
chosen allocation. It repeats this pattern until the dataset ends.

The timeline is:

    latest training close -> next observation's execution close -> first test close.

The saved rebalance log gives actual dates for all three. Each training window
excludes the returns it is about to evaluate. Earlier test observations can enter
a later training window once their dates are in the past: that is normal rolling
estimation, not a leak from the future.

The lag improves clarity about what information is available before a trade. It
does not transform adjusted closes into executable quotes, prove that the selected
asset universe is unbiased, or provide untouched holdout evidence after someone
has repeatedly changed the strategy in response to this period's results.

SPY is removed before fitting. It is a benchmark, not a sixth HRP investment.

## 10. Monthly target updates versus daily rebalancing

Weights change when prices change, even if no trades occur. Suppose wealth is
100, divided equally between assets A and B. A rises 10% on day 1; B is unchanged.
Wealth becomes 105, with positions worth 55 and 50. A's weight is now 55/105,
approximately 52.38%, not 50%.

If B rises 10% on day 2 and A is unchanged, holding those positions gives wealth
110. Restoring 50/50 at the end of day 1 instead gives wealth 110.25. Your old
daily dot-product calculation produced the latter trading rule while the report
described a monthly hold. The revised HRP test implements the former within each
21-observation block, then establishes the next target allocation.

The helper compounds each asset separately, scales its initial sleeve by the
target weight, sums sleeve values and converts total wealth changes to daily
portfolio returns. Its final sleeve values determine drifted weights for the next
rebalance. These approximate total-return sleeves reinvest distributions within
assets according to adjusted-price returns.

Inspect `hold_block` and `walk_forward`.

## 11. Costs and fair controls

The primary displayed chart uses zero costs, explicitly. A second case charges
5 basis points, or 0.0005, per dollar traded. The traded fraction is the sum of
absolute differences between drifted old weights and new target weights. Buying
and selling both count. Initial investment from cash counts as a traded fraction
of one.

Moving 20% of wealth from one asset to another trades 40% in total. At 5 bps,
the approximate wealth charge is 0.4*0.0005 = 0.0002, or 0.02%. The charge is
applied before the first return of the new block. This is an illustrative linear
cost assumption, not a complete execution, spread or market-impact model.

Equal weighting uses exactly the same assets, dates, holding blocks, lag and
cost model as HRP. It asks whether HRP's allocations add something relative to a
simple allocation rule. SPY remains a different exposure and provides market
context. A difference against SPY alone cannot be assigned to clustering.

## 12. How to interpret the output metrics

| Metric | Meaning | What it does not establish |
|---|---|---|
| Terminal wealth | Compounded value of initial wealth 1 | A future payoff |
| Annualized arithmetic mean | Daily average multiplied by 252 | Compounded annual growth |
| Annualized compound growth | Terminal wealth raised to 252/N, minus one | Exact calendar-year growth or a forecast |
| Annualized volatility | Daily sample standard deviation multiplied by sqrt(252) | Every aspect of risk; annualization assumes away serial dependence |
| Sharpe ratio | Annualized daily excess mean divided by daily volatility | Statistical superiority without further analysis |
| Maximum drawdown | Worst historical decline from a running wealth peak, including initial capital | A bound on a future loss |
| Historical 95% loss VaR | Negative of the empirical fifth percentile of daily returns | A confidence interval or guaranteed maximum daily loss |

For example, +10% followed by -10% gives wealth 0.99 from one. The arithmetic
average return is zero, while compounded growth is negative. That is why the two
annual return summaries are kept separate.

The MVO quantile is measured in the same sample used to choose its weights, and
assumes daily restoration of those fitted weights. The HRP metrics come from
subsequent blocks with drifting holdings. Do not compare them as if both were
matched out-of-sample strategy tests.

## 13. What you can reasonably conclude

You have demonstrated an in-sample constrained allocation calculation and a
separate rolling allocation/accounting experiment. You can describe the measured
differences among HRP, equal weighting and SPY on this particular historical
sample, under the recorded assumptions.

You have not demonstrated general HRP superiority over MVO, guaranteed safe
sector diversification, a forecast of future losses, or a new allocation method.
Recognizing those limits is part of a good technical explanation.

The later covariance study will ask a narrower question that can be answered
more cleanly: how estimation error and regularization affect allocations when
the true covariance is known in a controlled experiment.

For the corrected zero-cost rolling test, the measured results from 5 January
2021 to 31 December 2024 are:

| Rule | Annualized compound growth (252/N) | Annualized volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| HRP | 12.41% | 12.29% | 0.694 | -17.88% |
| Equal weight, same assets | 21.61% | 16.01% | 1.057 | -20.06% |
| SPY buy and hold | 13.95% | 16.48% | 0.637 | -24.50% |

HRP had lower measured volatility and drawdown than SPY, alongside lower growth.
Equal weighting grew more than HRP and had a higher measured Sharpe, alongside
greater volatility and drawdown. These are historical observations under the
recorded assumptions; no statistical significance or future ranking follows.
This is a useful trade-off to explain, rather than a result to hide because it
does not favor the more elaborate allocation rule.

## 14. Before discussing this with a referee

Try explaining these without reading the notes:

1. What is an allocation weight, and why must it be aligned by asset label?
2. Why is a weighted sum of simple returns exact for beginning-of-interval weights?
3. What does the covariance matrix contribute to the portfolio variance?
4. What is the maximum-Sharpe objective, and why is this fit in-sample?
5. What does HRP calculate from correlations and cluster variances?
6. What information is available before each allocation in the revised test?
7. How do portfolio weights drift when you do not trade?
8. Why is equal weighting a more informative algorithm control than SPY alone?
9. Which old claims have you withdrawn, and what would be needed to test them?

Rerun a small worked example, check the results by hand, then change one input
and explain the resulting change. Record your own questions and any revisions
you make after the discussion. That gives a referee evidence of your understanding
and response to feedback, rather than simply evidence that they received files.
