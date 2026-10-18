# Working through the corrected notebooks

Use this beside MVO.ipynb and HRP.ipynb. Try each checkpoint before opening its
worked answer. Then move to COVARIANCE_DERIVATIONS.md and the experiment notebook.

## MVO: follow its six executed code cells

| Cell | Calculation | Explain this yourself |
|---|---|---|
| 1 | Load the saved prices and align their dates | Why alignment comes before differencing |
| 2 | Calculate simple returns, sample means and covariance | Their units and the role of covariance |
| 3 | Fit labelled maximum-Sharpe weights | Objective, constraints and solver convergence |
| 4 | Sample 10,000 feasible allocations | Why the dots are not future outcomes |
| 5 | Evaluate the fitted allocation in-sample | Why those statistics are not independent validation |
| 6 | Compare portfolio log return with averaged asset logs | The different aggregation rules |

Follow the inputs and outputs in allocation.py before reading plotting code.

**M1.** Start with wealth 100, invested 50/50 in A and B. A rises 20%; B falls
10%. With no trade afterward, calculate wealth, portfolio return and A's weight.

**M2.** For daily covariance

$$S=\begin{pmatrix}0.0004&0.0001\\0.0001&0.0009\end{pmatrix},$$

calculate variance and volatility at weights (1/2,1/2). Identify the
cross-covariance term and explain why adding the individual variances is wrong.

**M3.** For M1, calculate exact portfolio log return and weighted asset logs.
Which agrees with the change in the total wealth?

**M4.** An optimizer selects weights from four years of observations, then those
weights are evaluated on the same four years. What is missing from an
out-of-sample performance claim?

<details>
<summary>Worked checks for M1-M4</summary>

M1: positions become 60 and 45. Wealth is 105, return is 5%, and A's weight is
60/105=57.1429%. The return calculation is 0.5(0.2)+0.5(-0.1)=0.05.

M2: variance is 0.25(0.0004)+2(0.25)(0.0001)+0.25(0.0009)=0.000375.
Daily volatility is about 1.9365%. Covariance contributes 0.00005 to variance.

M3: exact log return is log(1.05), about 0.048790. Averaged asset logs are
0.5 log(1.2)+0.5 log(0.9), about 0.038481. The first describes total wealth.

M4: weights must be selected before a genuinely later evaluation period, with
that separation repeated in a rolling design. Asset selection and repeated
experimentation remain potential sources of bias even after a chronological split.

</details>

## HRP: follow its five executed code cells

| Cell | Calculation | Explain this yourself |
|---|---|---|
| 1 | Load common returns and separate SPY | Why the benchmark must not enter fitting |
| 2 | Fit first-window HRP and print its dates | Clustering, cluster variance and information timing |
| 3 | Compare holding positions with daily restoration | Which calculation requires trading |
| 4 | Run HRP, equal weight and SPY under two cost cases | Fair controls and what the cost assumption means |
| 5 | Inspect rebalance records | Training cutoff, execution date and subsequent test returns |

The library orders assets through a correlation-based tree, then recursively
bisects that order and allocates using cluster variances. Describe that rule
precisely; it does not find safer sectors from sector labels.

**H1.** Training ends on 31 December 2020, execution is modeled at the 4 January
2021 close, and the first evaluated return ends on 5 January. Which price interval
produces the first return? Why does training stop before execution?

**H2.** Start 50/50. A doubles on day 1, B doubles on day 2, and the other asset
is unchanged on each day. Explain why holding ends at wealth 2, while daily
restoration ends at 2.25. Identify the trade between the two days.

**H3.** A portfolio changes from 40/60 to 70/30. At 5 bps per dollar traded,
what fraction of wealth is charged when both buying and selling count?

**H4.** Interpret equal weighting's higher historical growth and Sharpe alongside
HRP's lower volatility and drawdown in two sentences. Do not invent a causal
explanation or a future ranking.

<details>
<summary>Worked checks for H1-H4</summary>

H1: the first return runs from the 4 January close to the 5 January close. The
implementation leaves one observation between the information cutoff and execution.

H2: holdings after day 1 are worth 1 and 0.5. Keeping them gives 1 and 1 after
day 2. Restoring 50/50 instead changes them to 0.75 each, then gives 0.75 and
1.5. That intermediate restoration sells A and buys B.

H3: 30% is bought and 30% sold. The traded fraction is 0.6 and the charge is
0.6*0.0005=0.0003, or 0.03% of wealth (3 bps).

H4: these rules show a risk-return trade-off under the recorded assumptions.
They provide neither a matched MVO-versus-HRP test nor evidence that the ranking
persists in another period. Supply your own interpretation of the actual figures.

</details>

## Discuss the reasoning

Choose one correction, demonstrate the failure it fixes with a numerical example,
then explain why fixing it is insufficient to establish a performance claim.

Possible discussion topics include holding accounting, the minimum-variance
proof, known true covariance, paired comparisons and Monte Carlo precision.
Record feedback and how you revise your argument.
