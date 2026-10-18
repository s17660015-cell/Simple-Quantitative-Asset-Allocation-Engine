# Correction record

This record describes the technical corrections to the November 2025 project.
The original eight files are preserved at the repository root and in
`archive/2025-original`, with SHA-256 hashes in the archive manifest.
The corrections were prepared with Codex assistance on 5 October 2026.

| Earlier issue | Correction | Why it matters |
|---|---|---|
| Printed MVO weights against a different ticker order | Labelled Series and explicit alignment | The earlier claim that MSFT received zero weight was false |
| Shifted a mixed crypto/equity calendar before dropping missing prices | Align complete common price rows first | Retains valid returns spanning weekends rather than dropping Monday |
| Used weighted asset log returns as exact portfolio returns | Use simple returns for estimates and accounting | Weighted asset logs generally differ from the actual portfolio log return |
| Unspecified adjustment defaults | Set auto_adjust explicitly and record the library version and data hash | Makes the meaning and provenance of the input prices clear |
| Random allocation cloud called a computed efficient frontier/future simulation | Relabel and use seeded Dirichlet weight sampling | Separates a feasible set illustration from forecasting and frontier optimization |
| Solver output accepted without convergence checks | Check success, finite values, bounds and budget | Prevents reporting a failed numerical result as an optimum |
| Fixed daily weights while claiming monthly holding | Drift holdings inside each 21-observation block | Aligns return calculations with the described trading rule |
| Information available at an assumed execution close | Use a one-observation signal delay and save training/execution dates | Makes the signal-to-trade chronology auditable |
| No same-asset allocation baseline | Add equal weighting under the same holding and execution schedule | Provides a more informative control than SPY alone |
| Costs ignored without sensitivity results | Retain a declared zero-cost case and add 5-bps proportional costs | Shows the effect of an explicit illustrative trading-cost assumption |
| Risk and return statements made without supporting metrics | Save compound growth, volatility, Sharpe and drawdown | Distinguishes measured outcomes from qualitative claims |
| HRP described as escaping historical correlation and guaranteeing sector diversification | Explain its actual clustering and variance allocation rules | The library still uses covariance estimates and has no hard sector cap |
| Report implied a matched HRP-versus-MVO test | Present two separate demonstrations | Different universes and fitting procedures do not isolate the allocation method |
| No input snapshot, run record or saved HRP execution output | Cache/hash data, save settings and results, execute explanatory notebooks | Enables inspection and offline reruns |
| PyPortfolioOpt 1.6.0 fails with SciPy 1.18's removed private linkage registry | Pin a tested compatible SciPy version below 1.18 and record the lock | Prevents dependency drift from breaking the executed HRP example |
| Historical quantile presented as a future loss guarantee | Describe the empirical fifth percentile and loss-VaR convention | It is an estimated historical tail threshold, not a bound on future losses |

The corrected allocations will differ from the old output for several reasons:
correct labels, simple rather than weighted log returns, calendar alignment,
a consistent risk-free convention, newer data retrieval and recorded library
versions. The HRP test also changes holding accounting, adds a signal delay and
includes cost assumptions. These differences should not be attributed to one
isolated correction without a separate controlled ablation.

The maximum-Sharpe fit remains an in-sample illustration. A rolling MVO-versus-HRP
experiment and the covariance-shrinkage study are separate possible extensions,
not completed results of this correction.
