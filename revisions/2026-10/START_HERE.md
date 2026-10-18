# Current revision and remaining work

Published as a separate revision on 11 October 2026. The original files at the
repository root remain unchanged.

The maximum-Sharpe and rolling HRP calculations have been corrected and executed.
The report is a draft for review. The covariance extension has guided derivations,
an experiment scaffold and worked reference functions. Its two implementation
tasks and the full empirical study remain unfinished.

## Read the completed historical work

- [Report draft](Quantitative_Asset_Allocation_Paper.pdf)
- [Editable report](Quantitative_Asset_Allocation_Paper.md)
- [Maximum-Sharpe notebook](MVO.ipynb)
- [Rolling HRP notebook](HRP.ipynb)
- [Corrections](CHANGES.md)

## Continue the covariance extension

The question is whether improved covariance conditioning also produces lower
true portfolio risk. The model is fully invested minimum variance with short
positions allowed, which differs from the historical long-only maximum-Sharpe fit.

1. Study shrinkage, conditioning versus true risk, and the repeated-sampling design
   in [the derivations](COVARIANCE_DERIVATIONS.md).
2. Complete the GMV and shrinkage functions in [the experiment starter](COVARIANCE_EXPERIMENT.ipynb).
3. Validate them, run the preview, then run the predeclared full experiment and
   write an interpretation of the results and uncertainty.
4. Review a final report containing the completed study. The optional sensitivity
   and HRP cluster calculations can be left for later.

The supplied reference functions are teaching and verification material. Tests
using them check the infrastructure; they are not completed student study results.
The October corrections and study materials were prepared with Codex assistance.

Reproduction instructions, assumptions and data provenance are in [README.md](README.md).
Personal application and referee-planning documents are kept outside this public revision.
