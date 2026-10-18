"""Build a safe-to-run starter notebook, leaving two student implementation tasks."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Covariance estimation and portfolio stability: starter experiment

**Question:** Does better covariance conditioning produce lower true portfolio risk?

This is a scaffold prepared with assistance on 7 October 2026. Two implementation
tasks and your interpretation remain to be completed. No full study results are
claimed by running this notebook in its supplied state. Read
`COVARIANCE_DERIVATIONS.md` first. Optional worked code is in
`covariance_reference.py`; try your implementations before consulting it.

The model is equality-constrained minimum variance, with negative weights allowed.
It differs from the original long-only maximum-Sharpe fit. We evaluate variance,
not trading profitability."""),
    md("""## 1. Predeclare the design

There are three known true covariance structures, three sample sizes and five
fixed shrinkage strengths. All methods use the same sample within a replication.
The true covariance evaluates risk but never chooses the best shrinkage strength.
The planned 300 replications per setting are distinct from a five-replication
preview. Keep the complete grid even when a result is unfavorable."""),
    code("""from pathlib import Path
from dataclasses import replace, asdict
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd() / '.mpl-cache'))
import numpy as np
import pandas as pd
from IPython.display import display, Image
from covariance_lab import (ExperimentConfig, covariance_scenarios,
                            validate_implementations, run_experiment, save_experiment)
plan = ExperimentConfig()
display(pd.Series(asdict(plan), name='predeclared_design').to_frame())
print('Planned observations are IID Gaussian vectors with known covariance.')
print('A common 0.01 scale keeps return units small without changing GMV weights or risk ratios.')"""),
    md("""## 2. Inspect the true covariance structures

The identity case is a useful control: full identity shrinkage already matches
the true allocation structure. The unequal-variance and correlation-block cases
examine target misspecification. The two-block construction has within-block
correlation 0.7, between-block correlation 0.1 and positive eigenvalues.

**Your interpretation:** why would a study containing only the identity case
overstate the general appeal of full shrinkage?"""),
    code("""true_covariances = covariance_scenarios(plan.p, plan.scale)
structure_checks = []
for name, covariance in true_covariances.items():
    eigenvalues = np.linalg.eigvalsh(covariance)
    structure_checks.append({'scenario': name,
                             'smallest_eigenvalue': eigenvalues[0],
                             'largest_eigenvalue': eigenvalues[-1],
                             'true_condition_number': eigenvalues[-1]/eigenvalues[0]})
display(pd.DataFrame(structure_checks))"""),
    md(r"""## 3. Task A: implement GMV weights

Solve $Sv=1$, then normalize $v$ so the weights sum to one. Use a linear solve
rather than an explicit inverse. Keep negative weights when they arise; clipping
would silently change the optimization problem. Explain why the normalization
denominator is positive for a positive-definite covariance."""),
    code("""def fit_gmv(covariance):
    # TODO A: write the linear solve and normalize the vector.
    # Return a NumPy vector of length p, not a covariance matrix.
    raise NotImplementedError('Complete Task A from the GMV derivation.')"""),
    md(r"""## 4. Task B: implement identity shrinkage

Use $m=\operatorname{tr}(S)/p$ and
$S_\lambda=(1-\lambda)S+\lambda mI$. Return a matrix of the same shape.
Check the endpoints: strength zero returns the sample covariance; strength one
returns a scaled identity. A fixed strength is not a fitted Ledoit-Wolf estimator."""),
    code("""def shrink_covariance(sample_covariance, strength):
    # TODO B: calculate the target scale and the convex combination.
    # Use the input matrix's dimension, not a hard-coded 10.
    raise NotImplementedError('Complete Task B from the shrinkage definition.')"""),
    md("""## 5. Check the analytic example before simulating

The validation checks Sigma=diag(1,9), its GMV weights, and the strength-0.5
shrunk matrix and allocation. It also checks strength endpoints. Errors should
be investigated, not bypassed. In the supplied scaffold, missing tasks are
reported clearly and subsequent simulations remain inactive."""),
    code("""IMPLEMENTATIONS_READY = False
try:
    IMPLEMENTATIONS_READY = validate_implementations(fit_gmv, shrink_covariance)
except NotImplementedError as error:
    print(error)
    print('The scaffold is ready, but no experiment has been completed.')
else:
    print('Worked-example checks passed. Now explain why true variance rises from 0.9 to 1.3.')"""),
    md("""## 6. Preview the pipeline

Once the tasks pass, run five replications per setting to inspect file shapes,
metrics and plots. Those small preview outputs are not your final study.
The full grid retains 300 replications. Controls are equal weighting and
sample inverse-variance weighting, both evaluated under the same true covariance."""),
    code("""if IMPLEMENTATIONS_READY:
    preview_plan = replace(plan, replications=5)
    preview_records = run_experiment(fit_gmv, shrink_covariance, preview_plan)
    preview_folder = Path('results/covariance-extension/preview')
    preview_summary = save_experiment(preview_records, preview_plan, preview_folder, mode='preview')
    display(preview_summary.head(14))
    display(Image(filename=str(preview_folder / 'risk-inflation.png')))
    display(Image(filename=str(preview_folder / 'conditioning.png')))
else:
    print('Preview skipped: complete and check Tasks A and B first.')"""),
    md("""## 7. Run the planned study explicitly

Change the flag only after you have understood and checked the implementation.
The engine saves replication data, summary data, settings and two figures in
a separate directory. It does not overwrite the historical MVO/HRP outputs.
No lambda is selected from oracle outcomes."""),
    code("""RUN_FULL_EXPERIMENT = False
if RUN_FULL_EXPERIMENT and IMPLEMENTATIONS_READY:
    study_records = run_experiment(fit_gmv, shrink_covariance, plan)
    study_folder = Path('results/covariance-extension/study')
    study_summary = save_experiment(study_records, plan, study_folder, mode='planned study')
    display(study_summary)
elif RUN_FULL_EXPERIMENT:
    print('Full run blocked by incomplete implementation tasks.')
else:
    print('Full run not requested. Planned replications per setting:', plan.replications)"""),
    md(r"""## 8. Write your interpretation

**TODO C:** state which conclusions follow from each result, using the actual
output rather than a preferred story.

- Why is true risk evaluated using $\Sigma$ rather than the fitted covariance?
- Explain the same-constraint oracle ratio and why it should be at least one.
- Identify a case where conditioning improves while true risk worsens.
- Explain the identity control's structural advantage and the bias in a
  common-variance target when actual variances differ.
- Compare paired risk changes against sample GMV. Monte Carlo standard errors
  concern independent simulation replications, not future market performance.
- Discuss covariance error, weight error and gross exposure. Why can these
  measurements tell different stories?

Do not omit unfavorable strengths, use true risk to select an implementable
lambda, or present supplied worked code as independently authored work. Record
your implementation decisions and feedback-driven revisions."""),
]


if __name__ == '__main__':
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.12'},
    })
    destination = ROOT / 'COVARIANCE_EXPERIMENT.ipynb'
    nbf.write(nb, destination)
    print('Created starter notebook:', destination)
