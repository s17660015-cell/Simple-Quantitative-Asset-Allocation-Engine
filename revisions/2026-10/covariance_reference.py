"""Optional worked solutions for teaching and infrastructure checks.

Try the notebook exercises first. These supplied functions are not evidence
that you independently implemented the methods. No full study is run here.
"""

import numpy as np


def fit_gmv(covariance):
    covariance = np.asarray(covariance, dtype=float)
    v = np.linalg.solve(covariance, np.ones(covariance.shape[0]))
    return v / v.sum()


def shrink_covariance(sample_covariance, strength):
    sample_covariance = np.asarray(sample_covariance, dtype=float)
    p = sample_covariance.shape[0]
    target_scale = np.trace(sample_covariance) / p
    return (1 - strength) * sample_covariance + strength * target_scale * np.eye(p)
