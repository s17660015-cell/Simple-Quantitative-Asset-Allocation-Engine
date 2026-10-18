"""GMV and scaled-identity covariance-shrinkage estimators.

These implementations support the simulation engine and analytic checks.
The full empirical covariance study has not been run.
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
