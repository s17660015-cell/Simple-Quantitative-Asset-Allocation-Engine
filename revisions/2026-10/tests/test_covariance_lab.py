"""Check mathematical properties and paired-simulation accounting on small grids."""

import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from covariance_lab import covariance_scenarios, ExperimentConfig, run_experiment, summarize, validate_implementations
from covariance_reference import fit_gmv, shrink_covariance


class CovarianceLabTests(unittest.TestCase):
    def test_scenarios_have_declared_correlations_and_are_spd(self):
        covs = covariance_scenarios()
        for cov in covs.values():
            self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)
        cov = covs['unequal_variance_blocks']
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        self.assertAlmostEqual(corr[0, 1], 0.7)
        self.assertAlmostEqual(corr[0, 5], 0.1)
        np.testing.assert_allclose(np.diag(corr), 1)

    def test_worked_example_conditioning_improves_while_risk_worsens(self):
        validate_implementations(fit_gmv, shrink_covariance)
        true = np.diag([1., 9.])
        shrunk = shrink_covariance(true, 0.5)
        w0, w1 = fit_gmv(true), fit_gmv(shrunk)
        self.assertAlmostEqual(w0 @ true @ w0, 0.9)
        self.assertAlmostEqual(w1 @ true @ w1, 1.3)
        self.assertLess(np.linalg.cond(shrunk), np.linalg.cond(true))

    def test_scale_changes_variance_but_not_weights(self):
        cov = covariance_scenarios()['unequal_variance_blocks']
        np.testing.assert_allclose(fit_gmv(cov), fit_gmv(13*cov), atol=1e-10)
        np.testing.assert_allclose(fit_gmv(shrink_covariance(cov, 0.25)),
                                   fit_gmv(shrink_covariance(13*cov, 0.25)), atol=1e-10)

    def test_grid_is_reproducible_paired_and_obeys_oracle_bound(self):
        config = ExperimentConfig(sample_sizes=(40,), replications=4)
        records = run_experiment(fit_gmv, shrink_covariance, config)
        repeated = run_experiment(fit_gmv, shrink_covariance, config)
        pd.testing.assert_frame_equal(records, repeated)
        self.assertEqual(len(records), 3 * 4 * 7)
        counts = records.groupby(['scenario','n','trial']).size()
        self.assertTrue((counts == 7).all())
        self.assertTrue((records.risk_inflation >= 1-1e-8).all())
        for _, group in records.groupby(['scenario','n','trial']):
            full = group.loc[group.method=='shrinkage_1'].iloc[0]
            equal = group.loc[group.method=='equal_weight'].iloc[0]
            self.assertAlmostEqual(full.risk_inflation, equal.risk_inflation)
            self.assertAlmostEqual(full.covariance_condition_number, 1)

    def test_paired_standard_error_matches_replication_differences(self):
        records = run_experiment(fit_gmv, shrink_covariance,
                                 ExperimentConfig(sample_sizes=(40,),replications=5))
        summary = summarize(records)
        base = records.loc[(records.scenario=='unequal_variance_blocks') &
                           (records.method=='sample_gmv')].sort_values('trial')
        other = records.loc[(records.scenario=='unequal_variance_blocks') &
                            (records.method=='shrinkage_0.25')].sort_values('trial')
        diff = other.risk_inflation.to_numpy()-base.risk_inflation.to_numpy()
        row = summary.loc[(summary.scenario=='unequal_variance_blocks') &
                          (summary.method=='shrinkage_0.25')].iloc[0]
        self.assertAlmostEqual(row.paired_change_vs_sample, diff.mean())
        self.assertAlmostEqual(row.paired_mcse, diff.std(ddof=1)/np.sqrt(5))

    def test_invalid_budget_is_rejected_before_simulating(self):
        with self.assertRaises(ValueError):
            validate_implementations(lambda cov: np.zeros(cov.shape[0]), shrink_covariance)


if __name__ == '__main__':
    unittest.main()
