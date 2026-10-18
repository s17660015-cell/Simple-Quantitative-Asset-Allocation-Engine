"""Meaningful checks of corrected accounting, alignment, timing and optimization."""

import unittest
import numpy as np
import pandas as pd
from allocation import (
    prepare_prices, simple_returns, fixed_weight_returns, hold_block,
    max_sharpe_weights, walk_forward, performance_metrics,
)


class AllocationTests(unittest.TestCase):
    def test_calendar_alignment_preserves_monday(self):
        prices = pd.DataFrame(
            {"equity": [100, np.nan, np.nan, 110, 121],
             "crypto": [200, 202, 204, 210, 220]},
            index=pd.date_range("2024-01-05", periods=5),
        )
        returns = simple_returns(prepare_prices(prices, ["equity", "crypto"]))
        self.assertEqual(returns.index.day_name().tolist(), ["Monday", "Tuesday"])
        np.testing.assert_allclose(returns.equity, [0.1, 0.1])
        self.assertAlmostEqual(returns.crypto.iloc[0], 0.05)

    def test_weights_align_by_labels_after_column_permutation(self):
        returns = pd.DataFrame({"A": [0.1, 0.0], "B": [0.0, 0.2]})
        weights = pd.Series({"B": 0.25, "A": 0.75})
        expected = pd.Series([0.075, 0.05])
        pd.testing.assert_series_equal(fixed_weight_returns(returns, weights), expected)
        pd.testing.assert_series_equal(fixed_weight_returns(returns[["B", "A"]], weights), expected)

    def test_holding_block_matches_constant_positions(self):
        returns = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
        weights = pd.Series({"A": 0.5, "B": 0.5})
        held, end_weights = hold_block(returns, weights)
        self.assertAlmostEqual((1 + held).prod(), 2.0)
        self.assertAlmostEqual(held.iloc[1], 1 / 3)
        np.testing.assert_allclose(end_weights, [0.5, 0.5])
        daily = fixed_weight_returns(returns, weights)
        self.assertAlmostEqual((1 + daily).prod(), 2.25)

    def test_simple_portfolio_return_is_exact(self):
        returns = pd.DataFrame({"A": [0.1], "B": [-0.1]})
        weights = pd.Series({"A": 0.5, "B": 0.5})
        self.assertAlmostEqual(fixed_weight_returns(returns, weights).iloc[0], 0.0)

    def test_drawdown_includes_initial_capital(self):
        returns = pd.Series([-0.2, 0.0], index=pd.bdate_range("2024-01-01", periods=2))
        self.assertAlmostEqual(performance_metrics(returns)["maximum_drawdown"], -0.2)

    def test_missing_or_mislabelled_weights_fail(self):
        returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.01]})
        with self.assertRaises(ValueError):
            fixed_weight_returns(returns, pd.Series({"A": 1.0}))

    @staticmethod
    def sample_returns(n=320):
        rng = np.random.default_rng(7401)
        x = rng.normal([0.001, 0.0007, 0.0004], [0.02, 0.012, 0.007], size=(n, 3))
        return pd.DataFrame(x, columns=["A", "B", "C"],
                            index=pd.bdate_range("2020-01-01", periods=n))

    def test_optimizer_is_feasible_and_permutation_invariant(self):
        returns = self.sample_returns()
        w, info = max_sharpe_weights(returns)
        reordered, _ = max_sharpe_weights(returns[["C", "A", "B"]])
        self.assertTrue(info["success"])
        self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue((w >= 0).all())
        np.testing.assert_allclose(w, reordered.reindex(w.index), atol=1e-6)

    def test_future_data_cannot_change_earlier_target(self):
        returns = self.sample_returns()
        first = walk_forward(returns)
        changed = returns.copy()
        changed.iloc[253:] = changed.iloc[253:] * 2
        second = walk_forward(changed)
        pd.testing.assert_series_equal(first.target_weights.iloc[0], second.target_weights.iloc[0])
        log = first.rebalances.iloc[0]
        self.assertLess(pd.Timestamp(log.training_end), pd.Timestamp(log.execution_close))
        self.assertLess(pd.Timestamp(log.execution_close), pd.Timestamp(log.first_test_date))

    def test_costs_reduce_wealth_without_changing_targets(self):
        returns = self.sample_returns()
        zero = walk_forward(returns, cost_bps=0)
        costly = walk_forward(returns, cost_bps=5)
        pd.testing.assert_frame_equal(zero.target_weights, costly.target_weights)
        gross_wealth = (1 + zero.returns).prod()
        net_wealth = (1 + costly.returns).prod()
        expected = gross_wealth * (1 - costly.rebalances.cost_fraction).prod()
        self.assertAlmostEqual(net_wealth, expected)
        self.assertLess(net_wealth, gross_wealth)
        self.assertAlmostEqual(costly.rebalances.traded_weight.iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
