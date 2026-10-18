"""Experiment infrastructure for the covariance study, with student-supplied fits.

This module does not fill the notebook's GMV or shrinkage exercises. The oracle
is calculated independently from the known simulated covariance, and is never
used to choose a shrinkage strength. Prepared with assistance, 7 October 2026.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from importlib.metadata import version
import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    p: int = 10
    sample_sizes: tuple = (40, 120, 252)
    replications: int = 300
    lambdas: tuple = (0.0, 0.05, 0.25, 0.75, 1.0)
    seed: int = 20261007
    scale: float = 0.01


def covariance_scenarios(p=10, scale=0.01):
    """Known SPD covariances; a common scale does not change GMV weights."""
    if p < 2 or p % 2 or scale <= 0:
        raise ValueError("Use an even dimension >= 2 and a positive scale.")
    relative_vol = np.linspace(0.6, 1.8, p)
    d = np.diag(relative_vol)
    block = np.zeros((p, p))
    half = p // 2
    block[:half, :half] = 1
    block[half:, half:] = 1
    corr = 0.3 * np.eye(p) + 0.6 * block + 0.1 * np.ones((p, p))
    return {
        "equal_variance_independent": scale**2 * np.eye(p),
        "unequal_variance_independent": scale**2 * d @ d,
        "unequal_variance_blocks": scale**2 * d @ corr @ d,
    }


def _check_covariance(cov, p):
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (p, p) or not np.isfinite(cov).all():
        raise ValueError("Covariance has the wrong shape or nonfinite entries.")
    if not np.allclose(cov, cov.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Covariance is not symmetric.")
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues[0] <= 0:
        raise ValueError("This grid requires a positive-definite estimate.")
    return cov, eigenvalues


def _check_weights(w, p):
    w = np.asarray(w, dtype=float)
    if w.shape != (p,) or not np.isfinite(w).all():
        raise ValueError("GMV must return a finite length-p vector.")
    if not np.isclose(w.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError("The allocation violates the fully invested constraint.")
    # Negative weights are allowed in this equality-constrained study.
    return w


def validate_implementations(fit_gmv, shrink_covariance):
    """Check exercises against a worked analytic example before simulation."""
    s = np.diag([1.0, 9.0])
    fitted = _check_weights(fit_gmv(s.copy()), 2)
    np.testing.assert_allclose(fitted, [0.9, 0.1], atol=1e-8)
    shrunk, _ = _check_covariance(shrink_covariance(s.copy(), 0.5), 2)
    np.testing.assert_allclose(shrunk, np.diag([3.0, 7.0]), atol=1e-10)
    np.testing.assert_allclose(fit_gmv(shrunk), [0.7, 0.3], atol=1e-8)
    np.testing.assert_allclose(shrink_covariance(s.copy(), 0), s, atol=1e-10)
    np.testing.assert_allclose(shrink_covariance(s.copy(), 1), 5 * np.eye(2), atol=1e-10)
    return True


def run_experiment(fit_gmv, shrink_covariance, config=ExperimentConfig()):
    """Use identical samples across methods within each independent replication."""
    validate_implementations(fit_gmv, shrink_covariance)
    if config.replications < 2 or any(n <= config.p for n in config.sample_sizes):
        raise ValueError("Use >= 2 replications and n > p for the core SPD grid.")
    if 0.0 not in config.lambdas or any(not 0 <= x <= 1 for x in config.lambdas):
        raise ValueError("Declare lambda=0 and keep all strengths in [0,1].")
    if len(set(config.lambdas)) != len(config.lambdas):
        raise ValueError("Shrinkage strengths must be unique.")
    rng = np.random.default_rng(config.seed)
    records = []
    for scenario, true_cov in covariance_scenarios(config.p, config.scale).items():
        factor = np.linalg.cholesky(true_cov)
        v = np.linalg.solve(true_cov, np.ones(config.p))
        oracle = v / v.sum()
        oracle_variance = float(oracle @ true_cov @ oracle)
        for n in config.sample_sizes:
            for trial in range(config.replications):
                sample = rng.standard_normal((n, config.p)) @ factor.T
                s = np.cov(sample, rowvar=False, ddof=1)
                candidates = []
                for lam in config.lambdas:
                    estimate, eigenvalues = _check_covariance(
                        shrink_covariance(s.copy(), lam), config.p
                    )
                    w = _check_weights(fit_gmv(estimate.copy()), config.p)
                    method = "sample_gmv" if lam == 0 else f"shrinkage_{lam:g}"
                    cov_error = np.linalg.norm(estimate - true_cov, "fro")**2 / np.linalg.norm(true_cov, "fro")**2
                    candidates.append((method, lam, w, float(eigenvalues[-1] / eigenvalues[0]), float(cov_error)))

                inverse_variance = 1 / np.diag(s)
                inverse_variance /= inverse_variance.sum()
                candidates += [
                    ("equal_weight", np.nan, np.full(config.p, 1 / config.p), np.nan, np.nan),
                    ("inverse_variance", np.nan, inverse_variance, np.nan, np.nan),
                ]
                for method, lam, w, condition, cov_error in candidates:
                    true_variance = float(w @ true_cov @ w)
                    inflation = true_variance / oracle_variance
                    if inflation < 1 - 1e-8:
                        raise RuntimeError("Risk below the same-constraint oracle: check weights and metric.")
                    records.append({
                        "scenario": scenario, "p": config.p, "n": n, "trial": trial,
                        "method": method, "lambda": lam,
                        "covariance_condition_number": condition,
                        "relative_covariance_error": cov_error,
                        "true_variance": true_variance,
                        "oracle_variance": oracle_variance,
                        "risk_inflation": inflation,
                        "weight_error": float(np.linalg.norm(w - oracle)),
                        "gross_exposure": float(np.abs(w).sum()),
                    })
    return pd.DataFrame(records)


def summarize(records):
    """Report paired changes against sample GMV, with Monte Carlo standard errors."""
    keys = ["scenario", "p", "n", "trial"]
    base = records.loc[records.method == "sample_gmv", keys + ["risk_inflation"]]
    base = base.rename(columns={"risk_inflation": "sample_risk_inflation"})
    paired = records.merge(base, on=keys, how="left", validate="many_to_one")
    if paired.sample_risk_inflation.isna().any():
        raise ValueError("Every trial needs its paired sample-GMV result.")
    paired["change_vs_sample"] = paired.risk_inflation - paired.sample_risk_inflation
    rows = []
    for (scenario, p, n, method), g in paired.groupby(["scenario", "p", "n", "method"], sort=False):
        count = len(g)
        mcse = float(g.risk_inflation.std(ddof=1) / np.sqrt(count))
        paired_mcse = float(g.change_vs_sample.std(ddof=1) / np.sqrt(count))
        mean = float(g.risk_inflation.mean())
        change = float(g.change_vs_sample.mean())
        rows.append({
            "scenario": scenario, "p": p, "n": n, "method": method,
            "lambda": g["lambda"].iloc[0], "replications": count,
            "mean_risk_inflation": mean, "median_risk_inflation": float(g.risk_inflation.median()),
            "risk_inflation_q90": float(g.risk_inflation.quantile(0.9)),
            "risk_mcse": mcse, "risk_mc_lower": mean - 1.96 * mcse,
            "risk_mc_upper": mean + 1.96 * mcse,
            "median_condition_number": float(g.covariance_condition_number.median()),
            "mean_relative_covariance_error": float(g.relative_covariance_error.mean()),
            "mean_weight_error": float(g.weight_error.mean()),
            "mean_gross_exposure": float(g.gross_exposure.mean()),
            "paired_change_vs_sample": change, "paired_mcse": paired_mcse,
            "paired_mc_lower": change - 1.96 * paired_mcse,
            "paired_mc_upper": change + 1.96 * paired_mcse,
        })
    return pd.DataFrame(rows)


def plot_summary(summary, folder, mode):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    scenarios = list(summary.scenario.unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(13, 4.2), squeeze=False)
    fig2, axes2 = plt.subplots(1, len(scenarios), figsize=(13, 4.2), squeeze=False)
    labels = {"equal_variance_independent": "Equal variances, independent",
              "unequal_variance_independent": "Unequal variances, independent",
              "unequal_variance_blocks": "Unequal variances, correlation blocks"}
    for j, scenario in enumerate(scenarios):
        data = summary.loc[(summary.scenario == scenario) & summary["lambda"].notna()]
        for n, group in data.groupby("n"):
            group = group.sort_values("lambda")
            axes[0, j].errorbar(group["lambda"], group.mean_risk_inflation,
                               yerr=1.96 * group.risk_mcse, marker="o", capsize=3,
                               linewidth=1.3, label=f"n={n}")
            axes2[0, j].plot(group["lambda"], group.median_condition_number,
                             marker="o", linewidth=1.3, label=f"n={n}")
        axes[0, j].axhline(1, color="black", linestyle=":", linewidth=1, label="Oracle")
        for ax in (axes[0, j], axes2[0, j]):
            ax.set_xlabel("Shrinkage strength")
            ax.set_title(labels[scenario], fontsize=10)
            ax.grid(alpha=0.2)
        axes2[0, j].set_yscale("log")
    axes[0, 0].set_ylabel("Mean true risk / oracle risk")
    axes2[0, 0].set_ylabel("Median covariance condition number (log scale)")
    axes[0, -1].legend(fontsize=8)
    axes2[0, -1].legend(fontsize=8)
    for figure, name, title in [(fig, "risk-inflation.png", "True risk and Monte Carlo precision"),
                                (fig2, "conditioning.png", "Numerical conditioning")]:
        figure.suptitle(f"{title} - {mode}", fontsize=12)
        figure.tight_layout()
        figure.savefig(folder / name, dpi=180)
        plt.close(figure)


def save_experiment(records, config, folder, mode="preview"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    records.to_csv(folder / "replications.csv", index=False)
    summary.to_csv(folder / "summary.csv", index=False)
    metadata = {
        "mode": mode, "configuration": asdict(config),
        "note": "Fixed strengths; no oracle-based lambda selection. Monte Carlo intervals measure simulation precision.",
        "scale_note": "Relative standard deviations are multiplied by a common 0.01 scale; weights and risk ratios are invariant.",
        "versions": {name: version(name) for name in ["numpy", "pandas", "matplotlib"]},
    }
    (folder / "settings.json").write_text(json.dumps(metadata, indent=2))
    plot_summary(summary, folder, mode)
    return summary
