"""Run the unmodified authors' KR estimator on both bundled Treasury samples.

Usage in Jupyter: import importlib.util; load this file; call run().
No network calls or environment installations occur in this runner.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runtime_cache" / "matplotlib"))


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_sources():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for row in manifest["files"]:
        actual = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
        if actual != row["sha256"]:
            raise AssertionError(f"Original source changed: {row['path']}")
    return manifest


def run(show_plots=False):
    import numpy as np
    import pandas as pd
    import scipy
    import scipy.sparse as sps
    from scipy.optimize import brentq
    from threadpoolctl import threadpool_limits
    import matplotlib
    import matplotlib.pyplot as plt

    start = time.perf_counter()
    OUTPUT.mkdir(exist_ok=True)
    manifest = _verify_sources()
    original = ROOT / "original" / "author"
    model = _module("stripping_author_kr_model", original / "src" / "kr_model.py")
    utils = _module("stripping_author_kr_utils", original / "src" / "kr_utils.py")
    dates = ["1961-06-30", "2013-12-31"]
    alpha, delta, ridge, n = 0.05, 0.0, 1.0, 30 * 365
    print("[Stripping] Original source hashes verified; loading real bundled Treasury data.", flush=True)
    samples = {}
    checks = {"source_hashes_verified": True}
    for date in dates:
        b = np.load(original / f"example_data/price_{date}.npy", allow_pickle=False)
        c = sps.load_npz(original / f"example_data/cashflow_{date}.npz").toarray()
        assert c.shape[0] == len(b) and np.all(np.isfinite(b)) and np.all(b > 0)
        ytm, duration, maturities, ytm_exact = [], [], [], []
        for i, row in enumerate(c):
            ix = np.flatnonzero(row) + 1
            cf = row[ix - 1]
            y, dur = utils.get_ytm_and_duration(cf, ix, b[i])
            exact = brentq(lambda z: np.sum(cf * np.exp(-ix / 365 * z)) - b[i], -0.5, 1.0)
            ytm.append(y); duration.append(dur); maturities.append(ix[-1]); ytm_exact.append(exact)
        ytm, duration, maturities, ytm_exact = map(np.asarray, (ytm, duration, maturities, ytm_exact))
        np.testing.assert_allclose(ytm, ytm_exact, rtol=0, atol=1e-7)
        samples[date] = dict(B=b, C=c, ytm=ytm, dur=duration, ttm=maturities,
                             inv_w=(duration * b) ** 2 * len(b))
        checks[f"{date}_author_vs_brentq_ytm_max_abs"] = float(np.max(np.abs(ytm - ytm_exact)))
        print(f"  {date}: {len(b)} bonds; maximum maturity {maturities.max() / 365:.3f} years", flush=True)

    print(f"[Stripping] Generating the authors' exact {n} x {n} kernel (~{n*n*8/1e9:.2f} GB).", flush=True)
    kernel_start = time.perf_counter()
    k = model.generate_kernel_matrix(alpha, delta, n, n)
    kernel_seconds = time.perf_counter() - kernel_start
    indices = np.array([0, 6, 29, 89, 364, 729, 1824, 3649, 7299, n - 1])
    sample_kernel = k[np.ix_(indices, indices)]
    np.testing.assert_allclose(sample_kernel, sample_kernel.T, rtol=0, atol=1e-12)
    x = (indices[:, None] + 1) / 365
    z = (indices[None, :] + 1) / 365
    mn, mx = np.minimum(x, z), np.maximum(x, z)
    independent_kernel = (-mn / alpha**2 * np.exp(-alpha * mn)
                          - 2 / alpha**3 * np.expm1(-alpha * mn)
                          - mn / alpha**2 * np.exp(-alpha * mx))
    np.testing.assert_allclose(sample_kernel, independent_kernel, rtol=1e-8, atol=5e-11)
    checks["kernel_formula_max_abs_difference"] = float(np.max(np.abs(sample_kernel - independent_kernel)))
    checks["kernel_sample_symmetry_max_abs_difference"] = float(np.max(np.abs(sample_kernel - sample_kernel.T)))

    results, metrics, sensitivities = {}, [], {}
    print("[Stripping] Estimating both dates with unmodified authors' KR().", flush=True)
    with threadpool_limits(limits=2):
        for date, s in samples.items():
            b, c, inv_w = s["B"], s["C"], s["inv_w"]
            fit = model.KR(C=c, B=b, ridge=ridge, inv_w=inv_w, K=k)
            mask = np.flatnonzero(c.sum(axis=0))
            cm = c[:, mask]
            km = k[np.ix_(mask, mask)]
            lam = ridge / (mask[-1] + 1)
            q = cm @ km @ cm.T
            a = q + lam * np.diag(inv_w)
            rhs = b - cm @ np.ones(len(mask))
            dual = np.linalg.solve(a, rhs)
            independent_g = 1 + k[:, mask] @ cm.T @ dual
            np.testing.assert_allclose(fit["g_solved"], independent_g, rtol=1e-7, atol=1e-7)
            checks[f"{date}_inverse_vs_solve_curve_max_abs"] = float(np.max(np.abs(fit["g_solved"] - independent_g)))
            checks[f"{date}_normal_equation_relative_residual"] = float(np.linalg.norm(a @ dual - rhs) / np.linalg.norm(rhs))
            assert checks[f"{date}_normal_equation_relative_residual"] < 1e-8
            max_maturity = int(s["ttm"].max())
            assert np.all(np.isfinite(fit["g_solved"][:max_maturity]))
            assert np.all(fit["g_solved"][:max_maturity] > 0)
            fitted_prices = c @ fit["g_solved"][:c.shape[1]]
            fitted_yields = []
            for i, row in enumerate(c):
                ix = np.flatnonzero(row) + 1
                fitted_yields.append(utils.get_ytm_and_duration(row[ix - 1], ix, fitted_prices[i])[0])
            fitted_yields = np.asarray(fitted_yields)
            rprice_bps = (fitted_prices / b - 1) * 1e4
            yerr_bps = (fitted_yields - s["ytm"]) * 1e4
            row = dict(date=date, securities=len(b), max_maturity_years=max_maturity / 365,
                       price_rmse=float(np.sqrt(np.mean((fitted_prices - b)**2))),
                       relative_price_rmse_bps=float(np.sqrt(np.mean(rprice_bps**2))),
                       yield_rmse_bps=float(np.sqrt(np.mean(yerr_bps**2))),
                       yield_mae_bps=float(np.mean(np.abs(yerr_bps))),
                       discount_positive_within_observed_maturities=True)
            metrics.append(row)
            results[date] = dict(fit=fit, fitted_prices=fitted_prices, fitted_yields=fitted_yields)
            sensitivities[date] = {}
            for value in [0.1, 1.0, 10.0]:
                sensitivities[date][value] = (fit if value == ridge else model.KR(c, b, value, inv_w, k))["y_solved"]
            pd.DataFrame(dict(security_index=np.arange(len(b)), maturity_years=s["ttm"] / 365,
                              observed_price=b, fitted_price=fitted_prices, observed_ytm=s["ytm"],
                              fitted_ytm=fitted_yields, relative_price_error_bps=rprice_bps,
                              yield_error_bps=yerr_bps)).to_csv(OUTPUT / f"bond_fits_{date}.csv", index=False)
            pd.DataFrame(dict(maturity_days=np.arange(1, n+1), discount=fit["g_solved"],
                              log_yield=fit["y_solved"])).to_csv(OUTPUT / f"curves_{date}.csv", index=False)
            print(f"  {date}: yield RMSE {row['yield_rmse_bps']:.6f} bps; relative pricing RMSE {row['relative_price_rmse_bps']:.6f} bps", flush=True)

    # Original-example views: discount/yield curves and observed/fitted prices/YTM.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for j, date in enumerate(dates):
        lim = int(samples[date]["ttm"].max()); t = np.arange(1, lim+1) / 365
        axes[j, 0].plot(t, results[date]["fit"]["g_solved"][:lim])
        axes[j, 1].plot(t, results[date]["fit"]["y_solved"][:lim] * 100)
        axes[j, 0].set(title=f"Discount curve: {date}", xlabel="Maturity (years)", ylabel="Discount factor")
        axes[j, 1].set(title=f"Zero-coupon log yield: {date}", xlabel="Maturity (years)", ylabel="Yield (%)")
    for ax in axes.flat: ax.grid(alpha=0.2)
    fig.savefig(OUTPUT / "discount_yield_curves.png", dpi=160)

    fitfig, fitaxes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for j, date in enumerate(dates):
        s, r = samples[date], results[date]
        for ax, obs, pred, label in [(fitaxes[j, 0], s["B"], r["fitted_prices"], "Bond price"),
                                     (fitaxes[j, 1], s["ytm"] * 100, r["fitted_yields"] * 100, "YTM (%)")]:
            ax.scatter(s["ttm"] / 365, obs, s=15, label="Observed", alpha=0.7)
            ax.scatter(s["ttm"] / 365, pred, s=15, marker="x", label="Fitted")
            ax.set(title=f"{label}: {date}", xlabel="Maturity (years)", ylabel=label)
            ax.legend(); ax.grid(alpha=0.2)
    fitfig.savefig(OUTPUT / "observed_fitted_prices_yields.png", dpi=160)

    sensfig, sensaxes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for j, date in enumerate(dates):
        lim = int(samples[date]["ttm"].max()); t = np.arange(1, lim+1) / 365
        for value in [0.1, 1.0, 10.0]:
            sensaxes[j].plot(t, sensitivities[date][value][:lim] * 100, label=f"lambda={value:g}")
        sensaxes[j].set(title=f"Ridge sensitivity: {date}", xlabel="Maturity (years)", ylabel="Yield (%)")
        sensaxes[j].legend(); sensaxes[j].grid(alpha=0.2)
    sensfig.savefig(OUTPUT / "ridge_sensitivity.png", dpi=160)
    if show_plots: plt.show()
    else:
        for f in [fig, fitfig, sensfig]: plt.close(f)

    pd.DataFrame(metrics).to_csv(OUTPUT / "metrics.csv", index=False)
    summary = dict(
        paper="Stripping the Discount Curve — a Robust Machine Learning Approach",
        authors=["Damir Filipovic", "Markus Pelger", "Ye Ye"],
        repo=manifest["repo"], commit=manifest["commit"],
        paper_url="https://pubsonline.informs.org/doi/10.1287/mnsc.2023.01401",
        author_data_code_url="https://mpelger.people.stanford.edu/data-and-code",
        paper_version_formula_checked="Author March 10, 2023 draft hosted by Yale",
        paper_version_url="https://economics.yale.edu/sites/default/files/2023-03/Pelger.pdf",
        first_draft="2022-03-15", first_online_publication="2025-12-22",
        journal_issue="Management Science 72(9), September 2026, 8057-8120",
        scope="Exact original KR_example.ipynb estimator/settings on two bundled real U.S. Treasury dates; supplementary ridge sensitivity and numerical validation.",
        limitation="This is a two-date core-method replication, not the full historical empirical study, benchmark tables, or confidence intervals. Full raw Treasury panel requires WRDS/CRSP access. Journal-version figure numbering was not verified.",
        paper_mapping="2023 draft Theorem 2, Eqs (10)-(11) closed-form KR, Eq (14) kernel, Eq (17) duration weights; two of three dates in Appendix Figure A.5(a) ridge sensitivity and Figure A.9 observed/fitted YTM. Figure panels and grids are partial, not complete reproductions.",
        settings=dict(alpha=alpha, delta=delta, ridge=ridge, max_maturity_days=n, dates=dates, blas_threads=2),
        metrics=metrics, checks=checks, all_checks_passed=True,
        kernel_seconds=kernel_seconds, elapsed_seconds=time.perf_counter()-start,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         pandas=pd.__version__, matplotlib=matplotlib.__version__),
        plots=["discount_yield_curves.png", "observed_fitted_prices_yields.png", "ridge_sensitivity.png"],
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = dict(title=summary["paper"], authors=summary["authors"], publication_date=summary["first_online_publication"],
                    journal_issue=summary["journal_issue"], first_draft=summary["first_draft"], paper_url=summary["paper_url"],
                    scope=summary["scope"], limitations=summary["limitation"], formula_source_version=summary["paper_version_formula_checked"],
                    formula_source_url=summary["paper_version_url"], paper_mapping=summary["paper_mapping"])
    (ROOT / "paper_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Stripping] All numerical checks passed. Saved summary.json and 3 plots; {summary['elapsed_seconds']:.1f} seconds.", flush=True)
    return summary


if __name__ == "__main__":
    run()
