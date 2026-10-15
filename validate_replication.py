"""Independent numerical checks for the three replication projects.

Usage: python validate_replication.py [--source-root PATH] [--notebook PATH]
Writes validation_report.json below source-root; exits nonzero on failed checks.
Author source is never executed by this validator. Missing source is a failure,
so an output-only checkout cannot claim that source hashes were verified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import norm


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, project, name, passed, details=None):
        row = {"project": project, "check": name, "passed": bool(passed)}
        if details is not None:
            row["details"] = details
        self.rows.append(row)
        print(f"{'PASS' if passed else 'FAIL'} {project}: {name}", flush=True)

    def close(self, project, name, actual, expected, atol=1e-8, rtol=1e-8):
        actual, expected = np.asarray(actual), np.asarray(expected)
        finite = np.isfinite(actual).all() and np.isfinite(expected).all()
        difference = float(np.max(np.abs(actual - expected))) if finite else None
        self.add(project, name, finite and np.allclose(actual, expected, atol=atol, rtol=rtol),
                 {"max_absolute_difference": difference, "atol": atol, "rtol": rtol})


def read_json(path):
    return json.loads(path.read_text())


def check_source(project, checks):
    manifest = read_json(project / "source_manifest.json")
    files = manifest["files"]
    entries = (list(files.items()) if isinstance(files, dict)
               else [(row["path"], row["sha256"]) for row in files])
    missing, mismatches = [], []
    for relative, expected in entries:
        # Dictionaries and the Shrinking manifest are relative to source/;
        # Stripping's manifest explicitly starts with original/.
        path = project / relative
        if not path.is_file():
            path = project / "source" / relative
        if not path.is_file():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(relative)
    checks.add(project.name, "original source SHA256 hashes", not missing and not mismatches,
               {"expected_files": len(entries), "verified_files": len(entries) - len(missing) - len(mismatches),
                "missing_source": missing, "hash_mismatches": mismatches,
                "remedy": "Run python setup_sources.py to download pinned author sources." if missing else None,
                "commit": manifest.get("commit")})


def check_stripping(project, checks):
    name = project.name
    metrics = pd.read_csv(project / "results" / "metrics.csv")
    checks.add(name, "two expected dated metric rows", len(metrics) == 2 and set(metrics.date) == {"1961-06-30", "2013-12-31"})
    for row in metrics.to_dict("records"):
        date = row["date"]
        bonds = pd.read_csv(project / "results" / f"bond_fits_{date}.csv")
        numeric = bonds.select_dtypes(include="number").to_numpy()
        checks.add(name, f"{date} finite per-bond output and row count", np.isfinite(numeric).all() and len(bonds) == row["securities"])
        price_error = bonds.fitted_price.to_numpy() - bonds.observed_price.to_numpy()
        relative_error_bp = price_error / bonds.observed_price.to_numpy() * 10000
        yield_error_bp = (bonds.fitted_ytm.to_numpy() - bonds.observed_ytm.to_numpy()) * 10000
        recomputed = [np.sqrt(np.mean(price_error**2)), np.sqrt(np.mean(relative_error_bp**2)),
                      np.sqrt(np.mean(yield_error_bp**2)), np.mean(np.abs(yield_error_bp))]
        reported = [row[key] for key in ("price_rmse", "relative_price_rmse_bps", "yield_rmse_bps", "yield_mae_bps")]
        checks.close(name, f"{date} price and yield RMSE/MAE from bond CSV", recomputed, reported)
        checks.close(name, f"{date} per-bond price/yield errors", np.column_stack([relative_error_bp, yield_error_bp]),
                     bonds[["relative_price_error_bps", "yield_error_bps"]].to_numpy())


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy._core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Disallowed data constructor: {module}.{name}")
        return allowed[(module, name)]


def closed_form_kernel(days, alpha=.05):
    x = np.arange(1, days + 1, dtype=float) / 365
    mn, mx = np.minimum(x[:, None], x[None, :]), np.maximum(x[:, None], x[None, :])
    return (-mn / alpha**2 * np.exp(-alpha * mn)
            + 2 / alpha**3 * (1 - np.exp(-alpha * mn))
            - mn / alpha**2 * np.exp(-alpha * mx))


def analytic_loocv(gram, rx, Z, smoothing):
    """Exact ridge PRESS formula with the author's per-training-sample penalty.

    Explicit solves handle holdouts that change the maximum cash-flow day.
    This implementation uses no author fitting or LOOCV function.
    """
    n = len(rx)
    max_day = np.flatnonzero(Z.sum(axis=0))[-1] + 1
    penalty = (n - 1) * smoothing / max_day
    inverse = solve(gram + penalty * np.eye(n), np.eye(n), assume_a="pos")
    error = (inverse @ rx) / np.diag(inverse)
    for i in range(n):
        train = np.arange(n) != i
        training_max_day = np.flatnonzero(Z[train].sum(axis=0))[-1] + 1
        if training_max_day != max_day:
            training_penalty = (n - 1) * smoothing / training_max_day
            coefficients = solve(gram[np.ix_(train, train)] + training_penalty * np.eye(n - 1), rx[train], assume_a="pos")
            error[i] = rx[i] - gram[i, train] @ coefficients
    return float(np.sqrt(np.mean(error**2)) * 10000)


def check_shrinking(project, checks):
    name = project.name
    results = project / "results"
    metrics, bonds = pd.read_csv(results / "model_metrics.csv"), pd.read_csv(results / "bond_predictions.csv")
    expected_dates = {"1986-06-30", "2013-12-31"}
    checks.add(name, "22 dated Full KR / KR-1 through KR-10 rows", len(metrics) == 22 and set(metrics.date) == expected_dates and not metrics.duplicated(["date", "model"]).any())
    for date, group in bonds.groupby("date"):
        for model, column in [("Full KR", "fitted_full_kr"), ("KR-4", "fitted_kr4")]:
            error = group.excess_return.to_numpy() - group[column].to_numpy()
            actual = [np.sqrt(np.mean(error**2)) * 10000, 1 - np.sum(error**2) / np.sum(group.excess_return.to_numpy()**2)]
            reported = metrics[(metrics.date == date) & (metrics.model == model)].iloc[0]
            checks.close(name, f"{date} {model} RMSE/R2 from per-bond CSV", actual,
                         [reported.rmse_in_sample_bp, reported.r2_uncentered_in_sample])

    # Raw data are required for independent all-rank fitting and PRESS checks.
    with (project / "source" / "example_data" / "dict_data.pkl").open("rb") as handle:
        panels = NumpyOnlyUnpickler(handle).load()
    basis = np.load(results / "kernel_top10.npz", allow_pickle=False)
    V, eigenvalues = basis["V"], basis["eigenvalues"]
    K = closed_form_kernel(3650)
    checks.close(name, "kernel basis orthonormality", V.T @ V, np.eye(10), atol=1e-10)
    residual = np.linalg.norm(K @ V - V * eigenvalues) / np.linalg.norm(K @ V)
    checks.add(name, "saved basis satisfies independent published kernel eigen-equations", residual < 1e-10,
               {"relative_residual": float(residual)})
    for date, panel in panels.items():
        s, B = panel["date_s"], panel["B"]
        Z = panel["C"][:, s:] / B[:, None] * panel["g"][s:3650]
        rx = panel["Bc_shift"] / B - (1 + panel["rf_t"])**s
        columns = np.flatnonzero(Z.sum(axis=0))
        Zm = Z[:, columns]
        full_gram = Zm @ K[np.ix_(columns, columns)] @ Zm.T
        penalty = len(rx) * 10 / (columns[-1] + 1)
        selected_bonds = bonds[bonds.date == date].sort_values("bond_index")
        checks.close(name, f"{date} observed returns from raw author data", rx, selected_bonds.excess_return.to_numpy(), atol=1e-12)
        for rank in [None] + list(range(1, 11)):
            label = "Full KR" if rank is None else f"KR-{rank}"
            if rank is None:
                gram = full_gram
                fitted = gram @ solve(gram + penalty * np.eye(len(rx)), rx, assume_a="pos")
            else:
                loadings = Zm @ (V[:, :rank] * np.sqrt(eigenvalues[:rank]))[columns]
                # Primal r-coefficient ridge solve, independent of author dual fit.
                factors = solve(loadings.T @ loadings + penalty * np.eye(rank), loadings.T @ rx, assume_a="pos")
                fitted = loadings @ factors
                gram = loadings @ loadings.T
            error = rx - fitted
            actual = [np.sqrt(np.mean(error**2)) * 10000, 1 - np.sum(error**2) / np.sum(rx**2)]
            reported = metrics[(metrics.date == date) & (metrics.model == label)].iloc[0]
            checks.close(name, f"{date} independently fitted {label} RMSE/R2", actual,
                         [reported.rmse_in_sample_bp, reported.r2_uncentered_in_sample])
            if rank in (None, 4):
                loo_rmse = analytic_loocv(gram, rx, Z, 10)
                checks.close(name, f"{date} {label} independently derived analytic LOOCV", loo_rmse,
                             reported.rmse_leave_one_bond_out_bp)


def independent_sharpe_variance(sr, t, skew=0, kurt=3, rho=0):
    return ((1 + rho) / (1 - rho)
            - skew * sr * (1 + rho + rho**2) / (1 - rho**2)
            + (kurt - 1) * sr**2 / 4 * (1 + rho**2) / (1 - rho**2)) / t


def check_sharpe(project, checks):
    name, results = project.name, project / "results"
    numeric = pd.read_csv(results / "numeric_example.csv")
    checks.add(name, "numeric example has both rho cases and finite outputs", len(numeric) == 2 and set(numeric.rho) == {0, .2} and np.isfinite(numeric.to_numpy()).all())
    for row in numeric.to_dict("records"):
        variance = independent_sharpe_variance(row["observed_monthly_SR"], 24, -2.448, 10.164, row["rho"])
        null_variance = independent_sharpe_variance(0, 24, -2.448, 10.164, row["rho"])
        psr = norm.cdf(row["observed_monthly_SR"] / np.sqrt(null_variance))
        checks.close(name, f"rho={row['rho']} independent K=1 variance / PSR / p-value",
                     [variance, psr, 1 - psr],
                     [row["generalized_standard_error"]**2, row["PSR"], row["p_value"]], atol=1e-12)
    grid = pd.read_csv(results / "exhibit2_full_grid.csv")
    expected_sr = np.array([0, .15, .30, .45, .60])
    expected_rho = np.linspace(0, .5, 100)
    names = {"gaussian", "mild", "moderate", "severe"}
    checks.add(name, "complete unique 5 by 100 by 4 simulation grid", len(grid) == 2000 and set(grid.name) == names
               and not grid.duplicated(["name", "SR", "rho"]).any()
               and all(len(group) == 100 and np.allclose(np.sort(group.rho), expected_rho) for _, group in grid.groupby(["name", "SR"]))
               and np.array_equal(np.sort(grid.SR.unique()), expected_sr))
    numeric_columns = grid.select_dtypes(include="number")
    variance_columns = [column for column in grid.columns if column not in {"name", "SR", "rho"}]
    checks.add(name, "all 2000 grid outputs finite and variance estimates positive", np.isfinite(numeric_columns.to_numpy()).all()
               and (grid[variance_columns].to_numpy() > 0).all())
    checks.close(name, "grid Gaussian iid formula independently recomputed", independent_sharpe_variance(grid.SR, 60), grid["Gaussian iid"].to_numpy(), atol=1e-12)
    checks.close(name, "grid Gaussian AR(1) formula independently recomputed", independent_sharpe_variance(grid.SR, 60, rho=grid.rho), grid["Gaussian + autocorrelation"].to_numpy(), atol=1e-12)
    errors = pd.read_csv(results / "exhibit2_approximation_errors.csv")
    checks.add(name, "all 16 formula/distribution error-report rows", len(errors) == 16 and not errors.duplicated(["distribution", "formula"]).any())
    discrepancies = []
    for row in errors.to_dict("records"):
        group = grid[grid.name == row["distribution"]]
        difference = np.abs(group[row["formula"]].to_numpy() / group.simulation.to_numpy() - 1)
        checks.close(name, f"reported simulation discrepancy: {row['distribution']} / {row['formula']}",
                     [np.median(difference), np.max(difference)],
                     [row["median_absolute_relative_error"], row["max_absolute_relative_error"]], atol=1e-12)
        discrepancies.append({"distribution": row["distribution"], "formula": row["formula"],
                              "median_absolute_relative_error": float(np.median(difference)),
                              "max_absolute_relative_error": float(np.max(difference))})
    return {"reported_discrepancies": discrepancies,
            "interpretation": "The theoretical variance is approximate; finite-sample Monte Carlo and transformed copula dependence need not equal it. Discrepancies are measured, not forced to pass an equality assertion. PSR is a one-sided significance quantity, not a posterior probability."}


def check_notebook(path, checks):
    notebook = read_json(path)
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "".join(cell["source"]).strip()]
    errors = [{"code_cell": i, "ename": output.get("ename"), "evalue": output.get("evalue")}
              for i, cell in enumerate(code) for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    checks.add("notebook", "every code cell executed without error", bool(code) and all(cell.get("execution_count") is not None for cell in code) and not errors,
               {"code_cells": len(code), "executed_cells": sum(cell.get("execution_count") is not None for cell in code), "errors": errors})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args()
    checks, detail = Checks(), {}
    validators = {"stripping_discount_curve": check_stripping, "shrinking_term_structure": check_shrinking, "sharpe_ratio_inference": check_sharpe}
    for name, validator in validators.items():
        project = args.source_root / name
        try:
            check_source(project, checks)
        except Exception as error:
            checks.add(name, "original source SHA256 hashes", False, {"error": f"{type(error).__name__}: {error}"})
        try:
            result = validator(project, checks)
            if result is not None:
                detail[name] = result
        except Exception as error:
            checks.add(name, "independent numerical checks completed", False, {"error": f"{type(error).__name__}: {error}"})
    if args.notebook is not None:
        try:
            check_notebook(args.notebook, checks)
        except Exception as error:
            checks.add("notebook", "execution verified", False, {"error": f"{type(error).__name__}: {error}"})
    passed = all(row["passed"] for row in checks.rows)
    report = {"passed": passed, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "checks_passed": sum(row["passed"] for row in checks.rows), "checks_total": len(checks.rows),
              "checks": checks.rows, "details": detail,
              "scope": "Source-integrity and independent numerical-output verification of the stated core experiments; not full-paper empirical replication or a trading-performance certification."}
    destination = args.source_root / "validation_report.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Validation {'PASSED' if passed else 'FAILED'}: {report['checks_passed']}/{report['checks_total']} checks. Report: {destination}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
