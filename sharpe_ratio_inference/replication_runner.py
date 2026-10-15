"""Reproduce the author's numeric example and full Exhibit 2 simulation grid.

Source files are untouched. Only Ray scheduling is replaced by a serial loop,
with an explicit seed because the author's Exhibit 2 notebook has no seed.
The copula-based generator and approximate variance formula are the author's.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT / "source"))

import argparse
import ast
import contextlib
import hashlib
import io
import json
import time
from datetime import datetime, timezone
import nbformat
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
author = None  # Import third-party code only after validating its fixed-commit hashes.

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def check_source():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / "source" / name).read_bytes()).hexdigest() == expected, name
    print(f'Author source hashes verified: {manifest["commit"]}', flush=True)
    return manifest


def independent_variance(sr, t, skew=0, kurt=3, rho=0):
    # K=1: closed-form expression, independent of author's A/B/C computation.
    return ((1 + rho) / (1 - rho)
            - skew * sr * (1 + rho + rho**2) / (1 - rho**2)
            + (kurt - 1) * sr**2 / 4 * (1 + rho**2) / (1 - rho**2)) / t


def numeric_example():
    test_names = ["test_sharpe_ratio_variance", "test_minimum_track_record_length",
                  "test_probabilistic_sharpe_ratio", "test_sharpe_ratio_power",
                  "test_pFDR", "test_oFDR", "test_FDR_critical_value"]
    for name in test_names:
        getattr(author, name)()
    print(f"Author inference tests passed: {len(test_names)}", flush=True)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        author.test_numeric_example()
    (RESULTS / "author_numeric_example.log").write_text(output.getvalue())
    print(output.getvalue(), flush=True)
    rows = []
    sr, t, skew, kurt, alpha = .036 / .079, 24, -2.448, 10.164, .10
    for rho in [0, .2]:
        args = dict(gamma3=skew, gamma4=kurt, rho=rho)
        variance = author.sharpe_ratio_variance(sr, t, **args)
        independent = independent_variance(sr, t, skew, kurt, rho)
        independent_psr = scipy.stats.norm.cdf(sr / np.sqrt(independent_variance(0, t, skew, kurt, rho)))
        psr = author.probabilistic_sharpe_ratio(sr, 0, T=t, **args)
        alpha_fdr, beta_fdr, critical_fdr, estimated_fdr = author.control_for_FDR(.25, SR0=0, SR1=.5, p_H1=.1, T=t, **args)
        independent_fdr = .9 * alpha_fdr / (.9 * alpha_fdr + .1 * (1 - beta_fdr))
        assert abs(variance - independent) < 1e-12
        assert abs(psr - independent_psr) < 1e-12
        assert abs(estimated_fdr - independent_fdr) < 1e-12
        assert abs(estimated_fdr - .25) < 1e-8
        rows.append(dict(rho=rho, observed_monthly_SR=sr,
                         generalized_standard_error=np.sqrt(variance),
                         gaussian_iid_standard_error=np.sqrt(author.sharpe_ratio_variance(sr, t)),
                         minimum_track_record_months=author.minimum_track_record_length(sr, 0, alpha=alpha, **args),
                         PSR=psr, p_value=1-psr,
                         power=author.sharpe_ratio_power(0, .5, t, alpha=alpha, **args),
                         planned_FDR=author.pFDR(.1, alpha, 1-author.sharpe_ratio_power(0, .5, t, alpha=alpha, **args)),
                         FDR_controlled_critical_SR=critical_fdr,
                         target_FDR=.25, achieved_FDR=estimated_fdr,
                         variance_formula_absolute_error=abs(variance-independent)))
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "numeric_example.csv", index=False)
    return {"author_tests_passed": test_names, "independent_formula_check_passed": True,
            "rows": rows, "scope": "Author functions.py numeric example, plus independent K=1 formula checks"}


def exhibit2():
    notebook = nbformat.read(ROOT / "source" / "Exhibit_2_Variance_Plot.ipynb", as_version=4)
    cell = next(c.source for c in notebook.cells if c.cell_type == "code" and "def f3(" in c.source)
    tree = ast.parse(cell)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "f3")
    function.decorator_list = []  # remove Ray scheduling; keep calculation body verbatim
    scope = dict(np=np, scipy=__import__("scipy"),
                 generate_autocorrelated_non_gaussian_data=author.generate_autocorrelated_non_gaussian_data,
                 sharpe_ratio_variance=author.sharpe_ratio_variance)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "source" / "Exhibit_2_Variance_Plot.ipynb"), "exec"), scope)
    np.random.seed(20261015)
    sr_grid = [0, .15, .30, .45, .60]
    rho_grid = np.linspace(0, .5, 100)
    distributions = ["gaussian", "mild", "moderate", "severe"]
    rows, start = [], time.perf_counter()
    total = len(sr_grid) * len(rho_grid) * len(distributions)
    print(f"Exhibit 2 full grid: {total} cases, T=60, 10,000 paths/case, seed=20261015", flush=True)
    for sr in sr_grid:
        for rho in rho_grid:
            for name in distributions:
                row = scope["f3"](sr, rho, 60, name)
                rows.append(row)
                if len(rows) % 100 == 0:
                    print(f"Completed {len(rows)}/{total} | {time.perf_counter()-start:.1f}s", flush=True)
    table = pd.DataFrame(rows)
    assert len(table) == total
    assert np.isfinite(table.iloc[:, 1:].to_numpy()).all()
    assert (table.iloc[:, 3:].to_numpy() > 0).all()
    table.to_csv(RESULTS / "exhibit2_full_grid.csv", index=False)
    errors = []
    for name, group in table.groupby("name", sort=False):
        for formula in table.columns[4:]:
            relative = np.abs(group[formula] / group["simulation"] - 1)
            errors.append(dict(distribution=name, formula=formula,
                               median_absolute_relative_error=float(relative.median()),
                               max_absolute_relative_error=float(relative.max())))
    pd.DataFrame(errors).to_csv(RESULTS / "exhibit2_approximation_errors.csv", index=False)
    for name in distributions:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
        subset = table[table["name"] == name]
        for sr, ax in zip(sr_grid, axes.flat):
            group = subset[subset["SR"] == sr]
            for column in table.columns[3:]:
                ax.plot(group["rho"], group[column], label=column,
                        color="black" if column == "simulation" else None,
                        linewidth=2.5 if column == "simulation" else 1.3)
            ax.set(xlabel="Generator rho", ylabel="Sharpe estimator variance", title=f"Monthly SR={sr:.2f}")
            ax.grid(alpha=.2)
        axes.flat[-1].axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, labels, loc="center")
        fig.suptitle(f"Author Exhibit 2: {name}, T=60, 10,000 paths/case")
        fig.savefig(RESULTS / f"exhibit2_{name}.png", dpi=150)
        plt.close(fig)
    return dict(cases=total, paths_per_case=10000, T=60, seed=20261015,
                sr_grid=sr_grid, rho_grid_points=100, distributions=distributions,
                author_function_body_unchanged=True,
                scheduling="Serial instead of Ray, with explicit seed absent from original notebook",
                generator="Author Gaussian copula AR(1) + interpolated empirical mixture marginals; rho denotes the generator parameter",
                runtime_seconds=time.perf_counter()-start,
                approximation_errors=errors,
                limitation="This is the pinned February 2026 author-code experiment. The March 2026 revised paper PDF has not been checked figure by figure. Non-normal copula dependence does not exactly satisfy the closed-form AR(1) moment restrictions; approximation discrepancies are reported, not treated as exact-validation failures.")


def main():
    global author
    import importlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["numeric", "exhibit2", "all"], default="all")
    args = parser.parse_args()
    manifest = check_source()
    author = importlib.import_module("functions")
    path = RESULTS / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary.update(repository=manifest["repository"], commit=manifest["commit"],
                   paper_url="https://ssrn.com/abstract=5520741", interpreter=Path(sys.executable).name,
                   executed_at=datetime.now(timezone.utc).isoformat())
    if args.mode in ["numeric", "all"]:
        summary["numeric_example"] = numeric_example()
        path.write_text(json.dumps(summary, indent=2))
    if args.mode in ["exhibit2", "all"]:
        summary["exhibit2"] = exhibit2()
    summary["status"] = "completed" if "numeric_example" in summary and "exhibit2" in summary else "partial"
    path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results/summary.json | status={summary['status']}", flush=True)


if __name__ == "__main__":
    main()
