"""Run the authors' two-date Shrinking the Term Structure example locally.

No account, network connection or package installation is needed at run time.
The source tree is the unmodified author repository at the commit in metadata.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import pickle
import pickletools
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import solve
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(SOURCE / "source"))
author_kernel = author_models = None

COMMIT = "7cbfa4635c8461995ff9bcea74fff2818fe4618f"
ALPHA, DELTA, LAMBDA, N, MAX_FACTORS = 0.05, 0.0, 10.0, 3650, 10
DATES = ["1986-06-30", "2013-12-31"]


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Allow only the three NumPy constructors present in this audited file."""
    ALLOWED = {
        ("numpy.core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
        ("numpy", "dtype"): np.dtype,
        ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Disallowed global: {module}.{name}")
        return self.ALLOWED[(module, name)]


def validate_source():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for row in manifest["files"]:
        path = SOURCE / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
    data = SOURCE / "example_data" / "dict_data.pkl"
    ops = list(pickletools.genops(data.read_bytes()))
    assert not any(op.name in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"} for op, _, _ in ops)
    with data.open("rb") as handle:
        panels = NumpyOnlyUnpickler(handle).load()
    assert list(panels) == DATES
    for date, panel in panels.items():
        assert set(panel) == {"date_s", "B", "Bc_shift", "C", "rf_t", "g"}
        assert panel["C"].shape == (len(panel["B"]), N)
        for key in ("B", "Bc_shift", "C", "g", "rf_t"):
            assert np.isfinite(panel[key]).all(), (date, key)
        assert np.all(panel["B"] > 0)
    return panels


def prepare(panel):
    """The exact normalization and return construction in author cell 8."""
    B = panel["B"]
    Bc_shift = (1 / B) * panel["Bc_shift"]
    C = (1 / B)[:, np.newaxis] * panel["C"]
    s = panel["date_s"]
    rf = (1 + panel["rf_t"]) ** s - 1
    ret = Bc_shift - np.ones(len(B))
    rx = ret - rf
    Z_bar = C[:, s - 1:] @ np.diag(panel["g"][s - 1:N])
    return Z_bar[:, 1:], ret, rx, float(rf)


def independent_full_check(Z, rx, K):
    columns = np.where(Z.sum(axis=0) != 0)[0]
    Zm = Z[:, columns]
    penalty = len(rx) * LAMBDA / (columns[-1] + 1)
    Kmm = K[np.ix_(columns, columns)]
    system = Zm @ Kmm @ Zm.T + penalty * np.eye(len(rx))
    curve = K[:, columns] @ Zm.T @ solve(system, rx, assume_a="pos")
    return curve, Zm @ curve[columns]


def fast_exact_loocv(Z, rx, K, V, eigenvalues, rank):
    """Independent exact leave-one-bond-out errors; verified against author fits.

    Sherman-Morrison uses a common training-sample penalty. When leaving out a
    bond changes the maximum nonzero cash-flow day, call the author's estimator
    explicitly, preserving its penalty normalization and cash-flow selection.
    """
    n = len(rx)
    columns = np.where(Z.sum(axis=0) != 0)[0]
    max_day = columns[-1] + 1
    penalty = (n - 1) * LAMBDA / max_day
    if rank is None:
        Zm = Z[:, columns]
        gram = Zm @ K[np.ix_(columns, columns)] @ Zm.T
    else:
        loadings = V[:, :rank] * np.sqrt(eigenvalues[:rank])
        bond_loadings = Z[:, columns] @ loadings[columns]
        gram = bond_loadings @ bond_loadings.T
    inv_system = solve(gram + penalty * np.eye(n), np.eye(n), assume_a="pos")
    errors = (inv_system @ rx) / np.diag(inv_system)
    predictions = rx - errors
    changed = []
    for i in range(n):
        train = np.arange(n) != i
        train_columns = np.where(Z[train].sum(axis=0) != 0)[0]
        if train_columns[-1] + 1 != max_day:
            changed.append(i)
    checked = sorted(set([0, n // 2, n - 1] + changed))
    max_check_error = 0.0
    for i in checked:
        train = np.arange(n) != i
        if rank is None:
            out = author_models.one_fit(Z[train], LAMBDA, rx[train], 1, K)
            curve = out["rx_g_full"]
        else:
            out = author_models.FM_ridge_solution(
                Z[train], LAMBDA, rx[train], 1, V, eigenvalues, rank,
                fit_single_R=True,
            )
            curve = out["rx_g_fm"]
        prediction = float(Z[i] @ curve[:Z.shape[1]])
        if i in changed:
            predictions[i] = prediction
            errors[i] = rx[i] - prediction
        else:
            max_check_error = max(max_check_error, abs(prediction - predictions[i]))
    assert max_check_error < 1e-8, max_check_error
    return predictions, float(np.sqrt(np.mean(errors ** 2))), max_check_error, len(checked)


def main():
    global author_kernel, author_models
    import importlib
    start = time.perf_counter()
    print("[1/5] Verify fixed-commit source hashes and load two real Treasury panels", flush=True)
    panels = validate_source()
    author_kernel = importlib.import_module("kernel")
    author_models = importlib.import_module("models")
    print("        1986-06-30: 140 bonds; 2013-12-31: 239 bonds", flush=True)
    print("[2/5] Author kernel: alpha=.05, delta=0, maturity grid=3650 days", flush=True)
    K = author_kernel.generate_kernel_matrix(ALPHA, DELTA, N, N)
    assert K.shape == (N, N) and np.isfinite(K).all()
    # Check the closed-form kernel in the published paper, Appendix A.1.
    x, y = np.meshgrid(np.array([1, 365, 1825, 3650]) / 365, np.array([1, 365, 1825, 3650]) / 365, indexing="ij")
    mn, mx = np.minimum(x, y), np.maximum(x, y)
    manual = -mn / ALPHA**2 * np.exp(-ALPHA * mn) + 2 / ALPHA**3 * (1 - np.exp(-ALPHA * mn)) - mn / ALPHA**2 * np.exp(-ALPHA * mx)
    ind = np.array([0, 364, 1824, 3649])
    assert np.allclose(K[np.ix_(ind, ind)], manual, rtol=1e-12, atol=1e-12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    print("[3/5] Compute author full SVD (3650 by 3650); CPU only", flush=True)
    svd_start = time.perf_counter()
    U, eigenvalues, Vh = np.linalg.svd(K)
    V = Vh.T
    assert np.isclose(U[:, :10], V[:, :10]).all()
    print(f"        SVD completed in {time.perf_counter() - svd_start:.1f} sec", flush=True)
    np.savez_compressed(RESULTS / "kernel_top10.npz", eigenvalues=eigenvalues[:10], V=V[:, :10])

    rows, observations, curves, validation, date_summaries = [], [], [], [], []
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    print("[4/5] Run unchanged author full KR and KR-1 to KR-10 fits", flush=True)
    for row_index, date in enumerate(DATES):
        panel = panels[date]
        Z, ret, rx, rf = prepare(panel)
        full = author_models.one_fit(Z, LAMBDA, rx, panel["date_s"], K)
        fm = author_models.FM_ridge_solution(Z, LAMBDA, rx, panel["date_s"], V, eigenvalues, MAX_FACTORS)
        assert np.isfinite(full["rx_g_full"]).all()
        assert np.isfinite(full["rx_full"]).all()
        direct_curve, direct_fit = independent_full_check(Z, rx, K)
        direct_diff = float(np.max(np.abs(direct_curve - full["rx_g_full"])))
        assert np.allclose(direct_curve, full["rx_g_full"], rtol=1e-7, atol=1e-9)
        assert np.allclose(direct_fit, full["rx_full"], rtol=1e-7, atol=1e-9)
        max_maturity = np.apply_along_axis(lambda a: len(a) - (a != 0)[::-1].argmax(), 1, panel["C"])
        sort = np.argsort(max_maturity)
        ax, ax2 = axes[row_index]
        ax.plot(max_maturity[sort] / 365, rx[sort] * 10000, "o", markersize=3, alpha=.65, label="Observed bonds")
        ax.plot(max_maturity[sort] / 365, full["rx_full"][sort] * 10000, "--", label="Full KR")
        ax.plot(max_maturity[sort] / 365, fm[4]["rx_fm"][sort] * 10000, "-", label="KR-4")
        ax.set(title=f"{date}: observed and fitted excess returns", xlabel="Maturity (years)", ylabel="Excess return (bp)")
        ax.legend()
        maturity = np.arange(1, N + 1) / 365
        ax2.plot(maturity, full["rx_g_full_aligned"] * 10000, label="Full KR")
        ax2.plot(maturity, fm[4]["rx_g_fm_aligned"] * 10000, label="KR-4")
        ax2.set(title=f"{date}: discount-bond return curve", xlabel="Maturity (years)", ylabel="Excess return (bp)")
        ax2.legend()
        for axx in (ax, ax2): axx.grid(alpha=.2)

        # Recreate the authors' cell 10, preserving decimal-return axes and layout.
        original, original_axes = plt.subplots(1, 2, figsize=(15, 6))
        original_axes[0].plot(max_maturity[sort], rx[sort], "--o", label=r"True $R_{t+1}^e$")
        original_axes[0].plot(max_maturity[sort], full["rx_full"][sort], "--o", label=r"Fitted $R_{t+1}^e$")
        original_axes[0].set_title(f"Fitted excess returns of underlying securities on {date}")
        original_axes[1].plot(full["rx_g_full"], label=r"Fitted $R_{g,t+1}^e$")
        original_axes[1].set_title(f"Fitted excess return curve on {date}")
        for axx in original_axes:
            axx.set_xlabel("Time to Maturity in Days")
            axx.legend()
        original.savefig(RESULTS / f"author_example_{date}.png", dpi=130, bbox_inches="tight")
        plt.close(original)

        loo_results = {}
        print(f"        {date}: fit complete; verify exact bond leave-one-out errors", flush=True)
        for rank in (None, 4):
            predictions, rmse, check_error, checks = fast_exact_loocv(Z, rx, K, V, eigenvalues, rank)
            label = "Full KR" if rank is None else "KR-4"
            loo_results[label] = {"rmse_bp": rmse * 10000, "author_fit_max_abs_difference": check_error, "checked_holdouts": checks}
            validation.append({"date": date, "test": f"exact LOOCV versus author estimator ({label})", "max_absolute_difference": check_error, "passed": True})
        for rank in [None] + list(range(1, MAX_FACTORS + 1)):
            label = "Full KR" if rank is None else f"KR-{rank}"
            fitted = full["rx_full"] if rank is None else fm[rank]["rx_fm"]
            rmse = float(np.sqrt(np.mean((rx - fitted) ** 2)))
            rows.append({"date": date, "model": label, "factors": N if rank is None else rank, "bonds": len(rx), "rmse_in_sample_bp": rmse * 10000, "r2_uncentered_in_sample": 1 - float(np.sum((rx - fitted) ** 2) / np.sum(rx ** 2)), "rmse_leave_one_bond_out_bp": loo_results.get(label, {}).get("rmse_bp", None)})
        for i in range(len(rx)):
            observations.append({"date": date, "bond_index": i, "maturity_days": int(max_maturity[i]), "price": float(panel["B"][i]), "excess_return": float(rx[i]), "fitted_full_kr": float(full["rx_full"][i]), "fitted_kr4": float(fm[4]["rx_fm"][i])})
        for i in range(N):
            curves.append({"date": date, "maturity_days": i + 1, "full_kr": float(full["rx_g_full_aligned"][i]), "kr4": float(fm[4]["rx_g_fm_aligned"][i])})
        normalized_residual_full = float(np.sum((rx - full["rx_full"]) ** 2) / np.sum(rx ** 2))
        normalized_residual_4 = float(np.sum((rx - fm[4]["rx_fm"]) ** 2) / np.sum(rx ** 2))
        normalized_residual_1 = float(np.sum((rx - fm[1]["rx_fm"]) ** 2) / np.sum(rx ** 2))
        date_summaries.append({"date": date, "next_business_day": "1986-07-01" if row_index == 0 else "2014-01-02", "bonds": len(rx), "calendar_days": panel["date_s"], "compounded_risk_free_return": rf, "full_kr_rmse_bp": full["rmse"] * 10000, "kr4_rmse_bp": fm[4]["rmse_fm"] * 10000, "full_kr_r2_uncentered": 1 - normalized_residual_full, "kr4_r2_uncentered": 1 - normalized_residual_4, "it_vol_kr4": normalized_residual_4, "t_com_kr1_minus_kr4": normalized_residual_1 - normalized_residual_4, "loocv": loo_results, "independent_full_max_abs_difference": direct_diff})
        validation.append({"date": date, "test": "author inverse versus independent positive-definite solve", "max_absolute_difference": direct_diff, "passed": True})
        print(f"        {date}: Full KR RMSE={full['rmse'] * 10000:.4f} bp; KR-4={fm[4]['rmse_fm'] * 10000:.4f} bp; exact LOOCV Full/KR4={loo_results['Full KR']['rmse_bp']:.4f}/{loo_results['KR-4']['rmse_bp']:.4f} bp", flush=True)

    print("[5/5] Save results, source provenance and original-style figures", flush=True)
    figure.savefig(RESULTS / "shrinking_term_structure_replication.png", dpi=160)
    plt.close(figure)
    pd.DataFrame(rows).to_csv(RESULTS / "model_metrics.csv", index=False)
    pd.DataFrame(observations).to_csv(RESULTS / "bond_predictions.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS / "return_curves.csv", index=False)
    pd.DataFrame(validation).to_csv(RESULTS / "validation.csv", index=False)
    factor_figure, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    metrics = pd.DataFrame(rows)
    for date in DATES:
        subset = metrics[(metrics.date == date) & (metrics.model != "Full KR")]
        ax.plot(subset.factors, subset.rmse_in_sample_bp, "o-", label=date)
    ax.set(xlabel="Number of KR factors", ylabel="In-sample RMSE (bp)", title="Author KR-1 to KR-10 experiment on two bundled dates")
    ax.legend(); ax.grid(alpha=.2)
    factor_figure.savefig(RESULTS / "factor_count_rmse.png", dpi=160)
    plt.close(factor_figure)
    summary = {"status": "completed", "title": "Shrinking the Term Structure", "repository": "https://github.com/rosieiiiii/KR_return_example", "commit": COMMIT, "paper_publication_date": "2026-08-21", "scope": "Author's two-date, real-data core example; unchanged kernel/model functions, full 3650-day SVD, full KR and KR-1 to KR-10. Independent exact bond LOOCV is an added validation.", "limitations": ["Only the two dates bundled by the authors are reproduced, rather than the full daily 1961-2020 empirical sample.", "Expanded history requires WRDS access to CRSP Treasury prices and cash flows. No WRDS credentials were requested or used.", "Two cross sections cannot validate the four-factor time-series pricing results, recession complexity premium or full-paper tables.", "Reported factor-count RMSE and uncentered R2 are in sample. Leave-one-bond-out RMSE is cross-sectional out of sample on the same day, not a future-return forecast.", "The markdown in the author example mentions lambda=1; its actual executable cell and the 2026 paper use lambda=10, which is followed here."], "parameters": {"alpha": ALPHA, "delta": DELTA, "lambda_unscaled": LAMBDA, "maturity_grid_days": N, "factor_counts": list(range(1, MAX_FACTORS + 1))}, "dates": date_summaries, "validation_passed": True, "elapsed_seconds": time.perf_counter() - start, "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__}, "files": {"metrics_csv": "results/model_metrics.csv", "main_figure": "results/shrinking_term_structure_replication.png", "observed_and_fitted": "results/bond_predictions.csv", "curves": "results/return_curves.csv", "validation": "results/validation.csv", "factor_count_figure": "results/factor_count_rmse.png"}}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"COMPLETE: all source, kernel, full-fit and LOOCV checks passed; {summary['elapsed_seconds']:.1f} sec", flush=True)


if __name__ == "__main__":
    main()
