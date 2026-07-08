"""Diagnostics of Section 5.2: Geyer-IPS ESS, ESS/grad, ESJD/grad."""

import numpy as np

__all__ = ["ess_geyer", "summarize_chain"]


def ess_geyer(x):
    """Effective sample size, Geyer initial-positive-sequence estimator.

    FFT autocovariances; adjacent-lag pairs are summed and the sum is
    truncated at the first negative pair.  Note the estimate is clipped at
    the chain length N when the estimated integrated autocorrelation time
    falls below one.
    """
    x = np.asarray(x, float)
    n = len(x)
    x = x - x.mean()
    if n <= 1 or np.var(x) == 0.0:
        return float(n)
    f = np.fft.fft(x, 2 * n)
    pwr = (f * f.conj()).real
    acf = np.fft.ifft(pwr)[:n].real
    acf = acf / acf[0]
    npair = (n - 1) // 2
    if npair <= 0:
        return float(n)
    rp = acf[1:1 + 2 * npair:2] + acf[2:2 + 2 * npair:2]
    cut = npair
    for i, val in enumerate(rp):
        if val < 0.0:
            cut = i
            break
    tau = 1.0 + 2.0 * acf[1:2 * cut + 1].sum()
    return float(n / max(tau, 1.0))


def summarize_chain(tg, prod, sizes, depths, stop, built, selected_idx, ngrad):
    """Per-cell production diagnostics (the columns of raw_results.csv).

    ESS/grad and ESJD/grad follow eqs. (essgrad) and (esjdgrad) of the paper:
    G_N is the total number of gradient evaluations of the production run,
    charging all leaves built, including rejected and boundary leaves.
    ``cap_rate`` is the fraction of transitions hitting the orbit-size cap
    without a zero-weight leaf (stop == 4); ``candidate_reject_rate`` is the
    fraction with a rejected candidate (stop == 1).
    """
    stat_v = np.asarray(tg["stat_vector"], float)
    stat_v = stat_v / max(np.linalg.norm(stat_v), 1e-300)
    stat = prod @ stat_v
    coord_ess = np.array([ess_geyer(prod[:, j]) for j in range(tg["d"])])
    dprod = np.diff(prod, axis=0)
    dstat = np.diff(stat)
    out = {
        "ngrad": int(ngrad),
        "mean_grad_per_iter": float(ngrad / max(len(prod), 1)),
        "ess_stat": ess_geyer(stat),
        "ess_stat_per_kgrad": float(1000.0 * ess_geyer(stat) / max(ngrad, 1)),
        "ess_min_per_kgrad": float(1000.0 * np.min(coord_ess) / max(ngrad, 1)),
        "ess_median_per_kgrad": float(1000.0 * np.median(coord_ess) / max(ngrad, 1)),
        "esjd_stat_per_kgrad": float(1000.0 * np.mean(dstat * dstat) * max(len(dstat), 1) / max(ngrad, 1)),
        "esjd_total_per_kgrad": float(1000.0 * np.mean(np.sum(dprod * dprod, axis=1)) * max(len(dprod), 1) / max(ngrad, 1)),
        "median_L": float(np.median(sizes)),
        "p90_L": float(np.quantile(sizes, 0.90)),
        "p99_L": float(np.quantile(sizes, 0.99)),
        "max_L": int(np.max(sizes)),
        "median_depth": float(np.median(depths)),
        "cap_rate": float(np.mean(stop == 4)),
        "candidate_reject_rate": float(np.mean(stop == 1)),
        "mean_selected_index": float(np.mean(selected_idx)),
        "stat_mean": float(np.mean(stat)),
        "stat_var": float(np.var(stat, ddof=1)),
        "mean_abs_coord_mean": float(np.mean(np.abs(np.mean(prod, axis=0)))),
        "median_coord_var": float(np.median(np.var(prod, axis=0, ddof=1))),
    }
    if "true_mean" in tg:
        tm = np.asarray(tg["true_mean"])
        out["mean_l2_error"] = float(np.linalg.norm(np.mean(prod, axis=0) - tm))
    if "true_var" in tg:
        tv = np.asarray(tg["true_var"])
        out["median_var_ratio"] = float(np.median(np.var(prod, axis=0, ddof=1) / tv))
    if "true_cov" in tg:
        out["stat_true_var"] = float(stat_v @ tg["true_cov"] @ stat_v)
    return out
