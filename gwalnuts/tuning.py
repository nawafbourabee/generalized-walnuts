"""Macro-step tuning (eq. Gamma of the paper) and section-center estimation.

The step size h solves P(micro(theta, rho, h, delta) = 0) = Gamma on states
of a calibration chain, found by geometric bisection.  The section center /
radial anchor C for the first-return rules is the coordinate-wise median of
a Poincare calibration chain run at the tuned h.
"""

import numpy as np
from numba import njit

from .engine import FLOW_HAM, STOP_POINCARE, _random_unit, make_sampler

__all__ = ["make_nohalve_probe", "expand_gamma_bracket", "tune_macro_step",
           "estimate_section_center"]


def make_nohalve_probe(logp, grad_logp, flow_code):
    """Compiled estimator of P(micro(theta, rho, h, delta) = 0).

    ``micro = 0`` means one level-0 macro step keeps the effective-energy
    swing within delta.
    """
    @njit(cache=False)
    def ham_energy(theta, rho):
        return -logp(theta) + 0.5 * (rho @ rho)

    @njit(cache=False)
    def iso_b_step(theta, rho, s_macro, ngrad):
        g = grad_logp(theta, ngrad); xi = np.linalg.norm(g); d = theta.shape[0]
        if xi < 1e-300: return rho.copy(), 0.0
        e = g / xi; gamma = rho @ e
        delta = s_macro * xi / (2.0 * (d - 1))
        e_neg2 = np.exp(-2.0 * delta); e_neg = np.exp(-delta)
        pre = 0.5 * ((1.0 + gamma) + (1.0 - gamma) * e_neg2)
        if pre <= 0.0: return rho.copy(), 0.0
        log_denom = delta + np.log(pre)
        coef_pre = 0.5 * ((1.0 + gamma) - (1.0 - gamma) * e_neg2) - gamma * e_neg
        rho_new = rho * (e_neg / pre) + (coef_pre / pre) * e
        nm = np.linalg.norm(rho_new)
        if nm > 0.0:
            rho_new = rho_new / nm
        return rho_new, -(d - 1) * log_denom

    @njit(cache=False)
    def frac_no_halve(states, h, delta, seed):
        np.random.seed(seed)
        n = states.shape[0]; d = states.shape[1]
        cnt = 0
        ngrad = np.zeros(1, dtype=np.int64)
        for k in range(n):
            th = states[k].copy()
            if flow_code == FLOW_HAM:
                rh = np.random.standard_normal(d)
                H0 = ham_energy(th, rh); Hmax = H0; Hmin = H0
                g = grad_logp(th, ngrad)
                rh = rh + 0.5 * h * g
                th = th + h * rh
                g = grad_logp(th, ngrad)
                rh = rh + 0.5 * h * g
                H = ham_energy(th, rh)
                if H > Hmax: Hmax = H
                if H < Hmin: Hmin = H
                if Hmax - Hmin <= delta:
                    cnt += 1
            else:
                rh = _random_unit(d)
                H0 = -logp(th); Hmax = H0; Hmin = H0
                rh_m, dJ1 = iso_b_step(th, rh, h, ngrad)
                th = th + h * rh_m
                rh, dJ2 = iso_b_step(th, rh_m, h, ngrad)
                H = -logp(th) - (dJ1 + dJ2)
                if H > Hmax: Hmax = H
                if H < Hmin: Hmin = H
                if Hmax - Hmin <= delta:
                    cnt += 1
        return cnt / n
    return frac_no_halve


def expand_gamma_bracket(probe, states, delta, gamma, seed,
                         h_lo=1e-4, h_hi=2.0, max_expand=16):
    """Find a geometric bracket with f(h_lo) >= gamma and f(h_hi) <= gamma."""
    lo = float(h_lo); hi = float(h_hi)
    f_lo = float(probe(states, lo, delta, seed))
    f_hi = float(probe(states, hi, delta, seed + 1))
    n_expand = 0
    while f_lo < gamma and n_expand < max_expand:
        hi = lo; f_hi = f_lo
        lo *= 0.5
        f_lo = float(probe(states, lo, delta, seed + 10 + n_expand))
        n_expand += 1
    while f_hi > gamma and n_expand < max_expand:
        lo = hi; f_lo = f_hi
        hi *= 2.0
        f_hi = float(probe(states, hi, delta, seed + 100 + n_expand))
        n_expand += 1
    return lo, hi, f_lo, f_hi


def tune_macro_step(tg, flow_code, stop_code, seed, n_cal, h_warm, *,
                    delta, gamma, i_max, max_ell, bisection_iters=24,
                    h_lo=1e-4, h_hi=4.0):
    """Tune h for one target/flow/rule cell by the Gamma protocol.

    Runs an ``n_cal``-step calibration chain with the cell's own sampler at
    ``h_warm`` (first-return rules with C = 0), estimates the no-halving
    probability on its states, and bisects geometrically to
    P(micro = 0) = gamma.

    Returns ``(h, frac_nohalve_at_h, calibration_states, trace)``.
    """
    run = make_sampler(tg["logp"], tg["grad_logp"], flow_code, stop_code)
    C0 = np.zeros(tg["d"])
    warm = run(tg["theta0"].copy(), C0, n_cal, h_warm, delta, i_max, max_ell, seed)[0]
    probe = make_nohalve_probe(tg["logp"], tg["grad_logp"], flow_code)
    lo, hi, _, _ = expand_gamma_bracket(
        probe, warm, delta, gamma, seed + 10_000, h_lo=h_lo, h_hi=h_hi
    )
    trace = []
    for it in range(bisection_iters):
        mid = float(np.sqrt(lo * hi))
        frac = float(probe(warm, mid, delta, seed + 20_000 + it))
        trace.append((mid, frac))
        if frac > gamma:
            lo = mid
        else:
            hi = mid
    h = float(np.sqrt(lo * hi))
    frac = float(probe(warm, h, delta, seed + 30_000))
    return h, frac, warm, trace


def estimate_section_center(tg, flow_code, h, seed, n_cal, *,
                            delta, i_max, max_ell):
    """Coordinate-wise median of a Poincare calibration chain at the tuned h."""
    run = make_sampler(tg["logp"], tg["grad_logp"], flow_code, STOP_POINCARE)
    cal = run(tg["theta0"].copy(), np.zeros(tg["d"]), n_cal, h, delta, i_max, max_ell, seed)[0]
    return np.median(cal, axis=0)
