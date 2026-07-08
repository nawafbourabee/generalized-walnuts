"""Generalized WALNUTS core sampler.

This module implements the six production samplers of the companion paper

    Bou-Rabee, Carpenter, Kleppe.
    "WALNUTS with General Measure-Preserving Flows and
     Initial-Point-Symmetric Stopping Rules."

Function names follow the paper's pseudocode listings; see LISTINGS.md for
the full mapping.  In brief:

===========================  ==========================================
paper listing                function here
===========================  ==========================================
``p-micro``                  :func:`p_micro_sample`, :func:`p_micro_pmf`
``micro``                    forward-search loop inside ``build_leaf_rand``
``BAB``                      ``b_step`` + drift inside ``build_leaf_rand``
``build-leaf-rand``          ``build_leaf_rand`` (inside :func:`make_sampler`)
``cross-halfhyperplane``     :func:`cross_halfhyperplane`
``cross-radial-max``         :func:`cross_radial_max`
``walnuts-step`` driver      ``run`` returned by :func:`make_sampler`
===========================  ==========================================

The numerical core is byte-identical (same floating-point operations, same
random-number call sequence) to the code that produced the results reported
in the paper; only exploratory stopping rules that do not appear in the
paper have been removed, and public names have been aligned with the
listings.
"""

import numpy as np
from numba import njit

__all__ = [
    "FLOW_HAM", "FLOW_ISO",
    "STOP_UTURN", "STOP_POINCARE", "STOP_RADIAL_MAX",
    "p_micro_sample", "p_micro_pmf", "uturn", "b_step_update",
    "cross_halfhyperplane", "cross_radial_max",
    "make_sampler",
]

# ---------------------------------------------------------------------------
# Component codes: two flows x three stopping rules = six samplers.
# ---------------------------------------------------------------------------

FLOW_HAM = 0          # Hamiltonian flow, leapfrog micro steps
FLOW_ISO = 1          # isokinetic flow on R^d x S^{d-1}, BAB micro steps

STOP_UTURN = 0        # standard U-turn rule with sub-U-turn checks
STOP_POINCARE = 1     # first return to a random half-hyperplane section
STOP_RADIAL_MAX = 2   # first return, anchored radial max-to-max (oriented)


# ---------------------------------------------------------------------------
# Small numerical helpers.
# ---------------------------------------------------------------------------

@njit(cache=False)
def _logsumexp2(a, b):
    if a == -np.inf: return b
    if b == -np.inf: return a
    m = a if a > b else b
    return m + np.log(np.exp(a - m) + np.exp(b - m))


@njit(cache=False)
def uturn(l_th, l_rh, r_th, r_rh):
    """U-turn predicate on an orbit chord (paper, eq. for the U-turn rule)."""
    dtheta = r_th - l_th
    return (l_rh @ dtheta < 0.0) or (r_rh @ dtheta < 0.0)


@njit(cache=False)
def p_micro_sample():
    """Draw ell - ell* from the micro-step randomization (Listing p-micro)."""
    return 0 if np.random.random() < 2.0/3.0 else 1


@njit(cache=False)
def p_micro_pmf(j, i):
    """pmf of Listing p-micro: P(ell = j | ell* = i) = 2/3, 1/3 on {i, i+1}."""
    if j == i:     return 2.0/3.0
    if j == i + 1: return 1.0/3.0
    return 0.0


@njit(cache=False)
def _random_unit(d):
    z = np.random.standard_normal(d)
    return z / np.linalg.norm(z)


# ---------------------------------------------------------------------------
# Crossing predicates for the first-return stopping rules (Section 4).
# ---------------------------------------------------------------------------

@njit(cache=False)
def cross_halfhyperplane(t_prev, t_new, eta, gamma_v, C):
    """Listing cross-halfhyperplane / Definition pair-cross.

    Unoriented sign change of phi(theta) = eta^T (theta - C) with crossing
    parameter t in (0, 1], restricted to the side gamma^T (theta* - C) > 0.
    """
    bp = t_prev - C; bn = t_new - C
    ep = eta @ bp; en = eta @ bn
    denom = ep - en
    if denom == 0.0: return False
    tau = ep / denom
    if tau <= 0.0 or tau > 1.0: return False
    bs = tau * bn + (1.0 - tau) * bp
    return (gamma_v @ bs) > 0.0


@njit(cache=False)
def b_step_update(g, rho, s_macro):
    """Closed-form isokinetic half B kick given the gradient (Listing BAB).

    Realizes the B step of the BAB splitting over time ``s_macro / 2``:
    rotates the unit direction ``rho`` toward the normalized gradient and
    returns ``(rho_new, dlogJ)`` with
    ``dlogJ = -(d-1) log(cosh beta + gamma_0 sinh beta)`` computed in the
    numerically stable form ``-(d-1)(delta + log pre)``, where
    ``beta = delta = (s_macro / 2) ||g|| / (d-1)`` and
    ``gamma_0 = rho . g / ||g||`` (Proposition b-step of the paper).
    """
    xi = np.linalg.norm(g); d = rho.shape[0]
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
    if nm > 0: rho_new = rho_new / nm
    return rho_new, -(d - 1) * log_denom


@njit(cache=False)
def cross_radial_max(t_prev, r_prev, t_new, r_new, C, dir_):
    """Listing cross-radial-max / Definition pair-cross (anchored section).

    Oriented crossing of the anchored radial section: the radial derivative
    phi(z) = (theta - C)^T rho changes sign from + to - along the flow in
    forward physical time.  For a backward-grown arm (``dir_ == -1``) the
    pair arrives in generation order, so the test is applied with the roles
    of the two leaves exchanged (the sigma-reordering of the listing).
    """
    vp = (t_prev - C) @ r_prev
    vn = (t_new - C) @ r_new
    if dir_ == 1:
        return vp > 0.0 and vn < 0.0
    return vp < 0.0 and vn > 0.0


# ---------------------------------------------------------------------------
# Orbit-node bookkeeping for the U-turn sampler (Listing walnuts-step,
# extend-orbit).  A node stores the endpoints, size, categorical sample,
# total weight, and admissibility flag of a (sub)orbit; the on-the-fly merge
# stack realizes the binary-tree extension in O(log L) memory.
# ---------------------------------------------------------------------------

@njit(cache=False)
def _set_leaf(idx, th, rh, lw,
              lth, rth, lrh, rrh, sample, ncnt, logW, llogw, rlogw, good):
    d = th.shape[0]
    for i in range(d):
        lth[idx, i] = th[i]; rth[idx, i] = th[i]
        lrh[idx, i] = rh[i]; rrh[idx, i] = rh[i]
        sample[idx, i] = th[i]
    ncnt[idx] = 1
    logW[idx] = lw
    llogw[idx] = lw
    rlogw[idx] = lw
    good[idx] = True


@njit(cache=False)
def _merge_orbits(left, right, dest, stop_code,
                  lth, rth, lrh, rrh, sample, lidx, ridx, sidx, ncnt,
                  logW, llogw, rlogw, good, merge_mode):
    """Merge two adjacent (sub)orbits into ``dest`` (Listing extend-orbit).

    ``merge_mode``:
      -1  Barker selection between the two blocks (within-extension merges);
       1  biased selection toward ``right`` (top-level merge, orbit grown right);
       0  biased selection toward ``left``  (top-level merge, orbit grown left).

    For ``STOP_UTURN`` the merged node is admissible iff both children are
    and the merged chord passes the U-turn test (sub-U-turn property).
    """
    d = lth.shape[1]
    for i in range(d):
        lth[dest, i] = lth[left, i]
        rth[dest, i] = rth[right, i]
        lrh[dest, i] = lrh[left, i]
        rrh[dest, i] = rrh[right, i]
    ncnt[dest] = ncnt[left] + ncnt[right]
    lidx[dest] = lidx[left]
    ridx[dest] = ridx[right]

    tot = _logsumexp2(logW[left], logW[right])
    pick_right = False
    if merge_mode == -1:
        if tot > -np.inf and np.log(np.random.random()) <= logW[right] - tot:
            pick_right = True
    elif merge_mode == 1:
        if logW[right] >= logW[left] or np.log(np.random.random()) <= logW[right] - logW[left]:
            pick_right = True
    else:
        pick_left = False
        if logW[left] >= logW[right] or np.log(np.random.random()) <= logW[left] - logW[right]:
            pick_left = True
        pick_right = not pick_left
    for i in range(d):
        sample[dest, i] = sample[right, i] if pick_right else sample[left, i]
    sidx[dest] = sidx[right] if pick_right else sidx[left]
    logW[dest] = tot
    llogw[dest] = llogw[left]
    rlogw[dest] = rlogw[right]

    fire = False
    if stop_code == STOP_UTURN:
        dotL = 0.0
        dotR = 0.0
        for i in range(d):
            Delta = rth[dest, i] - lth[dest, i]
            dotL += lrh[dest, i] * Delta
            dotR += rrh[dest, i] * Delta
        fire = (dotL < 0.0 or dotR < 0.0)
    good[dest] = good[left] and good[right] and (not fire)


@njit(cache=False)
def _copy_orbit(src, dest,
                lth, rth, lrh, rrh, sample, lidx, ridx, sidx, ncnt,
                logW, llogw, rlogw, good):
    d = lth.shape[1]
    for i in range(d):
        lth[dest, i] = lth[src, i]
        rth[dest, i] = rth[src, i]
        lrh[dest, i] = lrh[src, i]
        rrh[dest, i] = rrh[src, i]
        sample[dest, i] = sample[src, i]
    ncnt[dest] = ncnt[src]
    logW[dest] = logW[src]
    llogw[dest] = llogw[src]
    rlogw[dest] = rlogw[src]
    good[dest] = good[src]
    lidx[dest] = lidx[src]
    ridx[dest] = ridx[src]
    sidx[dest] = sidx[src]


# ---------------------------------------------------------------------------
# Sampler factory.
# ---------------------------------------------------------------------------

def make_sampler(logp, grad_logp, flow_code, stop_code):
    """Compile one of the six generalized-WALNUTS samplers.

    Parameters
    ----------
    logp : numba-jitted ``f(theta) -> float``
        Unnormalized log target density.
    grad_logp : numba-jitted ``f(theta, ngrad) -> array``
        Gradient of ``logp``; must increment ``ngrad[0]`` by one per call
        (this side effect is the gradient accounting used for ESS/grad).
    flow_code : ``FLOW_HAM`` or ``FLOW_ISO``
    stop_code : ``STOP_UTURN``, ``STOP_POINCARE`` or ``STOP_RADIAL_MAX``

    Returns
    -------
    run : compiled function
        ``run(theta0, C, n_iter, h, delta, i_max, max_ell, seed)`` returning
        ``(samples, depths, sizes, ngrad, stop, built, selected_idx)``.

        * ``samples`` -- (n_iter, d) chain of positions;
        * ``depths``  -- last doubling round attempted per transition;
        * ``sizes``   -- number of selectable (positive-weight) leaves L;
        * ``ngrad``   -- total gradient evaluations (all leaves charged,
          including failed and crossing leaves);
        * ``stop``    -- termination code per transition:
          1 = candidate rejected (zero-weight leaf; for the U-turn rule also
          a sub-U-turn inside the extension), 2 = orbit-level U-turn,
          3 = both arms returned to the section, 4 = size cap 2^i_max;
        * ``built``   -- leaves built per transition (including discarded);
        * ``selected_idx`` -- orbit index of the selected leaf
          (U-turn rule; -1 for first-return rules).

        ``C`` is the section center / radial anchor (ignored by
        ``STOP_UTURN``; pass zeros).
    """

    @njit(cache=False)
    def ham_energy(theta, rho):
        return -logp(theta) + 0.5 * (rho @ rho)

    @njit(cache=False)
    def b_step(theta, rho, s_macro, ngrad):
        """Evaluate the gradient and apply :func:`b_step_update` (Listing BAB)."""
        g = grad_logp(theta, ngrad)
        rho_new, dlogJ = b_step_update(g, rho, s_macro)
        return rho_new, dlogJ, g

    @njit(cache=False)
    def build_leaf_rand(theta, rho, logw_start, dir_, h, delta, max_ell, ngrad):
        """One macro leaf with randomized micro level (Listing build-leaf-rand).

        Forward search = Listing micro fused with the integration: the
        smallest level ell* <= max_ell whose H_eff swing stays within delta
        (sentinel ell* = max_ell + 1 if none does); ell is then drawn from
        p-micro, the level-ell* integration being reused when ell = ell*
        (valid by time-symmetry).  The reverse search recomputes micro from
        the new leaf, and the weight update carries the Hastings ratio
        p(ell | ell*_rev) / p(ell | ell*); the weight is -inf when the
        reverse check fails or ell falls outside the reverse pmf's support.
        """
        q0 = rho.copy()
        d = theta.shape[0]
        if flow_code == FLOW_HAM:
            H_old = ham_energy(theta, q0)
            ell_star = max_ell + 1; fwd = False
            th_s = theta; rh_s = q0; sw_s = 0.0; g_s = np.empty(d)
            for ell in range(max_ell + 1):
                n = 1 << ell; s = dir_ * h / n
                th = theta.copy(); rh = q0.copy()
                H = ham_energy(th, rh); Hmax = H; Hmin = H
                g = grad_logp(th, ngrad)
                ok = True
                for _ in range(n):
                    rh = rh + 0.5 * s * g
                    th = th + s * rh
                    g = grad_logp(th, ngrad)
                    rh = rh + 0.5 * s * g
                    H = ham_energy(th, rh)
                    if H > Hmax: Hmax = H
                    if H < Hmin: Hmin = H
                    if Hmax - Hmin > delta:
                        ok = False
                        break
                if ok:
                    ell_star = ell; fwd = True
                    th_s = th; rh_s = rh; sw_s = Hmax - Hmin; g_s = g
                    break
            ell_p = ell_star + p_micro_sample()
            if fwd and ell_p == ell_star:
                th1 = th_s; rh_int = rh_s; sw_leaf = sw_s; g1 = g_s
            else:
                n = 1 << ell_p; s = dir_ * h / n
                th1 = theta.copy(); rh_int = q0.copy()
                H = ham_energy(th1, rh_int); Hmax = H; Hmin = H
                g1 = grad_logp(th1, ngrad)
                for _ in range(n):
                    rh_int = rh_int + 0.5 * s * g1
                    th1 = th1 + s * rh_int
                    g1 = grad_logp(th1, ngrad)
                    rh_int = rh_int + 0.5 * s * g1
                    H = ham_energy(th1, rh_int)
                    if H > Hmax: Hmax = H
                    if H < Hmin: Hmin = H
                sw_leaf = Hmax - Hmin
            D = H_old - ham_energy(th1, rh_int)
            ell_plus = ell_p; rev = (sw_leaf <= delta)
            for ell in range(ell_p):
                n = 1 << ell; s = -dir_ * h / n
                th = th1.copy(); rh = rh_int.copy()
                H = ham_energy(th, rh); Hmax = H; Hmin = H
                g = grad_logp(th, ngrad)
                ok = True
                for _ in range(n):
                    rh = rh + 0.5 * s * g
                    th = th + s * rh
                    g = grad_logp(th, ngrad)
                    rh = rh + 0.5 * s * g
                    H = ham_energy(th, rh)
                    if H > Hmax: Hmax = H
                    if H < Hmin: Hmin = H
                    if Hmax - Hmin > delta:
                        ok = False
                        break
                if ok:
                    ell_plus = ell; rev = True; break
            if rev:
                pn = p_micro_pmf(ell_p, ell_plus); pd = p_micro_pmf(ell_p, ell_star)
                lw = logw_start + D + np.log(pn) - np.log(pd) if pn > 0.0 and pd > 0.0 else -np.inf
            else:
                lw = -np.inf
            return th1, rh_int, lw, g1

        # Isokinetic BAB branch.
        l0 = logp(theta)
        ell_star = max_ell + 1; fwd = False
        th_s = theta; q_s = q0; lJ_s = 0.0; sw_s = 0.0; g_s = np.empty(d)
        for ell in range(max_ell + 1):
            n = 1 << ell; s = dir_ * h / n
            th = theta.copy(); q = q0.copy(); logJ = 0.0
            H = -logp(th); Hmax = H; Hmin = H
            ok = True; g_end = np.empty(d)
            for _ in range(n):
                if s >= 0.0:
                    qm, dJ1, _ = b_step(th, q, s, ngrad)
                    th = th + s * qm
                    q, dJ2, g_end = b_step(th, qm, s, ngrad)
                else:
                    sp = -s
                    qm, dJ1, _ = b_step(th, -q, sp, ngrad)
                    th = th + sp * qm
                    qtmp, dJ2, g_end = b_step(th, qm, sp, ngrad)
                    q = -qtmp
                logJ += dJ1 + dJ2
                H = -logp(th) - logJ
                if H > Hmax: Hmax = H
                if H < Hmin: Hmin = H
                if Hmax - Hmin > delta:
                    ok = False
                    break
            if ok:
                ell_star = ell; fwd = True
                th_s = th; q_s = q; lJ_s = logJ; sw_s = Hmax - Hmin; g_s = g_end
                break
        ell_p = ell_star + p_micro_sample()
        if fwd and ell_p == ell_star:
            th1 = th_s; q_int = q_s; lJ_leaf = lJ_s; sw_leaf = sw_s; g1 = g_s
        else:
            n = 1 << ell_p; s = dir_ * h / n
            th1 = theta.copy(); q_int = q0.copy(); lJ_leaf = 0.0
            H = -logp(th1); Hmax = H; Hmin = H
            g1 = np.empty(d)
            for _ in range(n):
                if s >= 0.0:
                    qm, dJ1, _ = b_step(th1, q_int, s, ngrad)
                    th1 = th1 + s * qm
                    q_int, dJ2, g1 = b_step(th1, qm, s, ngrad)
                else:
                    sp = -s
                    qm, dJ1, _ = b_step(th1, -q_int, sp, ngrad)
                    th1 = th1 + sp * qm
                    qtmp, dJ2, g1 = b_step(th1, qm, sp, ngrad)
                    q_int = -qtmp
                lJ_leaf += dJ1 + dJ2
                H = -logp(th1) - lJ_leaf
                if H > Hmax: Hmax = H
                if H < Hmin: Hmin = H
            sw_leaf = Hmax - Hmin
        D = logp(th1) + lJ_leaf - l0
        ell_plus = ell_p; rev = (sw_leaf <= delta)
        for ell in range(ell_p):
            n = 1 << ell; s = -dir_ * h / n
            th = th1.copy(); q = q_int.copy(); logJ = 0.0
            H = -logp(th); Hmax = H; Hmin = H
            ok = True; gtmp = np.empty(d)
            for _ in range(n):
                if s >= 0.0:
                    qm, dJ1, _ = b_step(th, q, s, ngrad)
                    th = th + s * qm
                    q, dJ2, gtmp = b_step(th, qm, s, ngrad)
                else:
                    sp = -s
                    qm, dJ1, _ = b_step(th, -q, sp, ngrad)
                    th = th + sp * qm
                    qtmp, dJ2, gtmp = b_step(th, qm, sp, ngrad)
                    q = -qtmp
                logJ += dJ1 + dJ2
                H = -logp(th) - logJ
                if H > Hmax: Hmax = H
                if H < Hmin: Hmin = H
                if Hmax - Hmin > delta:
                    ok = False
                    break
            if ok:
                ell_plus = ell; rev = True; break
        if rev:
            pn = p_micro_pmf(ell_p, ell_plus); pd = p_micro_pmf(ell_p, ell_star)
            lw = logw_start + D + np.log(pn) - np.log(pd) if pn > 0.0 and pd > 0.0 else -np.inf
        else:
            lw = -np.inf
        return th1, q_int, lw, g1

    @njit(cache=False)
    def run(theta0, C, n_iter, h, delta, i_max, max_ell, seed):
        """Run ``n_iter`` transitions (Listing walnuts-step per transition)."""
        np.random.seed(seed); d = theta0.shape[0]
        nslots = i_max + 4; cur = i_max + 1; tmp = i_max + 2; cand = i_max + 3
        lth = np.zeros((nslots, d)); rth = np.zeros((nslots, d))
        lrh = np.zeros((nslots, d)); rrh = np.zeros((nslots, d))
        sample = np.zeros((nslots, d))
        lidx = np.zeros(nslots, dtype=np.int64)
        ridx = np.zeros(nslots, dtype=np.int64)
        sidx = np.zeros(nslots, dtype=np.int64)
        ncnt = np.zeros(nslots, dtype=np.int64)
        logW = np.zeros(nslots); llogw = np.zeros(nslots); rlogw = np.zeros(nslots)
        good = np.zeros(nslots, dtype=np.bool_)
        valid = np.zeros(i_max + 1, dtype=np.bool_)
        samples = np.empty((n_iter, d))
        sizes = np.empty(n_iter, dtype=np.int64)
        depths = np.empty(n_iter, dtype=np.int64)
        stop = np.zeros(n_iter, dtype=np.int64)
        built = np.zeros(n_iter, dtype=np.int64)
        selected_idx = np.zeros(n_iter, dtype=np.int64)
        ngrad = np.zeros(1, dtype=np.int64); theta = theta0.copy()
        for t in range(n_iter):
            # refresh: draw the auxiliary variable and the seed leaf's weight.
            rho = np.random.standard_normal(d) if flow_code == FLOW_HAM else _random_unit(d)
            lw0 = logp(theta) - 0.5*(rho@rho) if flow_code == FLOW_HAM else logp(theta)
            _set_leaf(cur, theta, rho, lw0,
                      lth, rth, lrh, rrh, sample, ncnt, logW, llogw, rlogw, good)
            lidx[cur] = 0
            ridx[cur] = 0
            sidx[cur] = 0
            dr = 0; nbuilt = 0; reason = 0

            if stop_code == STOP_POINCARE or stop_code == STOP_RADIAL_MAX:
                # ---- first-return driver (Listing walnuts-fr) ----
                if stop_code == STOP_RADIAL_MAX:
                    eta = np.zeros(d)
                    Csec = np.zeros(d)
                else:
                    # draw-section: eta uniform on the sphere, gamma uniform
                    # on the sphere orthogonal to eta, section centered at C.
                    z1 = np.random.standard_normal(d); eta = z1/np.linalg.norm(z1)
                    Csec = C
                # The anchored radial rule needs no section direction, but the
                # published runs drew (and discarded) gamma here as well; the
                # draw is kept so that the random-number sequence -- and hence
                # every published chain -- is reproduced bit for bit.
                z2 = np.random.standard_normal(d); z2 = z2-(z2@eta)*eta; gamma_v = z2/np.linalg.norm(z2)
                crossed_left = False; crossed_right = False; failed_any = False
                theta_tilde = theta.copy(); gW = lw0; gEf = lw0; gEb = lw0
                npos = 1
                gl_t = theta.copy(); gl_r = rho.copy(); gr_t = theta.copy(); gr_r = rho.copy()
                for depth in range(i_max):
                    dir_ = 1 if np.random.random() < 0.5 else -1
                    if dir_ == 1 and crossed_right:
                        if crossed_left and crossed_right: break
                        continue
                    if dir_ == -1 and crossed_left:
                        if crossed_left and crossed_right: break
                        continue
                    th = gr_t.copy() if dir_ == 1 else gl_t.copy()
                    rh = gr_r.copy() if dir_ == 1 else gl_r.copy()
                    lw = gEf if dir_ == 1 else gEb
                    eW = -np.inf; cand_th = th.copy(); end_th = th.copy(); end_rh = rh.copy()
                    crossed = False; failed = False
                    for k in range(1 << depth):
                        th_new, rh_new, lw_new, g_new = build_leaf_rand(
                            th, rh, lw, dir_, h, delta, max_ell, ngrad)
                        if lw_new == -np.inf:
                            # zero-weight leaf: boundary of the attempted
                            # extension; not selectable, arm terminates.
                            nbuilt += 1
                            failed = True
                            failed_any = True
                            break
                        if stop_code == STOP_RADIAL_MAX:
                            crossed_now = cross_radial_max(th, rh, th_new, rh_new, C, dir_)
                        else:
                            crossed_now = cross_halfhyperplane(th, th_new, eta, gamma_v, Csec)
                        if crossed_now:
                            # First-return convention: the crossing leaf is a
                            # sentinel that terminates growth but is not
                            # retained as a selectable state (Def. frs).
                            crossed = True; break
                        nbuilt += 1
                        newW = _logsumexp2(eW, lw_new)
                        if eW == -np.inf or np.log(np.random.random()) <= lw_new - newW:
                            cand_th = th_new.copy()
                        eW = newW; th = th_new; rh = rh_new; lw = lw_new
                        end_th = th.copy(); end_rh = rh.copy()
                        npos += 1
                    if eW > -np.inf:
                        # biased-progressive merge toward the fresh extension.
                        if np.log(np.random.random()) <= eW - gW:
                            theta_tilde = cand_th.copy()
                        gW = _logsumexp2(gW, eW)
                    if dir_ == 1:
                        gr_t = end_th; gr_r = end_rh; gEf = lw
                        crossed_right = crossed_right or crossed or failed
                    else:
                        gl_t = end_th; gl_r = end_rh; gEb = lw
                        crossed_left = crossed_left or crossed or failed
                    dr = depth + 1
                    if crossed_left and crossed_right:
                        reason = 1 if failed_any else 3; break
                if reason == 0:
                    reason = 1 if failed_any else 4
                theta = theta_tilde; samples[t] = theta; sizes[t] = npos
                depths[t] = dr; stop[t] = reason; built[t] = nbuilt
                selected_idx[t] = -1
                continue

            # ---- standard U-turn driver (Listings walnuts-step, extend-orbit) ----
            for depth in range(i_max):
                dir_ = 1 if np.random.random() < 0.5 else -1
                for j in range(i_max + 1):
                    valid[j] = False
                th = rth[cur].copy() if dir_ == 1 else lth[cur].copy()
                rh = rrh[cur].copy() if dir_ == 1 else lrh[cur].copy()
                lw = rlogw[cur] if dir_ == 1 else llogw[cur]
                off = ridx[cur] if dir_ == 1 else lidx[cur]
                for k in range(ncnt[cur]):
                    th, rh, lw, g_new = build_leaf_rand(
                        th, rh, lw, dir_, h, delta, max_ell, ngrad)
                    off += dir_
                    if lw == -np.inf:
                        # zero-weight leaf: candidate subtree inadmissible.
                        nbuilt += 1
                        reason = 1
                        break
                    _set_leaf(tmp, th, rh, lw,
                              lth, rth, lrh, rrh, sample, ncnt, logW, llogw, rlogw, good)
                    lidx[tmp] = off
                    ridx[tmp] = off
                    sidx[tmp] = off
                    nbuilt += 1
                    j = 0
                    while j < depth and valid[j]:
                        # within-extension merges: Barker selection, with the
                        # sub-U-turn check applied to every merged subtree.
                        if dir_ == 1:
                            _merge_orbits(j, tmp, cand, stop_code,
                                          lth, rth, lrh, rrh, sample, lidx, ridx, sidx,
                                          ncnt, logW, llogw, rlogw, good, -1)
                        else:
                            _merge_orbits(tmp, j, cand, stop_code,
                                          lth, rth, lrh, rrh, sample, lidx, ridx, sidx,
                                          ncnt, logW, llogw, rlogw, good, -1)
                        _copy_orbit(cand, tmp,
                                    lth, rth, lrh, rrh, sample, lidx, ridx, sidx,
                                    ncnt, logW, llogw, rlogw, good)
                        valid[j] = False; j += 1
                    _copy_orbit(tmp, j,
                                lth, rth, lrh, rrh, sample, lidx, ridx, sidx,
                                ncnt, logW, llogw, rlogw, good)
                    valid[j] = True
                    if not good[j]:
                        reason = 1; break
                ext = depth
                if reason == 1:
                    break
                # top-level merge: biased toward the fresh extension.
                if dir_ == 1:
                    _merge_orbits(cur, ext, cand, stop_code,
                                  lth, rth, lrh, rrh, sample, lidx, ridx, sidx,
                                  ncnt, logW, llogw, rlogw, good, 1)
                else:
                    _merge_orbits(ext, cur, cand, stop_code,
                                  lth, rth, lrh, rrh, sample, lidx, ridx, sidx,
                                  ncnt, logW, llogw, rlogw, good, 0)
                dr = depth + 1
                enlarged_bad = not good[cand]
                _copy_orbit(cand, cur,
                            lth, rth, lrh, rrh, sample, lidx, ridx, sidx,
                            ncnt, logW, llogw, rlogw, good)
                if enlarged_bad:
                    reason = 2; break
            if reason == 0 and ncnt[cur] >= (1 << i_max):
                reason = 4
            theta = sample[cur].copy(); samples[t] = theta; sizes[t] = ncnt[cur]
            depths[t] = dr; stop[t] = reason; built[t] = nbuilt
            selected_idx[t] = sidx[cur]
        return samples, depths, sizes, ngrad[0], stop, built, selected_idx

    return run
