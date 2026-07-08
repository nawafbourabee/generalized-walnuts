"""The seven benchmark targets of Section 5.1 of the companion paper.

Each target is a dict with numba-jitted ``logp(theta)`` and
``grad_logp(theta, ngrad)`` (the latter increments ``ngrad[0]`` once per
call -- this side effect is the gradient accounting), the dimension ``d``,
the reference initialization ``theta0``, the reported slow statistic
(``stat_name``, unit vector ``stat_vector``), and, where available in
closed form, ``true_mean`` / ``true_var`` / ``true_cov``.
"""

import numpy as np
from numba import njit

__all__ = [
    "make_regression", "make_AR1", "make_ar1_target", "make_funnel",
    "make_banana_product", "make_eight", "make_sv",
    "wrap_builtin_target", "make_targets", "finite_difference_check",
    "ar1_slow_vector",
]


# ---- Gaussian regression (exact conjugate posterior), d = 20 --------------

def make_regression(p=20, n=120, sigma=1.0, prior_sd=5.0, seed=20260627):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    beta_true = rng.normal(scale=0.5, size=p)
    y = x @ beta_true + sigma * rng.normal(size=n)
    prec = (x.T @ x) / (sigma * sigma) + np.eye(p) / (prior_sd * prior_sd)
    cov = np.linalg.inv(prec)
    mean = cov @ (x.T @ y) / (sigma * sigma)
    stat = np.linalg.eigh(cov)[1][:, -1].copy()

    mean_const = np.asarray(mean)
    prec_const = np.asarray(prec)

    @njit(cache=False)
    def logp(th):
        z = th - mean_const
        return -0.5 * (z @ (prec_const @ z))

    @njit(cache=False)
    def grad_logp(th, ngrad):
        ngrad[0] += 1
        return -(prec_const @ (th - mean_const))

    return {
        "key": "regression",
        "name": f"Gaussian regression d={p}",
        "logp": logp,
        "grad_logp": grad_logp,
        "d": p,
        "theta0": mean.copy(),
        "stat_name": "soft posterior PC",
        "stat_vector": stat / np.linalg.norm(stat),
        "true_mean": mean,
        "true_cov": cov,
        "meta": {"n": n, "p": p, "sigma": sigma, "prior_sd": prior_sd},
    }


# ---- AR(1) Gaussian, covariance correlation phi -----------------------------

def make_AR1(d=100, phi=0.9):
    phi = float(phi); dd = d
    scale = 1.0 / (1.0 - phi * phi)

    @njit(cache=False)
    def logp(th):
        s = th[0]*th[0] + th[dd-1]*th[dd-1]          # endpoint diagonal = 1
        for i in range(1, dd-1):
            s += (1.0+phi*phi)*th[i]*th[i]
        for i in range(dd-1):
            s += -2.0*phi*th[i]*th[i+1]
        return -0.5*scale*s

    @njit(cache=False)
    def grad_logp(th, ngrad):
        g = np.empty(dd)
        for i in range(dd):
            d0 = 1.0 if (i == 0 or i == dd-1) else (1.0+phi*phi)
            gi = d0*th[i]
            if i > 0:    gi += -phi*th[i-1]
            if i < dd-1: gi += -phi*th[i+1]
            g[i] = -scale*gi
        ngrad[0] += 1
        return g

    return dict(logp=logp, grad_logp=grad_logp, d=dd, slow=0,
                theta0=np.zeros(dd), name=f"AR(1) d={dd}", slow_name="theta_1")


def ar1_slow_vector(d):
    j = np.arange(1, d + 1)
    v = np.sin(np.pi * j / (d + 1))
    return v / np.linalg.norm(v)


def make_ar1_target(d, phi):
    tg = make_AR1(d=d, phi=phi)
    stat = ar1_slow_vector(d)
    idx = np.arange(d)
    cov = phi ** np.abs(idx[:, None] - idx[None, :])
    tg = dict(tg)
    tg.update({
        "key": f"ar1_phi{phi:g}_d{d}",
        "name": f"AR(1), d={d}, corr={phi:g}",
        "stat_name": "slow sine mode",
        "stat_vector": stat,
        "true_mean": np.zeros(d),
        "true_cov": cov,
        "true_var": np.diag(cov),
        "meta": {"d": d, "phi": phi},
    })
    return tg


# ---- Neal's funnel, d = nx + 1: v ~ N(0, sv^2), x_i | v ~ N(0, e^v) --------

def make_funnel(nx=10, sv=3.0):
    dd = nx+1; sv2 = sv*sv

    @njit(cache=False)
    def logp(th):
        v = th[0]; ssq = 0.0
        for i in range(1, dd): ssq += th[i]*th[i]
        return -0.5*v*v/sv2 - 0.5*nx*v - 0.5*np.exp(-v)*ssq

    @njit(cache=False)
    def grad_logp(th, ngrad):
        v = th[0]; g = np.empty(dd); ssq = 0.0
        for i in range(1, dd): ssq += th[i]*th[i]
        ev = np.exp(-v)
        g[0] = -v/sv2 - 0.5*nx + 0.5*ev*ssq
        for i in range(1, dd): g[i] = -ev*th[i]
        ngrad[0] += 1
        return g

    return dict(logp=logp, grad_logp=grad_logp, d=dd, slow=0,
                theta0=np.zeros(dd), name=f"Neal funnel d={dd}", slow_name="v")


# ---- Product of two-dimensional banana factors -----------------------------

def make_banana_product(k=10, b=1.0):
    d = 2 * k

    @njit(cache=False)
    def logp(th):
        out = 0.0
        for j in range(k):
            x = th[2 * j]
            y = th[2 * j + 1]
            r = y - b * (x * x - 1.0)
            out += -0.5 * x * x - 0.5 * r * r
        return out

    @njit(cache=False)
    def grad_logp(th, ngrad):
        g = np.empty(d)
        for j in range(k):
            x = th[2 * j]
            y = th[2 * j + 1]
            r = y - b * (x * x - 1.0)
            g[2 * j] = -x + 2.0 * b * x * r
            g[2 * j + 1] = -r
        ngrad[0] += 1
        return g

    stat = np.zeros(d)
    stat[1] = 1.0
    true_mean = np.zeros(d)
    true_var = np.empty(d)
    true_var[0::2] = 1.0
    true_var[1::2] = 1.0 + 2.0 * b * b
    return {
        "key": f"banana{k}_b{b:g}",
        "name": f"Product of {k} bananas, b={b:g}",
        "logp": logp,
        "grad_logp": grad_logp,
        "d": d,
        "theta0": np.zeros(d),
        "stat_name": "y_1",
        "stat_vector": stat,
        "true_mean": true_mean,
        "true_var": true_var,
        "meta": {"k": k, "b": b},
    }


# ---- Rubin's eight schools (centered), d = 10: (mu, log tau, theta_1..8) ---
# priors: mu ~ N(0, 5^2), tau ~ HalfNormal(5); change of variables
# xi = log tau contributes + log tau.

_Y8 = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
_S8 = np.array([15., 10., 16., 11., 9., 11., 10., 18.])


def make_eight(Y=_Y8, S=_S8, mu_sd=5.0, tau_sd=5.0):
    J = Y.shape[0]; dd = J+2; iv = 1.0/(S*S); mv = 1.0/(mu_sd*mu_sd); tv = 1.0/(tau_sd*tau_sd)

    @njit(cache=False)
    def logp(th):
        mu = th[0]; xi = th[1]; t2 = np.exp(2.0*xi); itau2 = np.exp(-2.0*xi)
        lp = -0.5*mu*mu*mv - 0.5*t2*tv - 7.0*xi        # mu prior, tau prior, -8 xi + Jacobian xi
        for j in range(J):
            thj = th[2+j]
            lp += -0.5*(_Y8[j]-thj)*(_Y8[j]-thj)*iv[j] - 0.5*itau2*(thj-mu)*(thj-mu)
        return lp

    @njit(cache=False)
    def grad_logp(th, ngrad):
        mu = th[0]; xi = th[1]; itau2 = np.exp(-2.0*xi); t2 = np.exp(2.0*xi)
        g = np.empty(dd); ssum = 0.0; sq = 0.0
        for j in range(J):
            thj = th[2+j]; dmu = thj-mu; ssum += dmu; sq += dmu*dmu
            g[2+j] = (_Y8[j]-thj)*iv[j] - itau2*dmu
        g[0] = itau2*ssum - mu*mv
        g[1] = itau2*sq - 7.0 - t2*tv
        ngrad[0] += 1
        return g

    return dict(logp=logp, grad_logp=grad_logp, d=dd, slow=1,
                theta0=np.concatenate((np.array([0.0, 1.0]), Y.copy())),
                name=f"Eight schools d={dd}", slow_name="log tau")


# ---- Stochastic volatility (Kim-Shephard-Chib), d = T + 3 ------------------
# params (mu, alpha, gamma, h_1..h_T): phi = tanh(alpha), sigma = exp(gamma).
# h_1 ~ N(mu, sigma^2/(1-phi^2)); h_t ~ N(mu + phi (h_{t-1}-mu), sigma^2);
# y_t ~ N(0, e^{h_t}).  priors: mu ~ N(0, 5^2); phi ~ Uniform(-1, 1) (flat);
# sigma ~ HalfNormal(1).  Plus change-of-variable Jacobians.

def make_sv(T=30, seed=0, mu_true=-1.0, phi_true=0.95, sig_true=0.25, mu_sd=5.0, sig_sd=1.0):
    rng = np.random.default_rng(seed)
    h = np.empty(T); h[0] = mu_true+rng.standard_normal()*sig_true/np.sqrt(1-phi_true**2)
    for t in range(1, T): h[t] = mu_true+phi_true*(h[t-1]-mu_true)+sig_true*rng.standard_normal()
    Y = np.exp(h/2.0)*rng.standard_normal(T)          # observed returns
    dd = T+3; mv = 1.0/(mu_sd*mu_sd); sprv = 1.0/(sig_sd*sig_sd)

    @njit(cache=False)
    def logp(th):
        mu = th[0]; al = th[1]; ga = th[2]
        phi = np.tanh(al); sig = np.exp(ga); s2 = sig*sig; om2 = 1.0-phi*phi
        if s2 < 1e-300:
            s2 = 1e-300
        if om2 < 1e-300:
            om2 = 1e-300
        lp = -0.5*mu*mu*mv - 0.5*s2*sprv               # mu prior, sigma prior
        lp += np.log(om2) + ga                          # Jacobians: alpha->phi, gamma->sigma
        a1 = th[3]-mu
        lp += 0.5*np.log(om2) - ga - 0.5*om2*a1*a1/s2   # h_1
        for t in range(1, T):
            r = (th[3+t]-mu) - phi*(th[3+t-1]-mu)
            lp += -ga - 0.5*r*r/s2                       # h_t | h_{t-1}
        for t in range(T):
            ht = th[3+t]
            lp += -0.5*ht - 0.5*Y[t]*Y[t]*np.exp(-ht)    # y_t | h_t
        return lp

    @njit(cache=False)
    def grad_logp(th, ngrad):
        mu = th[0]; al = th[1]; ga = th[2]
        phi = np.tanh(al); sig = np.exp(ga); s2 = sig*sig; om2 = 1.0-phi*phi
        if s2 < 1e-300:
            s2 = 1e-300
        inv = 1.0/s2
        if om2 < 1e-300:
            om2 = 1e-300
        g = np.zeros(dd)
        a = np.empty(T)
        for t in range(T): a[t] = th[3+t]-mu
        # h gradients from the AR prior
        g[3+0] += -om2*a[0]*inv
        for t in range(1, T):
            r = a[t]-phi*a[t-1]
            g[3+t] += -r*inv
            g[3+t-1] += phi*r*inv
        # h gradients from likelihood
        for t in range(T):
            ht = th[3+t]
            g[3+t] += -0.5 + 0.5*Y[t]*Y[t]*np.exp(-ht)
        # mu
        dmu = om2*a[0]*inv
        for t in range(1, T):
            r = a[t]-phi*a[t-1]; dmu += (1.0-phi)*r*inv
        g[0] = dmu - mu*mv
        # phi then alpha (dphi/dalpha = 1 - phi^2 = om2)
        dphi = phi*a[0]*a[0]*inv - 3.0*phi/om2
        for t in range(1, T):
            r = a[t]-phi*a[t-1]; dphi += r*a[t-1]*inv
        g[1] = om2*dphi
        # gamma (sigma = e^gamma)
        ssr = om2*a[0]*a[0]
        for t in range(1, T):
            r = a[t]-phi*a[t-1]; ssr += r*r
        # -T from latent transition scales, +1 from gamma->sigma Jacobian.
        g[2] = -T*1.0 + 1.0 + inv*ssr - s2*sprv
        ngrad[0] += 1
        return g

    theta0 = np.concatenate((np.array([mu_true, np.arctanh(phi_true), np.log(sig_true)]), h.copy()))
    return dict(logp=logp, grad_logp=grad_logp, d=dd, slow=2,
                name=f"Stochastic vol d={dd}", slow_name="gamma=log sigma",
                theta0=theta0, Y=Y)


# ---- assembly ---------------------------------------------------------------

def wrap_builtin_target(key, tg):
    tg = dict(tg)
    stat = np.zeros(tg["d"])
    stat[int(tg["slow"])] = 1.0
    tg.update({
        "key": key,
        "stat_name": tg.get("slow_name", f"theta_{tg['slow']}"),
        "stat_vector": stat,
        "meta": {},
    })
    return tg


def make_targets(full_ar1=False):
    """The paper's benchmark suite, in the order used by the seed arithmetic."""
    targets = [make_regression()]
    if full_ar1:
        for phi in (0.9, 0.95):
            for d in (100, 400):
                targets.append(make_ar1_target(d, phi))
    else:
        targets.append(make_ar1_target(100, 0.9))
        targets.append(make_ar1_target(400, 0.95))
    targets.extend([
        wrap_builtin_target("funnel_d11", make_funnel(nx=10)),
        make_banana_product(k=10, b=1.0),
        wrap_builtin_target("eight_schools_centered", make_eight()),
        wrap_builtin_target("sv_T30", make_sv(T=30, seed=0)),
    ])
    return targets


def finite_difference_check(tg, eps=1e-5, seed=1):
    """Max relative error of grad_logp against central finite differences."""
    rng = np.random.default_rng(seed)
    x = tg["theta0"] + 0.2 * rng.normal(size=tg["d"])
    ng = np.zeros(1, dtype=np.int64)
    ga = tg["grad_logp"](x, ng)
    gn = np.empty(tg["d"])
    for i in range(tg["d"]):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        gn[i] = (tg["logp"](xp) - tg["logp"](xm)) / (2.0 * eps)
    return float(np.max(np.abs(ga - gn)) / (np.max(np.abs(gn)) + 1e-12))
