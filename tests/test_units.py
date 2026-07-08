"""Fast unit tests: pure predicates and closed forms (no sampler compile)."""

import numpy as np
import pytest

from gwalnuts import (b_step_update, cross_halfhyperplane, cross_radial_max,
                      p_micro_pmf, uturn)
from gwalnuts.targets import make_targets, finite_difference_check


# ---- p-micro ---------------------------------------------------------------

def test_p_micro_pmf_normalized_and_supported():
    for i in range(5):
        total = sum(p_micro_pmf(j, i) for j in range(i + 4))
        assert total == pytest.approx(1.0)
        assert p_micro_pmf(i, i) == pytest.approx(2.0 / 3.0)
        assert p_micro_pmf(i + 1, i) == pytest.approx(1.0 / 3.0)
        assert p_micro_pmf(i + 2, i) == 0.0
        assert p_micro_pmf(i - 1, i) == 0.0


# ---- crossing predicates ---------------------------------------------------

def test_halfhyperplane_unoriented_with_side_condition():
    d = 3
    eta = np.array([1.0, 0.0, 0.0])
    gam = np.array([0.0, 1.0, 0.0])
    C = np.zeros(d)
    a = np.array([-0.5, 1.0, 0.0])
    b = np.array([0.5, 1.0, 0.0])
    # sign change on the gamma > 0 side fires in either direction
    assert cross_halfhyperplane(a, b, eta, gam, C)
    assert cross_halfhyperplane(b, a, eta, gam, C)
    # same geometry on the gamma < 0 side does not
    am = a * np.array([1, -1, 1]); bm = b * np.array([1, -1, 1])
    assert not cross_halfhyperplane(am, bm, eta, gam, C)
    # no sign change -> no crossing
    assert not cross_halfhyperplane(b, b + eta, eta, gam, C)


def test_halfhyperplane_endpoint_convention():
    # crossing parameter t in (0, 1]: landing exactly on the section counts,
    # starting on it does not (prevents double counting at a shared leaf).
    eta = np.array([1.0, 0.0])
    gam = np.array([0.0, 1.0])
    C = np.zeros(2)
    off = np.array([-1.0, 1.0]); on = np.array([0.0, 1.0]); past = np.array([1.0, 1.0])
    assert cross_halfhyperplane(off, on, eta, gam, C)      # t = 1
    assert not cross_halfhyperplane(on, past, eta, gam, C)  # t = 0


def test_radial_max_oriented_and_reordered():
    C = np.zeros(2)
    th_in = np.array([1.0, 0.0])
    rho_out = np.array([1.0, 0.0])   # phi = +1 (moving away from C)
    th_out = np.array([1.5, 0.0])
    rho_in = np.array([-1.0, 0.0])   # phi = -1 (moving back toward C)
    # forward arm: + to - is a radial maximum -> crossing
    assert cross_radial_max(th_in, rho_out, th_out, rho_in, C, 1)
    # - to + is a radial minimum -> not a crossing (oriented rule)
    assert not cross_radial_max(th_out, rho_in, th_in, rho_out, C, 1)
    # backward arm: pair arrives in generation order; sigma-reordering makes
    # the same physical maximum fire with dir_ = -1
    assert cross_radial_max(th_out, rho_in, th_in, rho_out, C, -1)
    assert not cross_radial_max(th_in, rho_out, th_out, rho_in, C, -1)


def test_uturn_predicate():
    l_th = np.zeros(2); r_th = np.array([1.0, 0.0])
    ahead = np.array([1.0, 0.0]); behind = np.array([-1.0, 0.0])
    assert not uturn(l_th, ahead, r_th, ahead)
    assert uturn(l_th, behind, r_th, ahead)
    assert uturn(l_th, ahead, r_th, behind)


# ---- isokinetic B step -----------------------------------------------------

def test_b_step_unit_norm_and_closed_form():
    # Moderate beta: the implementation must agree with the paper's
    # unreduced closed form.  (At large beta the unreduced form suffers
    # catastrophic cancellation when gamma_0 is near -1 -- the reason the
    # implementation uses the stable rewrite; see the large-beta test.)
    rng = np.random.default_rng(0)
    for d in (2, 5, 50):
        for scale in (1e-3, 0.3, 1.0):
            g = scale * rng.normal(size=d)
            rho = rng.normal(size=d); rho /= np.linalg.norm(rho)
            s = 0.3
            rho_new, dlogJ = b_step_update(g, rho, s)
            assert np.linalg.norm(rho_new) == pytest.approx(1.0, abs=1e-12)
            # b_step_update realizes the half B kick of the BAB splitting:
            # beta = (s/2) ||g|| / (d - 1).
            xi = np.linalg.norm(g); e = g / xi; g0 = rho @ e
            beta = s * xi / (2.0 * (d - 1))
            denom = np.cosh(beta) + g0 * np.sinh(beta)
            expected = (rho + (np.sinh(beta) + g0 * (np.cosh(beta) - 1.0)) * e) / denom
            assert np.allclose(rho_new, expected, atol=1e-10)
            assert dlogJ == pytest.approx(-(d - 1) * np.log(denom), rel=1e-10)


def test_b_step_large_beta_stable():
    # Unconditional stability: huge gradients (deep in a funnel neck) must
    # produce a finite log-Jacobian and a unit direction aligned with g.
    rng = np.random.default_rng(3)
    d = 5
    g = 1e8 * rng.normal(size=d)
    rho = rng.normal(size=d); rho /= np.linalg.norm(rho)
    rho_new, dlogJ = b_step_update(g, rho, 1.0)
    assert np.isfinite(dlogJ)
    assert np.linalg.norm(rho_new) == pytest.approx(1.0, abs=1e-12)
    assert rho_new @ (g / np.linalg.norm(g)) == pytest.approx(1.0, abs=1e-8)


def test_b_step_reversibility():
    # B_s with the flipped output direction undoes the step and negates dlogJ.
    rng = np.random.default_rng(1)
    d = 7
    g = rng.normal(size=d)
    rho = rng.normal(size=d); rho /= np.linalg.norm(rho)
    rho_new, dJ = b_step_update(g, rho, 0.4)
    rho_back, dJb = b_step_update(g, -rho_new, 0.4)
    assert np.allclose(rho_back, -rho, atol=1e-12)
    assert dJb == pytest.approx(-dJ, rel=1e-10)


def test_b_step_jacobian_matches_numerical_determinant():
    # dlogJ is the log-determinant of the B step as a map of the sphere:
    # compare against a finite-difference Jacobian in tangent coordinates.
    rng = np.random.default_rng(2)
    d = 3
    g = rng.normal(size=d)
    rho = rng.normal(size=d); rho /= np.linalg.norm(rho)
    s = 0.5
    rho_new, dlogJ = b_step_update(g, rho, s)

    def tangent_basis(v):
        q, _ = np.linalg.qr(np.column_stack([v, np.eye(d)[:, :d - 1]]))
        return q[:, 1:]

    Q0 = tangent_basis(rho)
    Q1 = tangent_basis(rho_new)
    eps = 1e-6
    J = np.empty((d - 1, d - 1))
    for k in range(d - 1):
        vp = rho + eps * Q0[:, k]; vp /= np.linalg.norm(vp)
        vm = rho - eps * Q0[:, k]; vm /= np.linalg.norm(vm)
        fp, _ = b_step_update(g, vp, s)
        fm, _ = b_step_update(g, vm, s)
        J[:, k] = Q1.T @ (fp - fm) / (2 * eps)
    assert np.log(abs(np.linalg.det(J))) == pytest.approx(dlogJ, abs=1e-4)


# ---- target gradients ------------------------------------------------------

@pytest.mark.parametrize("idx", range(7))
def test_gradients_finite_difference(idx):
    tg = make_targets()[idx]
    assert finite_difference_check(tg) < 1e-6, tg["key"]
