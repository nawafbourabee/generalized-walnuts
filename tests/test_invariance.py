"""Invariance stress test: all six samplers must preserve N(0, I) exactly,
even when the macro step is far beyond the leapfrog stability limit and the
micro-level cap binds, so that reversibility failures (zero-weight leaves)
are frequent.  This is the end-to-end correctness test of the transition
kernel, including the handling of failed and crossing leaves.

The two cheapest variants run in the default test set; the remaining four
are marked ``slow`` (run them with ``pytest -m slow``).
"""

import numpy as np
import pytest
from numba import njit

from gwalnuts import (FLOW_HAM, FLOW_ISO, STOP_POINCARE, STOP_RADIAL_MAX,
                      STOP_UTURN, ess_geyer, make_sampler)

D = 2
N = 20000
DELTA = 0.05
I_MAX = 8
MAX_ELL = 3            # low cap: forward cap-outs occur
H_HAM = 2.0            # beyond the leapfrog stability limit at ell = 0
H_ISO = 5.0


@njit(cache=False)
def _logp(th):
    return -0.5 * (th @ th)


@njit(cache=False)
def _grad(th, ngrad):
    ngrad[0] += 1
    return -th


def _run_and_check(flow, stop, h):
    run = make_sampler(_logp, _grad, flow, stop)
    s, depths, sizes, ngrad, stopc, built, sel = run(
        np.zeros(D), np.zeros(D), N, h, DELTA, I_MAX, MAX_ELL, 12345
    )
    x = s[N // 10:]
    reject = float(np.mean(stopc == 1))
    assert reject > 0.02, "stress test should provoke frequent failures"
    for j in range(D):
        m = x[:, j].mean()
        v = x[:, j].var(ddof=1)
        z_m = m / np.sqrt(v / ess_geyer(x[:, j]))
        z_v = (v - 1.0) / np.sqrt(2.0 / ess_geyer(x[:, j] ** 2))
        assert abs(z_m) < 4.5, (flow, stop, j, m)
        assert abs(z_v) < 4.5, (flow, stop, j, v)


def test_ham_uturn():
    _run_and_check(FLOW_HAM, STOP_UTURN, H_HAM)


def test_iso_anchored():
    _run_and_check(FLOW_ISO, STOP_RADIAL_MAX, H_ISO)


@pytest.mark.slow
def test_ham_poincare():
    _run_and_check(FLOW_HAM, STOP_POINCARE, H_HAM)


@pytest.mark.slow
def test_ham_anchored():
    _run_and_check(FLOW_HAM, STOP_RADIAL_MAX, H_HAM)


@pytest.mark.slow
def test_iso_uturn():
    _run_and_check(FLOW_ISO, STOP_UTURN, H_ISO)


@pytest.mark.slow
def test_iso_poincare():
    _run_and_check(FLOW_ISO, STOP_POINCARE, H_ISO)
