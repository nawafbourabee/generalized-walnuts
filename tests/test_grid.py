"""Protocol tests: seed arithmetic (which defines the published runs) and a
smoke run of the production grid driver."""

import numpy as np
import pytest

from gwalnuts.grid import BASE_SEED, FLOWS, RULES, main
from gwalnuts.targets import make_targets


def test_seed_arithmetic_is_frozen():
    # The published cells are addressed by
    # seed = 20260627 + 100000*ti + 10000*fi + 1000*ri + si.
    # Freezing the constants and the orderings keeps every published cell
    # reachable by index.
    assert BASE_SEED == 20260627
    keys = [tg["key"] for tg in make_targets()]
    assert keys == ["regression", "ar1_phi0.9_d100", "ar1_phi0.95_d400",
                    "funnel_d11", "banana10_b1", "eight_schools_centered",
                    "sv_T30"]
    assert [f[0] for f in FLOWS] == ["Hamiltonian", "Isokinetic"]
    assert [f[2] for f in FLOWS] == [0.10, 0.50]        # h_warm
    assert [r[0] for r in RULES] == ["standard U-turn",
                                     "half-hyperplane Poincare",
                                     "anchored U-turn"]


def test_target_dimensions_and_inits():
    tgs = {tg["key"]: tg for tg in make_targets()}
    assert tgs["regression"]["d"] == 20
    assert tgs["ar1_phi0.9_d100"]["d"] == 100
    assert tgs["ar1_phi0.95_d400"]["d"] == 400
    assert tgs["funnel_d11"]["d"] == 11
    assert tgs["banana10_b1"]["d"] == 20
    assert tgs["eight_schools_centered"]["d"] == 10
    assert tgs["sv_T30"]["d"] == 33
    # eight schools initializes at (mu, xi, theta) = (0, 1, Y)
    th0 = tgs["eight_schools_centered"]["theta0"]
    assert th0[0] == 0.0 and th0[1] == 1.0 and th0[2] == 28.0


@pytest.mark.slow
def test_grid_smoke(tmp_path):
    main(["--smoke", "--target-key", "eight_schools_centered",
          "--flow", "Hamiltonian", "--rule", "standard U-turn",
          "--outdir", str(tmp_path)])
    assert (tmp_path / "raw_results.csv").exists()
    assert (tmp_path / "summary_results.csv").exists()
    assert (tmp_path / "summary_results.md").exists()
    assert (tmp_path / "gradient_checks.csv").exists()
