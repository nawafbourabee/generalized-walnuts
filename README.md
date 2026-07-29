# Generalized WALNUTS

Companion code for

> Nawaf Bou-Rabee, Bob Carpenter, and Tore Selland Kleppe.
> **WALNUTS with General Measure-Preserving Flows and Initial-Point-Symmetric
> Stopping Rules.** 2026.

The package implements the paper's six samplers, obtained by pairing two
measure-preserving flows with three initial-point-symmetric stopping rules.
It also includes the experiment driver used to produce every table in
Section 5.

|  | standard U-turn | half-hyperplane Poincaré | anchored radial max-to-max |
|---|---|---|---|
| **Hamiltonian flow** (leapfrog micro steps) | `walnuts-h` | `walnuts-ph` | `walnuts-ah` |
| **Isokinetic flow** (BAB micro steps, unconditionally stable) | `walnuts-i` | `walnuts-pi` | `walnuts-ai` |

All six use the same within-orbit refinement and orbit-selection procedure.
Each macro step is divided into 2^ℓ micro steps, where the level ℓ is
randomized and its probability is included in the Hastings ratio.  The six
samplers differ only in their flow and stopping rule.

**Headline result** (Section 5): the isokinetic anchored sampler
`walnuts-ai` was the most robust of the six variants.  On Neal's funnel, its
ESS per gradient was about **10.9×** that of `walnuts-h`.  It also performed
well on nonlinear and hierarchical targets, with about 10% overhead on an
easy Gaussian-regression target where the standard U-turn sampler was already
effective.

Function names follow the paper's pseudocode listings.
**[LISTINGS.md](LISTINGS.md)** shows where each listing appears in the code
and explains where several pseudocode operations are combined into one
routine.

This repository accompanies the paper and generalizes the original WALNUTS
algorithm; for the reference implementation of standard WALNUTS
(Hamiltonian flow, U-turn rule), see
[flatironinstitute/walnuts](https://github.com/flatironinstitute/walnuts).

## Install

```bash
pip install .          # or: pip install -e .[test]
```

Requires Python ≥ 3.10, NumPy, Numba, SciPy.

## Quick start

```python
import numpy as np
from gwalnuts import make_sampler, FLOW_ISO, STOP_RADIAL_MAX
from gwalnuts.targets import make_funnel, wrap_builtin_target

tg = wrap_builtin_target("funnel_d11", make_funnel(nx=10))
run = make_sampler(tg["logp"], tg["grad_logp"], FLOW_ISO, STOP_RADIAL_MAX)

samples, depths, sizes, ngrad, stop, built, sel = run(
    tg["theta0"],        # initial position
    np.zeros(tg["d"]),   # section center / radial anchor C
    2000,                # draws
    0.5,                 # macro step size h
    0.05,                # energy-error tolerance delta
    12,                  # i_max: at most 2^12 leaves per orbit
    10,                  # ell_max: at most 2^10 micro steps per macro step
    12345,               # seed
)
```

A custom target is specified by two Numba-compiled functions:
`logp(theta) -> float` and `grad_logp(theta, ngrad) -> ndarray`.  The gradient
function must increment `ngrad[0]` once per call.  This counter is used to
compute the ESS-per-gradient diagnostics.

## Reproducing the paper

The full experiment contains 7 targets × 6 samplers × 3 seeds.  Each
target–sampler combination uses 2,000 draws for calibration, 2,000 for
warmup, and 2,000 for production, together with a step-size search.  The full
run takes several hours on one CPU core; the Poincaré samplers account for
most of the cost.

```bash
gwalnuts-grid --outdir production_grid
```

You can run any subset without changing its seeds because each
target–flow–rule combination has a fixed seed:

```bash
gwalnuts-grid --target-key funnel_d11                       # one target
gwalnuts-grid --target-key eight_schools_centered \
              --flow Hamiltonian --rule "anchored U-turn"    # one cell x 3 seeds
gwalnuts-grid --smoke                                        # tiny smoke run
```

The command writes `raw_results.csv`, with one row per combination and seed,
and `summary_results.csv` and `summary_results.md`, with means and standard
deviations over seeds.  The paper's tables report the seed means from
`raw_results.csv`.

### Reproducibility notes

* The published seed is
  `20260627 + 100000·ti + 10000·fi + 1000·ri + si`, where the indices identify
  the target, flow, stopping rule, and replicate.  Unit tests fix both this
  formula and the ordering of those choices.
* Commit
  [`d667ab4`](https://github.com/nawafbourabee/generalized-walnuts/tree/d667ab449919de6cce5cc1cb9d724a654163c791)
  contains the implementation used for the reported experiments.  The
  current version corrects the reverse level search when no level at or below
  `ell_max` satisfies the effective-energy tolerance.  This correction can
  change a transition only when that cap is reached.
* With fixed NumPy, Numba, and BLAS versions, a run is deterministic.
  Different numerical-library versions can introduce small rounding
  differences, after which the trajectories may separate.  Results should
  therefore be compared statistically across software environments rather
  than sample by sample.
* Computational cost can vary substantially between seeds on funnel-shaped
  targets such as eight schools and stochastic volatility.  Some chains
  require much finer micro steps, especially near the funnel neck.  The
  `_sd` columns of `summary_results.csv` show this variation.

## Tests

```bash
pytest                # fast set: closed forms, predicates, gradients, 2 invariance runs
pytest -m slow        # full invariance suite (all six samplers) + grid smoke run
```

The invariance tests check the complete transition procedure on $N(0,I)$.
They use deliberately difficult settings: the macro step exceeds the
leapfrog stability limit, the micro-level cap is reached, and zero-weight
leaves occur frequently.

## Layout

```
gwalnuts/
  engine.py        six samplers; names follow the paper's listings
  targets.py       the seven benchmark targets of Section 5.1
  tuning.py        Gamma step-size calibration, section centers
  diagnostics.py   Geyer-IPS ESS, ESS/grad, ESJD/grad
  grid.py          production grid driver (console script: gwalnuts-grid)
tests/
LISTINGS.md        pseudocode <-> code map
```

## Citation

If you use this code, please cite the paper:

```bibtex
@unpublished{BouRabeeCarpenterKleppe2026,
  author = {Bou-Rabee, Nawaf and Carpenter, Bob and Kleppe, Tore Selland},
  title  = {{WALNUTS} with General Measure-Preserving Flows and
            Initial-Point-Symmetric Stopping Rules},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT — see [LICENSE](LICENSE).
