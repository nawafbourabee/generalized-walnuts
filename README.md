# Generalized WALNUTS

Companion code for

> Nawaf Bou-Rabee, Bob Carpenter, and Tore Selland Kleppe.
> **WALNUTS with General Measure-Preserving Flows and Initial-Point-Symmetric
> Stopping Rules.** 2026.

The package implements the paper's six samplers — two measure-preserving
flows crossed with three initial-point-symmetric stopping rules — and the
production grid that generates every table in Section 5.

|  | standard U-turn | half-hyperplane Poincaré | anchored radial max-to-max |
|---|---|---|---|
| **Hamiltonian flow** (leapfrog micro steps) | `walnuts-h` | `walnuts-ph` | `walnuts-ah` |
| **Isokinetic flow** (BAB micro steps, unconditionally stable) | `walnuts-i` | `walnuts-pi` | `walnuts-ai` |

All six share the WALNUTS within-orbit step-size adaptation (macro steps
split into 2^ℓ micro steps, with the randomized level selection and exact
Hastings correction of the paper) and the same orbit-selection mechanism;
only the flow and the stopping rule change.

**Headline result** (Section 5): the isokinetic anchored sampler
`walnuts-ai` is the most robust of the six variants, improving ESS per
gradient by about **10.8×** over `walnuts-h` on Neal's funnel and performing
strongly on nonlinear and hierarchical targets, while paying only ~10%
overhead on an easy Gaussian-regression target where the standard U-turn
sampler already excels.

Function names follow the paper's pseudocode listings; **[LISTINGS.md](LISTINGS.md)**
gives the complete listing-by-listing map, including the few places where a
listing is realized in fused or streaming form.

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

Custom targets are two Numba-jitted functions: `logp(theta) -> float` and
`grad_logp(theta, ngrad) -> ndarray`, where `grad_logp` increments
`ngrad[0]` once per call (this side effect is the gradient accounting behind
the ESS-per-gradient diagnostics).

## Reproducing the paper

The full grid (7 targets × 6 samplers × 3 seeds, each cell with 2000-draw
calibration, warmup, and production plus step-size bisection) is a long run —
hours on a single core, dominated by the Poincaré cells:

```bash
gwalnuts-grid --outdir production_grid
```

Any subset reproduces the corresponding published cells, because the cell
seeds are addressed by index, not by position in the filtered list:

```bash
gwalnuts-grid --target-key funnel_d11                       # one target
gwalnuts-grid --target-key eight_schools_centered \
              --flow Hamiltonian --rule "anchored U-turn"    # one cell x 3 seeds
gwalnuts-grid --smoke                                        # tiny smoke run
```

Outputs: `raw_results.csv` (one row per cell and seed, including the tuning
trace), `summary_results.csv` / `summary_results.md` (means ± sd over seeds).
The paper's tables are the seed means of `raw_results.csv`.

### Reproducibility notes

* The seed arithmetic `20260627 + 100000·ti + 10000·fi + 1000·ri + si` and
  the target/flow/rule orderings define the published runs and are frozen by
  a unit test.
* On a fixed NumPy/Numba/BLAS stack, reruns are deterministic and the
  sampler is bit-for-bit equivalent to the code that produced the paper's
  results (verified for all six samplers).  Across different
  versions, last-bit rounding differences are amplified by the chaotic
  dynamics, so individual cells reproduce statistically (within seed-to-seed
  spread) rather than bitwise.
* Per-seed costs are heavy-tailed on the funnel-like targets (eight schools,
  stochastic volatility): occasional seeds spend long stretches at deep
  micro-refinement, and the Γ-tuning root can be poorly conditioned when the
  calibration chain visits the funnel neck.  Expect visibly larger
  seed-to-seed variance there; the `_sd` columns of `summary_results.csv`
  quantify it.

## Tests

```bash
pytest                # fast set: closed forms, predicates, gradients, 2 invariance runs
pytest -m slow        # full invariance suite (all six samplers) + grid smoke run
```

The invariance tests are the end-to-end correctness check: each sampler must
recover N(0, I) exactly under stress settings (macro step beyond the
leapfrog stability limit, micro-level cap binding) where reversibility
failures occur in a large fraction of transitions.

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
