"""Generalized WALNUTS: measure-preserving flows + first-return stopping rules.

Companion code for

    Bou-Rabee, Carpenter, Kleppe.
    "WALNUTS with General Measure-Preserving Flows and
     Initial-Point-Symmetric Stopping Rules."

Quick start::

    import numpy as np
    from gwalnuts import make_sampler, FLOW_ISO, STOP_RADIAL_MAX
    from gwalnuts.targets import make_funnel, wrap_builtin_target

    tg = wrap_builtin_target("funnel_d11", make_funnel(nx=10))
    run = make_sampler(tg["logp"], tg["grad_logp"], FLOW_ISO, STOP_RADIAL_MAX)
    samples, depths, sizes, ngrad, stop, built, sel = run(
        tg["theta0"], np.zeros(tg["d"]), 2000,
        0.5, 0.05, 12, 10, 12345,
    )
"""

from .engine import (
    FLOW_HAM, FLOW_ISO,
    STOP_UTURN, STOP_POINCARE, STOP_RADIAL_MAX,
    b_step_update, cross_halfhyperplane, cross_radial_max,
    make_sampler, p_micro_pmf, p_micro_sample, uturn,
)
from .diagnostics import ess_geyer, summarize_chain
from .tuning import (estimate_section_center, expand_gamma_bracket,
                     make_nohalve_probe, tune_macro_step)

__version__ = "1.0.0"

__all__ = [
    "FLOW_HAM", "FLOW_ISO",
    "STOP_UTURN", "STOP_POINCARE", "STOP_RADIAL_MAX",
    "make_sampler", "p_micro_sample", "p_micro_pmf", "uturn",
    "cross_halfhyperplane", "cross_radial_max", "b_step_update",
    "ess_geyer", "summarize_chain",
    "tune_macro_step", "estimate_section_center",
    "make_nohalve_probe", "expand_gamma_bracket",
    "__version__",
]
