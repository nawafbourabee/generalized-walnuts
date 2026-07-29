# Pseudocode ↔ code map

Every listing in the companion paper corresponds to a function in
`gwalnuts/engine.py` (samplers), `gwalnuts/tuning.py` (calibration), or
`gwalnuts/grid.py` (experiments).  The table below shows this correspondence
and explains where the code combines several pseudocode operations.

| Paper listing or definition | Code | Notes |
|---|---|---|
| Listing `p-micro` | `engine.p_micro_sample`, `engine.p_micro_pmf` | Selects ℓ* with probability 2/3 and ℓ*+1 with probability 1/3. |
| Listing `micro` | Forward level-search loop inside `build_leaf_rand` | The code searches and integrates at the same time. It starts at level 0 and stops at the first level whose effective-energy range is at most δ. The accepted endpoint is reused. If no level through ℓ_max is acceptable, the search returns the default value ℓ_max+1; the subsequent random choice may therefore be as large as ℓ_max+2. |
| Listing `BAB` | `engine.b_step_update` and the position update inside `build_leaf_rand` | Applies a half B step, a full position update, and a second half B step. The B step uses the stable form of its closed solution and log-Jacobian. Backward integration uses time reversal: negate ρ, take a forward step, and negate ρ again. |
| Listing `build-leaf-rand` | `build_leaf_rand` inside `engine.make_sampler` | Performs the forward level search, samples ℓ′, and performs the same capped search from the proposed endpoint in reverse. If ℓ′ is within the cap, time-symmetry allows the forward effective-energy range to be reused. If no capped reverse level is acceptable, the reverse search returns ℓ_max+1. The leaf weight includes the Hastings ratio $p_{\mathrm{micro}}(\ell'\mid\ell^+)/p_{\mathrm{micro}}(\ell'\mid\ell^*)$; it is zero when the numerator is zero. |
| Listings `WALNUTS-STEP` and `extend-orbit` | First-return branch of `run` in `engine.make_sampler` | At doubling round i, the code chooses a direction and attempts at most 2^i new leaves. Growth in that direction stops at its first section crossing or zero-weight leaf. The boundary leaf is not selectable. The selected leaf is updated online, so memory use does not increase with the orbit length. The newly generated extension replaces the current selection with probability $\min(1,W_{\rm ext}/W)$. |
| Standard U-turn driver | U-turn branch of `run`, with `_set_leaf`, `_merge_orbits`, and `_copy_orbit` | Uses stochastic doubling and checks every completed subtree for a U-turn. A zero-weight leaf or a U-turn inside the new extension rejects that extension before it is joined to the existing orbit (stop code 1). A U-turn across the complete joined orbit stops the transition after selection (stop code 2). The merge stack uses O(log L) memory. |
| `draw-section` | First-return branch of `run` | At each transition, η is drawn uniformly from the unit sphere and γ uniformly from the unit sphere orthogonal to η; the section is centered at C. The anchored radial rule does not use η or γ. The code nevertheless retains these unused draws so that commit `d667ab4` reproduces the random-number sequence used for the reported experiments. |
| Listing `cross-halfhyperplane` / Definition *pair-cross* | `engine.cross_halfhyperplane` | Checks for a sign change across the half-hyperplane. The interpolated crossing time must lie in (0,1], and the crossing point must satisfy γᵀ(θ*−C)>0. |
| Listing `cross-radial-max` / Definition *pair-cross* | `engine.cross_radial_max` | Checks whether the radial derivative changes from positive to negative in forward physical time. For backward integration, `dir_` reverses the generation order before applying the test. |
| Definition *frs* (first-return segment) | First-return branch of `run` | A crossing or zero-weight leaf stops growth in that direction and is not selectable. Leaves accepted earlier in the same doubling round are kept, and growth in the other direction may continue. |
| Equation (Γ) tuning criterion | `tuning.tune_macro_step`, `tuning.make_nohalve_probe`, `tuning.expand_gamma_bracket` | Chooses h so that level 0 is acceptable with probability Γ=0.80, estimated from a calibration chain. The search uses 24 geometric-bisection iterations. First-return rules use C=0 during this calibration. |
| Section centers (Section 5.2) | `tuning.estimate_section_center` | Uses the coordinate-wise median of a Poincaré calibration chain. The center is then held fixed during warmup and production. |
| ESS and ESJD per gradient | `diagnostics.ess_geyer`, `diagnostics.summarize_chain` | Counts every production gradient evaluation, including evaluations for zero-weight and crossing leaves. Calibration and warmup are not included. |
| Experiment settings (Section 5.2) | `grid.run_cell`, `grid.main` | Uses seed `20260627 + 100000·ti + 10000·fi + 1000·ri + si`; 2,000 calibration, warmup, and production draws; 3 seeds; δ=0.05; i_max=12; ℓ_max=10; and initial step size 0.10 for Hamiltonian flow or 0.50 for isokinetic flow. |

## Stop codes returned by `run`

| Code | Meaning |
|---|---|
| 1 | A zero-weight leaf was encountered; for the U-turn rule, this code also covers a U-turn inside the proposed extension. |
| 2 | The complete joined orbit made a U-turn. |
| 3 | Both integration directions reached their first section crossing. |
| 4 | The orbit reached the limit 2^i_max without encountering a zero-weight leaf. |

`cap_rate` in the result tables is the fraction of transitions with code 4;
`candidate_reject_rate` is the fraction with code 1.  Code 1 has a broader
meaning for the U-turn sampler than for the first-return samplers, so this
rate should not be compared across the two families.
