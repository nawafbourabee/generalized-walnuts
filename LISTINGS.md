# Pseudocode ↔ code map

Every listing in the companion paper corresponds to a specific function in
`gwalnuts/engine.py` (samplers), `gwalnuts/tuning.py` (calibration), or
`gwalnuts/grid.py` (experimental protocol).  This file records the mapping
and the few places where the implementation realizes a listing in a fused or
streaming form; the realized transition kernel is identical.

| Paper listing / definition | Code | Notes |
|---|---|---|
| Listing `p-micro` | `engine.p_micro_sample`, `engine.p_micro_pmf` | pmf (2/3, 1/3) on {ℓ*, ℓ*+1}. |
| Listing `micro` | forward-search loop inside `build_leaf_rand` | Fused with the integration: the level search integrates each candidate level and exits at the first level whose H_eff swing stays within δ, rather than calling a separate `micro` routine; the accepted level's endpoint is reused (valid by time-symmetry). A fully failed search sets the sentinel ℓ* = ℓ_max + 1, so drawn levels reach ℓ_max + 2 (the listing's semantics). |
| Listing `BAB` | `engine.b_step_update` (B step) + position drift inside `build_leaf_rand` | Closed-form B step of Proposition *b-step*; ΔlogJ = −(d−1) log(cosh β + γ₀ sinh β) evaluated in the numerically stable form −(d−1)(δ + log pre). Backward steps negate ρ, step forward, and negate back (time reversal). |
| Listing `build-leaf-rand` | `build_leaf_rand` inside `engine.make_sampler` | Forward micro search, p-micro draw, reverse micro search over ℓ < ℓ_p (the forward swing is reused at ℓ_p itself, exact by time-symmetry), and the weight update with the Hastings ratio p(ℓ\|ℓ*₋)/p(ℓ\|ℓ*). Weight −∞ iff the reverse check fails or ℓ leaves the reverse pmf's support. |
| Listing `walnuts-step`, `extend-orbit` | U-turn branch of `run` in `engine.make_sampler`, with `_set_leaf`, `_merge_orbits`, `_copy_orbit` | Stochastic doubling; within-extension merges use Barker selection with sub-U-turn checks on every merged subtree (`merge_mode = −1`); the top-level old-orbit/extension merge is biased toward the fresh block (`merge_mode = 1` growing right, `0` growing left). A zero-weight leaf or sub-U-turn discards the whole candidate extension before any merge (stop = 1); a U-turn across the merged orbit terminates after the biased merge (stop = 2). O(log L) memory via the merge stack. |
| First-return driver (`walnuts-fr`) | first-return branch of `run` in `engine.make_sampler` | Streaming reservoir realizes the within-extension categorical draw in O(1) memory; the biased-progressive merge accepts the extension's candidate with probability min(1, W_ext/W). Rounds whose arm has terminated are skipped with the round counter still advancing. |
| `draw-section` | inline in the first-return branch | Fresh η uniform on the sphere and γ uniform on the sphere orthogonal to η per transition, section centered at C. For the anchored rule no section direction is needed, but the γ draw is retained (and discarded) because the published runs consumed it: keeping it reproduces every published chain bit for bit. |
| Listing `cross-halfhyperplane` / Def. *pair-cross* | `engine.cross_halfhyperplane` | Unoriented sign change with crossing parameter t ∈ (0, 1] and side condition γᵀ(θ\* − C) > 0. |
| Listing `cross-radial-max` / Def. *pair-cross* | `engine.cross_radial_max` | Oriented max-to-max test in forward physical time; the σ-reordering for backward-grown arms is realized by the `dir_` argument. |
| Def. *frs* (first-return segment) | first-return branch | The crossing leaf terminates growth but is **not** retained as a selectable state. A zero-weight leaf clips the arm at the preceding positive-weight leaf (leaves built earlier in the same round are kept); the opposite arm continues independently. |
| Eq. (Γ) tuning criterion | `tuning.tune_macro_step`, `tuning.make_nohalve_probe`, `tuning.expand_gamma_bracket` | P(micro = 0) = Γ = 0.80 estimated on a calibration chain (first-return rules calibrated with C = 0), geometric bisection, 24 iterations. |
| Section centers (Sec. 5.2) | `tuning.estimate_section_center` | Coordinate-wise median of a Poincaré calibration chain at the tuned h; frozen before warmup and production. |
| ESS (Geyer IPS), eqs. (essgrad), (esjdgrad) | `diagnostics.ess_geyer`, `diagnostics.summarize_chain` | G_N charges **all** production gradients, including failed and crossing leaves; calibration and warmup gradients are excluded. |
| Experimental protocol (Sec. 5.2) | `grid.run_cell`, `grid.main` | Seed arithmetic `20260627 + 100000·ti + 10000·fi + 1000·ri + si`; N_cal = N_warm = N_prod = 2000; K = 3 seeds; δ = 0.05, i_max = 12, ℓ_max = 10; h_warm = 0.10 (Hamiltonian) / 0.50 (isokinetic). |

## Stop codes reported by `run`

| code | meaning |
|---|---|
| 1 | candidate rejected: zero-weight (reversibility-failure) leaf; for the U-turn rule also a sub-U-turn inside the extension |
| 2 | U-turn across the merged orbit (normal termination, U-turn rule) |
| 3 | both arms returned to the section (normal termination, first-return rules) |
| 4 | orbit-size cap 2^i_max reached without a zero-weight leaf |

`cap_rate` in the result tables is the fraction of transitions with code 4;
`candidate_reject_rate` is the fraction with code 1 (note the code-1
semantics differ between the U-turn and first-return families, so this
column should not be compared across families).
