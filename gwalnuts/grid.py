"""Production grid runner: reproduces all experiments of Section 5.

Protocol (Section 5.2 of the paper): for every target x flow x stopping-rule
cell and every seed,

1. tune h to P(micro = 0) = 0.80 on a 2000-draw calibration chain,
2. for the first-return rules, freeze the section center C as the
   coordinate-wise median of a Poincare calibration chain at the tuned h,
3. run a 2000-draw warmup from theta_ref with the frozen (h, C),
4. run 2000 production draws from the last warmup state, and report the
   diagnostics of Section 5.2 averaged over K = 3 seeds.

Seed arithmetic (do not change; it defines the published runs):
``seed = 20260627 + 100000 * ti + 10000 * fi + 1000 * ri + si``
where ti, fi, ri, si index target, flow, rule, and seed replicate in the
orders below.  Filtering targets/flows/rules on the command line preserves
the indices, so any subset reproduces the corresponding published cells.

Usage::

    gwalnuts-grid --outdir production_grid            # full grid (long!)
    gwalnuts-grid --target-key funnel_d11             # one target
    gwalnuts-grid --smoke                             # tiny smoke run
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import engine
from .diagnostics import summarize_chain
from .targets import make_targets, finite_difference_check
from .tuning import estimate_section_center, tune_macro_step

DELTA = 0.05
GAMMA = 0.80
I_MAX = 12
MAX_ELL = 10

FLOWS = [
    ("Hamiltonian", engine.FLOW_HAM, 0.10),   # (name, code, h_warm)
    ("Isokinetic", engine.FLOW_ISO, 0.50),
]

RULES = [
    ("standard U-turn", engine.STOP_UTURN),
    ("half-hyperplane Poincare", engine.STOP_POINCARE),
    ("anchored U-turn", engine.STOP_RADIAL_MAX),
]

BASE_SEED = 20260627


def sampler_label(flow_code, stop_code):
    suffix = "h" if flow_code == engine.FLOW_HAM else "i"
    if stop_code == engine.STOP_UTURN:
        return f"walnuts-{suffix}"
    if stop_code == engine.STOP_POINCARE:
        return f"walnuts-p{suffix}"
    if stop_code == engine.STOP_RADIAL_MAX:
        return f"walnuts-a{suffix}"
    return f"walnuts-unknown-{suffix}"


def run_cell(tg, flow_name, flow_code, rule_name, stop_code, seed,
             n_cal, n_warm, n_prod, h_warm):
    h, frac, _, trace = tune_macro_step(
        tg, flow_code, stop_code, seed, n_cal, h_warm,
        delta=DELTA, gamma=GAMMA, i_max=I_MAX, max_ell=MAX_ELL,
    )
    if stop_code == engine.STOP_POINCARE or stop_code == engine.STOP_RADIAL_MAX:
        center = estimate_section_center(
            tg, flow_code, h, seed + 40_000, n_cal,
            delta=DELTA, i_max=I_MAX, max_ell=MAX_ELL,
        )
    else:
        center = np.zeros(tg["d"])
    run = engine.make_sampler(tg["logp"], tg["grad_logp"], flow_code, stop_code)
    warm = run(tg["theta0"].copy(), center, n_warm, h, DELTA, I_MAX, MAX_ELL,
               seed + 100_000)[0]
    prod, depths, sizes, ngrad, stop, built, selected_idx = run(
        warm[-1].copy(), center, n_prod, h, DELTA, I_MAX, MAX_ELL,
        seed + 200_000
    )
    row = summarize_chain(tg, prod, sizes, depths, stop, built, selected_idx, ngrad)
    row.update({
        "target_key": tg["key"],
        "target": tg["name"],
        "d": tg["d"],
        "stat": tg["stat_name"],
        "flow": flow_name,
        "rule": rule_name,
        "sampler": sampler_label(flow_code, stop_code),
        "seed": int(seed),
        "h": h,
        "frac_nohalve": frac,
        "center_norm": float(np.linalg.norm(center)),
        "tuning_trace_json": json.dumps(trace),
    })
    return row


def aggregate_rows(rows):
    groups = {}
    for r in rows:
        key = (r["target_key"], r["target"], r["d"], r["stat"], r["flow"], r["rule"], r.get("sampler", ""))
        groups.setdefault(key, []).append(r)
    agg = []
    metrics = [
        "h", "frac_nohalve", "ess_stat_per_kgrad", "ess_min_per_kgrad",
        "ess_median_per_kgrad", "esjd_stat_per_kgrad", "esjd_total_per_kgrad",
        "median_L", "p90_L", "p99_L", "cap_rate", "mean_grad_per_iter",
        "stat_mean", "stat_var", "mean_l2_error", "median_var_ratio",
    ]
    for key, vals in groups.items():
        item = {
            "target_key": key[0],
            "target": key[1],
            "d": key[2],
            "stat": key[3],
            "flow": key[4],
            "rule": key[5],
            "sampler": key[6],
            "n_seeds": len(vals),
        }
        for m in metrics:
            xs = [v[m] for v in vals if m in v and np.isfinite(v[m])]
            if xs:
                item[m + "_mean"] = float(np.mean(xs))
                item[m + "_sd"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        agg.append(item)
    return agg


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = []
    by_target = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)
    for target, vals in by_target.items():
        lines.append(f"## {target}\n")
        lines.append("| Sampler | Flow | Rule | h | ESS/stat/kg | ESSmin/kg | ESSmed/kg | ESJD/stat/kg | ESJD/total/kg | med L | p99 L | cap % |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in vals:
            lines.append(
                f"| {r.get('sampler', '')} | {r['flow']} | {r['rule']} | {r.get('h_mean', np.nan):.5g} | "
                f"{r.get('ess_stat_per_kgrad_mean', np.nan):.3g} | "
                f"{r.get('ess_min_per_kgrad_mean', np.nan):.3g} | "
                f"{r.get('ess_median_per_kgrad_mean', np.nan):.3g} | "
                f"{r.get('esjd_stat_per_kgrad_mean', np.nan):.3g} | "
                f"{r.get('esjd_total_per_kgrad_mean', np.nan):.3g} | "
                f"{r.get('median_L_mean', np.nan):.0f} | "
                f"{r.get('p99_L_mean', np.nan):.0f} | "
                f"{100*r.get('cap_rate_mean', np.nan):.1f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="production_grid")
    p.add_argument("--n-cal", type=int, default=2000)
    p.add_argument("--n-warm", type=int, default=2000)
    p.add_argument("--n-prod", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--full-ar1", action="store_true")
    p.add_argument("--target-key", action="append", default=None,
                   help="Run only targets with this key; repeat for multiple targets.")
    p.add_argument("--flow", action="append", default=None,
                   help="Run only flows with this name; repeat for multiple flows.")
    p.add_argument("--rule", action="append", default=None,
                   help="Run only stopping rules with this name; repeat for multiple rules.")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.n_cal = min(args.n_cal, 100)
        args.n_warm = min(args.n_warm, 100)
        args.n_prod = min(args.n_prod, 100)
        args.seeds = min(args.seeds, 1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    indexed_targets = list(enumerate(make_targets(full_ar1=args.full_ar1)))
    if args.target_key:
        wanted = set(args.target_key)
        indexed_targets = [(i, tg) for i, tg in indexed_targets if tg["key"] in wanted]
        missing = wanted.difference(tg["key"] for _, tg in indexed_targets)
        if missing:
            raise ValueError(f"Unknown target key(s): {sorted(missing)}")
    grad_rows = []
    for _, tg in indexed_targets:
        rel = finite_difference_check(tg)
        grad_rows.append({"target_key": tg["key"], "target": tg["name"],
                          "d": tg["d"], "fd_rel_grad_error": rel})
        print(f"GRAD {tg['key']}: {rel:.2e}", flush=True)
    write_csv(outdir / "gradient_checks.csv", grad_rows)

    raw_rows = []
    flows = list(enumerate(FLOWS))
    if args.flow:
        wanted_flows = set(args.flow)
        flows = [(i, x) for i, x in flows if x[0] in wanted_flows]
        missing = wanted_flows.difference(x[0] for _, x in flows)
        if missing:
            raise ValueError(f"Unknown flow name(s): {sorted(missing)}")
    rules = list(enumerate(RULES))
    if args.rule:
        wanted_rules = set(args.rule)
        rules = [(i, x) for i, x in rules if x[0] in wanted_rules]
        missing = wanted_rules.difference(x[0] for _, x in rules)
        if missing:
            raise ValueError(f"Unknown rule name(s): {sorted(missing)}")
    for ti, tg in indexed_targets:
        for fi, (flow_name, flow_code, h_warm) in flows:
            for ri, (rule_name, stop_code) in rules:
                for si in range(args.seeds):
                    seed = BASE_SEED + 100000 * ti + 10000 * fi + 1000 * ri + si
                    print(f"START {tg['key']} / {flow_name} / {rule_name} / seed {si}", flush=True)
                    row = run_cell(
                        tg, flow_name, flow_code, rule_name, stop_code, seed,
                        args.n_cal, args.n_warm, args.n_prod, h_warm
                    )
                    raw_rows.append(row)
                    write_csv(outdir / "raw_results_partial.csv", raw_rows)
                    print(
                        f"DONE {tg['key']} / {flow_name} / {rule_name}: "
                        f"h={row['h']:.5g}, ESSstat/kg={row['ess_stat_per_kgrad']:.3g}, "
                        f"ESSmin/kg={row['ess_min_per_kgrad']:.3g}, "
                        f"ESJDstat/kg={row['esjd_stat_per_kgrad']:.3g}, "
                        f"medL={row['median_L']:.0f}, p99={row['p99_L']:.0f}",
                        flush=True,
                    )

    agg_rows = aggregate_rows(raw_rows)
    write_csv(outdir / "raw_results.csv", raw_rows)
    write_csv(outdir / "summary_results.csv", agg_rows)
    write_markdown(outdir / "summary_results.md", agg_rows)
    print(outdir / "summary_results.md")
    print(outdir / "summary_results.csv")
    print(outdir / "raw_results.csv")


if __name__ == "__main__":
    main()
