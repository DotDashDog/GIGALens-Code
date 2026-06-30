r"""grid.py -- LAPS validation grid driver.

Runs a list of (target, config) cases through ``run_one``, saves each case's
JSON record + multi-panel PNG into ``results/<preset>/<run_id>/``, and writes
``summary.csv`` / ``summary.json`` (one row per run).

Usage
-----
    python grid.py <preset> [--only <substr>]

``--only <substr>`` runs only the cases whose ``run_id`` contains ``<substr>``
(used for the builder's single-cell sanity checks).

Presets
-------
  sanity : 1 run  (T_iso, paper schedule, self_calibrated x_i^2, warm, M=512).
  core   : default on all 4 beds + the E-A schedule x switch contrast.

  --- discriminating sweeps (COLD init + multi-seed, the ones that bite) ---
  mscale  : E-F unbiasedness. {T_iso,T_ill} x M{512,2048,8192} x seed{0,1,2},
            WARM.  b2_avg/b2_max vs the 1/M floor.                 (18 runs)
  ksweep  : E-B justify k. T_ill COLD, k{1.0,1.5,2.0,3.0} x seed{0,1,2}.
            switch_index, b2_*_at_switch, final b2_avg, steps_to_floor. (12)
  initmode: E-C warm/cold/off. T_ill x seed{0,1,2}, cells {warm, cold,
            off-qz warm}.  premature-switch test.                    (9)
  schedC  : schedule+C from cold. T_ill COLD, sched{paper,emaus} x
            C{0.025,0.1} x seed{0,1,2}.  steps_to_floor discriminates. (12)

  --- adversarial-grader follow-ups (over-claim controls) ---
  offqz    : F1 off-qz robustness. T_ill M=512 seed{0..9} x cells {warm, cold,
             off, off+persist2(switch_persist=2)}.  failure-rate + margin. (40)
  phase2off: F3 isolate Phase-2. T_ill COLD, sched{paper,emaus} x C{0.025,0.1}
             x phase2{on,off} x seed{0,1,2}.  unadjusted-bias control.     (24)

Each case is a dict: {"target": <name>, "qz_mode": ..., **laps_kwargs}. The
``run_id`` encodes target + every swept knob (seed, k, C, qz, init) so cells are
independent and the preset is resumable (an existing ``record.json`` is reused).
"""

import csv
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import targets as _targets       # noqa: E402
import run as _run               # noqa: E402
import plots as _plots           # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
SEEDS = [0, 1, 2]

# Default (paper-faithful) LAPS config for the validation beds. Small + CPU-fast.
DEFAULT_CFG = dict(
    schedule="paper", switch="paper", switch_mode="self_calibrated", switch_k=1.5,
    init_mode="warm", num_chains=512, num_unadjusted_steps=300,
    num_adjusted_steps=200, chunk_size=25, seed=0,
)

# Cold-init beds need a longer Phase 1 to equilibrate the high-variance dims;
# warm beds switch early. Budgets chosen so each CPU run stays well under ~60s.
COLD_CFG = dict(DEFAULT_CFG, init_mode="cold", num_unadjusted_steps=400,
                num_adjusted_steps=150, chunk_size=25)
# WARM unbiasedness sweep: budgets FIXED across M (so b2 floor scaling is clean).
WARM_M_CFG = dict(DEFAULT_CFG, init_mode="warm", num_unadjusted_steps=200,
                  num_adjusted_steps=100, chunk_size=25)
# P6a stiff beds (T_rot, T_banana_hi): M=1024, longer Phase 1 so a stiff target
# CAN equilibrate within the budget (else nothing about k/precond is learnable),
# generous Phase 2 so bisection has room to chase 0.70 on the ill-conditioned bed.
STIFF_CFG = dict(DEFAULT_CFG, num_chains=1024, num_unadjusted_steps=400,
                 num_adjusted_steps=200, chunk_size=25)


def _case(target, base=DEFAULT_CFG, **overrides):
    cfg = dict(base)
    cfg.update(overrides)
    cfg["target"] = target
    return cfg


def build_preset(preset):
    if preset == "sanity":
        return [_case("T_iso")]
    if preset == "core":
        cases = [_case(t) for t in _targets.ALL_TARGETS]
        for bed in ("T_iso", "T_ill"):
            for sched in ("paper", "emaus"):
                for sw in ("paper", "emaus"):
                    if bed == "T_iso" and sched == "paper" and sw == "paper":
                        continue
                    cases.append(_case(bed, schedule=sched, switch=sw))
        return cases

    if preset == "mscale":   # E-F unbiasedness (WARM, fixed budgets across M)
        cases = []
        for bed in ("T_iso", "T_ill"):
            for M in (512, 2048, 8192):
                for s in SEEDS:
                    cases.append(_case(bed, base=WARM_M_CFG, num_chains=M, seed=s,
                                       qz_mode="warm"))
        return cases

    if preset == "ksweep":   # E-B justify k (T_ill COLD)
        cases = []
        for k in (1.0, 1.5, 2.0, 3.0):
            for s in SEEDS:
                cases.append(_case("T_ill", base=COLD_CFG, switch_k=k, seed=s))
        return cases

    if preset == "initmode":  # E-C warm / cold / off-qz warm (T_ill)
        cases = []
        for s in SEEDS:
            cases.append(_case("T_ill", base=DEFAULT_CFG, init_mode="warm",
                               qz_mode="warm", seed=s))
            cases.append(_case("T_ill", base=COLD_CFG, seed=s))
            # off-qz warm = inflated-2x(std)+shifted fake qz, warm init -> the
            # premature-switch test (fires before equilibration?).
            cases.append(_case("T_ill", base=DEFAULT_CFG, init_mode="warm",
                               qz_mode="off", seed=s))
        return cases

    if preset == "schedC":   # schedule + C efficiency from COLD (T_ill)
        cases = []
        for sched in ("paper", "emaus"):
            for C in (0.025, 0.1):
                for s in SEEDS:
                    cases.append(_case("T_ill", base=COLD_CFG, schedule=sched,
                                       C=C, seed=s))
        return cases

    if preset == "offqz":   # F1: off-qz robustness + persistence guard (10 seeds)
        # T_ill, M=512, seeds 0..9; cells warm / cold / off / off+persist2.
        # Question: is the off-qz seed1 hard-fail a fluke or a pattern, and does
        # requiring 2 consecutive ripe chunks (switch_persist=2) fix it?
        cases = []
        for s in range(10):
            cases.append(_case("T_ill", base=DEFAULT_CFG, init_mode="warm",
                               qz_mode="warm", seed=s))
            cases.append(_case("T_ill", base=COLD_CFG, seed=s))
            cases.append(_case("T_ill", base=DEFAULT_CFG, init_mode="warm",
                               qz_mode="off", seed=s))
            cases.append(_case("T_ill", base=DEFAULT_CFG, init_mode="warm",
                               qz_mode="off", switch_persist=2, seed=s))
        return cases

    if preset == "phase2off":  # F3: isolate the Phase-2 correction (control)
        # T_ill COLD, schedule{paper,emaus} x C{0.025,0.1} x phase2{on,off} x
        # seed{0,1,2}. Prediction: phase2 OFF -> b^2 higher + C/schedule-dependent
        # (unadjusted asymptotic bias); phase2 ON -> b^2 drops to the 1/M floor.
        cases = []
        for sched in ("paper", "emaus"):
            for C in (0.025, 0.1):
                for p2 in (True, False):
                    for s in SEEDS:
                        cases.append(_case("T_ill", base=COLD_CFG, schedule=sched,
                                           C=C, phase2_enabled=p2, seed=s))
        return cases

    if preset == "stiff":   # P6a STIFF-PROXY VALIDATION (T_rot, T_banana_hi)
        # 2 beds x init{warm,cold} x k{1.5,3.0} x phase2{on,off} x seed{0,1,2}.
        # Questions: (a) does Phase-2 reduce bias on the banana? (b) does the
        # DIAGONAL precond FAIL on the rotated bed (accept can't reach 0.70 / b2
        # above floor) => is a dense metric needed? (c) does k finally bite on a
        # stiff bed where equilibration may exceed the window?
        cases = []
        for bed in ("T_rot", "T_banana_hi"):
            for init in ("warm", "cold"):
                for k in (1.5, 3.0):
                    for p2 in (True, False):
                        for s in SEEDS:
                            qz = "warm"  # warm-qz init (cold ignores qz)
                            cases.append(_case(
                                bed, base=STIFF_CFG, init_mode=init,
                                qz_mode=qz, switch_k=k, phase2_enabled=p2,
                                seed=s))
        return cases

    raise ValueError(f"unknown preset {preset!r}")


def _run_id(cfg):
    """Encode every swept knob so cells are unique/independent/resumable."""
    parts = [
        cfg["target"],
        f"sched-{cfg['schedule']}", f"sw-{cfg['switch']}",
        f"mode-{cfg['switch_mode']}", f"init-{cfg['init_mode']}",
        f"qz-{cfg.get('qz_mode', 'warm')}", f"M{cfg['num_chains']}",
        f"k{cfg.get('switch_k', 1.5)}",
    ]
    if cfg.get("C") is not None:
        parts.append(f"C{cfg['C']}")
    if int(cfg.get("switch_persist", 1)) != 1:
        parts.append(f"persist{int(cfg['switch_persist'])}")
    if "phase2_enabled" in cfg:
        parts.append("phase2on" if cfg["phase2_enabled"] else "phase2off")
    parts.append(f"seed{cfg.get('seed', 0)}")
    return "__".join(parts)


def _row(target_name, cfg, qz_mode, record, dt, rid):
    I, R = record["internals"], record["results"]
    return dict(
        target=target_name, schedule=cfg.get("schedule"), switch=cfg.get("switch"),
        switch_mode=cfg.get("switch_mode"), init_mode=cfg.get("init_mode"),
        qz_mode=qz_mode, M=record["config"]["num_chains"],
        switch_k=cfg.get("switch_k", 1.5), C=cfg.get("C"), seed=cfg.get("seed", 0),
        switch_persist=int(cfg.get("switch_persist", 1)),
        phase2_enabled=bool(cfg.get("phase2_enabled", True)),
        switch_index=I["switch_index"], switched=I["switched"],
        phase1_len=I["phase1_len"],
        b2_max=R["b2_max"], b2_avg=R["b2_avg"], b2_success=R["b2_success"],
        b2_floor=R["b2_floor"],
        # --- b^2-trajectory (the new discriminating metrics) ---
        b2_avg_at_switch=R["b2_avg_at_switch"],
        b2_max_at_switch=R["b2_max_at_switch"],
        steps_to_floor=R["steps_to_floor"], b2_floor2=R["b2_floor2"],
        max_var_rel_err=R["max_var_rel_err"],
        accept_final=float(I["p2_accept"][-1]),
        frozen=bool(np.asarray(I["p2_frozen"])[-1]),
        target_accept=I["target_accept"], nan_free=record["nan_free"],
        seconds=round(dt, 2), run_id=rid,
    )


def run_preset(preset, only=None):
    cases = build_preset(preset)
    out_root = os.path.join(RESULTS_DIR, preset)
    os.makedirs(out_root, exist_ok=True)
    if only is not None:
        toks = [t for t in only.split(",") if t]
        cases = [c for c in cases
                 if all(t in _run_id({**c}) for t in toks)]
    print(f"[grid] preset={preset}  {len(cases)} run(s) -> {out_root}"
          + (f"  (--only {only})" if only else ""), flush=True)

    summary_rows = []
    for i, cfg in enumerate(cases, 1):
        cfg = dict(cfg)
        target_name = cfg.pop("target")
        qz_mode = cfg.pop("qz_mode", "warm")
        rid = _run_id({**cfg, "target": target_name, "qz_mode": qz_mode})
        run_dir = os.path.join(out_root, rid)
        os.makedirs(run_dir, exist_ok=True)
        json_path = os.path.join(run_dir, "record.json")
        png_path = os.path.join(run_dir, "diagnostics.png")

        # resumability: reuse an existing completed record.
        if os.path.exists(json_path):
            print(f"[grid] ({i}/{len(cases)}) {rid}  [cached]", flush=True)
            with open(json_path) as fh:
                record = json.load(fh)
            summary_rows.append(_row(target_name, cfg, qz_mode, record, 0.0, rid))
            continue

        print(f"[grid] ({i}/{len(cases)}) {rid} ...", flush=True)
        t0 = time.time()
        target = _targets.make_target(target_name)
        record = _run.run_one(target, run_id=rid, qz_mode=qz_mode, **cfg)
        dt = time.time() - t0

        _run.save_record_json(record, json_path)
        _plots.plot_run(record, png_path)

        row = _row(target_name, cfg, qz_mode, record, dt, rid)
        summary_rows.append(row)
        print(f"[grid]   done {dt:.1f}s  switch={row['switch_index']} "
              f"b2_avg={row['b2_avg']:.3e} b2_avg@sw={row['b2_avg_at_switch']:.3e} "
              f"steps_to_floor={row['steps_to_floor']} "
              f"accept_final={row['accept_final']:.3f}", flush=True)

    summ_json = os.path.join(out_root, "summary.json")
    with open(summ_json, "w") as fh:
        json.dump(summary_rows, fh, indent=2)
    summ_csv = os.path.join(out_root, "summary.csv")
    if summary_rows:
        with open(summ_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"[grid] wrote {summ_csv}\n[grid] wrote {summ_json}", flush=True)
    return summary_rows


if __name__ == "__main__":
    args = sys.argv[1:]
    preset = args[0] if args else "sanity"
    only = None
    if "--only" in args:
        only = args[args.index("--only") + 1]
    run_preset(preset, only=only)
