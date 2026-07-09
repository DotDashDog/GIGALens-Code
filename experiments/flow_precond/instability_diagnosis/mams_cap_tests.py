"""Tests for the MAMS trajectory-length cap (max_num_integration_steps).

Design checkpoint (pre-registered; see mams_cap_notes.md for the full analysis):

  Hypothesis: yesterday's 3.6 h hang is n = L/eps explosion -- dual averaging
  drove eps down mid-transient (or a 20-step burnin froze a collapsed eps at
  the mode-0 boundary, where NO controller runs), with L fixed, so every
  transition costs ~L/eps integrator calls.

  Test A (healthy, cap never binds): patched module must be BIT-IDENTICAL to
    the unpatched snapshot for the same seed.
    Falsifier: any byte of position/step_size/L/num_integration_steps differs.
  Test B (pathological, mimics the hang: eps0=1e-8, 20-step burnin): the
    unpatched kernel's first transition alone needs ~L/eps ~ 3e8 integrator
    calls (shown arithmetically, and in projected wall-time from the measured
    per-call cost); the patched run must complete in bounded time (< ~5 min)
    with capped fraction near 1.
    Falsifier: patched run exceeds 10 min, or capped fraction < 0.5, or any
    realized n > 60.
  Test C (controller stability: eps0 = healthy/100, adequate burnin): the DA
    loop under the cap is self-correcting -- capped fraction ~0 over the last
    quarter of tuning, final eps within ~3x of the healthy run's final eps,
    and L does not wind up (final L <= (N_MAX/2) * final mean eps; only one L
    update ever happens, capped at 2x).
    Falsifier: capped fraction > 0.1 at end of tuning, or eps not recovered
    within 3x, or final L above the realizability bound.

Run:
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:<worktree>/src \
  <oldapi python> mams_cap_tests.py
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP = HERE / "tmp"
WORKTREE_SRC = HERE.parents[2] / "src"
SIDECAR = "/global/homes/l/linusu/sidecar_jax_upgrade"
SNAPSHOT = TMP / "mams_unpatched_snapshot.py"

N_MAX = 60
DIM = 8
N_CHAINS = 8
SEED = 3


# --------------------------------------------------------------------------
# Tiny duck-typed target: 8-dim anisotropic Gaussian (condition number 10).
# MAMS_JIT touches only model_seq.prob_model.log_prob(z)[0].
# --------------------------------------------------------------------------

def make_model_and_qz():
    import jax.numpy as jnp
    from tensorflow_probability.substrates import jax as tfp
    tfd = tfp.distributions

    scales = jnp.logspace(-0.5, 0.5, DIM)

    class GaussProbModel:
        def log_prob(self, z):
            lp = -0.5 * jnp.sum((z / scales) ** 2, axis=-1)
            return lp, None

    class TinySeq:
        prob_model = GaussProbModel()

    qz = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(DIM, dtype=jnp.float64),
        scale_diag=jnp.ones(DIM, dtype=jnp.float64),
    )
    return TinySeq(), qz


def run_mams(num_burnin, num_results, init_step_size=None, init_L=None,
             cap=None):
    """Run MAMS_JIT (whichever module PYTHONPATH resolves) with debug output."""
    from gigalens_research.inference import mams
    model_seq, qz = make_model_and_qz()
    kwargs = dict(
        n_hmc=N_CHAINS, num_burnin_steps=num_burnin, num_results=num_results,
        seed=SEED, debug_output=True, progress_bar=False,
        init_step_size=init_step_size, init_L=init_L,
    )
    if cap is not None:  # unpatched module has no such kwarg
        kwargs["max_num_integration_steps"] = cap
    t0 = time.perf_counter()
    hist = mams.MAMS_JIT(model_seq, qz, **kwargs)
    wall = time.perf_counter() - t0
    return hist, wall


def save_hist(hist, wall, out):
    import numpy as np
    arrays = dict(
        position=np.asarray(hist.position),
        step_size=np.asarray(hist.step_size),
        L=np.asarray(hist.L),
        num_integration_steps=np.asarray(hist.num_integration_steps),
        acceptance_rate=np.asarray(hist.acceptance_rate),
        wall=np.asarray(wall),
    )
    if hasattr(hist, "traj_capped"):
        arrays["traj_capped"] = np.asarray(hist.traj_capped)
    np.savez(out, **arrays)


# --------------------------------------------------------------------------
# Subprocess entry: run test A's configuration under the current PYTHONPATH.
# --------------------------------------------------------------------------

def main_runA(out):
    hist, wall = run_mams(num_burnin=200, num_results=200)
    save_hist(hist, wall, out)
    print(f"runA done in {wall:.1f}s -> {out}")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_orig_tree():
    """Symlink-copy of src/gigalens_research with mams.py = unpatched snapshot."""
    base = TMP / "src_orig"
    if base.exists():
        shutil.rmtree(base)
    dst_pkg = base / "gigalens_research"
    shutil.copytree(
        WORKTREE_SRC / "gigalens_research", dst_pkg,
        copy_function=os.symlink,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    target = dst_pkg / "inference" / "mams.py"
    target.unlink()
    shutil.copyfile(SNAPSHOT, target)
    return base


def subprocess_env(src_first):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SIDECAR}:{src_first}"
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    return env


def test_A():
    print("\n=== Test A: bit-identity when the cap never binds ===")
    import numpy as np
    orig_src = build_orig_tree()
    outs = {}
    for name, src in [("orig", orig_src), ("patched", WORKTREE_SRC)]:
        out = TMP / f"testA_{name}.npz"
        cmd = [sys.executable, str(Path(__file__).resolve()), "runA", str(out)]
        r = subprocess.run(cmd, env=subprocess_env(src), capture_output=True, text=True)
        print(f"[{name}] " + r.stdout.strip().replace("\n", " | "))
        if r.returncode != 0:
            print(r.stderr[-3000:])
            raise SystemExit(f"test A subprocess ({name}) failed")
        outs[name] = np.load(out)
    ok = True
    for key in ["position", "step_size", "L", "num_integration_steps", "acceptance_rate"]:
        a, b = outs["orig"][key], outs["patched"][key]
        same = (a.dtype == b.dtype) and (a.shape == b.shape) and (a.tobytes() == b.tobytes())
        print(f"  {key:24s} dtype={a.dtype} shape={a.shape} bit-identical={same}")
        ok &= same
    capped = outs["patched"]["traj_capped"]
    frac = capped.mean()
    nmax = outs["patched"]["num_integration_steps"].max()
    print(f"  patched capped fraction = {frac:.4f} (expect 0.0), max n = {nmax}")
    ok &= (frac == 0.0)
    print(f"Test A: {'PASS' if ok else 'FAIL'} [UNCERTIFIED]")
    return ok, outs["patched"]


def test_B():
    print("\n=== Test B: pathological 20-step burnin (hang class) ===")
    import numpy as np
    init_eps, burnin, results = 1e-8, 20, 100
    init_L = float(np.sqrt(DIM))  # the default the hang run used

    # Arithmetic for the UNPATCHED kernel (do not actually run it): the first
    # trajectory draw is halton(0)=0.5 -> n ~ rescale(L/eps)/2 + eps ~ L/eps.
    n_first = init_L / init_eps
    print(f"  unpatched: avg_n = L/eps = {init_L:.3f}/{init_eps:.0e} = {n_first:.3e} "
          f"integrator calls for a SINGLE transition")

    hist, wall = run_mams(burnin, results, init_step_size=init_eps, init_L=init_L)
    nis = np.asarray(hist.num_integration_steps)
    capped = np.asarray(hist.traj_capped)
    total_calls = int(nis[0].sum())  # shared across chains; per-chain trajectory cost
    per_call = wall / max(nis.sum(), 1)  # crude upper bound on per-call cost (all chains)
    print(f"  patched: wall = {wall:.1f}s for {burnin + results} steps, "
          f"total integrator calls/chain = {total_calls}")
    print(f"  patched: capped fraction = {capped.mean():.3f}, "
          f"max n = {nis.max()}, mean n = {nis.mean():.1f}")
    print(f"  measured cost <= {per_call * 1e6:.2f} us per chain-integrator-call "
          f"-> unpatched first transition alone >= {n_first * per_call * N_CHAINS / 60:.0f} min (projected)")
    final_eps = np.asarray(hist.step_size)[:, -1].mean()
    print(f"  final mean eps = {final_eps:.3e} (started {init_eps:.0e}; only 16 DA steps existed)")
    # Revised criterion (first run falsified the original "capped fraction > 0.5
    # over the WHOLE run" prediction -- see notes): the avg_n clamp binds through
    # the collapsed-eps burnin, then the L anti-windup clamp installs a realizable
    # L = (N_MAX/2)*eps at the L-update step, leaving the sampler PINNED at the
    # bound (visible via the >= flag) rather than clamped above it.
    burnin_capped = capped[:, :burnin].mean()
    print(f"  capped fraction over burnin = {burnin_capped:.3f}")
    ok = (wall < 600) and (burnin_capped > 0.5) and (capped.mean() > 0.5) \
        and (nis.max() <= N_MAX)
    print(f"Test B: {'PASS' if ok else 'FAIL'} [UNCERTIFIED]")
    return ok


def test_C(healthy):
    print("\n=== Test C: controller self-correction under the cap ===")
    import numpy as np
    healthy_final_eps = float(healthy["step_size"][:, -1].mean())
    init_eps = healthy_final_eps / 100.0
    burnin, results = 400, 200
    hist, wall = run_mams(burnin, results, init_step_size=init_eps)
    nis = np.asarray(hist.num_integration_steps)
    capped = np.asarray(hist.traj_capped)[0]  # shared across chains
    eps = np.asarray(hist.step_size)
    L = np.asarray(hist.L)[0]

    tuning = burnin  # frac_tune1+2+3 = 1.0 of burnin
    q = tuning // 4
    frac_by_quarter = [capped[k * q:(k + 1) * q].mean() for k in range(4)]
    frac_sampling = capped[tuning:].mean()
    final_eps = eps[:, -1].mean()
    ratio = final_eps / healthy_final_eps
    L_vals = np.unique(L)
    realizable = (N_MAX / 2) * final_eps
    print(f"  wall = {wall:.1f}s; init eps = {init_eps:.3e} (healthy final {healthy_final_eps:.3e})")
    print(f"  capped fraction by tuning quarter: "
          + ", ".join(f"{f:.2f}" for f in frac_by_quarter)
          + f"; sampling phase: {frac_sampling:.2f}")
    print(f"  final mean eps = {final_eps:.3e} (ratio to healthy final = {ratio:.2f}x)")
    print(f"  L trace unique values: {np.round(L_vals, 4)} "
          f"(init sqrt({DIM})={np.sqrt(DIM):.3f}; single ESS update, <=2x, "
          f"<= (N_MAX/2)*eps = {realizable:.3f})")
    print(f"  max n = {nis.max()}, mean n (sampling) = {nis[0, tuning:].mean():.1f}")
    ok = (frac_by_quarter[-1] <= 0.1) and (frac_sampling <= 0.1) \
        and (1 / 3 <= ratio <= 3) and (L[-1] <= realizable * (1 + 1e-9)) \
        and (nis.max() <= N_MAX)
    print(f"Test C: {'PASS' if ok else 'FAIL'} [UNCERTIFIED]")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "runA":
        main_runA(sys.argv[2])
        sys.exit(0)
    TMP.mkdir(exist_ok=True)
    okA, healthy = test_A()
    okB = test_B()
    okC = test_C(healthy)
    print(f"\nSummary: A={'PASS' if okA else 'FAIL'} B={'PASS' if okB else 'FAIL'} "
          f"C={'PASS' if okC else 'FAIL'}  [all UNCERTIFIED]")
    sys.exit(0 if (okA and okB and okC) else 1)
