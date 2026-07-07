"""Shared helpers for experiments T0/T1 (docs/logs/why-hard-to-sample.md).

Kept deliberately flat and dependency-light, matching the repo's existing
experiment style (see experiments/sim_carousel/_h1h2_diag/driver_h1h2.py).

Three responsibilities:
  1. run_standard_mclmc  -- invoke MCLMC_JIT with the standard config, save the
     debug history npz + a JSON provenance manifest, exactly like driver_h1h2.py.
  2. compute_diagnostics -- per-parameter rank-R-hat / bulk-ESS (arviz), matching
     the conventions in src/gigalens_research/inference_utils/posterior.py.
  3. load_carousel_target -- rebuild the carousel prob_model/model_seq and the
     MAP-centered "under"-mode qz exactly as driver_h1h2.py does.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np


# ---------------------------------------------------------------------------
# float64 guard (repo rule: sampling code assumes float64 and must RAISE if not)
# ---------------------------------------------------------------------------

def assert_x64() -> None:
    """Raise unless JAX is running in float64. Do NOT silently enable it here --
    a mis-launched job (missing JAX_ENABLE_X64=1) must fail loudly, not run in
    float32 and quietly corrupt the experiment (O5 in the log)."""
    import jax
    import jax.numpy as jnp

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "jax_enable_x64 is OFF. This experiment requires float64. "
            "Launch with JAX_ENABLE_X64=1 (see slurm/*.sh)."
        )
    got = jnp.zeros(1).dtype
    if got != jnp.float64:
        raise RuntimeError(f"float64 not active despite jax_enable_x64; got {got}.")


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def git_commit(cwd: str | None = None) -> str:
    """Best-effort git HEAD of the worktree; 'unknown' if git is unavailable."""
    cwd = cwd or os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 1. run the sampler
# ---------------------------------------------------------------------------

def run_standard_mclmc(model_seq, qz, config, seed, out_path,
                       *, target_desc=None, provenance=None):
    """Run MCLMC_JIT(debug_output=True) with the standard `config` and `seed`,
    save the debug history to `out_path` (.npz) plus a sidecar `.json` manifest.

    `target_desc` (str) and `provenance` (dict) are recorded verbatim in the
    manifest so an arm can be traced back to its target and source data.

    Returns the numpy `position` array of shape (chains, burnin+results, dim).
    """
    assert_x64()
    from gigalens_research.inference import MCLMC_JIT

    kwargs = config.mclmc_kwargs(seed=seed, debug_output=True)
    t0 = time.perf_counter()
    hist = MCLMC_JIT(model_seq=model_seq, qz=qz, **kwargs)
    wall = time.perf_counter() - t0
    print(f"[run_standard_mclmc] seed={seed} sampling wall {wall:.1f}s")

    pos = np.asarray(hist.position)          # (chains, total, dim)
    xi = np.asarray(hist.xi)
    ss = np.asarray(hist.step_size)
    L_ = np.asarray(hist.L)
    # Match driver_h1h2.py: save only the first chain's inverse mass matrix
    # history (they are shared across chains) to keep the npz small.
    imm = np.asarray(hist.inverse_mass_matrix[:1])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez(
        out_path,
        position=pos, xi=xi, step_size=ss, L=L_, inverse_mass_matrix=imm,
        nb=config.num_burnin_steps, nr=config.num_results, seed=int(seed),
    )

    manifest = {
        "out_npz": os.path.abspath(out_path),
        "config": config.to_dict(),
        "mclmc_kwargs": kwargs,
        "seed": int(seed),
        "target_desc": target_desc,
        "provenance": provenance or {},
        "position_shape": list(pos.shape),
        "sampling_wall_s": wall,
        "git_commit": git_commit(),
        "timestamp_utc": _now(),
    }
    man_path = os.path.splitext(out_path)[0] + ".manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[run_standard_mclmc] saved {out_path} and {man_path}")
    return pos


# ---------------------------------------------------------------------------
# 2. diagnostics (arviz, matching posterior.py conventions)
# ---------------------------------------------------------------------------

def _az_diag(sz, fn):
    """Apply an arviz diagnostic per parameter on a (chains, draws, params)
    array -- identical wrapping to posterior.py:_az_diag."""
    import arviz as az
    ds = az.convert_to_dataset(np.asarray(sz))
    var = list(ds.data_vars)[0]
    return np.atleast_1d(np.asarray(fn(ds)[var].values))


def _rank_rhat(sz):
    """max(rank-R-hat, folded-R-hat), matching SamplerPosterior._rhat."""
    import arviz as az
    rank = _az_diag(sz, lambda ds: az.rhat(ds, method="rank"))
    folded = _az_diag(sz, lambda ds: az.rhat(ds, method="folded"))
    return np.maximum(rank, folded)


def _bulk_ess(sz):
    """Rank-normalized bulk-ESS, matching SamplerPosterior._ess."""
    import arviz as az
    return _az_diag(sz, lambda ds: az.ess(ds, method="bulk"))


def compute_diagnostics(position, num_results, param_names=None):
    """Per-parameter bulk-ESS and rank-R-hat on POST-BURN-IN samples only.

    position   : (chains, burnin+results, dim) array (as saved by run_standard_mclmc)
    num_results: number of trailing (results-phase) steps to keep
    param_names: optional list of length dim; defaults to integer indices

    Returns a dict:
      names, ess (per-param), rhat (per-param),
      min_ess, min_ess_param, max_rhat, max_rhat_param,
      num_chains, num_results_used, table (list of per-row dicts)
    """
    position = np.asarray(position)
    if position.ndim != 3:
        raise ValueError(f"position must be (chains, steps, dim); got {position.shape}")
    post = position[:, -num_results:, :]      # results phase only
    n_chains, n_used, dim = post.shape

    ess = np.asarray(_bulk_ess(post), dtype=float)
    rhat = np.asarray(_rank_rhat(post), dtype=float)

    if param_names is None:
        names = [str(i) for i in range(dim)]
    else:
        names = [str(n) for n in param_names]
        if len(names) != dim:
            raise ValueError(f"param_names length {len(names)} != dim {dim}")

    imin = int(np.argmin(ess))
    imax = int(np.argmax(rhat))
    table = [
        {"param": names[i], "ess": float(ess[i]), "rhat": float(rhat[i])}
        for i in range(dim)
    ]
    return {
        "names": names,
        "ess": ess,
        "rhat": rhat,
        "min_ess": float(ess[imin]),
        "min_ess_param": names[imin],
        "max_rhat": float(rhat[imax]),
        "max_rhat_param": names[imax],
        "num_chains": int(n_chains),
        "num_results_used": int(n_used),
        "table": table,
    }


def print_diagnostics(diag, header=""):
    if header:
        print(f"\n=== diagnostics: {header} ===")
    print(f"  chains={diag['num_chains']} results_used={diag['num_results_used']}")
    print(f"  min bulk-ESS = {diag['min_ess']:.3g}  (param {diag['min_ess_param']})")
    print(f"  max rank-Rhat = {diag['max_rhat']:.4f}  (param {diag['max_rhat_param']})")


# ---------------------------------------------------------------------------
# 3. carousel target + qz (exactly as driver_h1h2.py "under" mode)
# ---------------------------------------------------------------------------

def load_target(data_dir):
    """Dispatch a data-dir to its target loader.

    Two data-dir shapes are supported, distinguished by which file is present:
      * a `system.py` exposing `load_target()` (the per-system modules under
        experiments/why_hard_to_sample/systems/) -> call it. Used for the
        pre-registered T0/T1 systems (sys60, vela) whose qz is a saved
        SVI/bootstrap Gaussian, not a MAP-centered ball.
      * otherwise the legacy carousel shape (build_model.py + z_best.npy) ->
        load_carousel_target (unchanged fallback).

    Both return the SAME 5-tuple (model_seq, qz, z_center, dim, param_names).
    """
    if not data_dir:
        raise ValueError("data_dir is required (no default).")
    data_dir = os.path.abspath(data_dir)
    sys_py = os.path.join(data_dir, "system.py")
    if os.path.isfile(sys_py):
        import importlib.util
        mod_name = "_whts_system_" + os.path.basename(data_dir).replace(".", "_")
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        spec = importlib.util.spec_from_file_location(mod_name, sys_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_target()
    return load_carousel_target(data_dir)


def load_carousel_target(data_dir):
    """Rebuild the system's prob_model/model_seq and the MAP-centered "under"
    qz, exactly as driver_h1h2.py.

    data_dir fully defines the system: it must hold build_model.py (exposing
    `model_seq`) AND z_best.npy (and optionally names.npy) for the SAME system,
    so MAP and model cannot be silently mixed across systems. Required; no
    hidden default. For the full carousel, point it at the MAIN checkout
    _h1h2_diag dir (its build_model.py loads data via absolute paths, so
    importing it from anywhere is safe).

    Returns (model_seq, qz, z_best, dim, param_names_or_None).
    """
    if not data_dir:
        raise ValueError("data_dir is required (no default).")
    data_dir = os.path.abspath(data_dir)
    z_best_path = os.path.join(data_dir, "z_best.npy")
    if not os.path.isfile(z_best_path):
        raise FileNotFoundError(f"z_best.npy not found in data_dir: {z_best_path}")
    if not os.path.isfile(os.path.join(data_dir, "build_model.py")):
        raise FileNotFoundError(
            f"build_model.py not found in data_dir: {data_dir} -- data_dir must "
            "contain the system's build_model.py alongside its z_best.npy.")

    # build_model imports gigalens and enables x64 on import.
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    from build_model import model_seq  # noqa: E402

    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    z_best = jnp.asarray(np.load(z_best_path))
    dim = int(z_best.shape[0])
    # driver_h1h2.py "under" mode: tight isotropic ball centered at the MAP.
    qz = tfd.MultivariateNormalDiag(loc=z_best, scale_diag=jnp.full(dim, 1e-2))

    param_names = load_param_names(data_dir, dim)

    return model_seq, qz, z_best, dim, param_names


def load_param_names(data_dir, dim):
    """Load the parameter-name SET from names.npy and return it in sampler
    column order, or None if unavailable/mismatched.

    C-8 FIX (docs/logs/carousel-mclmc-sampling.md): names.npy was written via
    the bijector output dict's iteration order, which for TFP is
    REVERSED-alphabetical (verified empirically for _h1h2_diag/names.npy), so
    its stored order labels column i with the name for column dim-1-i. The
    validated mapping (plotting/diagnostics.py:_z_space_labels "COLUMN ORDER
    NOTE") is: sampler column i = alphabetically i-th unique key. So we use
    names.npy only as the key SET and sort it; this is correct regardless of
    the file's stored order (scene-API flat scalar-key form only).
    """
    if not data_dir:
        return None
    p = os.path.join(data_dir, "names.npy")
    if not os.path.isfile(p):
        return None
    names = sorted(str(n) for n in np.load(p, allow_pickle=True))
    if len(names) != dim or len(set(names)) != len(names):
        return None
    return names
