"""Shared, jax-FREE config + numpy helpers for T23/T24 (both registered
2026-07-03 in docs/logs/why-hard-to-sample.md).

This module is deliberately import-safe under the login python (numpy only): it
carries the SINGLE authoritative copy of

  * the xi <-> displacement alignment convention (traced from source, cited),
  * the preconditioning-metric meaning (traced from source, cited),
  * the arm/seed/clone data-path table + recoverable z_init locators,
  * the deterministic spike/calm selection machinery,
  * the bounded-parameter derivation (from the prior spec in source).

Both GPU stages (t23_momentum_gpu.py, t24_census_gpu.py) AND the login-node
analyzer (t23_t24_analyze.py) import it, so the alignment/metric/selection
conventions CANNOT drift between the compute and the read-back stages. (This
5th file is a small, justified deviation from the 4-file deliverable list: the
alternative -- three independent copies of the alignment trace -- is exactly the
correctness hazard the pre-registration flags.)

--------------------------------------------------------------------------------
TRACE 1 -- xi <-> displacement alignment (resolved in SOURCE, not fitted)
--------------------------------------------------------------------------------
Kernel: gigalens.jax.experimental.mclmc.MCLMC_JIT -> full_mclmc_with_adapt_sharded
        (src/gigalens_research/inference/mclmc.py).

In `step_batched` (mclmc.py:281) the scan carry holds `states` = the state BEFORE
this transition. The kernel is applied as
    new_state, info = kernel_fn(state=prev_state, ...)         # mclmc.py:298
and the diagnostic history row for this scan index i is written as
    h.position = new_states.position                          # mclmc.py:448
    h.xi       = xis                                          # mclmc.py:453
where `xis` is computed from `info.energy_change` of THIS transition
(skip_adapt: `xi_val = energy_change**2/(dim*var)+1e-8`, mclmc.py:334; the
adapt path uses the same `info.energy_change`, mclmc.py:204). `info.energy_change`
is the energy difference of the transition prev_state -> new_state produced by
the blackjax isokinetic integrator (blackjax/mcmc/mclmc.py MCLMCInfo.energy_change
= "difference in energy between the current and previous step"; integrators.py
with_isokinetic_maruyama computes it across the step that yields new_state).

Because history row i stores new_state.position AND the energy_change of the
transition that PRODUCED it, and history row i-1 stored the state that was fed in
as prev_state at step i, we have:

    xi[t] belongs to the transition INTO position[t], i.e. z_{t-1} -> z_t.
    => the displacement that generated xi[t] is  Dz_t := position[t] - position[t-1].

This is the convention used EVERYWHERE below (spike step t is paired with the
INCOMING displacement Dz_t = z_t - z_{t-1}); the Hessian is centered at the
endpoint z_t (the recorded position at which xi[t] is logged). The pre-reg's
generic "momentum proxy u_t ~ z_{t+1}-z_t" is the OUTGOING direction; we do not
use it to pair xi -- the source pairing above governs.

--------------------------------------------------------------------------------
TRACE 2 -- meaning of the saved `inverse_mass_matrix` (the preconditioner)
--------------------------------------------------------------------------------
Despite its name, the saved array is the COVARIANCE-LIKE preconditioner Sigma,
NOT Sigma^{-1}:
  * it is initialised to  qz.covariance()  (mclmc.py:67),
  * during windowed adaptation it is set to the sample COVARIANCE
    `sample_cov = welford_cov(...)` (mclmc.py:392-394),
  * the kernel Cholesky-factors it and pushes gradients/positions THROUGH the
    factor: `chol = cholesky(inverse_mass_matrix)` and
    `flatten_grads = chol.T @ flatten_grads`, `gr = chol @ momentum`
    (blackjax_updated_utils.py:416,436,448) -- i.e. it acts as the covariance
    metric Sigma (position increments live in the Sigma metric).

Therefore the registered curvature-along-motion uses Sigma^{-1} in the
denominator (squared preconditioned step length):
    c_t      = Dz' (-H) Dz / (Dz' Sigma^{-1} Dz)      [primary]
    c_t_eucl = Dz' (-H) Dz / ||Dz||^2                  [secondary]
with Sigma = inverse_mass_matrix[0, t] (frozen over the results phase: verified
max|Sigma_t - Sigma_2000| == 0 for both arms/all seeds; asserted at run time).
"""
from __future__ import annotations

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# registered constants (NEVER tune)
# ---------------------------------------------------------------------------
SEED = 20260703                 # numpy RandomState seed for all subsampling
XI_SPIKE_THRESH = 10.0          # registered tail threshold
XI_TOP_FRAC = 0.001             # "top 0.1%" of xi -> always included in spike set
N_SPIKE_MAX = 512               # per-arm cap on the subsampled spike set
N_CALM_MAX = 512                # per-arm cap on the calm control set
SEEDS = (1, 2, 3)               # real-run seeds present in t21
CLONE_SEED = 1                  # clone control seed present in t21

NB = 2000                       # burnin steps (t21)
NR = 2000                       # results steps (t21)
RESULTS_LO = NB                 # first results-phase GLOBAL step index (2000)
RESULTS_HI = NB + NR            # one past last (4000)

# T24 census segment: fixed contiguous 256-step results-phase window, SAME for
# all chains/arms. Choice: start 500 steps into the results phase (global 2500)
# so it is clear of the burnin->results boundary transient; length 256.
CENSUS_START = 2500
CENSUS_LEN = 256                # -> global steps 2500..2755 inclusive
CENSUS_N_TOP = 12               # power-iteration at top-12 pooled census spikes
CENSUS_N_CALM = 12              # + 12 matched same-segment median-c calm points
POWER_ITERS = 12                # fixed power-iteration count for lambda1(-H)

HVP_CHUNK_DEFAULT = 8           # HVPs cost ~2-3x logp+grad; logp chunk is <=16

# T24 registered stats thresholds
CENSUS_SPIKE_FACTOR = 3.0       # census spike := c_t > 3x segment median

# output dirs (HARDCODED -- never passed on the CLI; see run_t21.sh "0" bug note)
T23_OUT = os.path.join(_HERE, "results_carousel", "phaseC", "t23")
T24_OUT = os.path.join(_HERE, "results_carousel", "phaseC", "t24")
T21_DIR = os.path.join(_HERE, "results_carousel", "phaseC", "t21")

# short strings recorded verbatim in every output json
XI_ALIGNMENT = (
    "xi[t] belongs to the transition INTO position[t] (z_{t-1} -> z_t); the "
    "displacement generating xi[t] is Dz_t = position[t]-position[t-1]; Hessian "
    "centered at endpoint z_t. Traced from mclmc.py:281,298,448,453,334,204 and "
    "blackjax MCLMCInfo.energy_change; NOT fitted."
)
METRIC_NOTE = (
    "saved inverse_mass_matrix is the covariance-like preconditioner Sigma (NOT "
    "its inverse): init=qz.covariance() (mclmc.py:67), adapt=sample_cov "
    "(mclmc.py:392-394), kernel Cholesky-factors it as the metric "
    "(blackjax_updated_utils.py:416,436,448). c_t denominator uses Sigma^{-1}; "
    "Sigma frozen over results phase (asserted)."
)

# ---------------------------------------------------------------------------
# arm data-path table
# ---------------------------------------------------------------------------
_SIM = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests"

ARMS = {
    "old": {
        "system_dir": os.path.join(_HERE, "systems", "carousel_min_old"),
        "ref_samples": os.path.join(_SIM, "minimal_case_oldbij", "mclmc", "arrays.npz"),
        "z_init_ct": (5, 1371),      # samples_z[chain,t] used as typical-set init (t21 ledger)
        "z_init_logp": -119509.12633386465,
        "clone_target_npz": os.path.join(_HERE, "results_carousel", "old", "t1", "clone.npz"),
    },
    "new": {
        "system_dir": os.path.join(_HERE, "systems", "carousel_min_new"),
        "ref_samples": os.path.join(_SIM, "minimal_case_newbij", "mclmc", "arrays.npz"),
        "z_init_ct": (6, 6147),
        "z_init_logp": -119526.5363246717,
        "clone_target_npz": os.path.join(_HERE, "results_carousel", "new", "t1", "clone.npz"),
    },
}


def seed_npz(arm, seed):
    return os.path.join(T21_DIR, arm, f"t0_seed{seed}.npz")


def clone_run_npz(arm):
    return os.path.join(T21_DIR, arm, "t1_clone_typical_seed1.npz")


# ---------------------------------------------------------------------------
# bounded-parameter derivation (from the prior spec in source)
# ---------------------------------------------------------------------------
# Scalar token -> is it a HARD-BOUNDED (Uniform / TruncatedNormal) param?
# Derived from:
#   systems/carousel_min_common.py:116-117  gamma1,gamma2 ~ TruncatedNormal(0,0.1,-0.3,0.3)
#   systems/carousel_min_common.py:122-129  src center_x/center_y ~ Normal ; beta ~ LogNormal
#   systems/carousel_min_old/system.py:52-58 Rs~U(20,100), alpha_Rs~U(10,40),
#                                            e1,e2~TruncatedNormal(0,0.05,-0.2,0.2),
#                                            center_x/center_y ~ Normal
#   systems/carousel_min_new/system.py:52-58 Rs~U(20,100), theta_E~Normal(13,1),
#                                            e1,e2~TruncatedNormal, center_x/center_y ~ Normal
# LogNormal (beta) is bounded below by 0 but is NOT a hard prior wall in the
# unconstrained (log) coordinate, so it is NOT counted (matches the pre-reg
# "every Uniform and TruncatedNormal param is hard-bounded").
_BOUNDED_TOKEN = {
    "Rs": True,        # Uniform
    "alpha_Rs": True,  # Uniform (old only)
    "theta_E": False,  # Normal  (new only) -- UNBOUNDED
    "e1": True,        # TruncatedNormal
    "e2": True,        # TruncatedNormal
    "gamma1": True,    # TruncatedNormal
    "gamma2": True,    # TruncatedNormal
    "center_x": False, # Normal
    "center_y": False, # Normal
    "beta": False,     # LogNormal -- NOT hard-bounded
}
# expected number of hard-bounded columns per arm (sanity assertion)
EXPECTED_N_BOUNDED = {"old": 6, "new": 5}


def _token_of(name):
    """Longest scalar token that is a suffix of the (namespaced) param key.
    Longest-suffix wins so 'alpha_Rs' is not mis-tokenised as 'Rs'. Delimiter
    agnostic (handles '/', '.', '_' separators)."""
    cand = [t for t in _BOUNDED_TOKEN if name == t or name.endswith(t)]
    if not cand:
        return None
    return max(cand, key=len)


def derive_bounded(param_names, arm):
    """Return dict with the derived hard-bounded columns for `param_names`
    (authoritative sampler-column order). Asserts the derived count matches the
    source-expected count for the arm. Prints the derived list for review."""
    tokens = [_token_of(n) for n in param_names]
    unknown = [param_names[i] for i, t in enumerate(tokens) if t is None]
    if unknown:
        raise ValueError(f"[t23/24:{arm}] param keys with no known scalar token "
                         f"(update _BOUNDED_TOKEN from source): {unknown}")
    bounded_idx = [i for i, t in enumerate(tokens) if _BOUNDED_TOKEN[t]]
    bounded_names = [param_names[i] for i in bounded_idx]
    bounded_tokens = [tokens[i] for i in bounded_idx]
    exp = EXPECTED_N_BOUNDED.get(arm)
    if exp is not None and len(bounded_idx) != exp:
        raise AssertionError(
            f"[t23/24:{arm}] derived {len(bounded_idx)} bounded params "
            f"{bounded_names} != source-expected {exp}. Re-check priors.")
    # locate the registered wall coordinate(s)
    rs_index = next((i for i, t in enumerate(tokens) if t == "Rs"), None)
    alpha_rs_index = next((i for i, t in enumerate(tokens) if t == "alpha_Rs"), None)
    print(f"[t23/24:{arm}] param_names (sampler-column order): {list(param_names)}")
    print(f"[t23/24:{arm}] derived HARD-BOUNDED columns ({len(bounded_idx)}): "
          f"{list(zip(bounded_idx, bounded_names))}")
    print(f"[t23/24:{arm}] z_Rs column index = {rs_index}; "
          f"alpha_Rs column index = {alpha_rs_index}")
    return {
        "param_names": list(param_names),
        "tokens": tokens,
        "bounded_idx": bounded_idx,
        "bounded_names": bounded_names,
        "bounded_tokens": bounded_tokens,
        "rs_index": rs_index,
        "alpha_rs_index": alpha_rs_index,
    }


# ---------------------------------------------------------------------------
# run loading + results-phase slicing (numpy only)
# ---------------------------------------------------------------------------

def load_run(npz_path):
    """Load a t21 run npz. Returns dict with position (C,4000,14), xi (C,4000),
    step_size (C,4000), inverse_mass_matrix (1,4000,14,14)."""
    d = np.load(npz_path)
    return {
        "position": np.asarray(d["position"], dtype=np.float64),
        "xi": np.asarray(d["xi"], dtype=np.float64),
        "step_size": np.asarray(d["step_size"], dtype=np.float64),
        "inverse_mass_matrix": np.asarray(d["inverse_mass_matrix"], dtype=np.float64),
        "nb": int(d["nb"]), "nr": int(d["nr"]), "seed": int(d["seed"]),
    }


def frozen_sigma(run, arm="?", seed="?"):
    """Return the (14,14) covariance-like preconditioner Sigma, asserting it is
    frozen over the results phase (max abs diff over steps RESULTS_LO..RESULTS_HI-1)."""
    imm = run["inverse_mass_matrix"][0]           # (4000,14,14)
    res = imm[RESULTS_LO:RESULTS_HI]              # (2000,14,14)
    dmax = float(np.max(np.abs(res - res[0])))
    if dmax > 1e-10:
        raise AssertionError(
            f"[{arm}/seed{seed}] Sigma NOT frozen over results phase: "
            f"max abs diff {dmax:.3e} (expected 0).")
    return res[0], dmax


def sigma_inv(Sigma):
    """Sigma^{-1} via Cholesky (Sigma is SPD covariance-like)."""
    L = np.linalg.cholesky(Sigma)
    Linv = np.linalg.solve(L, np.eye(Sigma.shape[0]))
    return Linv.T @ Linv


# ---------------------------------------------------------------------------
# displacement + geometry (numpy only) -- uses the traced alignment
# ---------------------------------------------------------------------------

def incoming_disp(position, t):
    """Dz_t = position[:, t] - position[:, t-1]  (the displacement generating
    xi[t]). position is (C,4000,14); returns (C,14)."""
    return position[:, t, :] - position[:, t - 1, :]


def turn_cos(a, b):
    """cosine of the turn angle between two displacement vectors a,b (per row).
    Returns nan where either has ~zero norm."""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = na * nb
    out = np.full(na.shape, np.nan)
    good = denom > 0
    out[good] = np.sum(a[good] * b[good], axis=-1) / denom[good]
    return out


# ---------------------------------------------------------------------------
# deterministic spike/calm selection (numpy RandomState)
# ---------------------------------------------------------------------------

def _proportional_alloc(counts, n_target):
    """Largest-remainder proportional allocation of n_target across strata."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= n_target:
        return counts.copy()
    exact = n_target * counts / total
    alloc = np.floor(exact).astype(np.int64)
    rem = int(n_target - alloc.sum())
    if rem > 0:
        order = np.argsort(-(exact - alloc))
        for j in order[:rem]:
            alloc[j] += 1
    return np.minimum(alloc, counts)


def select_spike_calm(runs, arm, n_spike=N_SPIKE_MAX, n_calm=N_CALM_MAX):
    """Deterministic spike + matched calm selection pooled across the given
    runs (list of (seed, run-dict)).

    Spike set: results-phase steps with xi > XI_SPIKE_THRESH, subsampled to
    <= n_spike stratified proportionally across (seed,chain); PLUS all steps in
    the top XI_TOP_FRAC of the pooled results xi (always included).

    Calm set: results-phase steps with xi < that chain's per-(seed,chain)
    results-median, matched per (seed,chain) to the spike allocation, <= n_calm.

    Returns two structured dicts with fields seed,chain,step,xi (1-D arrays),
    plus bookkeeping in a meta dict.
    """
    rng = np.random.RandomState(SEED)

    # --- gather all results-phase candidates in a FIXED order ----------------
    seed_arr, chain_arr, step_arr, xi_arr = [], [], [], []
    per_chain_median = {}
    for seed, run in runs:
        xi = run["xi"]                       # (C,4000)
        C = xi.shape[0]
        for c in range(C):
            xr = xi[c, RESULTS_LO:RESULTS_HI]
            per_chain_median[(seed, c)] = float(np.median(xr))
            steps = np.arange(RESULTS_LO, RESULTS_HI)
            seed_arr.append(np.full(len(steps), seed))
            chain_arr.append(np.full(len(steps), c))
            step_arr.append(steps)
            xi_arr.append(xr)
    seed_arr = np.concatenate(seed_arr)
    chain_arr = np.concatenate(chain_arr)
    step_arr = np.concatenate(step_arr)
    xi_arr = np.concatenate(xi_arr)

    strat = seed_arr.astype(np.int64) * 100 + chain_arr.astype(np.int64)

    # --- spike ---------------------------------------------------------------
    top_thr = float(np.quantile(xi_arr, 1.0 - XI_TOP_FRAC))
    is_spike = xi_arr > XI_SPIKE_THRESH
    is_top = xi_arr >= top_thr
    always = np.where(is_spike & is_top)[0]
    pool = np.where(is_spike & ~is_top)[0]

    n_from_pool = max(0, int(n_spike) - len(always))
    pool_strat = strat[pool]
    uniq = np.unique(pool_strat)
    counts = np.array([(pool_strat == u).sum() for u in uniq])
    alloc = _proportional_alloc(counts, n_from_pool)
    spike_alloc = {}       # (seed,chain) -> n selected (for calm matching)
    sampled = []
    for u, a in zip(uniq, alloc):
        idx = pool[pool_strat == u]
        a = int(min(a, len(idx)))
        chosen = idx if a >= len(idx) else np.sort(rng.choice(idx, size=a, replace=False))
        sampled.append(chosen)
        spike_alloc[(int(u // 100), int(u % 100))] = a
    # count the always-included top spikes into the per-stratum tally too, so
    # calm matching mirrors the FULL spike chain-composition
    for u in np.unique(strat[always]):
        s, c = int(u // 100), int(u % 100)
        spike_alloc[(s, c)] = spike_alloc.get((s, c), 0) + int((strat[always] == u).sum())
    spike_idx = np.sort(np.concatenate([always] + sampled)) if sampled else np.sort(always)

    # --- calm (matched per (seed,chain) to the spike allocation) -------------
    calm_sel = []
    for (s, c), n_take in spike_alloc.items():
        n_take = min(int(n_take), int(n_calm))
        med = per_chain_median[(s, c)]
        cand = np.where((seed_arr == s) & (chain_arr == c) & (xi_arr < med))[0]
        if len(cand) == 0 or n_take == 0:
            continue
        a = int(min(n_take, len(cand)))
        chosen = cand if a >= len(cand) else np.sort(rng.choice(cand, size=a, replace=False))
        calm_sel.append(chosen)
    calm_idx = np.sort(np.concatenate(calm_sel)) if calm_sel else np.array([], dtype=int)
    if len(calm_idx) > n_calm:                    # global cap (matched alloc can exceed)
        calm_idx = np.sort(rng.choice(calm_idx, size=int(n_calm), replace=False))

    def _pack(idx):
        return {
            "seed": seed_arr[idx].astype(np.int64),
            "chain": chain_arr[idx].astype(np.int64),
            "step": step_arr[idx].astype(np.int64),
            "xi": xi_arr[idx].astype(np.float64),
        }

    meta = {
        "xi_top_thresh": top_thr,
        "n_always_top": int(len(always)),
        "n_spike": int(len(spike_idx)),
        "n_calm": int(len(calm_idx)),
        "spike_alloc": {f"{s}_{c}": int(n) for (s, c), n in sorted(spike_alloc.items())},
        "per_chain_median": {f"{s}_{c}": v for (s, c), v in sorted(per_chain_median.items())},
    }
    return _pack(spike_idx), _pack(calm_idx), meta


# ---------------------------------------------------------------------------
# clone target precision (analytic, constant Hessian) -- numpy
# ---------------------------------------------------------------------------

def clone_precision(arm):
    """Return (P, cholP_lower) for the clone target: the clone samples a Gaussian
    N(mean, cov) in z-space, so its target log-density Hessian is CONSTANT,
    H = -cov^{-1}, hence -H = cov^{-1} = P (the precision). No HVPs needed.
    Uses clone.npz['cholesky'] (cov = L L^T) so quad = ||L^{-1} Dz||^2."""
    d = np.load(ARMS[arm]["clone_target_npz"])
    L = np.asarray(d["cholesky"], dtype=np.float64)     # lower, cov = L L^T
    Linv = np.linalg.solve(L, np.eye(L.shape[0]))
    P = Linv.T @ Linv
    return P, L


def quad(M, v):
    """v' M v for a single vector v or row-wise for v of shape (N,d)."""
    v = np.asarray(v)
    if v.ndim == 1:
        return float(v @ M @ v)
    Mv = v @ M.T
    return np.einsum("ij,ij->i", v, Mv)
