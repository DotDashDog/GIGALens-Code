"""System-batched point-source log-prob for SBC campaigns (GPU batching, phase A).

Scope
-----
SBC on SIMULATED point-source systems only — never real data, never anything
with an imaging likelihood. The whole approach rests on a property that only a
simulation campaign guarantees: every system shares the model structure, the
free-parameter space, and the prior (the SBC builder enforces sim prior ==
inference prior from one persisted config), so two systems' log-posteriors
differ ONLY through a small pytree of observed-data constants. That is why this
module lives in GIGALens-Code and not in the gigalens package.

Why batching at all: MCLMC is a sequential walker over tiny state (16 chains x
12 params), so a single system cannot occupy a GPU — each step is
kernel-launch-latency bound and a lone system runs no faster (measured: slower)
than on CPU. But launch cost is nearly flat in batch width, so evaluating all
~100 systems' likelihoods in ONE traced computation costs roughly what one
system costs. This module provides that batched evaluation; the batched
samplers (phase B) build on it.

Mechanism
---------
The gigalens ``PointSourceObsLikelihoodTerm`` stores its observed data as
attributes read inside traced methods (``self.x``, ``self.sig_x``, ...,
``self.dataset.trust_radius_arcsec``). We build ONE template ``ProbModel`` and,
inside a ``jax.vmap``-ed function, shallow-copy the term and swap those
attributes for traced per-system rows. Attribute reads happen at trace time, so
the vmapped program evaluates every system's exact solo likelihood — same code
path, same float64 ops — with the system axis batched. Nothing in gigalens is
modified.

The per-system data surface (everything that may differ between systems) is
enumerated EXPLICITLY in ``_ARRAY_FIELDS``/``_SCALAR_FIELDS`` and everything
else that the traced code reads is asserted equal across systems at
construction (M3: raise, never default). If a future gigalens change adds a new
data-derived attribute, the equivalence test in
``simtests/tests/batched_ps_test.py`` is the tripwire.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence

import numpy as np


# Traced per-system data: (owner, attribute) -> per-system array of fixed shape.
# Owner "term" = the likelihood term; "dataset" = its PointSourceObsData.
_ARRAY_FIELDS = {
    "x": "term",              # (n,) observed image x [arcsec]
    "y": "term",              # (n,) observed image y
    "sig_x": "term",          # (n,) astrometric sigma x
    "sig_y": "term",          # (n,) astrometric sigma y
    "cap": "term",            # (n,) honesty saturation cap (10 x rms sigma)
    "inv_flux_obs": "term",   # (n,) observed 1/F        (flux channel)
    "sig_inv_flux": "term",   # (n,) sigma_F / F^2       (flux channel; F-dependent!)
    "td_obs": "term",         # (n,) observed delays, td[0] == 0 (td channel)
    "sig_td": "term",         # (n-1,) delay sigmas      (td channel)
}
_SCALAR_FIELDS = {
    "_log_norm": "term",              # Gaussian normalization; per-system via sig_inv_flux
    "trust_radius_arcsec": "dataset", # trust_region_frac x min image separation
}

# Static attributes the traced code also reads: these must be IDENTICAL across
# systems (they select trace-time branches or enter as Python constants), so a
# mismatch means the systems are not batchable and we must refuse loudly.
_STATIC_DATASET_FIELDS = ("n_images", "newton_steps", "trust_region_frac",
                          "has_flux", "has_td", "src_anchor_sigma", "event_size",
                          # multiplicity constraint: trace-time constants shared
                          # across a wave (all-None = off)
                          "mc_n_obs", "mc_mu_min", "mc_eps", "mc_lam",
                          "mc_grid_n", "mc_n_tiles", "mc_window",
                          "mc_window_scale")
_STATIC_TERM_FIELDS = ("event_size", "src_anchor_sigma", "lens_i", "src_i",
                       "src_j", "absolute", "mc_on")
_STATIC_TERM_TD_FIELDS = ("z_lens", "z_source", "cosmology_mode")


def _get_prob(seq_or_prob: Any) -> Any:
    """Accept a scene ``ProbModel`` (or a legacy wrapper exposing one)."""
    return getattr(seq_or_prob, "prob_model", seq_or_prob)


def extract_system_row(prob: Any) -> Dict[str, Any]:
    """The per-system data row of a solo ``ProbModel``: every traced quantity
    that may differ between SBC systems, as plain numpy (stacked later)."""
    import jax.numpy as jnp  # noqa: F401  (import kept local, matching the builder style)

    term = prob.terms[0]
    row: Dict[str, Any] = {}
    for name, owner in _ARRAY_FIELDS.items():
        src = term if owner == "term" else term.dataset
        if hasattr(src, name):
            row[name] = np.asarray(getattr(src, name), dtype=float)
    for name, owner in _SCALAR_FIELDS.items():
        src = term if owner == "term" else term.dataset
        row[name] = float(getattr(src, name))
    return row


def _static_signature(prob: Any) -> Dict[str, Any]:
    term = prob.terms[0]
    ds = term.dataset
    sig: Dict[str, Any] = {"term_type": type(term).__name__,
                           "n_terms": len(prob.terms)}
    for f in _STATIC_DATASET_FIELDS:
        sig[f"ds.{f}"] = getattr(ds, f)
    for f in _STATIC_TERM_FIELDS:
        sig[f"term.{f}"] = getattr(term, f)
    if ds.has_td:
        for f in _STATIC_TERM_TD_FIELDS:
            sig[f"term.{f}"] = getattr(term, f)
    return sig


class BatchedPointSourceProb:
    """All systems' log-posteriors as one vmapped computation.

    ``log_prob(z)`` / ``log_like(z)`` take ``z`` of shape
    ``(n_systems, *batch, n_params)`` — e.g. ``(S, C, P)`` for C chains per
    system — and return values shaped ``(n_systems, *batch)``, exactly equal
    (same float64 code path) to calling each system's solo ``ProbModel`` on its
    own slice. The prior/bijector are the template's, shared by construction.
    """

    def __init__(self, template_prob: Any, rows: List[Dict[str, Any]]):
        import jax
        import jax.numpy as jnp

        if not rows:
            raise ValueError("BatchedPointSourceProb needs at least one system.")
        keys = set(rows[0])
        for i, r in enumerate(rows):
            if set(r) != keys:
                raise ValueError(
                    f"System {i} has data fields {sorted(set(r))} but system 0 "
                    f"has {sorted(keys)} — mixed channel configurations are not "
                    "batchable (all systems must share fit_flux/fit_td).")
        self.prob = template_prob
        self.n_systems = len(rows)
        self.data = {k: jnp.stack([jnp.asarray(r[k], dtype=float) for r in rows])
                     for k in sorted(keys)}
        self._vlog_prob = jax.vmap(self._one_log_prob, in_axes=(0, 0))
        self._vlog_like = jax.vmap(self._one_log_like, in_axes=(0, 0))

    # ------------------------------------------------------------------ build
    @classmethod
    def from_probs(cls, seqs_or_probs: Sequence[Any]) -> "BatchedPointSourceProb":
        """Batch solo models built by the SBC builder (one per system).

        Refuses (raises, never defaults) if any system differs from the first
        in anything the traced code reads besides the enumerated data rows —
        image multiplicity, channel configuration, solver knobs, anchor scale,
        redshifts/cosmology mode, or the free-parameter list.
        """
        probs = [_get_prob(s) for s in seqs_or_probs]
        if not probs:
            raise ValueError("from_probs got an empty sequence of models.")
        ref_sig = _static_signature(probs[0])
        ref_names = list(probs[0].model.z_param_names)
        for i, p in enumerate(probs[1:], start=1):
            if len(p.terms) != 1:
                raise ValueError(
                    f"System {i} has {len(p.terms)} likelihood terms; batching "
                    "supports exactly the single point-source term.")
            sig = _static_signature(p)
            if sig != ref_sig:
                diff = {k: (ref_sig.get(k), sig.get(k))
                        for k in set(ref_sig) | set(sig)
                        if ref_sig.get(k) != sig.get(k)}
                raise ValueError(
                    f"System {i} is not batchable with system 0 — static "
                    f"likelihood structure differs: {diff}. Batched SBC "
                    "requires identical model structure across systems.")
            names = list(p.model.z_param_names)
            if names != ref_names:
                raise ValueError(
                    f"System {i} free-parameter list {names} != system 0 "
                    f"{ref_names}; the shared-prior requirement is violated.")
        rows = [extract_system_row(p) for p in probs]
        shapes0 = {k: np.shape(v) for k, v in rows[0].items()}
        for i, r in enumerate(rows[1:], start=1):
            bad = {k: (shapes0[k], np.shape(r[k]))
                   for k in r if np.shape(r[k]) != shapes0[k]}
            if bad:
                raise ValueError(
                    f"System {i} data shapes differ from system 0: {bad}.")
        return cls(probs[0], rows)

    @classmethod
    def from_systems(cls, systems: Sequence[Any], **builder_kwargs: Any
                     ) -> "BatchedPointSourceProb":
        """Convenience: run the solo SBC builder per system, then batch."""
        from gigalens_research.simtests.experiments.lenstronomy_point_source import (
            build_epl_shear_point_source_obs,
        )
        return cls.from_probs(
            [build_epl_shear_point_source_obs(s, **builder_kwargs)
             for s in systems])

    # ------------------------------------------------------------------- swap
    def _swapped(self, row: Dict[str, Any]) -> Any:
        """A shallow-copied ProbModel whose term reads this system's (traced)
        data. Called at trace time inside vmap — plain attribute assignment."""
        term = copy.copy(self.prob.terms[0])
        ds = copy.copy(term.dataset)
        term.dataset = ds
        for name, owner in {**_ARRAY_FIELDS, **_SCALAR_FIELDS}.items():
            if name in row:
                setattr(term if owner == "term" else ds, name, row[name])
        p = copy.copy(self.prob)
        p.terms = [term]
        return p

    def _one_log_prob(self, z: Any, row: Dict[str, Any]):
        return self._swapped(row).log_prob(z)

    def _one_log_like(self, z: Any, row: Dict[str, Any]):
        return self._swapped(row).log_like(z)

    # -------------------------------------------------------------------- api
    def log_prob(self, z: Any):
        """``(log_posterior, reduced_chi2)``, each ``(n_systems, *batch)`` for
        ``z`` of shape ``(n_systems, *batch, n_params)``."""
        return self._vlog_prob(z, self.data)

    def log_like(self, z: Any):
        """``(log_like, reduced_chi2)``, same shapes as :meth:`log_prob`."""
        return self._vlog_like(z, self.data)
