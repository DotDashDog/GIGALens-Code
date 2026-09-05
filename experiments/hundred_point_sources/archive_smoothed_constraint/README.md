# Archive — smoothed multiplicity constraint (P-8 … P-11)

These files drove the *smoothed* image-count constraint: the differentiable
proxy `N_eff = N ⊛ g_eps` added to the point-source likelihood as
`-lam (N_eff - n_obs)^2` (gigalens `PointSourcePositionData(multiplicity_constraint=…)`,
plus the eps-annealed batched MAP phase `batched_pipeline.batched_map_anneal`).

**The approach was REFUTED** by P-11 on 2026-08-04 (see
`docs/logs/point-source-sbc.md`, entries 2026-07-31 and 2026-08-04): the proxy
is the integer count blurred over the source plane, so a truth within ~eps of a
caustic reads a fractional count and is penalized at *any* resolution
(measured: 26/100 systems at eps = 0.05"), while the quadrature converges more
slowly as eps shrinks — the refinement direction is closed off on principle,
not on tuning.

**Retired 2026-09-04** (P-12, a design decision): the smoothed term is replaced
by the discrete, discontinuous
`gigalens.jax.point_source_multiplicity.PointSourceMultiplicityData`, which is
exact at every admissible truth and is screened by MAP's argmax and sampled by
MAMS.

These files are kept as **history only and are NOT runnable**: the
`multiplicity_constraint=` kwarg, the `mc_*` dataset attributes and
`batched_pipeline.batched_map_anneal` no longer exist in gigalens or in this
repo. The successor configs are `../campaign_v3_quad.yaml` and
`../campaign_v3_double.yaml`.

| file | what it was |
|---|---|
| `campaign_v2_double_mc.yaml` | P-10 double-arm rerun with the coarse constraint (eps 0.1", grid 384), sweep `{mc: 1}` |
| `campaign_v2_double_mc_fine.yaml` | P-11 fine-operator remedy (eps 0.05", grid 768), sweep `{mc: 2}` |
| `p11_remedy.sbatch` | the P-11 Lawrencium job (4× 2080 Ti) |
| `p10_analysis.py` | P-10 analysis: basin occupancy, SBC re-rank, penalty-at-truth audit |
| `p11_analysis.py` | P-11 analysis: fine-rung re-rank, unpenalized loglik PIT, window-12 truth recount |
