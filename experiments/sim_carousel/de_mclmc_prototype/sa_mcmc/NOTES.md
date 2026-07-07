# SA-MCMC mode-hop move — working notes (for the orchestrator; canonical log is theirs)

## 1. The algorithm (VERIFIED from the readable NeurIPS PDF)

Source (machine-readable, FlateDecode-decoded locally):
https://papers.nips.cc/paper/2019/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf
Michael H. Zhu, "Sample Adaptive MCMC", NeurIPS 2019. Algorithm 1, Section 3,
Proposition 1, Theorem 1 all read verbatim from the decoded text (see sa_move.py
docstring for the transcribed Algorithm 1 and the symbol decoding).

State = N points S=(θ1..θN). Target over the ensemble = product πᴺ = Ππ(θn).
μ(S)=sample mean, Σ(S)=sample covariance of the N points; proposal q(·|μ(S),Σ(S))
is Gaussian or a Gaussian scale-mixture. One iteration:
 1. draw θ_{N+1} ~ q(·|μ(S),Σ(S))
 2. T = (θ1..θN, θ_{N+1});  S_{-n} = T with θn removed (all n=1..N+1; S_{-(N+1)}=S)
 3. λn = q(θn | μ(S_{-n}),Σ(S_{-n})) / p(θn)              ← self-inclusive weights
 4. pick j with P[J=n] = λn / Σλ ;  next state = S_{-j}
    (j≤N → substitute proposal into slot j; j=N+1 → reject)
Only ONE new target eval/iter (p(θ_{N+1})); the λ's use cheap proposal densities.

**Proposition 1 (VERIFIED statement): the chain satisfies detailed balance w.r.t.
πᴺ.**  Theorem 1 (VERIFIED): ergodic for a DIAGONAL-covariance proposal under the
same assumptions as Metropolis–Hastings.

One-line balance (own-knowledge reconstruction, reproduces Prop 1 exactly): a move
S→S′ replacing θa by y has augmented set T=S∪{y}=S′∪{θa} identical both ways; with
Z(T)=Σ_{t∈T} q(t|T∖t)/p(t) symmetric in T and π(θ)/p(θ)=1/C,
 πᴺ(S)P(S→S′) = (1/C) Π_{n≠a}π(θn)·q(y|S)·q(θa|S′)/Z(T)
which is symmetric under S↔S′. **Crucial: the proof uses ONLY that q(·|S) depends
on the unordered SET S — so ANY set-symmetric proposal is admissible, including a
self-inclusive KDE/mixture.**  This is what licenses our curvature-aware variant.

Self-inclusive ⇒ no "valley at your own location" (the kernel-hop self-EXCLUSION
failure): θn's keep/delete weight is evaluated on sets that INCLUDE the proposal
and the other points, and the deletion mechanism (not a plain Hastings ratio)
carries reversibility.

## 2. Two proposal families implemented (sa_move.py)
- "gaussian": single fitted N(μ(S), prop_scale²Σ(S)) — the literal Zhu variant.
- "mixture": self-inclusive KDE q(·|S)=(1/N)Σ N(·;θm, bw²·kernel_cov) — curvature-aware.

## 3. Key MEASURED findings
- Both variants are UNBIASED in isolation. Pure SA (K=0) from a truth init holds
  the weight exactly (mixture bw=0.05 frozen at 0.594; bw=0.20 fluctuates about truth).
- DISCOVERY asymmetry (analytic, modes ±5, gap empty): single GAUSSIAN cold-discovers
  the empty far mode (V1 weight→0.70) via global-covariance EXPANSION feedback; the
  MIXTURE does NOT (local width-1 kernels cannot bridge a 10-unit empty gap). MCLMC
  randomness is identical between the two (same kmc split) and vanilla MCLMC is fully
  trapped (1.0), so the SA move itself does the bridging in the gaussian case.
  → gaussian passes V1; mixture FAILS V1 (cold discovery) but PASSES all invariance
    gates V2/V3/V4. This matches the method's stated purpose (equilibration across
    POPULATED modes), and predicts gaussian is over-dispersed on curved ridges.
- CURVED ridge: the only on-manifold-preserving blur is near-delta (a straight
  Gaussian step either moves along the curved tangent→off-ridge, or in perp dims
  >s_thin→off-ridge). So curvature-robust hop = propose ESSENTIALLY AT an existing
  other-mode on-ridge point (tight bw). Self-inclusion removes the reversibility
  penalty that killed kernel-hop at tight bw; the remaining cost is tiny-mode pinning.
- STEP-SIZE artifact: MCLMC at step 0.5 under-resolves the extreme curvature
  (mean logp −43 vs −5 exact = off-ridge); step 0.2 resolves it (logp −0.7). Use 0.2
  on the curved testbed. The tight-bw composite "drift" was this artifact, not SA bias.

## 4. Files
- sa_move.py            — the SA-MCMC mover + composite (gaussian & mixture). Imports
                          real MCLMC primitives read-only. Swap-in API.
- curved_testbed.py     — curved bimodal target (banana warp on n_curve dims), exact
                          logp (unit Jacobian), exact draws, classifier,
                          static_linear_de_acceptance (GATE-1), within_mode_cov.
- validate_sa_analytic.py <gaussian|mixture>  — Gate A (V1..V4) on easy analytic.
- validate_curved.py    — Gate C (+GATE-1) on the curved testbed.
- tiny_mode_test.py <w_minor> <prop> <scale>  — Gate B (T3) tiny-mode occupancy.
- gate1_sweep.py        — curvature calibration sweep.
Run env: shifter jax image, PYTHONPATH per brief, JAX_PLATFORMS=cpu, x64.
