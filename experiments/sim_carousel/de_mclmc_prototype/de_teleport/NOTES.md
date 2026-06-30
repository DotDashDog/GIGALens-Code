# de_teleport — periodic gamma=1 teleport + snooker, on a curved testbed

Standalone module wrapping the REAL MCLMC kernel (read-only import, same as
de_mclmc.py). Modifies no shared module. CPU-only. Verdicts are PROPOSED /
UNCERTIFIED for the orchestrator to grade.

## 1. Research — the faithful constructions (sources labeled)

### gamma=1 teleport — VERIFIED (secondary sources)
ter Braak (2006), *A Markov Chain Monte Carlo version of the genetic algorithm
Differential Evolution* (Stat. Comput. 16:239-249). DE-MC proposal
`x' = x_i + gamma*(x_a - x_b) + eps`, with `gamma = 2.38/sqrt(2d)` normally and
**`gamma = 1.0` every ~10th generation** to jump between modes. Ordered pairs
(a,b) drawn uniformly ⇒ proposal is SYMMETRIC ⇒ Metropolis acceptance
`min(1, pi(x')/pi(x_i))`, no Hastings term. The "teleport onto chain a" reading is
exact only in the limit `z_b → z_i`: then `z' = z_i + (z_a - z_b) ≈ z_a`.
[Verified from Wageningen/secondary sources; primary PDF not machine-readable here.]

### snooker — Jacobian VERIFIED from primary source code
ter Braak & Vrugt (2008), *Differential Evolution Markov Chain with snooker updater
and fewer chains* (Stat. Comput. 18:435-446). I could not read the PDF, but I read
the authoritative DREAM-Suite MATLAB source (Vrugt), which is the reference
implementation:

`Calc_proposal.m`:
```
c = R(i,3);                       % projection anchor
F  = X(i,:) - Z(c,:);             % line through current point and anchor
zp = F * ( ((Z(a,:)-Z(b,:)) . F) / (F.F) );   % project (z_a - z_b) onto the line
gamma_s = 1.2 + rand;             % U(1.2, 2.2)
Xp(i,:) = X(i,:) + gamma_s*zp + eps;
log_alfa_sn(i,1) = (DREAMPar.d - 1) * log( XpZ / XZ );   % XpZ=||Xp-Zc||, XZ=||X-Zc||
```
`Metropolis_rule.m`: `alfa = exp(log_alfa_sn) * a_L * a_PR`.
⇒ EXACT acceptance (as implemented here):
```
alpha = min(1,  ( ||z' - z_c|| / ||z_i - z_c|| )^(D-1)  *  pi(z') / pi(z_i) )
```
The `(D-1)` radial Jacobian is mandatory; OMITTING it BIASES the sampler (demo
below). [Jacobian + proposal VERIFIED from DREAM-Suite source, links in report.]

### Goodman & Weare (2010) stretch move — context, VERIFIED form
`Y = X_j + Z*(X_k - X_j)`, `Z ~ g(z) ∝ 1/sqrt(z)` on `[1/a, a]`, acceptance
`min(1, Z^(D-1) * pi(Y)/pi(X_k))`. Same `Z^{D-1}` radial-Jacobian family as snooker.
[Verified via secondary sources; used only as cross-check, not implemented.]

## 2. Files
- `curved_testbed.py` — 2-Gaussian mixture (masses 0.6/0.4) pushed through a smooth
  invertible triangular banana warp `y_k = x_k + b*c_k*(x0^2 - m^2)` (|detJ|=1, exact
  logp). Within-mode = curved thin ridge. `offridge_decomp` splits any point's
  Mahalanobis (pulled back to latent x) into along-ridge / OFF-ridge (on-manifold
  off-ridge ≈ sqrt(D-1) ≈ 3.0). Run: `bash run_cpu.sh curved_testbed.py`.
- `de_teleport.py` — `make_teleport_composite(move=...)`: 'gamma1' (symmetric
  periodic teleport), 'near' (gamma=1 nearest-neighbour near-teleport with the EXACT
  computable Hastings ratio), 'snooker' (verified (D-1) Jacobian; `drop_jacobian`
  flag for the bias demo). Same API as de_mclmc.make_composite.
- `offridge_diag.py` — cheap decisive geometry: GATE-1 b-calibration + off-ridge of
  every move type. `bash run_cpu.sh offridge_diag.py`.
- `curved_gates.py` — GATE-1 (real de_mclmc linear-DE acceptance) + GATE C (move
  comparison) on the curved target. `bash run_cpu.sh curved_gates.py [Rscale]`.
- `validate_easy.py` — GATE A unbiasedness on the easy separated mixture for all 3
  moves + snooker drop-Jacobian bias demo. `bash run_cpu.sh validate_easy.py [Rscale]`.
- `easy_hop.py` — mode-hopping (round-trips) on the benign separated case.
- `tiny_mode_T3.py` — GATE B tiny-mode occupancy (0.03, 0.001).
- `snooker_eps_diag.py` — drill-down resolving the snooker weight question.

## 3. Results vs gates (numbers; on-manifold off-ridge ≈ 3.0)

GATE-1 (faithful, real de_mclmc linear DE, curved target, balanced 16-chain init):
  b=6 → 0.68%, b=9 → 0.55%, b=12 → 0.44% within-step DE acceptance — reproduces the
  carousel's ~0.6% (≈8x below the Gaussian carousel_testbed's 8.7%). Calibrated b*=6.
  ⇒ testbed is faithful (NOT ill-posed).

GATE A unbiasedness (easy separated mixture, V2 invariance-from-truth):
  gamma1 : w@truth 0.717±0.018 (0.70 ✓), moments ±4.99/var~1.0 ✓, KS 0.44/0.66 ✓ → UNBIASED
  near   : w@truth 0.696±0.008 ✓, moments ±5.0/var~0.98 ✓, KS 0.86/0.44 ✓ → UNBIASED
           (the near-teleport Hastings q-ratio is correct.)
  snooker: within-mode unbiased (moments var 0.97-1.03, KS 0.69) ✓; cross-mode weight
           just tracks the (near-frozen) init because snooker barely hops modes
           (easy_hop: 0 round-trips) — a MIXING limitation, not a bias.
  snooker drop-Jacobian DEMO: with Jac w→0.66/0.69 (≈init), DROP Jac w→0.99
           (drains the minor mode) — VISIBLE BIAS ⇒ the (D-1) Jacobian is correct & needed.

Easy-case mode-hopping (populated init, R=1500, 32 chains):
  gamma1 round-trips=115, near round-trips=950, snooker round-trips=0 (acc 27% but
  no mode jumps), vanilla MCLMC round-trips=0 (barrier 12.5 traps it). ⇒ gamma1/near
  hop strongly on benign geometry and ARE necessary; snooker is a within-mode move,
  not a mode-jumper.

OFF-RIDGE diagnostic at carousel-faithful curvature (median off-ridge Mahalanobis;
on-manifold ≈ 3.0):
  true draws 2.9 | gamma1-near(within) 5.9 | near(cross,100% reach target) 17 |
  gamma_big 30 | gamma1-rand(within) 63 | snooker(within) 116 | gamma1-rand(cross) 703.
  Accept falls monotonically with off-ridge. The near-teleport is the LEAST-bad chord
  move yet still ~6x off-manifold: its residual `(z_i - z_b*)` (median 2.7) is a
  within-mode chord that the TARGET mode's differently-curved ridge amplifies off-ridge.

GATE C (curvature, full sampler, b*=6, R=1200, balanced init):
  round-trips: linDE 0, gamma1 0, near 0, snooker 0 (crossings 0-3). NO affine move —
  including gamma=1 teleport, near-teleport, snooker — mode-hops at carousel-faithful
  curvature. None beats the wall (target was ≥10x linDE round-trips).

GATE B / T3 tiny-mode occupancy (separated geometry, seeded 1 chain in minor):
  w_tiny=0.03  : gamma1 0.090±0.011, near 0.083±0.003 (truth 0.03) → PIN (≈1-1.5 chains)
  w_tiny=0.001 : gamma1 0.056±0.002, near 0.064±0.001 (truth 0.001) → PIN (≈1/16)
  ⇒ both OVER-occupy tiny modes (the user's exact worry). Mechanism: a lone chain in
  a tiny mode cannot be given an EXIT difference-vector because the frozen complement
  rarely contains a same-mode partner ⇒ pinned at ~1/16 regardless of true weight.

## 4. VERDICT (PROPOSED / UNCERTIFIED)
Periodic gamma=1 teleport, the gamma=1 near-teleport, and the snooker update are all
UNBIASED (gamma1/near by symmetric / exact-Hastings construction; snooker by the
verified (D-1) Jacobian — dropping it visibly biases). They hop SEPARATED modes well
(gamma1/near) but DO NOT beat the curvature wall: at carousel-faithful curvature every
affine proposal lands 6-700x off the on-manifold off-ridge scale and achieves 0 mode
round-trips (Gate C). The near-teleport — the curvature-robust candidate — is the
least-bad but still fails, by a MEASURED mechanism: its residual chord is amplified
off-ridge by the target mode's curvature. The only fully curvature-robust limit
(residual→0 = land exactly on a chain = the kernel_hop/NKC independence sampler) trades
the curvature wall for the tiny-mode PINNING pathology, which both 'near' and gamma1
also exhibit here (Gate B/T3 fail). No simpler variant avoids both walls.
