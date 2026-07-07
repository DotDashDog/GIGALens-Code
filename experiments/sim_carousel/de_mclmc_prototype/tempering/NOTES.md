# Tempering for MCLMC discovery + tiny-mode avoidance — working notes

Standalone wrapper around the REAL MCLMC kernel (imports
`_build_kernel_shardmap`, `isokinetic_mclachlan_smart`, `_single_init`
read-only; shared modules untouched). CPU only. PROPOSED / UNCERTIFIED — for
orchestrator audit.

## Research (labeled)
- **VERIFIED (MCLMC paper arXiv:2303.18221v3):** dynamics `dx=u dt`,
  `du=P(u)f(x)dt+ηP(u)dW`, `f=-∇S/(d-1)`, `S=-log p`, `|u|=1`; configuration
  stationary `ρ∝e^{-S}=p(x)`.
- **OWN-KNOWLEDGE (algebraic consequence):** tempering `log p → β log p`
  (`S→βS`) scales the force by β and makes stationary `∝ p^β` — the exact
  tempered target. Subtlety: the EEVPD-tuned step is target-dependent (force∝β);
  reusing the β=1 step at β<1 is CONSERVATIVE at *equilibrium* (hotter→less
  energy error) BUT transient off-ridge excursions during barrier crossing can
  still blow up EEVPD at a near-stability-knee step — must check the *anneal* max
  EEVPD, not just equilibrium (see Gate D).
- **VERIFIED (search):** PT swap accept `min(1,exp((β_i−β_j)(E_i−E_j)))`, E=−logp.
  AIS (Neal 2001) `π_k∝p^{β_k}q^{1−β_k}`, weights `∏π_k(x_k)/π_{k−1}(x_k)`.

## Mechanisms implemented
1. `tempered_mclmc.py` — TEMPERED BURN-IN (simplest-first): anneal β small→1, K
   real MCLMC steps/stage, momentum refreshed at stage boundaries, DISCARD hot
   samples, sample cold at β=1 (NO importance weights needed). + EEVPD step tuner.
2. `parallel_tempering.py` — replica-exchange PT (escalation): R replicas, K
   MCLMC steps/level + even/odd adjacent swaps; cold replica kept.

## Results vs gates (numbers; thresholds derived)
- **EEVPD steps:** easy mixture benign-Gaussian → energy error never binds in
  step≤1 (EEVPD(0.5)=1.5e-9 ≪ 5e-4); use step 0.5 (validated config).
  CURVED target → EEVPD BINDS: β=1 interp-tuned step 0.19 sits at the stability
  knee (anneal EEVPD 5.8e-2, 127× over) → use a FAITHFUL step 0.05 (anneal
  EEVPD ~1e-7).

- **Gate A discovery (easy, wrong-basin init):** tempered burn-in reaches the
  dominant basin (occ_+ 0.0→0.51); vanilla stays 0.0. DISCOVERY ✓. BUT one-shot
  WEIGHT freezes out: occ_+≈0.51 vs 0.70; drill (drill_schedule.py) shows a soft
  freeze-out floor that recedes toward 0.70 only with dense-near-1 ladder + huge
  effort (0.672 at 72k anneal steps, still 4.4 SE short). → escalate to PT.
- **Gate A weight (PT, pt_weight.py):** cold occ_+ = **0.6986 ± 0.0122**
  (|err|=0.0014 ≪ 3SE=0.037), no drift, within-mode moments exact, swap accept
  0.27–0.28 all rungs, EEVPD ≤2.8e-9. PASS.
- **Gate C unbiased cold (easy, truth init):** within-mode mean ±5 var ~1, axis1
  var 1.0, KS axis0 p=0.082 / axis1 p=0.70, zero drift (frozen). PASS.
- **Gate D curved barrier:** GATE-1 within-mode linear-DE acc 1.85% (carousel-
  faithful regime). At FAITHFUL step 0.05 discovery is BUDGET-limited:
  occ_A 0.19→0.41→0.58→0.53 over steps/stage 300→6000 (saturates ~0.55 ≈ freeze-
  out weight, truth 0.6); vanilla 0; affine DE 0 round-trips (C-14). Genuine
  faithful curved crossing (the coarse-step occ_A=0.516 was partly NUMERICAL
  HEATING — flagged). Cost: ~9× steps/stage vs easy (the EEVPD-fine-step price).
- **Gate B tiny drain:** one-shot tempered burn-in is freeze-out/seed dependent
  (drill_tiny.py: w=1e-3 → occ 0.06–0.31, mean 0.167, quantized at k/16; even a
  heroic dense-decouple cool only reaches 0.031). A frozen integer-chain ensemble
  CANNOT represent 1e-3. PT (mixing cold replica) DRAINS correctly (pt_drain.py):
  w=1e-3 → occ_minor **0.00125 ± 0.00017** (truth 1e-3); w=1e-5 → **0.0** (truth
  1e-5); both ≪ pin 1/24=0.0417; swap accept 0.27–0.28. PASS — the key
  differentiator from every empirical-ensemble hop (C-16 pin at 1/n).
  [PT result filled below]

## Verdict (draft)
- DISCOVERY across straight AND curved barriers from a wrong-basin start:
  tempered burn-in delivers (vanilla 0; every ensemble hop 0, C-16). Faithful
  step mandatory on curved targets (coarse step fakes discovery via heating).
- TINY-MODE DRAIN and exact comparable-mass WEIGHT: one-shot freeze-out fails;
  PARALLEL TEMPERING delivers (continuously-mixing cold replica), where every
  empirical-ensemble hop PINS at 1/n (C-16). Cost = R× replicas at sampling.
