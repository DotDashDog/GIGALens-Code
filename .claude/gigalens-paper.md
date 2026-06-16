# GIGA-Lens — Gu et al. arXiv:2202.07663 (v1, Feb 2022)

**Citations:** [arXiv:2202.07663](https://arxiv.org/abs/2202.07663); journal [ApJ 934:96 (2022)](https://doi.org/10.3847/1538-4357/ac6de4).

**Local PDF in this repo:** `GIGALens-Code/2202.07663v1.pdf` (arXiv v1). Use it for equations, figure numbers, and any claim not reproduced below.

**Name:** *Gradient-Informed, GPU-Accelerated* lens modeling — implemented in **TensorFlow** and **JAX**, with **TensorFlow Probability** for VI/HMC (and autotuning). The paper notes that **multi-GPU distribution** was only supported in the **JAX** pipeline at submission time (TensorFlow lacked general distributed computing outside neural networks).

---

## 1. Motivation (Introduction)

- **Science:** Strong lenses probe dark matter (e.g. substructure), and time delays / lensed SNe feed $H_0$ and cosmology. ML has accelerated lens finding; DESI Legacy Imaging work cited finds **>1500** new lenses. Rubin LSST, Euclid, and Roman motivate **$\mathcal{O}(10^5)$** lenses.
- **Computational gap:** Example: Rojas et al. (2021) report **~4.3 hours per system** (41 DES systems) with a **lenstronomy**-style pipeline. **PSO** (lenstronomy) parallelizes but has **no strong convergence guarantees**, especially in high dimension. **emcee** is widely used but can behave poorly in **high-dimensional** posteriors (§2.5.2; refs. to Huijser et al. 2017, Betancourt 2018).
- **Claim:** GIGA-Lens targets **robust**, **much faster**, **scalable** inference using **gradients + GPUs + AD**.

---

## 2. Physical forward model (§2.1.1)

- Parameters $\Theta$ predict **counts per second** on the image plane $I_{\rm model}(x,y;\Theta)$: deflection → ray trace to source plane → surface brightness conservation → **lens light** → **PSF convolution**. Optional sky in $\Theta$; the paper **assumes sky subtracted**.
- **Demonstration setup:** "Reference system" (Fig. 1) from lenstronomy's starting notebook, but **PSF** from **TinyTim** for **HST WFC3 F140W** at **0.13″**, subsampled to **0.065″** (more realistic than a Gaussian PSF in lenstronomy's notebook).
- **Mass:** **Elliptical power-law (EPL)** convergence $\kappa$ (Eq. 1; equivalent to PEMD) + **external shear** (Eq. 3). Deflection uses **Gaussian hypergeometric** form (Tessore & Metcalf 2015); truncation error $\lesssim 10^{-16}$ in $\lesssim 35$ iterations for $q \gtrsim 0.5$.
- **Light:** Elliptical **Sérsic** for lens and source (Eq. 4); **eccentricity** parameterization $(e_1,e_2)$ (Eq. 5). Lens light center **not** tied to mass center.
- **Pixels:** Model evaluated on a grid; **supersampling** by integer $k_{\rm super}$ (**$k_{\rm super}=2$** in the paper), then **average downsample** to data pixels.

---

## 3. Probabilistic model (§2.1.2)

- **Gaussian likelihood** with **per-pixel variance** combining **Gaussian background** and **Poisson** contribution from the **model** (Eq. 6–7):
  - $\sigma^2_{\rm tot}(x,y;\Theta) = \sigma^2_{\rm bkg} + I_{\rm model}(x,y;\Theta)/(G \cdot t_{\rm exp})$.
- They **reject** the common shortcut $\sigma^2_{\rm tot} \propto I_{\rm obs}$ for fixed $\Theta$: it can **bias** at low S/N (Horne 1986) and raises issues with negative pixels in lenstronomy's handling; they prefer the **model-based** Poisson term (small extra cost with AD).
- **Prior:** Product of independent distributions; **simulation distribution** for generating **100** training systems vs **broader prior** for inference (Eq. 8). **22 parameters** total (mass, shear, lens light, source light). Notation `a / b`: simulator uses `a`, prior uses `b`. **Log-normals** for several scales/amplitudes; **truncated normals** for some ellipticities; **uniforms** for Sérsic indices. Light amplitudes tuned so arc **S/N ~ 100** (range ~30–200). **Rotational symmetry** motivates **Gaussian** ellipticities vs **uniform** (Fig. 3).
- **Prior constraint:** They set priors so $\sqrt{(x_s-x_{\rm epl})^2+(y_s-y_{\rm epl})^2} \sim \theta_E$ to avoid extreme misalignment where gradients **vanish** (footnote §2.2).

---

## 4. Reparameterization and gradients (§2.2)

- **Bijectors** $g$ map **unconstrained** $\tilde{\Theta}\in\mathbb{R}^d$ to **physical** support (e.g. $\theta_E$: exponential map; bounded slope: sigmoid). **Change of variables** on the log-posterior: $\log \tilde{p}(\tilde{\Theta}|I) = \log p(\Theta|I) + \log|J|$ (Eq. 11–13). Likelihood unchanged in form but evaluated at $g(\tilde{\Theta})$.
- **AD** (TF/JAX): gradient cost **does not scale with dimension** like finite differences.

---

## 5. Step 1 — MAP via multi-start gradient descent (§2.3)

- **Goal:** Find **global** mode $\tilde{\Theta}^*_{\rm MAP}$ for **HMC initialization** (high-D posteriors → random starts impractical).
- **Method:** **$n_{\rm MAP}=300$** independent starts drawn from **prior** in unconstrained space; each run **$K_{\rm MAP}=300$** iterations of **Adam**; learning rate **$10^{-2}$** linear decay to **$10^{-3}$** over 300 steps. Best of 300 trajectories = MAP.
- **Multimodality:** Many **local** minima of $f=-\log\tilde{p}$ exist, but **only the global mode matters** for sampling — other modes have **negligible** posterior mass (footnote).
- **Empirics (reference system):** **~5%** of chains hit global mode (≥**1%** at low S/N); **17 s** on **four A100 GPUs** for this step (Table 1). **Weaker initialization sensitivity** than PSO (§2.6).

---

## 6. Step 2 — Variational inference for covariance (§2.4)

- **Goal:** Estimate **posterior covariance** $\tilde{\Sigma}$ to set **HMC mass matrix** (preconditioning) and **initialize chains**.
- **Ansatz:** **Multivariate normal** surrogate $\tilde{q}=\mathcal{N}(\tilde{\mu},\tilde{\Sigma})$; minimize **$\mathrm{KL}(\tilde{q}\Vert \tilde{p})$** ⇔ maximize **ELBO** (Eq. 16–18). **Stochastic VI:** Monte Carlo samples from $\tilde{q}$ at each step; **Adam** on $(\tilde{\mu},\tilde{\Sigma})$.
- **$\tilde{\Sigma}$:** **Cholesky bijector** to enforce PSD (lower-triangular $L$, **exp** on diagonal, $\tilde{\Sigma}=LL^\top$).
- **Init:** $\tilde{\mu}^{(1)}=\tilde{\Theta}^*_{\rm MAP}$, $\tilde{\Sigma}^{(1)}=10^{-6}I$ (start **narrow**). **$K_{\rm VI}=1000$**, **$n_{\rm VI}=500$**; learning rate **0 → $10^{-3}$** quadratically over first **500** steps (stability). **TF:** TFP VI; **JAX:** implement Eq. (18) directly.
- **Caveat:** True posterior is **not** Gaussian; Fig. 6 shows **banana** cross-sections though marginals look near-Gaussian. VI vs HMC marginals **do not** always match; authors still find VI covariance **usually adequate** for preconditioning. **~52 s** on four A100s (Table 1).

---

## 7. Step 3 — Hamiltonian Monte Carlo (§2.5)

- **Sampling:** **$n_{\rm HMC}=50$** parallel chains; init from **$\mathcal{N}(\tilde{\mu}^*_{\rm VI},\tilde{\Sigma}^*_{\rm VI})$**; **$n_{\rm burn}=250$**, **$n_{\rm sample}=750$**.
- **Tuning:** TFP **dual averaging** / NUTS-related tooling: adapt **step size** and **trajectory length** during **first 80% of burn-in**, then **fix** (adaptation breaks stationarity). Target acceptance **0.75** (range 0.6–0.8 per Betancourt 2018).
- **Mass matrix:** **$\tilde{M}=(\tilde{\Sigma}^*_{\rm VI})^{-1}$** (preconditioned HMC). On-the-fly mass adaptation could **remove** the VI step in future work (not yet trusted in their pipeline).
- **Diagnostics:** **ESS** and **Gelman–Rubin $\hat{R}$**; reference system: **ESS > 26000**, **$\hat{R}<1.01$** per parameter (Fig. 7); **~36 s** HMC on four A100s.
- **vs emcee:** Same reference system, **22-D**, fair comparison (lenstronomy-style emcee init on **CPU** vs HMC on **one A100**). HMC **~40 ESS/iter** (~**300 ESS/s**); emcee **~0.2 ESS/iter** (~**0.04 ESS/s**). emcee: poor mixing, **high $\hat{R}$** (~1.5 example Fig. 8), **long autocorrelation** (~300 lags vs HMC ≪10). **Gradient-free samplers** scale badly for **shapelets / wavelets / pixelized sources** and **many-parameter** perturbers (§2.5.2).

---

## 8. End-to-end timing (Table 1, §2.6)

| Step | Time (4× A100) | Key settings |
|------|----------------|--------------|
| MAP | **17 s** | $n_{\rm MAP}=300$, $K_{\rm MAP}=300$, Adam LR schedule |
| VI | **52 s** | $K_{\rm VI}=1000$, $n_{\rm VI}=500$ |
| HMC | **36 s** | 50 chains, 250+750, mass = VI$^{-1}$ |
| **Total** | **105 s** | Same hyperparameters used for **reference + 100** systems |

- **Single A100:** **~6 minutes** total (**~3.5×** slower than 4-GPU). **Perlmutter** early access cited.

---

## 9. Ensemble test — 100 simulated lenses (§3)

- **Simulation:** lenstronomy; parameters from **simulator** side of Eq. (8). **$\sigma_{\rm bkg}=0.2$**, **$t_{\rm exp}=100$ s**, **$G=1$**, PSF as Fig. 1; **80×80** pixels, **0.065″** pix, **5.2″** field (Fig. 10).
- **Inference:** Table 1 **unchanged** from reference tuning → **good recovery** vs ground truth (Fig. 11); scaled errors $\mu_z$ **consistent with no bias** (Table 2). **Worst** $\hat{R}$ over all systems/params **1.017**; **min ESS** **26822**.
- **Phenomenology:** Uncertainties on **$n_l$, $n_s$** grow at high Sérsic index (degeneracy with size).

---

## 10. Discussion and outlook (§4)

- Framework is **general**: any differentiable parameterized model; **shapelets** for source cited as **~+10%** compute vs Sérsic.
- **Real data:** **51** DESI-discovered systems with **HST** (prog. 15867); follow-up paper **in prep.**
- **Multimodality:** Not severe for their **EPL+Sérsic** model; **unclear** for richer models → future: **parallel tempering**, **adiabatic MC**, **AIS** (also for **Bayes factors** / substructure).
- **Speedups:** **8× A100** ~**half** time; **on-the-fly mass adaptation** might **drop VI** (~2×); **normalizing flows** could fit posterior well enough to **drop HMC** (Kingma et al. 2017; Papamakarios et al. 2018).

---

## 11. How this relates to follow-on inference research

The paper's design is explicitly **modular**: differentiable simulator + **MAP → VI (Gaussian surrogate) → preconditioned HMC**. Extensions (new samplers, flow-based posteriors, real-data systematics, substructure) should preserve **traceable** likelihoods/priors and **honest** diagnostics (ESS, $\hat{R}$, prior/likelihood checks). When changing any stage, re-read the **local PDF** for the original assumptions and baselines.
