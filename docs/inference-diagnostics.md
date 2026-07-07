# Inference Diagnostics
Accumulated experience about diagnostics for running inference with gigalens.

> The distilled **decision table** (signal → reading → required first action) lives on the
> operating card (`docs/agent-operating-card.md`), which is auto-injected into every session —
> this file is the depth behind it: what each diagnostic means and how to read the plots. If
> you change a threshold here, update the card's table to match. The ordered diagnostic
> *workflow* is the `/diagnose-sampling` skill (`.claude/skills/diagnose-sampling/SKILL.md`).

# Numerical Diagnostics
## Point Diagnostics
### Reduced Chi-Squared
In comparing a simulated lens to a the truth, you can get a reduced chi-squared value. Ideally, this value should be near 1. 
All 'good' fits should be at least below a reduced chi-squred of 1.2, but it depends on the context.
In most cases I've seen, values above 3 or so (in the final sample of MAP, or any samples from SVI or HMC) indicate that there's something pathological with the model. It could be a NaN-related issue in MAP (which results in particles not moving), a complete model mismatch, or something else.

## MCMC Diagnostics
### R-hat (Gelman Rubin Statistic)
This measures how well MCMC chains have converged (are sampling the same distribution). Closer to 1 is better. It is reported for each parameter, and to assess how well sampling is occurring, you should look at the worst R-hat. To trust a result, the Max(R-hat) must be at least below 1.1 and ideally below 1.01. For very rough guessing, R-hats below 1.3 are passable.

### ESS (Effective Sample Size)
This measures how many truly independent samples would have the same constraining power as the (often highly) autocorrelated samples produced by your MCMC sampling. Higher is better. It is reported for each parameter, and to assess how well sampling is occurring, you should look at the worst ESS.
Min(ESS) of close to the number of chains usually means you are failing to sample well.
There is one special case here. If there's bi- or multi- modality in your posterior, each individual mode can be well-sampled, even if the ESS and R-hat look bad. This is caused by your chains not mixing or mixing slowly between modes.

# Plotted Diagnostics
It may be easier for an AI agent to do some of these diagnostic tasks with purely numerical diagnostics. You should be honest about any uncertainties/difficulties you have in reading these plots. 
However, the visual ones are tried-and-true and invaluable for human users. A picture is worth a thousand words, and it's very easy to miss things with numerical diagnostics that are very obvious with plot-based ones.
## Posterior-Side Plot Diagnostics
### Cornerplots
These are one of the most comprehensive diagnostics you can use. You can overplot results from any inference stage: point estimates from MAP, samples from a surrogate posterior, or MCMC samples.
However, they can be hard to read and slow to create, especially for models with many parameters. If you are an AI agent, DO NOT create cornerplots that are more than 4x4. Your vision model downsamples too much for them to be readable.
#### How to read cornerplots of MCMC samples:
- Fuzzy/irregular edges typically come along with low ESS
- Chains that are frozen, separated, or otherwise not mixing will often show up as self-contained (frequently small) blobs. This won't always happen, depending on how many chains there are and their proximity.
- If there are long, winding tracks, the chains are likely migrating towards an area of higher probability density.
- Don't equate the shape of the samples and the true shape of the posterior until at least Rhat < 1.1, ideally < 1.01.
- You can often spot a second mode on these plots. Note that this is a mode in the samples, not necessarily a true, high-probability mode. Also, be careful, since the KDE that draws the contours can make poorly sampled posteriors appear multimodal when they aren't.

#### Things to do with cornerplots
- See how far different inference stages migrate from previous ones. If HMC is far (multiple standard deviations) from the SVI results, that suggests SVI didn't find a very good surrogate posterior. This can be fine, but the sampling algorithm will have to work harder.
- Compare inverse mass matrix estimates with true posterior. If the inverse mass matrix being used (can plot as a multivariate normal surrogate with a mean at the mean of the real samples) is very different in spread from the true samples, that's a suggestion that sampling may be being held back by inverse mass matrix estimation.
- Check bias of sampling methods. Plotting the converged results from a known MCMC method on the same cornerplot as an experimental one can give a one-look way to check if they're similar.

### MCMC Trace Plots
Show the tracks of the parallel sampling chains in a single parameter. Usually less useful than cornerplots, but easier and faster to produce.
The main thing you can do with them that you can't do with cornerplots is look at individual chain history over time.
This has a few applications:
- Inspecting mixing. If chains mix well, the chain traces should blend together. If they don't, the chains will look at least somewhat separated. This will likely come along with poor R-hat
- In cases where you suspect multimodality, you should be able to see fairly discrete jumps where the chains hop from one mode to the other, while mixing well within modes.
- If the chains are 'frozen' (not moving at all or barely moving), this will be obvious in the trace plots.

## Point-Level Plot Diagnostics
### Comparison Between Observed and Model
Plotting the observed data and the modeled image side-by-side can be useful. It's not as sensitive to small differences as normalized residuals, but it's easy to see catastrophic model problems or large mismatches. 
With some effort, you can pick out smaller differences from these plots as well.
Seeing the full lensed image from the model is also a good first sanity check.

### Normalized Residual Plots
These are very useful for comparing a lensing model to the observed data. If your model is perfect, the normalized residual will just look like N(0, 1) gaussian noise.
But often, the model isn't perfect, and you'll have spatial structure in your residuals. This will often manifest near bright spots, like the center of the lens galaxy light or the lensing arcs. 
Residual structure indicates that you model isn't placing light where it should be, or it's making it too bright/too dim.
Common types of patterns in residuals:
- 'Dipole structure' shows up (usually) when a lensed image is misplaced by the model. It typically looks like an overestimation right next to an underestimation. It's an underestimation of the brightness where the observed lensing image is and an overestimation where the model's lensing image is.
- 'Point-in-ring structure' typically shows up when the model's lensed image can't reproduce an very sharp peak in the brightness of the observed lensed image. It looks like a ring of overestimation surrounding a smaller 'dot' of underestimation.
Plenty of residual structure doesn't fit into these categories, but still points to some level of model-data mismatch. Whether it's a significant level is a situation-dependent judgement call, but large clumps of 4+ sigma pixels are typically a sign that it's significant.

### Source-Plane Plots
Often, the mass model can compensate for a pathological or unphysical source light profile. Plotting just the source plane (not convolved with a PSF) can be helpful