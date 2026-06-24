from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
# NOTE: the old gigalens API (ForwardProbModel / BackwardProbModel / LensSimulator) was
# imported here but never used in any function body; removed with the old API. The
# functions below (get_noise_image / get_chisq / log_prob_image*) are pure and stay on
# the working path (e.g. plotting/systems.py uses get_noise_image/get_chisq).
import jax
import tensorflow_probability.substrates.jax as tfp
import os
import time
tfd = tfp.distributions

def get_noise_image(image, background_rms, exp_time):
    # Clip to non-negative before adding the Poisson term so a single noise-induced
    # negative pixel cannot drive the radicand below zero (which would produce a NaN
    # err_map and, via 1/err_map in lstsq_simulate, poison every entry of the
    # Gram matrix). Matches the convention used in BackwardProbModel.__init__.
    return jnp.sqrt(jnp.clip(image, 0, jnp.inf) / exp_time + background_rms**2)

def get_chisq(true_img, predicted_img, background_rms=0.2, exp_time=100):
    emap = get_noise_image(predicted_img, background_rms, exp_time)

    return jnp.sum(jnp.square((true_img-predicted_img)/emap))


def log_prob_image(prob_model, simulator, z):
    z = list(z.T)
    x = prob_model.bij.forward(z)
    im_sim = simulator.simulate(x)
    err_map = jnp.sqrt(prob_model.background_rms ** 2 + im_sim / prob_model.exp_time)
    # log_like = tfd.Independent(
    #     tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
    # ).log_prob(prob_model.observed_image)
    return tfd.Normal(im_sim, err_map).log_prob(prob_model.observed_image)
    # log_prior = prob_model.prior.log_prob(x) + prob_model.bij.forward_log_det_jacobian(z)
    # return log_like + log_prior, jnp.mean(
    #     ((im_sim - prob_model.observed_image) / err_map) ** 2, axis=(-2, -1)
    # )

def log_prob_image_patched(prob_model, simulator, z, kernel_size=3):
    z = list(z.T)
    x = prob_model.bij.forward(z)
    im_sim = simulator.simulate(x)
    err_map = jnp.sqrt(prob_model.background_rms ** 2 + im_sim / prob_model.exp_time)
    log_prob_img = tfd.Normal(im_sim, err_map).log_prob(prob_model.observed_image)

    window = jnp.ones((kernel_size, kernel_size))

    return jax.scipy.signal.convolve(log_prob_img, window, mode='full')

def bridge_sampler(rng_key, samples, log_posterior_fn, n_iter=50):
    """
    Estimates the log-marginal likelihood using the Meng-Wong bridge sampler.
    
    Args:
        rng_key: JAX random key.
        samples: MCMC samples of shape (n_samples, n_dim).
        log_posterior_fn: Function mapping (theta) -> log_posterior(theta).
        n_iter: Number of iterations for the Meng-Wong algorithm.
    """
    n_samples, n_dim = samples.shape
    
    # 1. Fit a Gaussian proposal to the MCMC samples
    mu = jnp.mean(samples, axis=0)
    cov = jnp.cov(samples, rowvar=False)
    proposal = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
    
    # 2. Draw 'fake' samples from the proposal
    prop_key, _ = jax.random.split(rng_key)
    gen_samples = proposal.sample(n_samples, seed=prop_key)
    
    l1 = log_posterior_fn(samples)
    l2 = log_posterior_fn(gen_samples)
    

    q1 = proposal.log_prob(samples)
    q2 = proposal.log_prob(gen_samples)
    
    # 4. Iterative Meng-Wong Algorithm
    log_r_guess = jnp.log(1/n_samples) + jax.nn.logsumexp(l2-q2, axis=0)
    log_r = log_r_guess

    
    h = []
    s1 = 0.5 # Proportion of samples from posterior (assuming n1 == n2)
    s2 = 0.5 # Proportion of samples from proposal
    
    for _ in range(n_iter):
        l_num = l2 - jnp.logaddexp(jnp.log(s1) + l2, jnp.log(s2) + log_r + q2)
        
        l_den = q1 - jnp.logaddexp(jnp.log(s1) + l1, jnp.log(s2) + log_r + q1)
        
        # Update log_r
        log_r_new = jax.nn.logsumexp(l_num, axis=0) - jax.nn.logsumexp(l_den, axis=0)
        
        # For stability, we can track the delta or use a small dampening factor if needed
        log_r = log_r_new
        h.append(log_r)

    return log_r, h