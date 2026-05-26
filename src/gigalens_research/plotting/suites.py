import matplotlib.pyplot as plt
from .corner import cornerplot_results
from .systems import plot_image_results
import os


def plot_loss_histories(fig, axs, map_chisq_hist, svi_loss_hist):
    """
    Plot the loss histories of the MAP and SVI stages of the inference pipeline.
    """
    axs[0].plot(map_chisq_hist)
    axs[0].set_title("MAP Loss History")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Chi-squared Loss")
    axs[0].set_ylim(bottom=0, top=3)

    axs[1].plot(svi_loss_hist)
    axs[1].set_title("SVI Loss History")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("ELBO")

def display_results(r, true_img, lens_sim, true_params=None, save_dir=None, 
    show=True, make_cornerplot=True, plot_caustics=False, model_seq=None):
    """
    Display all results of the inference pipeline, including:
    - Comparing predicted images to true images (MAP best sample and HMC median)
    - Plotting the loss histories of the MAP and SVI stages
    - Plotting the cornerplot of the results, including the MAP best fit, SVI samples, and HMC samples
    """
    
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(12,3)
    plot_image_results(fig, axs, true_img, prefix="MAP",
                       lens_sim=lens_sim, predicted_params=r['MAP'].MAP_best, 
                       resimulate=True, true_params=true_params, plot_caustics=plot_caustics, model_seq=model_seq)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'map_results.png'))
    if show:
        plt.show()
    plt.close(fig)

    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(12,3)
    plot_image_results(fig, axs, true_img, prefix="HMC",
                       lens_sim=lens_sim, predicted_params=r['HMC'].HMC_median, 
                       resimulate=True, true_params=true_params, plot_caustics=plot_caustics, model_seq=model_seq)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'hmc_results.png'))
    if show:
        plt.show()
    plt.close(fig)
    
    fig, axs = plt.subplots(1, 2)
    plot_loss_histories(fig, axs, r['MAP'].MAP_chisq_hist, r['SVI'].SVI_loss_hist)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'loss_histories.png'))
    if show:
        plt.show()
    plt.close(fig)
    
    if make_cornerplot:
        # HMC_samp_reduced = jax.random.choice(jax.random.PRNGKey(0), r['HMC'].HMC_samples, (1000,), replace=False)
        cornerplot_results(r['MAP'].MAP_best, r['SVI'].SVI_samples, r['HMC'].HMC_samples, true_params=true_params, hmc_median=r['HMC'].HMC_median)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'cornerplot.png'))
        if show:
            plt.show()
        plt.close()