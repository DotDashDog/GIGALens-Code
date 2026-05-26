def get_errors_diff(HMC_samples, true_params):
    
    lower_err = jax.tree.map(lambda x: -(jnp.percentile(x, 16)-jnp.median(x)), HMC_samples)
    upper_err = jax.tree.map(lambda x: jnp.percentile(x, 84)-jnp.median(x), HMC_samples)
    median = jax.tree.map(lambda x: jnp.median(x), HMC_samples)

    median_diff = jax.tree.map(lambda x, y: x-y, median, true_params)

    # flat_lower_err, _ = jax.tree.flatten(lower_err)
    # flat_upper_err, _ = jax.tree.flatten(upper_err)
    # flat_median_diff, _ = jax.tree.flatten(median_diff)
    # flat_true_params, _ = jax.tree.flatten(true_params)

    return median_diff, lower_err, upper_err

def normalize_residuals(median_diff, upper_err, lower_err):
    pos_res = median_diff > 0
    scale = np.zeros_like(median_diff)
    scale[pos_res] = lower_err[pos_res]
    scale[~pos_res] = upper_err[~pos_res]

    residual_norm = median_diff/scale
    chisq = 1/(residual_norm.shape[0]-1) * np.sum(np.square(residual_norm))
    return residual_norm, chisq

def residualplot_params(save_dirs, true_params_all, prob_models, make_hist=False, figsize=(20,15), plot_kwargs={}):

    median_diffs = []
    upper_errs = []
    lower_errs = []
    for i, save_dir in enumerate(save_dirs):
        true_params = jax.tree.map(lambda a : a[i], true_params_all)

        samples = np.load(os.path.join(save_dir, 'hmc_samples_z.npy')).reshape((-1, 22))
        HMC_samples = prob_models[i].bij.forward(list(samples.T))

        diff, low, high = get_errors_diff(HMC_samples, true_params)

        median_diffs.append(diff)
        lower_errs.append(low)
        upper_errs.append(high)

    
    #* Turn list of trees into a single tree of jnp arrays
    def list_to_tree(list_of_trees):
        return jax.tree.map(lambda *xs: jnp.array(xs), *list_of_trees)

    median_diffs = flatten_params_to_labeled_dict(list_to_tree(median_diffs))
    upper_errs = flatten_params_to_labeled_dict(list_to_tree(upper_errs))
    lower_errs = flatten_params_to_labeled_dict(list_to_tree(lower_errs))
    true_params_flat = flatten_params_to_labeled_dict(true_params_all)

    # labels = cornerplot_labels(true_params_all, latex=True)
    # n_params = len(labels)
    n_params = len(median_diffs)
    ncols = 5
    fig, axs = plt.subplots(n_params//ncols + 1, ncols)
    fig.set_size_inches(*figsize)
    fig.tight_layout()
    axs = axs.flatten()

    num_systems = len(save_dirs)

    for i, label in enumerate(median_diffs.keys()):
        axs[i].axhline(y=0, color='red', linestyle=':', alpha=0.5)
        axs[i].errorbar(true_params_flat[label], median_diffs[label], yerr=[lower_errs[label], upper_errs[label]], **(plot_kwargs |  dict(fmt='o', linestyle='')))
        axs[i].set_title(latex_label(label))
        yabs_max = abs(max(axs[i].get_ylim(), key=abs))
        axs[i].set_ylim(ymin=-yabs_max, ymax=yabs_max)

    # Turn off unused axes in the subplot grid
    for j in range(len(median_diffs.keys()), len(axs)):
        axs[j].axis('off')
    if not make_hist:
        return fig, axs
    else:
        plt.show()

    if make_hist:
        fig, axs = plt.subplots(n_params//3 + 1, 3)
        fig.set_size_inches(20,25)
        axs = axs.flatten()

        
        for i, label in enumerate(median_diffs.keys()):
            z_scores, chisq = normalize_residuals(median_diffs[label], upper_errs[label], lower_errs[label])
            outliers = jnp.where(jnp.abs(z_scores) > 5)[0]
            if len(outliers) > 0:
                print(f"{label} has outliers at indices: {outliers}")
            histogram_residuals(fig, axs[i], z_scores, f'{label}, chisq: {chisq:.3f}', bins=10)

def latex_label(label):

    latex_label_map = {
        # Mass
        "theta_E": r"$\theta_E$",
        "gamma": r"$\gamma_{epl}$",
        "e1": r"$\epsilon_{epl,1}$",
        "e2": r"$\epsilon_{epl,2}$",
        "center_x": r"$x_{epl}$",
        "center_y": r"$y_{epl}$",
        "gamma1": r"$\gamma_{ext,1}$",
        "gamma2": r"$\gamma_{ext,2}$",

        # Lens Light
        "lens_R_sersic": r"$R_{l}$",
        "lens_n_sersic": r"$n_{l}$",
        "lens_e1": r"$\epsilon_{l,1}$",
        "lens_e2": r"$\epsilon_{l,2}$",
        "lens_center_x": r"$x_{l}$",
        "lens_center_y": r"$y_{l}$",
        "lens_Ie": r"$I_l$",

        # Source Light
        "src_R_sersic": r"$R_{s}$",
        "src_n_sersic": r"$n_{s}$",
        "src_e1": r"$\epsilon_{s,1}$",
        "src_e2": r"$\epsilon_{s,2}$",
        "src_center_x": r"$x_{s}$",
        "src_center_y": r"$y_{s}$",
        "src_Ie": r"$I_s$",
    }

    return latex_label_map[label]