# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df_long = pd.read_csv('benchmarking_results/benchmark_results_final_gather_Nico.csv').sort_values(by='devices')
df_harry_long = pd.read_csv('benchmarking_results/benchmark_results_final_Harry.csv').sort_values(by='devices')

df = df_long.groupby(['devices', 'slurm_job_id', 'map_n_samples']).mean().reset_index()
df_harry = df_harry_long.groupby(['devices', 'slurm_job_id', 'map_n_samples']).mean().reset_index()

# df_nico_modified = pd.read_csv('benchmarking_results/benchmark_results_NicoModified.csv').sort_values(by='devices')
fig, axs= plt.subplots(3, 2, figsize=(10, 10), sharex=True)
axs = axs.T
axsN,axsH = axs

for i in range(3):
    axs[0][i].sharey(axs[1][i])

# MAP Performance - plot each unique map_n_samples value as a separate line


def independent_lines(ax, df, fn_to_plot, sample_column, line_label, label, linestyle='-', color_map=None, add_to_legend=True):
    # Create color map if not provided
    if color_map is None:
        unique_samples = sorted(df[sample_column].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_samples)))
        color_map = dict(zip(unique_samples, colors))
    
    for samples in sorted(df[sample_column].unique()):
        filtered_data = df[df[sample_column] == samples]
        legend_label = f'{sample_column}={samples}' if add_to_legend else None
        ax.plot(filtered_data['devices'], fn_to_plot(filtered_data), 
                label=legend_label, marker='o', 
                linestyle=linestyle, color=color_map[samples])
    
    return color_map

map_it_per_sec = lambda df: 1000/df['map_time']
svi_it_per_sec = lambda df: 1500/df['svi_time']
map_it_per_sec_no_first_epoch = lambda df: 999/(df['map_time'] - df['map_time_1epoch'])
svi_it_per_sec_no_first_epoch = lambda df: 1499/(df['svi_time'] - df['svi_time_1epoch'])

hmc_smp_per_sec = lambda df: df['n_hmc']*750/(df['hmc_time'] - df['hmc_time_burnin'])

#* Plot Nico's Pipeline Speeds
independent_lines(axsN[0], df, map_it_per_sec, 'map_n_samples', 'GL2.0', 'MAP')

independent_lines(axsN[1], df, svi_it_per_sec, 'n_vi', 'GL2.0', 'SVI')

independent_lines(axsN[2], df, hmc_smp_per_sec, 'n_hmc', 'GL2.0', 'HMC')

#* Plot Harry's Pipeline Speeds
independent_lines(axsH[0], df_harry, map_it_per_sec, 'map_n_samples', 'Harry', 'MAP')

independent_lines(axsH[1], df_harry, svi_it_per_sec, 'n_vi', 'Harry', 'SVI')

independent_lines(axsH[2], df_harry, hmc_smp_per_sec, 'n_hmc', 'Harry', 'HMC')

#* Plot Nico's Modified Pipeline Speeds
# independent_lines(axsN_modified[0], df_nico_modified, 'map_time', 'map_n_samples', 'Nico Mod SVI', 'MAP', 350)

# independent_lines(axsN_modified[1],  df_nico_modified, 'svi_time', 'n_vi', 'Nico Mod SVI', 'SVI', 1500)

# independent_lines(axsN_modified[2], df_nico_modified, 'hmc_time', 'n_hmc', 'Nico Mod SVI', 'HMC', 1)

# for i in range(2):
for ax in axs.flatten():
    ax.set_xlabel('Number of devices')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend()

for i in range(2):
    axs[i][0].set_ylabel('Iterations per second')

    axs[i][1].set_ylabel('Iterations per second')
    axs[i][1].set_title(f'SVI Performance ({1500} steps)')


    axs[i][2].set_ylabel('Samples/second')
    axs[i][2].set_title(f'HMC Performance')


plt.tight_layout()

plt.legend()
plt.show()

# %%
# Combined comparison plot - Harry vs Nico on same axes
fig_combined, axs_combined = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Create consistent color maps for each parameter type
# Get all unique values from both datasets to ensure consistent coloring
all_map_samples = sorted(set(df['map_n_samples'].unique()) | set(df_harry['map_n_samples'].unique()))
all_svi_samples = sorted(set(df['n_vi'].unique()) | set(df_harry['n_vi'].unique()))
all_hmc_samples = sorted(set(df['n_hmc'].unique()) | set(df_harry['n_hmc'].unique()))

map_colors = dict(zip(all_map_samples, plt.cm.tab10(np.linspace(0, 1, len(all_map_samples)))))
svi_colors = dict(zip(all_svi_samples, plt.cm.tab10(np.linspace(0, 1, len(all_svi_samples)))))
hmc_colors = dict(zip(all_hmc_samples, plt.cm.tab10(np.linspace(0, 1, len(all_hmc_samples)))))

#* Plot MAP Performance Comparison
independent_lines(axs_combined[0], df, map_it_per_sec, 'map_n_samples', 'GL2.0', 'MAP', linestyle='-', color_map=map_colors, add_to_legend=True)
independent_lines(axs_combined[0], df_harry, map_it_per_sec, 'map_n_samples', 'Harry', 'MAP', linestyle='--', color_map=map_colors, add_to_legend=False)
axs_combined[0].set_ylabel('Iterations per second')
axs_combined[0].set_title('MAP Performance Comparison (1000 iterations)')
axs_combined[0].legend()
axs_combined[0].text(0.98, 0.02, 'Solid: GL2.0, Dashed: Harry', transform=axs_combined[0].transAxes, 
                     verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#* Plot SVI Performance Comparison
independent_lines(axs_combined[1], df, svi_it_per_sec, 'n_vi', 'GL2.0', 'SVI', linestyle='-', color_map=svi_colors, add_to_legend=True)
independent_lines(axs_combined[1], df_harry, svi_it_per_sec, 'n_vi', 'Harry', 'SVI', linestyle='--', color_map=svi_colors, add_to_legend=False)
axs_combined[1].set_ylabel('Iterations per second')
axs_combined[1].set_title('SVI Performance Comparison (1500 steps)')
axs_combined[1].legend()
axs_combined[1].text(0.98, 0.02, 'Solid: GL2.0, Dashed: Harry', transform=axs_combined[1].transAxes, 
                     verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#* Plot HMC Performance Comparison
independent_lines(axs_combined[2], df, hmc_smp_per_sec, 'n_hmc', 'GL2.0', 'HMC', linestyle='-', color_map=hmc_colors, add_to_legend=True)
independent_lines(axs_combined[2], df_harry, hmc_smp_per_sec, 'n_hmc', 'Harry', 'HMC', linestyle='--', color_map=hmc_colors, add_to_legend=False)
axs_combined[2].set_ylabel('Samples per second')
axs_combined[2].set_title('HMC Performance Comparison')
axs_combined[2].set_xlabel('Number of devices')
axs_combined[2].legend()
axs_combined[2].text(0.98, 0.02, 'Solid: GL2.0, Dashed: Harry', transform=axs_combined[2].transAxes, 
                     verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

for ax in axs_combined:
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

plt.suptitle('Performance Comparison (Including 1st epoch for MAP, SVI)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%

#* Do the same but subtracting the first epoch from the MAP and SVI times
fig_combined, axs_combined = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Create consistent color maps for each parameter type
# Get all unique values from both datasets to ensure consistent coloring
all_map_samples = sorted(set(df['map_n_samples'].unique()) | set(df_harry['map_n_samples'].unique()))
all_svi_samples = sorted(set(df['n_vi'].unique()) | set(df_harry['n_vi'].unique()))
all_hmc_samples = sorted(set(df['n_hmc'].unique()) | set(df_harry['n_hmc'].unique()))

map_colors = dict(zip(all_map_samples, plt.cm.tab10(np.linspace(0, 1, len(all_map_samples)))))
svi_colors = dict(zip(all_svi_samples, plt.cm.tab10(np.linspace(0, 1, len(all_svi_samples)))))
hmc_colors = dict(zip(all_hmc_samples, plt.cm.tab10(np.linspace(0, 1, len(all_hmc_samples)))))

#* Plot MAP Performance Comparison
independent_lines(axs_combined[0], df, map_it_per_sec_no_first_epoch, 'map_n_samples', 'GL2.0', 'MAP', linestyle='-', color_map=map_colors, add_to_legend=True)
independent_lines(axs_combined[0], df_harry, map_it_per_sec_no_first_epoch, 'map_n_samples', 'Harry', 'MAP', linestyle='--', color_map=map_colors, add_to_legend=False)
axs_combined[0].set_ylabel('Iterations per second')
axs_combined[0].set_title('MAP Performance Comparison (1000 iterations)')
axs_combined[0].legend()
axs_combined[0].text(0.98, 0.02, 'Solid: GL2.0, Dashed: Harry', transform=axs_combined[0].transAxes, 
                     verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#* Plot SVI Performance Comparison
independent_lines(axs_combined[1], df, svi_it_per_sec_no_first_epoch, 'n_vi', 'GL2.0', 'SVI', linestyle='-', color_map=svi_colors, add_to_legend=True)
independent_lines(axs_combined[1], df_harry, svi_it_per_sec_no_first_epoch, 'n_vi', 'Harry', 'SVI', linestyle='--', color_map=svi_colors, add_to_legend=False)
axs_combined[1].set_ylabel('Iterations per second')
axs_combined[1].set_title('SVI Performance Comparison (1500 steps)')
axs_combined[1].legend()
axs_combined[1].text(0.98, 0.02, 'Solid: GL2.0, Dashed: Harry', transform=axs_combined[1].transAxes, 
                     verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#* Plot HMC Performance Comparison
independent_lines(axs_combined[2], df, hmc_smp_per_sec, 'n_hmc', 'GL2.0', 'HMC', linestyle='-', color_map=hmc_colors, add_to_legend=True)
independent_lines(axs_combined[2], df_harry, hmc_smp_per_sec, 'n_hmc', 'Harry', 'HMC', linestyle='--', color_map=hmc_colors, add_to_legend=False)
axs_combined[2].set_ylabel('Samples per second')
axs_combined[2].set_title('HMC Performance Comparison')
axs_combined[2].set_xlabel('Number of devices')
axs_combined[2].legend()
axs_combined[2].text(0.98, 0.02, 'Solid: GL2.0, Dashed: Harry', transform=axs_combined[2].transAxes, 
                     verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

for ax in axs_combined:
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

plt.suptitle('Performance Comparison (Not including 1st epoch for MAP, SVI)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%

#* Just plot Harry's speeds, with and without first epoch
fig_combined, axs_combined = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Create consistent color maps for each parameter type
# Get all unique values from both datasets to ensure consistent coloring
all_map_samples = sorted(set(df['map_n_samples'].unique()) | set(df_harry['map_n_samples'].unique()))
all_svi_samples = sorted(set(df['n_vi'].unique()) | set(df_harry['n_vi'].unique()))
all_hmc_samples = sorted(set(df['n_hmc'].unique()) | set(df_harry['n_hmc'].unique()))

map_colors = dict(zip(all_map_samples, plt.cm.tab10(np.linspace(0, 1, len(all_map_samples)))))
svi_colors = dict(zip(all_svi_samples, plt.cm.tab10(np.linspace(0, 1, len(all_svi_samples)))))
hmc_colors = dict(zip(all_hmc_samples, plt.cm.tab10(np.linspace(0, 1, len(all_hmc_samples)))))

#* Plot MAP Performance Comparison
# independent_lines(axs_combined[0], df, map_it_per_sec, 'map_n_samples', 'GL2.0', 'MAP', linestyle='-', color_map=map_colors, add_to_legend=True)
independent_lines(axs_combined[0], df_harry, map_it_per_sec, 'map_n_samples', 'Harry', 'MAP', linestyle='--', color_map=map_colors, add_to_legend=True)
axs_combined[0].set_ylabel('Iterations per second')
axs_combined[0].set_title('MAP Performance (1000 iterations)')
axs_combined[0].legend()

#* Plot SVI Performance Comparison
# independent_lines(axs_combined[1], df, svi_it_per_sec, 'n_vi', 'GL2.0', 'SVI', linestyle='-', color_map=svi_colors, add_to_legend=True)
independent_lines(axs_combined[1], df_harry, svi_it_per_sec, 'n_vi', 'Harry', 'SVI', linestyle='--', color_map=svi_colors, add_to_legend=True)
axs_combined[1].set_ylabel('Iterations per second')
axs_combined[1].set_title('SVI Performance (1500 steps)')
axs_combined[1].legend()

#* Plot HMC Performance Comparison
# independent_lines(axs_combined[2], df, hmc_smp_per_sec, 'n_hmc', 'GL2.0', 'HMC', linestyle='-', color_map=hmc_colors, add_to_legend=True)
independent_lines(axs_combined[2], df_harry, hmc_smp_per_sec, 'n_hmc', 'Harry', 'HMC', linestyle='--', color_map=hmc_colors, add_to_legend=True)
axs_combined[2].set_ylabel('Samples per second')
axs_combined[2].set_title('HMC Performance Comparison')
axs_combined[2].set_xlabel('Number of devices')
axs_combined[2].legend()

for ax in axs_combined:
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

plt.suptitle('Performance (Including 1st epoch for MAP, SVI)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



fig_combined, axs_combined = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Create consistent color maps for each parameter type
# Get all unique values from both datasets to ensure consistent coloring
all_map_samples = sorted(set(df['map_n_samples'].unique()) | set(df_harry['map_n_samples'].unique()))
all_svi_samples = sorted(set(df['n_vi'].unique()) | set(df_harry['n_vi'].unique()))
all_hmc_samples = sorted(set(df['n_hmc'].unique()) | set(df_harry['n_hmc'].unique()))

map_colors = dict(zip(all_map_samples, plt.cm.tab10(np.linspace(0, 1, len(all_map_samples)))))
svi_colors = dict(zip(all_svi_samples, plt.cm.tab10(np.linspace(0, 1, len(all_svi_samples)))))
hmc_colors = dict(zip(all_hmc_samples, plt.cm.tab10(np.linspace(0, 1, len(all_hmc_samples)))))

#* Plot MAP Performance Comparison
# independent_lines(axs_combined[0], df, map_it_per_sec_no_first_epoch, 'map_n_samples', 'GL2.0', 'MAP', linestyle='-', color_map=map_colors, add_to_legend=True)
independent_lines(axs_combined[0], df_harry, map_it_per_sec_no_first_epoch, 'map_n_samples', 'Harry', 'MAP', linestyle='--', color_map=map_colors, add_to_legend=True)
axs_combined[0].set_ylabel('Iterations per second')
axs_combined[0].set_title('MAP Performance (1000 iterations)')
axs_combined[0].legend()

#* Plot SVI Performance Comparison
# independent_lines(axs_combined[1], df, svi_it_per_sec_no_first_epoch, 'n_vi', 'GL2.0', 'SVI', linestyle='-', color_map=svi_colors, add_to_legend=True)
independent_lines(axs_combined[1], df_harry, svi_it_per_sec_no_first_epoch, 'n_vi', 'Harry', 'SVI', linestyle='--', color_map=svi_colors, add_to_legend=True)
axs_combined[1].set_ylabel('Iterations per second')
axs_combined[1].set_title('SVI Performance (1500 steps)')
axs_combined[1].legend()

#* Plot HMC Performance Comparison
# independent_lines(axs_combined[2], df, hmc_smp_per_sec, 'n_hmc', 'GL2.0', 'HMC', linestyle='-', color_map=hmc_colors, add_to_legend=True)
independent_lines(axs_combined[2], df_harry, hmc_smp_per_sec, 'n_hmc', 'Harry', 'HMC', linestyle='--', color_map=hmc_colors, add_to_legend=True)
axs_combined[2].set_ylabel('Samples per second')
axs_combined[2].set_title('HMC Performance Comparison')
axs_combined[2].set_xlabel('Number of devices')
axs_combined[2].legend()

for ax in axs_combined:
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

plt.suptitle('Performance (Not including 1st epoch for MAP, SVI)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# %%
