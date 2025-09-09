#%%
"""
Optimizer Testing Script: SGD from Optax

This script tests various configurations of Stochastic Gradient Descent (SGD) 
from the optax library as optimizers for both MAP and SVI phases in the 
GigaLens inference pipeline.

The script tests:
1. Basic SGD with different learning rates
2. SGD with momentum
3. SGD with Nesterov momentum
4. Comparison with default Adam optimizer

Results are saved and compared to establish baseline performance.
"""

import corner
import time
import os
import sys
import json
import numpy as np
from datetime import datetime
from os.path import expanduser
import matplotlib.pyplot as plt

# Setup code version and paths
code_version = "Harry"  # Change to "Nico" if using Nico's implementation

home = expanduser("~/")
if code_version == "Harry":
    srcdir = os.path.join(home, 'gigalens/src/')
elif code_version == "Nico":
    srcdir = os.path.join(home, "gigalens-multinode/gigalens_hackathon/src/")
else:
    raise ValueError(f"Invalid code version: {code_version}")

sys.path.insert(0, srcdir)
sys.path.insert(0, home+'/GIGALens-Code')
print(f'{code_version} GIGALENS IMPLEMENTATION')

import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from jax import random

# Import GigaLens components
if code_version == "Harry":
    from gigalens.jax.inference import HarryModellingSequence
elif code_version == "Nico":
    from gigalens.jax.inference import ModellingSequenceMultinode

from gigalens.jax.model import ForwardProbModel, BackwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

tfd = tfp.distributions

# Import helper functions
import helpers
from helpers import *

# Initialize JAX distributed if available
try:
    jax.distributed.initialize()
    print(f"JAX distributed initialized with {jax.device_count()} devices")
except:
    print("Running without JAX distributed initialization")

# Setup demo data and models
kernel = np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)
observed_img = np.load(os.path.join(srcdir, 'gigalens/assets/demo.npy'))

prior = helpers.make_default_prior()
phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])
prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=1, kernel=kernel)

# Initialize the modeling sequence
if code_version == "Harry":
    model_seq = HarryModellingSequence(phys_model, prob_model, sim_config)
elif code_version == "Nico":
    model_seq = ModellingSequenceMultinode(phys_model, prob_model, sim_config)

print("Setup complete. Beginning SGD optimizer testing...")

#%%
# Define SGD optimizer configurations to test
def create_sgd_optimizers():
    """
    Create different SGD optimizer configurations for testing.
    
    Returns:
        dict: Dictionary of optimizer configurations with descriptive names
    """
    
    optimizers = {}
    
    # Basic SGD with different learning rates
    learning_rates = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    
    for lr in learning_rates:
        optimizers[f"SGD_lr_{lr:.0e}"] = {
            'map_optimizer': optax.sgd(learning_rate=lr),
            'svi_optimizer': optax.sgd(learning_rate=lr * 0.1),  # Lower LR for SVI
            'description': f'Basic SGD with MAP lr={lr:.0e}, SVI lr={lr*0.1:.0e}'
        }
    
    # SGD with momentum
    momentum_values = [0.9, 0.95, 0.99]
    base_lr = 1e-2
    
    for momentum in momentum_values:
        optimizers[f"SGD_momentum_{momentum}"] = {
            'map_optimizer': optax.sgd(learning_rate=base_lr, momentum=momentum),
            'svi_optimizer': optax.sgd(learning_rate=base_lr * 0.1, momentum=momentum),
            'description': f'SGD with momentum={momentum}, MAP lr={base_lr:.0e}'
        }
    
    # SGD with Nesterov momentum
    for momentum in [0.9, 0.95]:
        optimizers[f"SGD_nesterov_{momentum}"] = {
            'map_optimizer': optax.sgd(learning_rate=base_lr, momentum=momentum, nesterov=True),
            'svi_optimizer': optax.sgd(learning_rate=base_lr * 0.1, momentum=momentum, nesterov=True),
            'description': f'SGD with Nesterov momentum={momentum}, MAP lr={base_lr:.0e}'
        }
    
    # SGD with learning rate schedules
    # Polynomial decay schedule
    schedule_fn_map = optax.polynomial_schedule(
        init_value=base_lr, end_value=base_lr/3, power=0.5, transition_steps=200
    )
    schedule_fn_svi = optax.polynomial_schedule(
        init_value=base_lr * 0.1, end_value=base_lr * 0.1/3, power=2, transition_steps=750
    )
    
    optimizers["SGD_scheduled"] = {
        'map_optimizer': optax.sgd(learning_rate=schedule_fn_map, momentum=0.9),
        'svi_optimizer': optax.sgd(learning_rate=schedule_fn_svi, momentum=0.9),
        'description': 'SGD with polynomial decay schedule and momentum=0.9'
    }
    
    # Chained SGD with gradient clipping
    optimizers["SGD_clipped"] = {
        'map_optimizer': optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(learning_rate=base_lr, momentum=0.9)
        ),
        'svi_optimizer': optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.sgd(learning_rate=base_lr * 0.1, momentum=0.9)
        ),
        'description': 'SGD with gradient clipping and momentum=0.9'
    }
    
    # Default Adam for comparison
    optimizers["Adam_default"] = {
        'map_optimizer': None,  # Use default
        'svi_optimizer': None,  # Use default
        'description': 'Default Adam optimizer (baseline)'
    }
    
    return optimizers

#%%
def run_sgd_test(optimizer_name, optimizer_config, reduced_steps=True):
    """
    Run a single SGD test configuration.
    
    Args:
        optimizer_name (str): Name of the optimizer configuration
        optimizer_config (dict): Configuration dictionary with optimizers
        reduced_steps (bool): Whether to use reduced steps for faster testing
        
    Returns:
        dict: Results including timing and convergence metrics
    """
    
    print(f"\n{'='*60}")
    print(f"Testing: {optimizer_name}")
    print(f"Description: {optimizer_config['description']}")
    print(f"{'='*60}")
    
    # Configure reduced steps for faster testing
    if reduced_steps:
        map_steps = 200
        svi_steps = 500
        n_samples = 200
        n_vi = 200
        n_hmc = 20
        hmc_burnin = 100
        hmc_results = 200
    else:
        map_steps = 350
        svi_steps = 1500
        n_samples = 500
        n_vi = 1000
        n_hmc = 50
        hmc_burnin = 250
        hmc_results = 750
    
    # Setup pipeline configuration
    pipeline_config = PipelineConfig(
        steps=["MAP", "SVI", "HMC"],
        map_steps=map_steps,
        map_n_samples=n_samples,
        map_optimizer=optimizer_config['map_optimizer'],
        n_vi=n_vi,
        svi_steps=svi_steps,
        svi_optimizer=optimizer_config['svi_optimizer'],
        hmc_burnin_steps=hmc_burnin,
        hmc_num_results=hmc_results,
        n_hmc=n_hmc
    )
    
    # Run the pipeline
    start_time = time.time()
    try:
        results = run_pipeline(model_seq, pipeline_config)
        total_time = time.time() - start_time
        
        # Extract performance metrics
        map_time = results["MAP"].time_taken
        svi_time = results["SVI"].time_taken  
        hmc_time = results["HMC"].time_taken
        
        # Get final losses
        map_final_loss = results["MAP"].MAP_chisq_hist[-1] if len(results["MAP"].MAP_chisq_hist) > 0 else np.inf
        svi_final_loss = results["SVI"].SVI_loss_hist[-1] if len(results["SVI"].SVI_loss_hist) > 0 else np.inf
        
        # Calculate convergence metrics
        map_convergence = np.mean(np.diff(results["MAP"].MAP_chisq_hist[-10:])) if len(results["MAP"].MAP_chisq_hist) >= 10 else 0
        svi_convergence = np.mean(np.diff(results["SVI"].SVI_loss_hist[-10:])) if len(results["SVI"].SVI_loss_hist) >= 10 else 0
        
        test_results = {
            'optimizer_name': optimizer_name,
            'description': optimizer_config['description'],
            'total_time': total_time,
            'map_time': map_time,
            'svi_time': svi_time,
            'hmc_time': hmc_time,
            'map_final_loss': float(map_final_loss),
            'svi_final_loss': float(svi_final_loss),
            'map_convergence': float(map_convergence),
            'svi_convergence': float(svi_convergence),
            'success': True,
            'error': None
        }
        
        print(f"✓ SUCCESS")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  MAP time: {map_time:.2f}s")
        print(f"  SVI time: {svi_time:.2f}s") 
        print(f"  HMC time: {hmc_time:.2f}s")
        print(f"  MAP final loss: {map_final_loss:.4e}")
        print(f"  SVI final loss: {svi_final_loss:.4e}")
        
        # Add diagnostic analysis for SGD optimizers
        if "SGD" in optimizer_name or np.isnan(svi_final_loss) or np.isinf(svi_final_loss):
            analyze_sgd_convergence_issues(results, optimizer_name)
        
        return test_results, results
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"✗ FAILED after {total_time:.2f}s")
        print(f"  Error: {str(e)}")
        
        test_results = {
            'optimizer_name': optimizer_name,
            'description': optimizer_config['description'],
            'total_time': total_time,
            'map_time': np.nan,
            'svi_time': np.nan,
            'hmc_time': np.nan,
            'map_final_loss': np.inf,
            'svi_final_loss': np.inf,
            'map_convergence': np.nan,
            'svi_convergence': np.nan,
            'success': False,
            'error': str(e)
        }
        
        return test_results, None

#%%
def save_results(all_results, save_dir):
    """
    Save test results to files.
    
    Args:
        all_results (list): List of test result dictionaries
        save_dir (str): Directory to save results
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(save_dir, f"sgd_optimizer_test_results_{timestamp}.json")
    
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {json_file}")
    
    # Create summary CSV
    import pandas as pd
    
    df = pd.DataFrame(all_results)
    csv_file = os.path.join(save_dir, f"sgd_optimizer_summary_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    
    print(f"Summary saved to: {csv_file}")
    
    return json_file, csv_file

#%%
def plot_results(all_results, save_dir):
    """
    Create visualization plots of the test results.
    
    Args:
        all_results (list): List of test result dictionaries
        save_dir (str): Directory to save plots
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Filter successful results
    successful_results = [r for r in all_results if r['success']]
    
    if len(successful_results) == 0:
        print("No successful results to plot")
        return
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    names = [r['optimizer_name'] for r in successful_results]
    map_times = [r['map_time'] for r in successful_results]
    svi_times = [r['svi_time'] for r in successful_results]
    map_losses = [r['map_final_loss'] for r in successful_results]
    svi_losses = [r['svi_final_loss'] for r in successful_results]
    
    # Plot 1: MAP timing comparison
    axes[0, 0].bar(names, map_times, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('MAP Phase Timing Comparison')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: SVI timing comparison  
    axes[0, 1].bar(names, svi_times, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('SVI Phase Timing Comparison')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MAP final loss comparison
    axes[1, 0].bar(names, map_losses, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('MAP Final Loss Comparison')
    axes[1, 0].set_ylabel('Final Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: SVI final loss comparison
    axes[1, 1].bar(names, svi_losses, alpha=0.7, color='plum')
    axes[1, 1].set_title('SVI Final Loss Comparison')
    axes[1, 1].set_ylabel('Final Loss')
    axes[1, 1].set_yscale('log')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(save_dir, f"sgd_optimizer_comparison_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to: {plot_file}")

#%%
def plot_loss_histories(detailed_results, save_dir):
    """
    Create detailed plots of loss histories during optimization.
    
    Args:
        detailed_results (dict): Dictionary of detailed pipeline results
        save_dir (str): Directory to save plots
    """
    
    import matplotlib.pyplot as plt
    
    successful_results = {name: result for name, result in detailed_results.items() 
                         if result is not None}
    
    if len(successful_results) == 0:
        print("No successful results with detailed histories to plot")
        return
    
    # Create subplots - 2 rows (MAP and SVI), multiple columns for different optimizers
    n_optimizers = len(successful_results)
    n_cols = min(4, n_optimizers)  # Max 4 columns for readability
    n_rows = 2 * ((n_optimizers + n_cols - 1) // n_cols)  # 2 rows per set of optimizers
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot MAP and SVI loss histories
    for idx, (opt_name, result) in enumerate(successful_results.items()):
        col = idx % n_cols
        row_map = 2 * (idx // n_cols)
        row_svi = row_map + 1
        
        # MAP loss history
        if row_map < n_rows and hasattr(result["MAP"], 'MAP_chisq_hist'):
            map_hist = result["MAP"].MAP_chisq_hist
            if len(map_hist) > 0:
                axes[row_map, col].plot(map_hist, 'b-', linewidth=2, alpha=0.7)
                axes[row_map, col].set_title(f'{opt_name}\nMAP Loss History')
                axes[row_map, col].set_xlabel('Step')
                axes[row_map, col].set_ylabel('Chi-squared Loss')
                axes[row_map, col].set_yscale('log')
                axes[row_map, col].grid(True, alpha=0.3)
                
                # Add final value annotation
                final_val = map_hist[-1]
                if not np.isnan(final_val) and not np.isinf(final_val):
                    axes[row_map, col].annotate(f'Final: {final_val:.2e}', 
                                               xy=(len(map_hist)-1, final_val),
                                               xytext=(0.7, 0.9), textcoords='axes fraction',
                                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
            else:
                axes[row_map, col].text(0.5, 0.5, 'No MAP history', 
                                       transform=axes[row_map, col].transAxes, ha='center')
                axes[row_map, col].set_title(f'{opt_name}\nMAP Loss History')
        
        # SVI loss history  
        if row_svi < n_rows and hasattr(result["SVI"], 'SVI_loss_hist'):
            svi_hist = result["SVI"].SVI_loss_hist
            if len(svi_hist) > 0:
                axes[row_svi, col].plot(svi_hist, 'r-', linewidth=2, alpha=0.7)
                axes[row_svi, col].set_title(f'{opt_name}\nSVI Loss History')
                axes[row_svi, col].set_xlabel('Step')
                axes[row_svi, col].set_ylabel('ELBO Loss')
                axes[row_svi, col].grid(True, alpha=0.3)
                
                # Check for problematic values
                has_nan = np.any(np.isnan(svi_hist))
                has_inf = np.any(np.isinf(svi_hist))
                
                if has_nan or has_inf:
                    # Find where problems start
                    if has_nan:
                        first_nan = np.where(np.isnan(svi_hist))[0]
                        if len(first_nan) > 0:
                            axes[row_svi, col].axvline(first_nan[0], color='orange', 
                                                      linestyle='--', alpha=0.8, 
                                                      label=f'First NaN at step {first_nan[0]}')
                    if has_inf:
                        first_inf = np.where(np.isinf(svi_hist))[0]
                        if len(first_inf) > 0:
                            axes[row_svi, col].axvline(first_inf[0], color='red', 
                                                      linestyle='--', alpha=0.8,
                                                      label=f'First Inf at step {first_inf[0]}')
                    axes[row_svi, col].legend()
                    
                    # Use log scale only for finite values
                    finite_vals = svi_hist[np.isfinite(svi_hist)]
                    if len(finite_vals) > 0:
                        axes[row_svi, col].set_yscale('log')
                else:
                    axes[row_svi, col].set_yscale('log')
                
                # Add final value annotation
                final_val = svi_hist[-1]
                status = "NaN" if np.isnan(final_val) else "Inf" if np.isinf(final_val) else f"{final_val:.2e}"
                color = "red" if (np.isnan(final_val) or np.isinf(final_val)) else "green"
                axes[row_svi, col].annotate(f'Final: {status}', 
                                           xy=(len(svi_hist)-1, final_val if np.isfinite(final_val) else 1),
                                           xytext=(0.7, 0.9), textcoords='axes fraction',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.5))
            else:
                axes[row_svi, col].text(0.5, 0.5, 'No SVI history', 
                                       transform=axes[row_svi, col].transAxes, ha='center')
                axes[row_svi, col].set_title(f'{opt_name}\nSVI Loss History')
    
    # Hide empty subplots
    total_plots_needed = 2 * len(successful_results)
    total_subplots = n_rows * n_cols
    for idx in range(total_plots_needed, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        if row < n_rows:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(save_dir, f"sgd_loss_histories_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Loss history plots saved to: {plot_file}")

#%%
def plot_convergence_analysis(detailed_results, save_dir):
    """
    Create convergence analysis plots focusing on SGD behavior.
    
    Args:
        detailed_results (dict): Dictionary of detailed pipeline results  
        save_dir (str): Directory to save plots
    """
    
    import matplotlib.pyplot as plt
    
    successful_results = {name: result for name, result in detailed_results.items() 
                         if result is not None}
    
    if len(successful_results) == 0:
        print("No successful results for convergence analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prepare data
    optimizer_names = []
    map_convergence_rates = []
    svi_convergence_rates = []
    svi_final_status = []
    map_final_losses = []
    
    for opt_name, result in successful_results.items():
        optimizer_names.append(opt_name)
        
        # MAP convergence rate (negative mean of last differences indicates convergence)
        if hasattr(result["MAP"], 'MAP_chisq_hist') and len(result["MAP"].MAP_chisq_hist) >= 10:
            map_hist = result["MAP"].MAP_chisq_hist
            map_conv = -np.mean(np.diff(map_hist[-10:]))  # Negative because we want decreasing loss
            map_convergence_rates.append(map_conv)
            map_final_losses.append(map_hist[-1])
        else:
            map_convergence_rates.append(0)
            map_final_losses.append(np.inf)
        
        # SVI analysis
        if hasattr(result["SVI"], 'SVI_loss_hist') and len(result["SVI"].SVI_loss_hist) >= 10:
            svi_hist = result["SVI"].SVI_loss_hist
            
            # Check final status
            final_val = svi_hist[-1]
            if np.isnan(final_val):
                svi_final_status.append('NaN')
                svi_convergence_rates.append(0)
            elif np.isinf(final_val):
                svi_final_status.append('Inf')
                svi_convergence_rates.append(0)
            else:
                svi_final_status.append('Finite')
                # Only compute convergence rate for finite values
                finite_vals = svi_hist[np.isfinite(svi_hist)]
                if len(finite_vals) >= 10:
                    svi_conv = -np.mean(np.diff(finite_vals[-10:]))
                    svi_convergence_rates.append(svi_conv)
                else:
                    svi_convergence_rates.append(0)
        else:
            svi_final_status.append('No Data')
            svi_convergence_rates.append(0)
    
    # Plot 1: MAP convergence rates
    colors_map = ['green' if rate > 0 else 'red' for rate in map_convergence_rates]
    bars1 = axes[0, 0].bar(range(len(optimizer_names)), map_convergence_rates, 
                           color=colors_map, alpha=0.7)
    axes[0, 0].set_title('MAP Convergence Rates\n(Positive = Converging)')
    axes[0, 0].set_ylabel('Convergence Rate')
    axes[0, 0].set_xticks(range(len(optimizer_names)))
    axes[0, 0].set_xticklabels(optimizer_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 2: SVI final status breakdown
    status_counts = {}
    for status in svi_final_status:
        status_counts[status] = status_counts.get(status, 0) + 1
    
    colors_svi = {'Finite': 'green', 'NaN': 'red', 'Inf': 'orange', 'No Data': 'gray'}
    pie_colors = [colors_svi.get(status, 'blue') for status in status_counts.keys()]
    
    axes[0, 1].pie(status_counts.values(), labels=status_counts.keys(), 
                   autopct='%1.0f%%', colors=pie_colors)
    axes[0, 1].set_title('SVI Final Status Distribution')
    
    # Plot 3: SVI convergence rates (only for finite results)
    finite_indices = [i for i, status in enumerate(svi_final_status) if status == 'Finite']
    finite_names = [optimizer_names[i] for i in finite_indices]
    finite_rates = [svi_convergence_rates[i] for i in finite_indices]
    
    if finite_names:
        colors_svi_conv = ['green' if rate > 0 else 'red' for rate in finite_rates]
        bars3 = axes[1, 0].bar(range(len(finite_names)), finite_rates, 
                               color=colors_svi_conv, alpha=0.7)
        axes[1, 0].set_title('SVI Convergence Rates (Finite Results Only)\n(Positive = Converging)')
        axes[1, 0].set_ylabel('Convergence Rate')
        axes[1, 0].set_xticks(range(len(finite_names)))
        axes[1, 0].set_xticklabels(finite_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Finite SVI Results', 
                        transform=axes[1, 0].transAxes, ha='center', va='center')
        axes[1, 0].set_title('SVI Convergence Rates (No Finite Results)')
    
    # Plot 4: MAP final losses (log scale)
    finite_map_losses = [loss for loss in map_final_losses if np.isfinite(loss)]
    if finite_map_losses:
        axes[1, 1].bar(range(len(optimizer_names)), 
                       [loss if np.isfinite(loss) else np.nan for loss in map_final_losses],
                       alpha=0.7, color='skyblue')
        axes[1, 1].set_title('MAP Final Losses')
        axes[1, 1].set_ylabel('Final Loss (log scale)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xticks(range(len(optimizer_names)))
        axes[1, 1].set_xticklabels(optimizer_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Finite MAP Results', 
                        transform=axes[1, 1].transAxes, ha='center', va='center')
        axes[1, 1].set_title('MAP Final Losses (No Finite Results)')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(save_dir, f"sgd_convergence_analysis_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Convergence analysis plots saved to: {plot_file}")

#%%
def analyze_sgd_convergence_issues(result, optimizer_name):
    """
    Analyze common SGD convergence issues and provide diagnostic information.
    
    Args:
        result: Pipeline result dictionary
        optimizer_name (str): Name of the optimizer being tested
    """
    
    print(f"\nDiagnostic Analysis for {optimizer_name}:")
    print("-" * 50)
    
    # MAP Analysis
    if "MAP" in result and hasattr(result["MAP"], 'MAP_chisq_hist'):
        map_hist = result["MAP"].MAP_chisq_hist
        if len(map_hist) > 0:
            print(f"MAP Results:")
            print(f"  Initial loss: {map_hist[0]:.4e}")
            print(f"  Final loss: {map_hist[-1]:.4e}")
            print(f"  Improvement: {(map_hist[0] - map_hist[-1]) / map_hist[0] * 100:.1f}%")
            
            # Check for oscillations
            if len(map_hist) > 10:
                recent_std = np.std(map_hist[-10:])
                recent_mean = np.mean(map_hist[-10:])
                if recent_std / recent_mean > 0.1:
                    print(f"  ⚠️  High oscillation detected (CV={recent_std/recent_mean:.3f})")
                else:
                    print(f"  ✓ Stable convergence (CV={recent_std/recent_mean:.3f})")
    
    # SVI Analysis
    if "SVI" in result and hasattr(result["SVI"], 'SVI_loss_hist'):
        svi_hist = result["SVI"].SVI_loss_hist
        if len(svi_hist) > 0:
            print(f"SVI Results:")
            initial_val = svi_hist[0]
            final_val = svi_hist[-1]
            initial_str = f"{initial_val:.4e}" if np.isfinite(initial_val) else "Non-finite"
            final_str = f"{final_val:.4e}" if np.isfinite(final_val) else "Non-finite"
            print(f"  Initial loss: {initial_str}")
            print(f"  Final loss: {final_str}")
            
            # Find where problems start
            finite_mask = np.isfinite(svi_hist)
            if not np.all(finite_mask):
                first_nonfinite = np.where(~finite_mask)[0]
                if len(first_nonfinite) > 0:
                    print(f"  ❌ Non-finite values start at step {first_nonfinite[0]}/{len(svi_hist)}")
                    
                    # Check what happened before the failure
                    if first_nonfinite[0] > 0:
                        last_finite_idx = first_nonfinite[0] - 1
                        print(f"  Last finite value: {svi_hist[last_finite_idx]:.4e}")
                        
                        # Check for rapid growth
                        if first_nonfinite[0] >= 5:
                            recent_finite = svi_hist[max(0, first_nonfinite[0]-5):first_nonfinite[0]]
                            growth_rate = np.mean(np.diff(recent_finite))
                            if growth_rate > 0:
                                print(f"  ⚠️  Loss was increasing before failure (avg rate: {growth_rate:.4e})")
                            else:
                                print(f"  ✓ Loss was decreasing before failure (avg rate: {growth_rate:.4e})")
            else:
                # All finite - check convergence quality
                if len(svi_hist) > 1:
                    improvement = (svi_hist[0] - svi_hist[-1]) / abs(svi_hist[0]) * 100
                    print(f"  Improvement: {improvement:.1f}%")
                    
                    # Check for oscillations
                    if len(svi_hist) > 10:
                        recent_std = np.std(svi_hist[-10:])
                        recent_mean = np.mean(svi_hist[-10:])
                        if abs(recent_mean) > 1e-10:  # Avoid division by very small numbers
                            cv = recent_std / abs(recent_mean)
                            if cv > 0.1:
                                print(f"  ⚠️  High oscillation detected (CV={cv:.3f})")
                            else:
                                print(f"  ✓ Stable convergence (CV={cv:.3f})")
    
    # Suggest potential fixes for SGD
    print(f"Potential SGD fixes for {optimizer_name}:")
    if "SGD_lr" in optimizer_name:
        lr = float(optimizer_name.split("_")[-1].replace("e-", "e-").replace("e+", "e+"))
        if lr > 1e-2:
            print(f"  💡 Try lower learning rate (current: {lr:.0e}, try: {lr/5:.0e})")
        elif lr < 1e-4:
            print(f"  💡 Try higher learning rate (current: {lr:.0e}, try: {lr*5:.0e})")
    
    if "momentum" not in optimizer_name.lower():
        print(f"  💡 Try adding momentum (e.g., momentum=0.9)")
    
    if "nesterov" not in optimizer_name.lower() and "momentum" in optimizer_name.lower():
        print(f"  💡 Try Nesterov momentum for better convergence")
    
    if "clipped" not in optimizer_name.lower():
        print(f"  💡 Try gradient clipping to prevent instability")
    
    print()

#%%
def main():
    """
    Main function to run all SGD optimizer tests.
    """
    
    print("Starting SGD Optimizer Testing for GigaLens")
    print(f"Code version: {code_version}")
    print(f"JAX devices: {jax.device_count()}")
    
    # Create save directory
    save_dir = os.path.join(home, "GIGALens-Code/sgd_optimizer_results")
    
    # Get optimizer configurations
    optimizers = create_sgd_optimizers()
    
    print(f"\nWill test {len(optimizers)} optimizer configurations:")
    for name, config in optimizers.items():
        print(f"  - {name}: {config['description']}")
    
    # Run tests
    all_results = []
    detailed_results = {}
    
    for optimizer_name, optimizer_config in optimizers.items():
        test_result, pipeline_result = run_sgd_test(
            optimizer_name, 
            optimizer_config, 
            reduced_steps=True  # Set to False for full testing
        )
        
        all_results.append(test_result)
        if pipeline_result is not None:
            detailed_results[optimizer_name] = pipeline_result
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    json_file, csv_file = save_results(all_results, save_dir)
    
    # Create plots
    print(f"\n{'='*60}")
    print("CREATING PLOTS")
    print(f"{'='*60}")
    
    plot_results(all_results, save_dir)
    plot_loss_histories(detailed_results, save_dir)
    plot_convergence_analysis(detailed_results, save_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in all_results if r['success']]
    failed_tests = [r for r in all_results if not r['success']]
    
    print(f"Total tests: {len(all_results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if len(successful_tests) > 0:
        print(f"\nBest performing configurations:")
        
        # Sort by MAP final loss
        best_map = min(successful_tests, key=lambda x: x['map_final_loss'])
        print(f"  MAP (lowest loss): {best_map['optimizer_name']} - {best_map['map_final_loss']:.4e}")
        
        # Sort by SVI final loss  
        best_svi = min(successful_tests, key=lambda x: x['svi_final_loss'])
        print(f"  SVI (lowest loss): {best_svi['optimizer_name']} - {best_svi['svi_final_loss']:.4e}")
        
        # Sort by total time
        fastest = min(successful_tests, key=lambda x: x['total_time'])
        print(f"  Fastest: {fastest['optimizer_name']} - {fastest['total_time']:.2f}s")
    
    if len(failed_tests) > 0:
        print(f"\nFailed configurations:")
        for failed in failed_tests:
            print(f"  - {failed['optimizer_name']}: {failed['error']}")
    
    print(f"\nResults saved in: {save_dir}")
    
    return all_results, detailed_results

#%%
if __name__ == "__main__":
    # Run main function only if this script is executed directly
    results, detailed = main()
    
    print("\n" + "="*60)
    print("SGD OPTIMIZER TESTING COMPLETE")
    print("="*60)
