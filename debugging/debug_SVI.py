import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os

def parse_svi_log(log_file):
    """
    Parse SVI debug log file and extract various metrics.
    
    Returns:
        dict: Dictionary containing arrays of parsed metrics
    """
    data = {
        'step': [],
        'loss': [],
        'grad_mean_norm': [],
        'grad_cov_norm': [],
        'update_mean_norm': [],
        'update_cov_norm': [],
        'mean_norm_before': [],
        'mean_norm_after': [],
        'cov_norm_before': [],
        'cov_norm_after': [],
        'cov_condition_number': [],
        'cov_forward_condition_number': [],
        'lps_mean': [],
        'model_log_prob_mean': [],
        'cov_eigenvalue_min': [],
        'cov_eigenvalue_max': [],
        'large_loss_changes': [],  # Store step numbers where large changes occur
        'timestamps': []
    }
    
    current_step = None
    
    with open(log_file, 'r') as f:
        for line in f:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
            if timestamp_match:
                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
            else:
                timestamp = None
            
            # Parse step number
            step_match = re.search(r'--- STEP (\d+) ---', line)
            if step_match:
                current_step = int(step_match.group(1))
                continue
            
            if current_step is None:
                continue
            
            # Parse loss
            loss_match = re.search(r'UPDATE - loss: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if loss_match:
                data['step'].append(current_step)
                data['loss'].append(float(loss_match.group(1)))
                data['timestamps'].append(timestamp)
                continue
            
            # Parse gradient norms
            grad_mean_match = re.search(r'UPDATE - grad_mean norm: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if grad_mean_match:
                data['grad_mean_norm'].append(float(grad_mean_match.group(1)))
                continue
            
            grad_cov_match = re.search(r'UPDATE - grad_cov norm: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|inf)', line)
            if grad_cov_match:
                val = grad_cov_match.group(1)
                data['grad_cov_norm'].append(float('inf') if val == 'inf' else float(val))
                continue
            
            # Parse update norms
            update_mean_match = re.search(r'UPDATE - update_mean norm: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if update_mean_match:
                data['update_mean_norm'].append(float(update_mean_match.group(1)))
                continue
            
            update_cov_match = re.search(r'UPDATE - update_cov norm: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if update_cov_match:
                data['update_cov_norm'].append(float(update_cov_match.group(1)))
                continue
            
            # Parse before/after norms
            mean_before_match = re.search(rf'STEP {current_step} - Before update - mean norm: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if mean_before_match:
                data['mean_norm_before'].append(float(mean_before_match.group(1)))
                continue
            
            mean_after_match = re.search(rf'STEP {current_step} - After update - mean norm: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if mean_after_match:
                data['mean_norm_after'].append(float(mean_after_match.group(1)))
                continue
            
            cov_before_match = re.search(rf'STEP {current_step} - Before update - cov norm: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if cov_before_match:
                data['cov_norm_before'].append(float(cov_before_match.group(1)))
                continue
            
            cov_after_match = re.search(rf'STEP {current_step} - After update - cov norm: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if cov_after_match:
                data['cov_norm_after'].append(float(cov_after_match.group(1)))
                continue
            
            # Parse condition numbers
            cov_cond_match = re.search(r'ELBO - cov condition number: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if cov_cond_match:
                data['cov_condition_number'].append(float(cov_cond_match.group(1)))
                continue
            
            cov_forward_cond_match = re.search(r'ELBO - cov_forward condition number: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if cov_forward_cond_match:
                data['cov_forward_condition_number'].append(float(cov_forward_cond_match.group(1)))
                continue
            
            # Parse ELBO components
            lps_mean_match = re.search(r'ELBO - lps mean: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if lps_mean_match:
                data['lps_mean'].append(float(lps_mean_match.group(1)))
                continue
            
            model_log_prob_mean_match = re.search(r'ELBO - model_log_prob mean: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if model_log_prob_mean_match:
                data['model_log_prob_mean'].append(float(model_log_prob_mean_match.group(1)))
                continue
            
            # Parse eigenvalues
            eigenval_match = re.search(r'ELBO - cov_eigenvalues min/max: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)/([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if eigenval_match:
                data['cov_eigenvalue_min'].append(float(eigenval_match.group(1)))
                data['cov_eigenvalue_max'].append(float(eigenval_match.group(2)))
                continue
            
            # Parse large loss changes
            large_change_match = re.search(r'Large loss change detected!.*Current: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if large_change_match:
                data['large_loss_changes'].append(current_step)
                continue
    
    # Convert lists to numpy arrays
    for key in data:
        if key != 'large_loss_changes':
            data[key] = np.array(data[key])
    
    return data

def find_loss_blowup_point(data, threshold_factor=10):
    """
    Find the point where loss "blows up" by detecting large jumps.
    
    Args:
        data: Parsed data dictionary
        threshold_factor: Factor to determine what constitutes a "blowup"
    
    Returns:
        int or None: Step number where blowup occurs
    """
    if len(data['loss']) < 2:
        return None
    
    # Calculate loss differences
    loss_abs = np.abs(data['loss'])
    loss_diffs = np.abs(np.diff(loss_abs))
    
    # Find significant jumps (more than threshold_factor times the median difference)
    median_diff = np.median(loss_diffs[:min(50, len(loss_diffs))])  # Use early differences as baseline
    
    # Look for points where loss becomes very large (> 1e10 absolute value)
    large_loss_indices = np.where(loss_abs > 1e10)[0]
    
    if len(large_loss_indices) > 0:
        return data['step'][large_loss_indices[0]]
    
    # Fallback: look for large jumps
    large_jump_indices = np.where(loss_diffs > threshold_factor * median_diff)[0]
    if len(large_jump_indices) > 0:
        return data['step'][large_jump_indices[0] + 1]  # +1 because diff is offset by 1
    
    return None

def plot_svi_metrics(data, output_dir='GIGALens-Code'):
    """
    Create comprehensive plots of SVI metrics over epochs.
    
    Args:
        data: Parsed data dictionary
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find blowup point
    blowup_step = find_loss_blowup_point(data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('SVI Training Metrics Over Epochs', fontsize=16)
    
    # Plot 1: Loss over epochs
    ax = axes[0, 0]
    ax.plot(data['step'], data['loss'], 'b-', linewidth=1, alpha=0.7)
    if blowup_step is not None:
        ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=2, 
                   label=f'Loss Blowup (Step {blowup_step})')
        ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (ELBO)')
    ax.set_title('Loss Over Training Steps')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Covariance eigenvalues (log scale)
    ax = axes[0, 1]
    if len(data['cov_eigenvalue_min']) > 0:
        eigenval_steps = data['step'][:len(data['cov_eigenvalue_min'])]
        ax.plot(eigenval_steps, data['cov_eigenvalue_min'], 'b-', linewidth=1, alpha=0.7, 
                label='Min Eigenvalue')
        ax.plot(eigenval_steps, data['cov_eigenvalue_max'], 'r-', linewidth=1, alpha=0.7, 
                label='Max Eigenvalue')
        ax.legend()
    if blowup_step is not None:
        ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Covariance Eigenvalues (Log Scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norms
    ax = axes[0, 2]
    if len(data['grad_mean_norm']) > 0:
        # Filter out infinite values for plotting
        finite_indices = np.isfinite(data['grad_cov_norm'])
        ax.plot(data['step'][:len(data['grad_mean_norm'])], data['grad_mean_norm'], 
                'g-', label='Mean Grad Norm', alpha=0.7)
        if np.any(finite_indices):
            ax.plot(data['step'][:len(data['grad_cov_norm'])][finite_indices], 
                    data['grad_cov_norm'][finite_indices], 
                    'r-', label='Cov Grad Norm', alpha=0.7)
    if blowup_step is not None:
        ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Update norms
    ax = axes[1, 0]
    if len(data['update_mean_norm']) > 0:
        ax.plot(data['step'][:len(data['update_mean_norm'])], data['update_mean_norm'], 
                'g-', label='Mean Update Norm', alpha=0.7)
        ax.plot(data['step'][:len(data['update_cov_norm'])], data['update_cov_norm'], 
                'r-', label='Cov Update Norm', alpha=0.7)
    if blowup_step is not None:
        ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Update Norm')
    ax.set_title('Parameter Update Norms')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Parameter norms (before/after updates)
    ax = axes[1, 1]
    if len(data['mean_norm_before']) > 0:
        # Handle potential length mismatch between before/after arrays
        min_len_mean = min(len(data['mean_norm_before']), len(data['mean_norm_after']))
        min_len_cov = min(len(data['cov_norm_before']), len(data['cov_norm_after']))
        
        # Use the appropriate step indices
        mean_steps = data['step'][:min_len_mean]
        cov_steps = data['step'][:min_len_cov]
        
        ax.plot(mean_steps, data['mean_norm_before'][:min_len_mean], 
                'b-', label='Mean Norm (Before)', alpha=0.7)
        ax.plot(mean_steps, data['mean_norm_after'][:min_len_mean], 
                'b--', label='Mean Norm (After)', alpha=0.7)
        ax.plot(cov_steps, data['cov_norm_before'][:min_len_cov], 
                'r-', label='Cov Norm (Before)', alpha=0.7)
        ax.plot(cov_steps, data['cov_norm_after'][:min_len_cov], 
                'r--', label='Cov Norm (After)', alpha=0.7)
    if blowup_step is not None:
        ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Parameter Norm')
    ax.set_title('Parameter Norms Before/After Updates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Condition numbers
    ax = axes[1, 2]
    if len(data['cov_condition_number']) > 0:
        cond_steps = data['step'][:len(data['cov_condition_number'])]
        cond_forward_steps = data['step'][:len(data['cov_forward_condition_number'])]
        
        ax.plot(cond_steps, data['cov_condition_number'], 
                'purple', label='Cov Condition Number', alpha=0.7)
        ax.plot(cond_forward_steps, data['cov_forward_condition_number'], 
                'orange', label='Cov Forward Condition Number', alpha=0.7)
    if blowup_step is not None:
        ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Condition Number')
    ax.set_title('Covariance Matrix Condition Numbers')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 7: ELBO components
    ax = axes[2, 0]
    if len(data['lps_mean']) > 0:
        lps_steps = data['step'][:len(data['lps_mean'])]
        model_prob_steps = data['step'][:len(data['model_log_prob_mean'])]
        
        ax.plot(lps_steps, data['lps_mean'], 
                'cyan', label='LPS Mean', alpha=0.7)
        ax.plot(model_prob_steps, data['model_log_prob_mean'], 
                'magenta', label='Model Log Prob Mean', alpha=0.7)
    if blowup_step is not None:
        ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Log Probability')
    ax.set_title('ELBO Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Loss difference (to highlight sudden changes)
    ax = axes[2, 1]
    if len(data['loss']) > 1:
        loss_diff = np.diff(np.abs(data['loss']))
        ax.plot(data['step'][1:], loss_diff, 'brown', alpha=0.7)
        if blowup_step is not None:
            ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=2, 
                       label=f'Loss Blowup (Step {blowup_step})')
            ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('|Loss| Difference')
    ax.set_title('Loss Change Between Steps')
    ax.set_yscale('symlog')  # Symmetric log scale to handle negative values
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Summary statistics
    ax = axes[2, 2]
    ax.text(0.1, 0.9, f'Total Steps: {len(data["step"])}', transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.8, f'Final Loss: {data["loss"][-1]:.2e}' if len(data['loss']) > 0 else 'No loss data', 
            transform=ax.transAxes, fontsize=12)
    if blowup_step is not None:
        ax.text(0.1, 0.7, f'Loss Blowup at Step: {blowup_step}', transform=ax.transAxes, 
                fontsize=12, color='red', weight='bold')
        loss_at_blowup = data['loss'][data['step'] == blowup_step][0] if len(data['loss']) > 0 else 'N/A'
        ax.text(0.1, 0.6, f'Loss at Blowup: {loss_at_blowup:.2e}', transform=ax.transAxes, 
                fontsize=12, color='red')
    else:
        ax.text(0.1, 0.7, 'No Loss Blowup Detected', transform=ax.transAxes, fontsize=12, color='green')
    
    if len(data['large_loss_changes']) > 0:
        ax.text(0.1, 0.5, f'Large Changes at Steps: {data["large_loss_changes"][:5]}', 
                transform=ax.transAxes, fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Training Summary')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'svi_training_metrics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    return fig, axes, blowup_step

def plot_blowup_zoom(data, blowup_step, output_dir='GIGALens-Code', window_size=20):
    """
    Create zoomed-in plots around the loss blowup point.
    
    Args:
        data: Parsed data dictionary
        blowup_step: Step number where blowup occurs
        output_dir: Directory to save plots
        window_size: Number of steps before and after blowup to include
    """
    if blowup_step is None:
        print("No blowup detected, skipping zoom plots.")
        return None, None
    
    # Find the index of the blowup step
    blowup_idx = np.where(data['step'] == blowup_step)[0]
    if len(blowup_idx) == 0:
        print(f"Blowup step {blowup_step} not found in data.")
        return None, None
    
    blowup_idx = blowup_idx[0]
    
    # Define zoom window
    start_idx = max(0, blowup_idx - window_size)
    end_idx = min(len(data['step']), blowup_idx + window_size + 1)
    
    # Create masks for different data arrays (they might have different lengths)
    def get_zoom_mask(array_len):
        array_start = max(0, min(array_len - 1, blowup_idx - window_size))
        array_end = min(array_len, blowup_idx + window_size + 1)
        return slice(array_start, array_end)
    
    # Extract zoomed data
    zoom_steps = data['step'][start_idx:end_idx]
    zoom_loss = data['loss'][start_idx:end_idx]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create zoomed figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f'SVI Training Metrics - Zoomed Around Loss Blowup (Step {blowup_step})', fontsize=16)
    
    # Plot 1: Loss around blowup
    ax = axes[0, 0]
    ax.plot(zoom_steps, zoom_loss, 'b-', linewidth=2, marker='o', markersize=4)
    ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=3, 
               label=f'Loss Blowup (Step {blowup_step})')
    ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (ELBO)')
    ax.set_title('Loss Around Blowup')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Covariance eigenvalues around blowup (log scale)
    ax = axes[0, 1]
    eigenval_mask = get_zoom_mask(len(data['cov_eigenvalue_min']))
    if len(data['cov_eigenvalue_min']) > 0 and eigenval_mask.stop > eigenval_mask.start:
        zoom_eigenval_steps = data['step'][:len(data['cov_eigenvalue_min'])][eigenval_mask]
        zoom_eigenval_min = data['cov_eigenvalue_min'][eigenval_mask]
        zoom_eigenval_max = data['cov_eigenvalue_max'][eigenval_mask]

        # print(zoom_eigenval_steps)
        # print(f"zoom_eigenval_min: {zoom_eigenval_min}")
        
        ax.plot(zoom_eigenval_steps, zoom_eigenval_min, 'b-', linewidth=2, marker='o', 
                label='Min Eigenvalue', markersize=4)
        ax.plot(zoom_eigenval_steps, zoom_eigenval_max, 'r-', linewidth=2, marker='s', 
                label='Max Eigenvalue', markersize=4)
        ax.legend()
    ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Covariance Eigenvalues Around Blowup (Log Scale)')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-16)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norms around blowup
    ax = axes[0, 2]
    grad_mask = get_zoom_mask(len(data['grad_mean_norm']))
    if len(data['grad_mean_norm']) > 0 and grad_mask.stop > grad_mask.start:
        zoom_grad_steps = data['step'][:len(data['grad_mean_norm'])][grad_mask]
        zoom_grad_mean = data['grad_mean_norm'][grad_mask]
        zoom_grad_cov = data['grad_cov_norm'][grad_mask]
        
        ax.plot(zoom_grad_steps, zoom_grad_mean, 'g-', linewidth=2, marker='o', 
                label='Mean Grad Norm', markersize=4)
        
        # Only plot finite cov grad norms
        finite_mask = np.isfinite(zoom_grad_cov)
        if np.any(finite_mask):
            ax.plot(zoom_grad_steps[finite_mask], zoom_grad_cov[finite_mask], 'r-', 
                    linewidth=2, marker='s', label='Cov Grad Norm', markersize=4)
    
    ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms Around Blowup')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Update norms around blowup
    ax = axes[1, 0]
    update_mask = get_zoom_mask(len(data['update_mean_norm']))
    if len(data['update_mean_norm']) > 0 and update_mask.stop > update_mask.start:
        zoom_update_steps = data['step'][:len(data['update_mean_norm'])][update_mask]
        zoom_update_mean = data['update_mean_norm'][update_mask]
        zoom_update_cov = data['update_cov_norm'][update_mask]
        
        ax.plot(zoom_update_steps, zoom_update_mean, 'g-', linewidth=2, marker='o', 
                label='Mean Update Norm', markersize=4)
        ax.plot(zoom_update_steps, zoom_update_cov, 'r-', linewidth=2, marker='s', 
                label='Cov Update Norm', markersize=4)
    
    ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Update Norm')
    ax.set_title('Parameter Update Norms Around Blowup')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Parameter norms around blowup
    ax = axes[1, 1]
    param_mask = get_zoom_mask(len(data['mean_norm_before']))
    if len(data['mean_norm_before']) > 0 and param_mask.stop > param_mask.start:
        # Handle potential length mismatch between before/after arrays
        mean_before_len = len(data['mean_norm_before'])
        mean_after_len = len(data['mean_norm_after'])
        cov_before_len = len(data['cov_norm_before'])
        cov_after_len = len(data['cov_norm_after'])
        
        # Get zoom masks for each array type
        mean_before_mask = get_zoom_mask(mean_before_len)
        mean_after_mask = get_zoom_mask(mean_after_len)
        cov_before_mask = get_zoom_mask(cov_before_len)
        cov_after_mask = get_zoom_mask(cov_after_len)
        
        # Create step arrays for each data type
        if mean_before_mask.stop > mean_before_mask.start:
            zoom_mean_before_steps = data['step'][:mean_before_len][mean_before_mask]
            zoom_mean_before = data['mean_norm_before'][mean_before_mask]
            ax.plot(zoom_mean_before_steps, zoom_mean_before, 'b-', linewidth=2, marker='o', 
                    label='Mean Norm (Before)', markersize=4)
        
        if mean_after_mask.stop > mean_after_mask.start:
            zoom_mean_after_steps = data['step'][:mean_after_len][mean_after_mask]
            zoom_mean_after = data['mean_norm_after'][mean_after_mask]
            ax.plot(zoom_mean_after_steps, zoom_mean_after, 'b--', linewidth=2, marker='o', 
                    label='Mean Norm (After)', markersize=4)
        
        if cov_before_mask.stop > cov_before_mask.start:
            zoom_cov_before_steps = data['step'][:cov_before_len][cov_before_mask]
            zoom_cov_before = data['cov_norm_before'][cov_before_mask]
            ax.plot(zoom_cov_before_steps, zoom_cov_before, 'r-', linewidth=2, marker='s', 
                    label='Cov Norm (Before)', markersize=4)
        
        if cov_after_mask.stop > cov_after_mask.start:
            zoom_cov_after_steps = data['step'][:cov_after_len][cov_after_mask]
            zoom_cov_after = data['cov_norm_after'][cov_after_mask]
            ax.plot(zoom_cov_after_steps, zoom_cov_after, 'r--', linewidth=2, marker='s', 
                    label='Cov Norm (After)', markersize=4)
    
    ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Parameter Norm')
    ax.set_title('Parameter Norms Around Blowup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Condition numbers around blowup
    ax = axes[1, 2]
    cond_mask = get_zoom_mask(len(data['cov_condition_number']))
    if len(data['cov_condition_number']) > 0 and cond_mask.stop > cond_mask.start:
        zoom_cond_steps = data['step'][:len(data['cov_condition_number'])][cond_mask]
        zoom_cov_cond = data['cov_condition_number'][cond_mask]
        zoom_cov_forward_cond = data['cov_forward_condition_number'][cond_mask]
        
        ax.plot(zoom_cond_steps, zoom_cov_cond, 'purple', linewidth=2, marker='o', 
                label='Cov Condition Number', markersize=4)
        ax.plot(zoom_cond_steps, zoom_cov_forward_cond, 'orange', linewidth=2, marker='s', 
                label='Cov Forward Condition Number', markersize=4)
    
    ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Condition Number')
    ax.set_title('Condition Numbers Around Blowup')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 7: ELBO components around blowup
    ax = axes[2, 0]
    elbo_mask = get_zoom_mask(len(data['lps_mean']))
    if len(data['lps_mean']) > 0 and elbo_mask.stop > elbo_mask.start:
        zoom_elbo_steps = data['step'][:len(data['lps_mean'])][elbo_mask]
        zoom_lps_mean = data['lps_mean'][elbo_mask]
        zoom_model_log_prob = data['model_log_prob_mean'][elbo_mask]
        
        ax.plot(zoom_elbo_steps, zoom_lps_mean, 'cyan', linewidth=2, marker='o', 
                label='LPS Mean', markersize=4)
        ax.plot(zoom_elbo_steps, zoom_model_log_prob, 'magenta', linewidth=2, marker='s', 
                label='Model Log Prob Mean', markersize=4)
    
    ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Log Probability')
    ax.set_title('ELBO Components Around Blowup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Loss change around blowup
    ax = axes[2, 1]
    if len(zoom_loss) > 1:
        loss_diff = np.diff(np.abs(zoom_loss))
        ax.plot(zoom_steps[1:], loss_diff, 'brown', linewidth=2, marker='o', markersize=4)
        ax.axvline(x=blowup_step, color='red', linestyle='--', linewidth=3, 
                   label=f'Loss Blowup (Step {blowup_step})')
        ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('|Loss| Difference')
    ax.set_title('Loss Change Around Blowup')
    ax.set_yscale('symlog')
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Detailed summary
    ax = axes[2, 2]
    ax.text(0.1, 0.9, f'Zoom Window: Steps {zoom_steps[0]} to {zoom_steps[-1]}', 
            transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.8, f'Blowup Step: {blowup_step}', transform=ax.transAxes, 
            fontsize=12, color='red', weight='bold')
    
    if len(zoom_loss) > 0:
        loss_before = zoom_loss[zoom_steps < blowup_step]
        loss_after = zoom_loss[zoom_steps >= blowup_step]
        
        if len(loss_before) > 0:
            ax.text(0.1, 0.7, f'Avg Loss Before: {np.mean(loss_before):.2e}', 
                    transform=ax.transAxes, fontsize=10)
        if len(loss_after) > 0:
            ax.text(0.1, 0.6, f'Avg Loss After: {np.mean(loss_after):.2e}', 
                    transform=ax.transAxes, fontsize=10)
            
        loss_at_blowup = zoom_loss[zoom_steps == blowup_step]
        if len(loss_at_blowup) > 0:
            ax.text(0.1, 0.5, f'Loss at Blowup: {loss_at_blowup[0]:.2e}', 
                    transform=ax.transAxes, fontsize=10, color='red')
    
    # Count large loss changes in zoom window
    large_changes_in_window = [step for step in data['large_loss_changes'] 
                              if zoom_steps[0] <= step <= zoom_steps[-1]]
    if large_changes_in_window:
        ax.text(0.1, 0.4, f'Large Changes in Window: {large_changes_in_window}', 
                transform=ax.transAxes, fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Zoom Summary')
    
    plt.tight_layout()
    
    # Save zoomed plot
    output_file = os.path.join(output_dir, f'svi_blowup_zoom_step_{blowup_step}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Zoomed plot saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    return fig, axes

def main():
    parser = argparse.ArgumentParser(description='Plot SVI training metrics from debug log')
    parser.add_argument('log_file', help='Path to the SVI debug log file')
    parser.add_argument('--output-dir', default='plots', help='Directory to save plots (default: plots)')
    parser.add_argument('--threshold', type=float, default=10.0, 
                       help='Threshold factor for detecting loss blowup (default: 10.0)')
    parser.add_argument('--zoom-window', type=int, default=20, 
                       help='Number of steps before and after blowup to include in zoom plots (default: 20)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found!")
        return
    
    print(f"Parsing log file: {args.log_file}")
    data = parse_svi_log(args.log_file)
    
    print(f"Parsed {len(data['step'])} training steps")
    if len(data['step']) == 0:
        print("No training steps found in log file!")
        return
    
    print("Creating plots...")
    fig, axes, blowup_step = plot_svi_metrics(data, args.output_dir)
    
    # Print summary
    if blowup_step is not None:
        print(f"\n*** Loss blowup detected at step {blowup_step} ***")
        loss_at_blowup = data['loss'][data['step'] == blowup_step][0]
        print(f"Loss value at blowup: {loss_at_blowup:.2e}")
    else:
        print("\nNo significant loss blowup detected.")
    
    if len(data['large_loss_changes']) > 0:
        print(f"Large loss changes detected at steps: {data['large_loss_changes']}")

    # Create zoomed-in plots around the loss blowup point
    if blowup_step is not None:
        print(f"\nCreating zoomed plots around blowup (±{args.zoom_window} steps)...")
    plot_blowup_zoom(data, blowup_step, args.output_dir, args.zoom_window)

if __name__ == '__main__':
    main()
