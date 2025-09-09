"""
SGD Learning Rate Sweep for GigaLens

This script performs a systematic sweep of SGD learning rates to find the
optimal configuration that avoids NaN/Inf issues in SVI while maintaining
good convergence.

Usage:
    python sgd_lr_sweep.py --min_lr 1e-5 --max_lr 1e-1 --num_points 10
"""

import sys
import os
from os.path import expanduser
import time
import optax
import numpy as np
import matplotlib.pyplot as plt

# Setup paths and imports (same as main script)
code_version = "Harry"
home = expanduser("~/")
if code_version == "Harry":
    srcdir = os.path.join(home, 'gigalens/src/')
elif code_version == "Nico":
    srcdir = os.path.join(home, "gigalens-multinode/gigalens_hackathon/src/")

sys.path.insert(0, srcdir)
sys.path.insert(0, home+'/GIGALens-Code')

from optimizer_testing import run_sgd_test, analyze_sgd_convergence_issues
import helpers

def sgd_lr_sweep(min_lr=1e-5, max_lr=1e-1, num_points=10, include_momentum=True):
    """
    Perform a systematic sweep of SGD learning rates.
    
    Args:
        min_lr (float): Minimum learning rate to test
        max_lr (float): Maximum learning rate to test  
        num_points (int): Number of learning rates to test
        include_momentum (bool): Whether to test momentum variants
    """
    
    print("SGD Learning Rate Sweep for GigaLens")
    print("=" * 50)
    print(f"Testing {num_points} learning rates from {min_lr:.0e} to {max_lr:.0e}")
    
    # Generate learning rates (log scale)
    learning_rates = np.logspace(np.log10(min_lr), np.log10(max_lr), num_points)
    
    results = []
    
    # Test basic SGD
    print(f"\n🔍 Testing Basic SGD")
    print("-" * 30)
    
    for i, lr in enumerate(learning_rates):
        print(f"\nTest {i+1}/{num_points}: Learning Rate = {lr:.0e}")
        
        config = {
            'map_optimizer': optax.sgd(learning_rate=lr),
            'svi_optimizer': optax.sgd(learning_rate=lr * 0.1),  # Lower LR for SVI
            'description': f'Basic SGD lr={lr:.0e}'
        }
        
        test_name = f"SGD_lr_{lr:.0e}"
        result, pipeline_result = run_sgd_test(test_name, config, reduced_steps=True)
        
        result['learning_rate'] = lr
        result['optimizer_type'] = 'basic_sgd'
        results.append(result)
        
        # Quick status
        if result['success']:
            svi_status = "NaN" if np.isnan(result['svi_final_loss']) else "Finite"
            print(f"    ✓ MAP: {result['map_final_loss']:.2e}, SVI: {svi_status}")
        else:
            print(f"    ✗ Failed: {result['error']}")
    
    # Test SGD with momentum if requested
    if include_momentum:
        print(f"\n🔍 Testing SGD with Momentum (0.9)")
        print("-" * 30)
        
        for i, lr in enumerate(learning_rates):
            print(f"\nTest {i+1}/{num_points}: Learning Rate = {lr:.0e} (momentum)")
            
            config = {
                'map_optimizer': optax.sgd(learning_rate=lr, momentum=0.9),
                'svi_optimizer': optax.sgd(learning_rate=lr * 0.1, momentum=0.9),
                'description': f'SGD with momentum lr={lr:.0e}'
            }
            
            test_name = f"SGD_momentum_lr_{lr:.0e}"
            result, pipeline_result = run_sgd_test(test_name, config, reduced_steps=True)
            
            result['learning_rate'] = lr
            result['optimizer_type'] = 'momentum_sgd'
            results.append(result)
            
            # Quick status
            if result['success']:
                svi_status = "NaN" if np.isnan(result['svi_final_loss']) else "Finite"
                print(f"    ✓ MAP: {result['map_final_loss']:.2e}, SVI: {svi_status}")
            else:
                print(f"    ✗ Failed: {result['error']}")
    
    return results, learning_rates

def plot_lr_sweep_results(results, learning_rates, save_dir=None):
    """
    Plot the results of the learning rate sweep.
    
    Args:
        results (list): List of test results
        learning_rates (array): Array of learning rates tested
        save_dir (str, optional): Directory to save plots
    """
    
    # Separate basic SGD and momentum SGD results
    basic_results = [r for r in results if r['optimizer_type'] == 'basic_sgd']
    momentum_results = [r for r in results if r['optimizer_type'] == 'momentum_sgd']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: SVI Success Rate vs Learning Rate
    basic_lrs = [r['learning_rate'] for r in basic_results]
    basic_svi_success = [1 if (r['success'] and np.isfinite(r['svi_final_loss'])) else 0 
                        for r in basic_results]
    
    axes[0, 0].semilogx(basic_lrs, basic_svi_success, 'bo-', label='Basic SGD', alpha=0.7)
    
    if momentum_results:
        momentum_lrs = [r['learning_rate'] for r in momentum_results]
        momentum_svi_success = [1 if (r['success'] and np.isfinite(r['svi_final_loss'])) else 0 
                               for r in momentum_results]
        axes[0, 0].semilogx(momentum_lrs, momentum_svi_success, 'ro-', label='SGD + Momentum', alpha=0.7)
    
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('SVI Success Rate')
    axes[0, 0].set_title('SVI Convergence Success vs Learning Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    # Plot 2: MAP Final Loss vs Learning Rate (for successful runs)
    successful_basic = [r for r in basic_results if r['success'] and np.isfinite(r['map_final_loss'])]
    if successful_basic:
        basic_map_lrs = [r['learning_rate'] for r in successful_basic]
        basic_map_losses = [r['map_final_loss'] for r in successful_basic]
        axes[0, 1].loglog(basic_map_lrs, basic_map_losses, 'bo-', label='Basic SGD', alpha=0.7)
    
    if momentum_results:
        successful_momentum = [r for r in momentum_results if r['success'] and np.isfinite(r['map_final_loss'])]
        if successful_momentum:
            momentum_map_lrs = [r['learning_rate'] for r in successful_momentum]
            momentum_map_losses = [r['map_final_loss'] for r in successful_momentum]
            axes[0, 1].loglog(momentum_map_lrs, momentum_map_losses, 'ro-', label='SGD + Momentum', alpha=0.7)
    
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('MAP Final Loss')
    axes[0, 1].set_title('MAP Convergence vs Learning Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: SVI Final Loss vs Learning Rate (for successful, finite runs)
    successful_basic_svi = [r for r in basic_results if r['success'] and np.isfinite(r['svi_final_loss'])]
    if successful_basic_svi:
        basic_svi_lrs = [r['learning_rate'] for r in successful_basic_svi]
        basic_svi_losses = [r['svi_final_loss'] for r in successful_basic_svi]
        axes[1, 0].loglog(basic_svi_lrs, basic_svi_losses, 'bo-', label='Basic SGD', alpha=0.7)
    
    if momentum_results:
        successful_momentum_svi = [r for r in momentum_results if r['success'] and np.isfinite(r['svi_final_loss'])]
        if successful_momentum_svi:
            momentum_svi_lrs = [r['learning_rate'] for r in successful_momentum_svi]
            momentum_svi_losses = [r['svi_final_loss'] for r in successful_momentum_svi]
            axes[1, 0].loglog(momentum_svi_lrs, momentum_svi_losses, 'ro-', label='SGD + Momentum', alpha=0.7)
    
    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('SVI Final Loss')
    axes[1, 0].set_title('SVI Final Loss vs Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Timing vs Learning Rate
    basic_times = [r['total_time'] for r in basic_results if r['success']]
    basic_time_lrs = [r['learning_rate'] for r in basic_results if r['success']]
    if basic_times:
        axes[1, 1].semilogx(basic_time_lrs, basic_times, 'bo-', label='Basic SGD', alpha=0.7)
    
    if momentum_results:
        momentum_times = [r['total_time'] for r in momentum_results if r['success']]
        momentum_time_lrs = [r['learning_rate'] for r in momentum_results if r['success']]
        if momentum_times:
            axes[1, 1].semilogx(momentum_time_lrs, momentum_times, 'ro-', label='SGD + Momentum', alpha=0.7)
    
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('Total Time (seconds)')
    axes[1, 1].set_title('Execution Time vs Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(save_dir, f"sgd_lr_sweep_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Learning rate sweep plots saved to: {plot_file}")
    
    plt.show()

def analyze_lr_sweep_results(results):
    """
    Analyze the learning rate sweep results and provide recommendations.
    
    Args:
        results (list): List of test results
    """
    
    print(f"\n{'='*60}")
    print("LEARNING RATE SWEEP ANALYSIS")
    print(f"{'='*60}")
    
    # Separate by optimizer type
    basic_results = [r for r in results if r['optimizer_type'] == 'basic_sgd']
    momentum_results = [r for r in results if r['optimizer_type'] == 'momentum_sgd']
    
    # Analysis for basic SGD
    print(f"\n📊 Basic SGD Analysis:")
    successful_basic = [r for r in basic_results if r['success']]
    svi_finite_basic = [r for r in successful_basic if np.isfinite(r['svi_final_loss'])]
    
    print(f"  Success rate: {len(successful_basic)}/{len(basic_results)} ({len(successful_basic)/len(basic_results)*100:.1f}%)")
    print(f"  SVI finite rate: {len(svi_finite_basic)}/{len(successful_basic)} ({len(svi_finite_basic)/len(successful_basic)*100:.1f}% of successful)")
    
    if svi_finite_basic:
        best_svi_basic = min(svi_finite_basic, key=lambda x: x['svi_final_loss'])
        print(f"  Best SVI result: lr={best_svi_basic['learning_rate']:.0e}, loss={best_svi_basic['svi_final_loss']:.4e}")
        
        # Find safe learning rate range
        safe_lrs = [r['learning_rate'] for r in svi_finite_basic]
        print(f"  Safe LR range: {min(safe_lrs):.0e} to {max(safe_lrs):.0e}")
    
    # Analysis for momentum SGD
    if momentum_results:
        print(f"\n📊 SGD + Momentum Analysis:")
        successful_momentum = [r for r in momentum_results if r['success']]
        svi_finite_momentum = [r for r in successful_momentum if np.isfinite(r['svi_final_loss'])]
        
        print(f"  Success rate: {len(successful_momentum)}/{len(momentum_results)} ({len(successful_momentum)/len(momentum_results)*100:.1f}%)")
        print(f"  SVI finite rate: {len(svi_finite_momentum)}/{len(successful_momentum)} ({len(svi_finite_momentum)/len(successful_momentum)*100:.1f}% of successful)")
        
        if svi_finite_momentum:
            best_svi_momentum = min(svi_finite_momentum, key=lambda x: x['svi_final_loss'])
            print(f"  Best SVI result: lr={best_svi_momentum['learning_rate']:.0e}, loss={best_svi_momentum['svi_final_loss']:.4e}")
            
            # Find safe learning rate range
            safe_lrs_momentum = [r['learning_rate'] for r in svi_finite_momentum]
            print(f"  Safe LR range: {min(safe_lrs_momentum):.0e} to {max(safe_lrs_momentum):.0e}")
    
    # Overall recommendations
    print(f"\n💡 Recommendations:")
    
    all_finite = svi_finite_basic + (svi_finite_momentum if momentum_results else [])
    if all_finite:
        overall_best = min(all_finite, key=lambda x: x['svi_final_loss'])
        optimizer_type = "Basic SGD" if overall_best['optimizer_type'] == 'basic_sgd' else "SGD + Momentum"
        print(f"  🏆 Best overall: {optimizer_type} with lr={overall_best['learning_rate']:.0e}")
        
        # Safe learning rate recommendation
        all_safe_lrs = [r['learning_rate'] for r in all_finite]
        if len(all_safe_lrs) > 1:
            median_safe_lr = np.median(all_safe_lrs)
            print(f"  🎯 Recommended starting LR: {median_safe_lr:.0e} (median of safe range)")
        
        # Check if momentum helps
        if momentum_results and len(svi_finite_momentum) > len(svi_finite_basic):
            print(f"  ✅ Momentum appears to improve stability")
        elif momentum_results:
            print(f"  ⚖️  Momentum shows mixed results - test case by case")
        
    else:
        print(f"  ❌ No configurations achieved finite SVI convergence")
        print(f"  💡 Try: Lower learning rates (< {min([r['learning_rate'] for r in results]):.0e})")
        print(f"  💡 Try: Gradient clipping or different initialization")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SGD Learning Rate Sweep for GigaLens')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate to test')
    parser.add_argument('--max_lr', type=float, default=1e-1, 
                        help='Maximum learning rate to test')
    parser.add_argument('--num_points', type=int, default=8,
                        help='Number of learning rates to test')
    parser.add_argument('--no_momentum', action='store_true',
                        help='Skip momentum tests')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Run the sweep
    results, learning_rates = sgd_lr_sweep(
        min_lr=args.min_lr,
        max_lr=args.max_lr, 
        num_points=args.num_points,
        include_momentum=not args.no_momentum
    )
    
    # Analyze results
    analyze_lr_sweep_results(results)
    
    # Create plots
    save_dir = args.save_dir or os.path.join(home, "GIGALens-Code/sgd_optimizer_results")
    plot_lr_sweep_results(results, learning_rates, save_dir)
    
    print(f"\nLearning rate sweep complete!") 