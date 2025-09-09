"""
Quick SGD Optimizer Test for GigaLens

This is a simplified version of the full optimizer_testing.py script.
Use this to quickly test specific SGD configurations or when you want
to run fewer tests for faster results.

Example usage:
    python sgd_quick_test.py
"""

import sys
import os
from os.path import expanduser
import time
import optax
import numpy as np

# Setup paths and imports (same as main script)
code_version = "Harry"
home = expanduser("~/")
if code_version == "Harry":
    srcdir = os.path.join(home, 'gigalens/src/')
elif code_version == "Nico":
    srcdir = os.path.join(home, "gigalens-multinode/gigalens_hackathon/src/")

sys.path.insert(0, srcdir)
sys.path.insert(0, home+'/GIGALens-Code')

# Import the helper functions from the main testing script
from optimizer_testing import (create_sgd_optimizers, run_sgd_test, save_results, 
                              plot_results, plot_loss_histories, analyze_sgd_convergence_issues)
import helpers

def quick_sgd_test():
    """
    Run a quick test of a few selected SGD configurations.
    """
    
    print("Quick SGD Test for GigaLens")
    print("=" * 40)
    
    # Create a subset of optimizers for quick testing
    selected_optimizers = {
        "SGD_basic": {
            'map_optimizer': optax.sgd(learning_rate=1e-2),
            'svi_optimizer': optax.sgd(learning_rate=1e-3),
            'description': 'Basic SGD with lr=1e-2 (MAP) and lr=1e-3 (SVI)'
        },
        "SGD_momentum": {
            'map_optimizer': optax.sgd(learning_rate=1e-2, momentum=0.9),
            'svi_optimizer': optax.sgd(learning_rate=1e-3, momentum=0.9),
            'description': 'SGD with momentum=0.9'
        },
        "SGD_nesterov": {
            'map_optimizer': optax.sgd(learning_rate=1e-2, momentum=0.9, nesterov=True),
            'svi_optimizer': optax.sgd(learning_rate=1e-3, momentum=0.9, nesterov=True),
            'description': 'SGD with Nesterov momentum=0.9'
        },
        "Adam_baseline": {
            'map_optimizer': None,  # Use default Adam
            'svi_optimizer': None,  # Use default Adam
            'description': 'Default Adam optimizer (baseline)'
        }
    }
    
    print(f"Testing {len(selected_optimizers)} configurations:")
    for name, config in selected_optimizers.items():
        print(f"  - {name}: {config['description']}")
    
    # Run tests with reduced steps for speed
    results = []
    
    for optimizer_name, optimizer_config in selected_optimizers.items():
        print(f"\n{'='*50}")
        print(f"Testing: {optimizer_name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        test_result, _ = run_sgd_test(
            optimizer_name, 
            optimizer_config, 
            reduced_steps=False  # Fast testing
        )
        
        results.append(test_result)
        
        if test_result['success']:
            print(f"✓ Completed in {time.time() - start_time:.1f}s")
            print(f"  MAP loss: {test_result['map_final_loss']:.4e}")
            print(f"  SVI loss: {test_result['svi_final_loss']:.4e}")
        else:
            print(f"✗ Failed: {test_result['error']}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("QUICK TEST SUMMARY")
    print(f"{'='*50}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        print(f"\nPerformance comparison:")
        for result in successful:
            print(f"  {result['optimizer_name']:15} | "
                  f"MAP: {result['map_final_loss']:.2e} | "
                  f"SVI: {result['svi_final_loss']:.2e} | "
                  f"Time: {result['total_time']:.1f}s")
        
        # Find best performers
        best_map = min(successful, key=lambda x: x['map_final_loss'])
        best_svi = min(successful, key=lambda x: x['svi_final_loss'])
        fastest = min(successful, key=lambda x: x['total_time'])
        
        print(f"\nBest results:")
        print(f"  Best MAP loss:  {best_map['optimizer_name']} ({best_map['map_final_loss']:.2e})")
        print(f"  Best SVI loss:  {best_svi['optimizer_name']} ({best_svi['svi_final_loss']:.2e})")
        print(f"  Fastest:        {fastest['optimizer_name']} ({fastest['total_time']:.1f}s)")
    
    if failed:
        print(f"\nFailed tests:")
        for result in failed:
            print(f"  {result['optimizer_name']}: {result['error']}")
    
    return results

def test_specific_sgd(learning_rate=1e-2, momentum=None, nesterov=False):
    """
    Test a specific SGD configuration.
    
    Args:
        learning_rate (float): Learning rate for SGD
        momentum (float, optional): Momentum parameter
        nesterov (bool): Whether to use Nesterov momentum
    """
    
    print(f"Testing specific SGD configuration:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Momentum: {momentum}")
    print(f"  Nesterov: {nesterov}")
    
    # Create optimizer configuration
    map_opt = optax.sgd(
        learning_rate=learning_rate, 
        momentum=momentum, 
        nesterov=nesterov
    )
    svi_opt = optax.sgd(
        learning_rate=learning_rate * 0.1,  # Lower LR for SVI
        momentum=momentum, 
        nesterov=nesterov
    )
    
    config = {
        'map_optimizer': map_opt,
        'svi_optimizer': svi_opt,
        'description': f'Custom SGD (lr={learning_rate}, momentum={momentum}, nesterov={nesterov})'
    }
    
    # Run test
    result, pipeline_result = run_sgd_test("Custom_SGD", config, reduced_steps=True)
    
    if result['success']:
        print(f"\n✓ SUCCESS!")
        print(f"  MAP final loss: {result['map_final_loss']:.4e}")
        print(f"  SVI final loss: {result['svi_final_loss']:.4e}")
        print(f"  Total time: {result['total_time']:.2f}s")
        
        # Add diagnostic analysis
        if pipeline_result and (np.isnan(result['svi_final_loss']) or np.isinf(result['svi_final_loss'])):
            analyze_sgd_convergence_issues(pipeline_result, "Custom_SGD")
        
        return result, pipeline_result
    else:
        print(f"\n✗ FAILED: {result['error']}")
        return result, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick SGD testing for GigaLens')
    parser.add_argument('--mode', choices=['quick', 'custom'], default='quick',
                        help='Test mode: quick (predefined configs) or custom')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate for custom test')
    parser.add_argument('--momentum', type=float, default=None,
                        help='Momentum for custom test')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use Nesterov momentum for custom test')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        results = quick_sgd_test()
    else:
        result, detailed = test_specific_sgd(
            learning_rate=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov
        )
    
    print(f"\nTest complete!") 