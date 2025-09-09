# SGD Optimizer Testing for GigaLens

This directory contains scripts for testing Stochastic Gradient Descent (SGD) optimizers from the Optax library as alternatives to the default Adam optimizer in the GigaLens inference pipeline.

## Scripts Overview

### 1. `optimizer_testing.py` - Comprehensive Testing Suite

This is the main testing script that runs extensive tests of various SGD configurations:

**Features:**
- Tests multiple SGD variants: basic SGD, SGD with momentum, SGD with Nesterov momentum
- Tests different learning rates and learning rate schedules
- Includes gradient clipping configurations
- Compares against default Adam optimizer as baseline
- Saves detailed results to JSON and CSV files
- Generates comparison plots
- Provides comprehensive performance metrics

**SGD Configurations Tested:**
- Basic SGD with learning rates: [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
- SGD with momentum: [0.9, 0.95, 0.99]
- SGD with Nesterov momentum: [0.9, 0.95]
- SGD with polynomial decay learning rate schedule
- SGD with gradient clipping
- Default Adam optimizer (baseline)

### 2. `sgd_quick_test.py` - Quick Testing Script

A simplified version for quick testing and custom configurations:

**Features:**
- Quick test mode with 4 predefined configurations
- Custom test mode for specific SGD parameters
- Command-line interface for easy usage
- Faster execution with reduced steps

## Usage

### Running the Full Test Suite

```bash
cd GIGALens-Code
python optimizer_testing.py
```

This will:
1. Test all SGD configurations (~14 different setups)
2. Save results to `sgd_optimizer_results/` directory
3. Generate comparison plots
4. Print summary of best performing configurations

### Running Quick Tests

```bash
# Quick test with predefined configurations
python sgd_quick_test.py

# Test specific SGD configuration
python sgd_quick_test.py --mode custom --lr 0.01 --momentum 0.9
python sgd_quick_test.py --mode custom --lr 0.02 --momentum 0.95 --nesterov
```

### Configuration Options

You can modify the `code_version` variable in both scripts:
- `"Harry"` - Uses Harry's 2025 multinode implementation
- `"Nico"` - Uses Nico's 2024 multinode implementation

## Output Files

The testing generates several output files:

### Results Directory: `sgd_optimizer_results/`

1. **JSON Results** (`sgd_optimizer_test_results_TIMESTAMP.json`)
   - Detailed results for each optimizer configuration
   - Includes timing, loss values, convergence metrics, and error information

2. **CSV Summary** (`sgd_optimizer_summary_TIMESTAMP.csv`)
   - Tabular summary suitable for spreadsheet analysis
   - Easy to import into data analysis tools

3. **Comparison Plots** (`sgd_optimizer_comparison_TIMESTAMP.png`)
   - 4-panel visualization comparing:
     - MAP phase timing
     - SVI phase timing  
     - MAP final loss values
     - SVI final loss values

## Performance Metrics

Each test measures:

- **Timing**: MAP time, SVI time, HMC time, total time
- **Convergence**: Final loss values for MAP and SVI phases
- **Stability**: Convergence rates and error handling
- **Success Rate**: Which configurations complete successfully

## Understanding the Results

### What to Look For

1. **Final Loss Values**: Lower values indicate better convergence to optimal solutions
2. **Timing**: Faster execution while maintaining good convergence
3. **Stability**: Configurations that consistently complete without errors
4. **Convergence Rate**: How quickly the loss decreases during optimization

### Expected Behavior

- **SGD vs Adam**: SGD may converge slower but can be more stable
- **Momentum**: Usually improves convergence speed and stability
- **Nesterov**: Often provides better convergence than standard momentum
- **Learning Rate**: Higher rates converge faster but may be unstable

## Customization

### Adding New Configurations

To test additional SGD configurations, modify the `create_sgd_optimizers()` function in `optimizer_testing.py`:

```python
# Example: Add SGD with weight decay
optimizers["SGD_weight_decay"] = {
    'map_optimizer': optax.chain(
        optax.sgd(learning_rate=1e-2, momentum=0.9),
        optax.add_decayed_weights(weight_decay=1e-4)
    ),
    'svi_optimizer': optax.chain(
        optax.sgd(learning_rate=1e-3, momentum=0.9),
        optax.add_decayed_weights(weight_decay=1e-4)
    ),
    'description': 'SGD with momentum and weight decay'
}
```

### Adjusting Test Parameters

Modify these parameters in the scripts for different testing scenarios:

- `reduced_steps=True/False`: Use faster/slower testing
- `map_steps`, `svi_steps`: Number of optimization steps
- `n_samples`, `n_vi`: Number of particles/samples
- Learning rate ranges and schedules

## Technical Notes

### Optimizer Integration

The SGD optimizers are integrated into the GigaLens pipeline through the `PipelineConfig` class:

```python
pipeline_config = PipelineConfig(
    map_optimizer=optax.sgd(learning_rate=1e-2, momentum=0.9),
    svi_optimizer=optax.sgd(learning_rate=1e-3, momentum=0.9),
    # ... other parameters
)
```

### Default vs Custom Optimizers

- When `map_optimizer=None` or `svi_optimizer=None`, the default Adam optimizers are used
- Default Adam uses polynomial scheduling: `init_value=-1e-2, end_value=-1e-2/3` for MAP
- Default Adam for SVI uses: `init_value=-1e-6, end_value=-3e-3`

### Memory and Performance

- SGD typically uses less memory than Adam (no second moment accumulation)
- SGD may require more iterations to converge
- Momentum variants add minimal memory overhead
- The testing uses reduced problem sizes for faster execution

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure GigaLens is properly installed and paths are correct
2. **Memory Issues**: Reduce `n_samples` or `n_vi` if running out of memory
3. **Convergence Failures**: Try lower learning rates or add momentum
4. **JAX Distributed**: May need to modify JAX initialization for your cluster

### Error Handling

The scripts include comprehensive error handling:
- Failed tests are recorded with error messages
- Successful tests continue even if some fail
- Results are saved even for partially completed test runs

## Example Results Interpretation

```
Best performing configurations:
  MAP (lowest loss): SGD_momentum_0.9 - 1.23e+02
  SVI (lowest loss): SGD_nesterov_0.95 - 4.56e-01
  Fastest: SGD_lr_5e-02 - 145.2s
```

This indicates:
- SGD with momentum=0.9 achieved the best MAP convergence
- SGD with Nesterov momentum=0.95 achieved the best SVI convergence  
- SGD with learning rate 5e-2 was fastest (though may have sacrificed accuracy)

Choose the configuration that best balances your needs for accuracy, speed, and stability. 