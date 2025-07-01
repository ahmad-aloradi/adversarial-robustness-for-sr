# BregmanPruner: Consistent Sparsity-Based Regularization

This directory contains the implementation of `BregmanPruner`, a PyTorch Lightning callback that applies Bregman divergence-based regularization to encourage sparsity in neural networks.

## Problem Solved

The original issue involved inconsistent sparsity calculations that resulted in training logs showing sparsity swinging between `0.000%` and unrealistic values like `99.765%`. This caused:

- Incorrect lambda scheduler adjustments
- Confusion about actual model sparsity
- Poor interaction between different pruning systems

## Solution

`BregmanPruner` provides:

1. **Consistent Sparsity Calculation**: Accurate measurement across different pruning states
2. **Adaptive Lambda Scheduling**: Automatic adjustment based on current vs target sparsity
3. **Seamless Integration**: Compatible with existing `SafeModelPruning` infrastructure
4. **Robust Error Handling**: Graceful handling of edge cases

## Usage

### Basic Usage

```python
from src.callbacks.pruning.bregman.bregman_pruner import BregmanPruner

# Create the callback
bregman_pruner = BregmanPruner(
    target_sparsity=0.8,      # 80% target sparsity
    initial_lambda=1.0,       # Starting regularization strength
    lambda_update_rate=0.001, # Rate of lambda adjustment
    update_frequency=100,     # Steps between updates
    verbose=1                 # Enable logging
)

# Add to PyTorch Lightning trainer
trainer = pl.Trainer(callbacks=[bregman_pruner])
```

### Integration with Model

In your PyTorch Lightning model:

```python
def training_step(self, batch, batch_idx):
    # Your standard loss calculation
    loss = self.compute_loss(batch)
    
    # Add Bregman regularization
    if self.bregman_pruner is not None:
        reg_loss = self.bregman_pruner.get_regularization_loss(self)
        loss = loss + reg_loss
    
    return loss
```

### Combined with Structured Pruning

```python
from src.callbacks.prune import SafeModelPruning

callbacks = [
    # Structured pruning (removes weights)
    SafeModelPruning(
        amount=0.2,
        scheduled_pruning=True,
        final_amount=0.6,
        epochs_to_ramp=10
    ),
    # Bregman regularization (encourages sparsity)
    BregmanPruner(
        target_sparsity=0.8,
        lambda_update_rate=0.001
    )
]

trainer = pl.Trainer(callbacks=callbacks)
```

## Key Features

### Accurate Sparsity Measurement

- Handles PyTorch pruning masks correctly
- Counts natural zeros in parameters
- Consistent results across multiple calls
- Proper parameter validation and caching

### Adaptive Lambda Scheduling

- Increases lambda when sparsity < target
- Decreases lambda when sparsity ≥ target
- Configurable update rate and frequency
- Bounded values to prevent instability

### Comprehensive Logging

Example logs produced:

```
[2025-07-01 16:14:43,870] Epoch 0: Sparsity of pruned modules = 12.543%
[2025-07-01 16:14:48,242] Sparsity 12.543% vs target 80.0% → Lambda ↗ 1.00000000 → 1.00067457
[2025-07-01 16:14:48,243] Step 100: Sparsity=12.543%, lambda=1.0007
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_sparsity` | float | 0.8 | Target sparsity ratio (0.0 to 1.0) |
| `initial_lambda` | float | 1.0 | Initial regularization strength |
| `lambda_update_rate` | float | 0.001 | Rate at which lambda is updated |
| `update_frequency` | int | 100 | Steps between lambda updates |
| `parameters_to_prune` | List[Tuple] | None | Specific parameters (auto-discover if None) |
| `verbose` | int | 0 | Verbosity level (0=silent, 1=basic, 2=detailed) |

## Testing

The implementation includes comprehensive tests covering:

- Basic functionality and parameter validation
- Sparsity calculation accuracy with various pruning states
- Lambda scheduling behavior
- Integration with SafeModelPruning
- Edge cases and error conditions

Run tests with:
```bash
python -m pytest tests/test_bregman_pruner.py -v
```

## Example

See `examples/bregman_pruner_usage.py` for a complete example of using BregmanPruner in a training pipeline.

## Troubleshooting

### Common Issues

1. **Sparsity shows 0.000% consistently**
   - ✅ Fixed: This was the original issue, now resolved
   - Verify parameters are being discovered correctly
   - Check if any actual pruning/sparsity is present

2. **Lambda grows too quickly**
   - Reduce `lambda_update_rate`
   - Increase `update_frequency`

3. **Incompatible with existing pruning**
   - Ensure both callbacks use same `parameters_to_prune` list
   - BregmanPruner is compatible with PyTorch's pruning masks

### Debugging

Enable verbose logging to see detailed sparsity calculations:

```python
bregman_pruner = BregmanPruner(verbose=2)
```

This will show parameter-by-parameter sparsity information and lambda adjustment details.