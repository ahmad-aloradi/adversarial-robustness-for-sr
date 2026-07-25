# Neural Network Compression Methods

This project implements two complementary approaches for neural network compression to create efficient speaker recognition models:

1. **Bregman Learning Framework** - A sparsity-inducing training approach
2. **Magnitude-Based Pruning** - A classical pruning method with advanced scheduling

---

## 1. Bregman Learning Framework

### Overview

The Bregman learning framework implements sparsity-inducing optimization during training, based on the work by [Bungert et al. (2022)](https://www.jmlr.org/papers/volume23/21-0545/21-0545.pdf). This approach achieves compression by incorporating adaptive regularization that encourages sparse weight distributions while maintaining model performance.

**Reference Implementation:** [TimRoith/BregmanLearning](https://github.com/TimRoith/BregmanLearning/tree/main/notebooks)

### Key Components

#### 1.1 Bregman Optimizers

Located in `src/callbacks/pruning/bregman/bregman_optimizers.py`:

- **LinBreg**: Linear Bregman iteration optimizer
  - Momentum-based variant of standard gradient descent

- **AdaBreg**: Adaptive Bregman iteration optimizer
  - Adam-style adaptive learning rates with Bregman regularization

Both optimizers support **parameter groups** with different regularization strategies.

#### 1.2 Bregman Regularizers

Located in `src/callbacks/pruning/bregman/bregman_regularizers.py` (others are not used):

- **RegNone**: No regularization (standard training)
- **RegL1**: L1 norm regularizer for unstructured sparsity
- **RegL1L2Conv**: Specialized group sparsity for convolutional layers

Each regularizer implements:
- `__call__(x)`: Computes regularization value
- `prox(x, delta)`: Proximal operator for Bregman updates
- `sub_grad(v)`: Subgradient for gradient-based methods

#### 1.3 Lambda Scheduler

Located in `src/callbacks/pruning/bregman/lambda_scheduler.py`:

A feedback controller that adjusts the regularization strength $\lambda$ to drive sparsity toward a target **supplied by `BregmanPruner` each step** ($\lambda$ rises when measured sparsity is below the target, falls when above). The scheduler holds no target of its own, so fixed and progressive targets both work unchanged.

```python
lambda_scheduler:
  _target_: src.callbacks.pruning.bregman.lambda_scheduler.LambdaScheduler
  initial_lambda: 1e-2
  acceleration_factor: 1.0
  update_frequency: 50       # steps between λ updates
```

The target sparsity lives on the pruner (`model_pruning.target_sparsity`); validation is suppressed until the model reaches its band (see the sparsity-gated callbacks).

**Key Parameters:**
- `initial_lambda`
- `target_sparsity`
- `acceleration_factor`: Controls how aggressively λ adapts; any value >= 0, shipped configs use 1.0
- `update_frequency`: Steps between λ updates
- λ trust region (no knob): `BregmanPruner` binds the scheduler to the run, after which each update obeys `|Δλ| ≤ λ0 · (lr/base_lr) · (1 - step/total_steps)`. `λ0` sets the scale (λ grows at most linearly, one λ0 per update), the lr ratio keeps the cap meaningful under warmup/annealing, and the taper bounds λ's total variation (`Σ|Δλ| ≤ λ0·(K+1)/2`) with the increment exactly 0 from the last step on, so λ converges and the tail of training is a fixed-λ Bregman iteration. Logged per step as `bregman/lambda_cap`

The update is defined as: `λ *= 1+a·gap` if s < target and `λ /= 1+a·|gap|` if sparsity > target.

**Note**: Depending on many factors (Bregman optimizer type, `lr` value, `lr_scheduler`, etc.), the target sparsity is not guaranteed to be reached. There is a balancing act between the contribution of the optimzier and regularizer terms in the weights updates.

#### 1.3b Progressive target ramp

`TargetScheduler` (same file) ramps the controller's target from `initial_sparsity` up to `final_sparsity` over `epochs_to_ramp` epochs (log-space by default, `schedule_type: linear` for a straight line), then holds. It is independent of `LambdaScheduler` — it only computes the moving target each epoch, which the feedback loop tracks. The gates and feasibility check still key off the final `target_sparsity`, so validation opens only once the model reaches the final band.

Set it under `model_pruning.target_scheduler` and start the weights dense (`initial_sparsity` = ramp start). Run:

```bash
python src/train.py experiment=img/bregman_adabreg_progressive
```

#### 1.4 Pruning Manager

Located in `src/callbacks/pruning/bregman/utils/pruning_manager.py`:

Manages parameter groups and applies structured/unstructured pruning based on sparsity thresholds (for the algorithm's initalization). The fine0grained control of initilization might be later deprecated and hardcode untructured pruning as it could be an overkill.

#### 1.5 BregmanPruner Callback

Located in `src/callbacks/pruning/bregman/bregman_pruner.py`:

Orchestrates the entire Bregman learning process:
- Initializes pruning manager
- Steps the lambda scheduler
- Logs sparsity metrics
- Synchronizes optimizer parameter groups
- Handles checkpoint save/load

### Usage Example

See `configs/experiment/sv/sv_bregman_adabreg.yaml` for a complete configuration:

```yaml
callbacks:
  model_pruning:
    _target_: src.callbacks.pruning.bregman.bregman_pruner.BregmanPruner
    sparsity_threshold: 1e-12
    verbose: 2
    target_sparsity: 0.9        # controller setpoint + gate band centre
    lambda_scheduler:
      _target_: src.callbacks.pruning.bregman.lambda_scheduler.LambdaScheduler
      _partial_: true
      initial_lambda: 1e-2
      acceleration_factor: 1.0
      update_frequency: 50

module:
  optimizer:
    _target_: src.callbacks.pruning.bregman.bregman_optimizers.AdaBreg
    _partial_: true
    lr: 1e-4

  model:
    pruning_groups:
      # Group 1: Convolutional layers with group sparsity
      - name: conv_layers
        layer_types: ["torch.nn.Conv1d", "torch.nn.Conv2d"]
        param_names: ["weight"]
        module_name_patterns: ['.*conv.*']
        optimizer_settings:
          reg:
            _target_: src.callbacks.pruning.bregman.bregman_regularizers.RegL1
            lamda: 1e-2
          lambda_scale: 1.0
        pruning_config:
          pruning_type: "unstructured"
          sparsity_rate: 0.99

      # Group 2: Linear layers with L1 sparsity
      - name: linear_layers
        layer_types: ["torch.nn.Linear"]
        param_names: ["weight"]
        optimizer_settings:
          reg:
            _target_: src.callbacks.pruning.bregman.bregman_regularizers.RegL1
            lamda: 1e-2
          lambda_scale: 1.0
        pruning_config:
          pruning_type: "unstructured"
          sparsity_rate: 0.99

      # Group 3: Protected layers (no pruning)
      - name: norm_params
        layer_types: ['torch.nn.BatchNorm1d', 'torch.nn.BatchNorm2d', 'torch.nn.LayerNorm']
        module_name_patterns: ['.*norm.*']
        optimizer_settings:
          reg:
            _target_: src.callbacks.pruning.bregman.bregman_regularizers.RegNone
          lambda_scale: 0.0
```

### Training Workflow

1. **Initialization**: Model parameters are assigned to groups based on layer type and name patterns
2. **Training Loop**:
   - Forward pass computes loss
   - Backward pass computes gradients
   - Bregman optimizer applies proximal operator using the regularizer
   - Lambda scheduler adjusts regularization strength based on current sparsity
3. **Sparsity Tracking**: BregmanPruner logs sparsity metrics per group and overall
4. **Checkpoint Handling**: Scheduler state is saved/restored for resuming training
---

## 2. Magnitude-Based Pruning

### Overview

Classical pruning method that removes weights with the smallest magnitudes, either all at once or gradually over training epochs. This implementation includes advanced features like checkpoint compatibility and metric tracker management.

### Key Components

#### 2.1 MagnitudePruner Callback

Located in `src/callbacks/pruning/prune.py`:

Main callback that orchestrates the pruning process with the following features:

- **Pruning Methods**: L1 unstructured, L1 structured, Ln structured
- **Scheduled Pruning**: Gradual sparsity ramping over epochs
- **Global/Local Pruning**: Prune across all parameters or per-layer
- **Permanent Pruning**: Option to make pruning permanent at training end
- **Checkpoint Compatibility**: Handles resumption from pruned checkpoints

**Key Parameters:**
```python
callbacks:
  model_pruning:
    _target_: src.callbacks.pruning.prune.MagnitudePruner
    pruning_fn: "l1_unstructured"    # Pruning strategy
    amount: 0.5                      # 50% final sparsity
    initial_amount: 0.0              # Starting sparsity for scheduled pruning (to be deprecated --> always 0)
    scheduled_pruning: true          # Enable gradual ramping
    schedule_type: "linear"          # The rate of increasing sparsity [linear, constant]
    epochs_to_ramp: 10               # Epochs to reach target sparsity
    use_global_unstructured: true    # Global vs local pruning
    make_pruning_permanent: true     # Fuse masks at training end
    min_param_elements: 100          # Skip layers with small number of parameters
    verbose: 1
```

#### 2.2 Parameter Manager

Located in `src/callbacks/pruning/parameter_manager.py`:

Manages parameter selection and validation using a hybrid strategy:

1. **Allowlist**: Standard prunable layers (Conv, Linear, LSTM, GRU, Embedding)
2. **Blocklist**: Protected layers (BatchNorm, LayerNorm, etc.)
3. **Duck Typing**: Custom layers not in Blocklist with `weight` parameter are considered prunable

Features:
- Automatic parameter discovery
- Size-based filtering (skip small parameters)
- Bias handling (optional)
- Detailed logging of prunable and skipped parameters

#### 2.3 Pruning Scheduler

Located in `src/callbacks/pruning/scheduler.py`:

Implements various sparsity ramping schedules:

- **Linear**: Uniformly increase from initial to final sparsity
- **Constant**: Prune the same amount of weights in each epoch


```python
scheduler = PruningScheduler(
    schedule_type="linear",
    final_sparsity=0.8,
    epochs_to_ramp=20,
    initial_sparsity=0.0
)
```

#### 2.4 Checkpoint Handler

Located in `src/callbacks/pruning/checkpoint_handler.py`:

It has two primary tasks:
1. Handling the saving/loading ckpt when training was interrupted during training (e.g., loading state dict of the model and optimizer).
2. It tracks the pruning schedule and ensures it is resumed from where it stopped.

Both goals ensure seamless loading of pruned checkpoints into unpruned models:

- Detects pruned checkpoints (parameters ending with `_orig`)
- Reconstructs pruning structure with `Identity` masks
- Auto-fuses weights when loading pruned checkpoint into clean model
- Maintains parameter order for optimizer compatibility

### Pruning Workflow

#### Training from Scratch

1. **Epoch Start** (`on_train_epoch_start`):
   - Compute target sparsity from scheduler
   - Remove existing masks (if resuming)
   - Apply pruning to reach target
   - Verify sparsity jump is monotonic

2. **Epoch End** (`on_train_epoch_end`):
   - Log sparsity metrics
   - Manage metric trackers (Early Stopping, Model Checkpoint)
   - Reset trackers during ramp-up phase to avoid premature stopping

3. **Training End** (`on_train_end`):
   - Optionally make pruning permanent (fuse masks into weights)

#### Resuming from Checkpoint

1. **Load Checkpoint** (`on_load_checkpoint`):
   - Restore pruning structure via `PrunedCheckpointHandler`
   - Load scheduler state
   - Reconstruct `Identity` masks for pruned parameters

2. **Continue Training**:
   - Scheduler continues from saved state
   - Sparsity maintained or increased (never decreased)

### Advanced Features

#### Metric Tracker Management

The pruner manages PyTorch Lightning EarlyStopping and ModelCheckpoint callbacks during the ramping phase:

```python
def _manage_metric_trackers(self, trainer, current_sparsity):
    target_reached = current_sparsity >= (self.final_amount - 1e-4)

    if not target_reached:
        # Disable trackers during ramp-up
        for callback in trainer.callbacks:
            if isinstance(callback, (EarlyStopping, ModelCheckpoint)):
                # Reset internal state, reduce save_top_k
    else:
        # Re-enable trackers once target is reached
```

This effectively treats the ramping up phase as a warmup phase and only starts tracking the metrics for early stopping after the warup phase is finished. It also disables saving the `best.ckpt` before reaching the target sparsity

#### Monotonic Sparsity Enforcement

```python
def _verify_sparsity_jump(self, old_sparsity, new_sparsity, applied_amount):
    if old_sparsity > 0.1 and new_sparsity < old_sparsity - 0.05:
        raise RuntimeError(
            f"Pruning Error: Current sparsity ({old_sparsity:.4f}) > "
            f"new sparsity ({new_sparsity:.4f}). Cannot un-prune weights."
        )
```

Ensures weights are never "un-pruned" during training.


---

## Comparison: Bregman vs Magnitude Pruning

| Aspect | Bregman Learning | Magnitude Pruning |
|--------|------------------|-------------------|
| **Approach** | Regularization-based | Weight removal |
| **Timing** | During training | Before/during training |
| **Adaptivity** | Automatic λ adjustment | Manual schedule design |
| **Granularity** | Parameter group control | Global/local options |
| **Flexibility** | Multiple regularizer types | Limited to magnitude |
| **Recommended Use** | Training from scratch | Fine-tuning or iterative pruning |


---

## Troubleshooting

### Bregman Learning

**Issue**: Sparsity not reaching target
- **Solution**: Increase `acceleration_factor`

**Issue**: Training unstable
- **Solution**: Decrease `initial_lambda` or try using a different regularizer (e.g., `RegL1L2` instead of `RegL1`)

**Issue**: Some layers not being pruned
- **Solution**: Check `lambda_scale` in group config (should be > 0)

### Magnitude Pruning

**Issue**: Checkpoint loading fails
- **Solution**: Ensure `PrunedCheckpointHandler` is in callbacks list

**Issue**: Sparsity jumps unexpectedly
- **Solution**: Check for checkpoint resumption issues; verify scheduler state

**Issue**: Early stopping triggered during ramp-up
- **Solution**: Increase `patience` or ensure metric tracker management is enabled

---

## File Structure

```
src/callbacks/pruning/
├── bregman/
│   ├── bregman_optimizers.py      # LinBreg, AdaBreg optimizers
│   ├── bregman_regularizers.py    # L1, L1L2, etc. regularizers
│   ├── bregman_pruner.py          # Main Bregman callback
│   ├── lambda_scheduler.py        # Adaptive λ scheduling
│   └── utils/
│       └── pruning_manager.py     # Parameter group management
├── prune.py                        # MagnitudePruner callback
├── parameter_manager.py            # Parameter selection for magnitude pruning
├── scheduler.py                    # Sparsity scheduling
└── checkpoint_handler.py           # Checkpoint compatibility

scripts/
└── make_pruning_permanent.py      # Post-training weight fusion

configs/experiment/sv/
├── sv_bregman_adabreg.yaml        # Bregman learning config (using adabreg)
└── sv_pruning_magnitude.yaml      # Magnitude pruning config (if exists)
```
