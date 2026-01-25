# Pruning Reference Implementations

**Domain:** Neural Network Magnitude Pruning for Speaker Verification
**Researched:** 2026-01-25
**Overall Confidence:** HIGH (verified against official PyTorch documentation and peer-reviewed papers)

---

## Reference Implementations

### 1. PyTorch Built-in: `torch.nn.utils.prune` (PRIMARY REFERENCE)

The official PyTorch pruning module is the authoritative reference for magnitude-based pruning.

**Key Functions:**

| Function | Description | Use Case |
|----------|-------------|----------|
| `prune.l1_unstructured` | Prunes weights by L1 magnitude per-layer | Local unstructured pruning |
| `prune.global_unstructured` | Prunes lowest magnitude across entire model | Global unstructured pruning |
| `prune.ln_structured` | Prunes channels/neurons by Ln norm | Structured pruning |
| `prune.remove` | Makes pruning permanent (fuses mask into weights) | Finalization |
| `prune.is_pruned` | Checks if module has pruning hooks | Verification |

**Implementation Details:**
- Stores original weight as `weight_orig` parameter
- Stores binary mask as `weight_mask` buffer
- Uses `forward_pre_hook` to apply mask before each forward pass
- The visible `weight` attribute is computed dynamically as `weight_orig * weight_mask`

**Reference Code for Global L1 Unstructured:**
```python
import torch.nn.utils.prune as prune

parameters_to_prune = [
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,  # 50% sparsity
)
```

**Source:** [PyTorch Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)

### 2. Torch-Pruning Library (STRUCTURAL PRUNING REFERENCE)

For structural pruning that actually removes parameters (not just masks them), Torch-Pruning is the standard.

**Key Difference from PyTorch:**
- PyTorch: Zeroizes parameters via masking (no speedup without sparse kernels)
- Torch-Pruning: Physically removes parameters using DepGraph dependency analysis

**Installation:** `pip install torch-pruning`

**Basic Usage:**
```python
import torch_pruning as tp

DG = tp.DependencyGraph().build_dependency(
    model,
    example_inputs=torch.randn(1, 80, 300)  # Mel spectrogram input
)
group = DG.get_pruning_group(
    model.conv1,
    tp.prune_conv_out_channels,
    idxs=[2, 6, 9]
)
if DG.check_pruning_group(group):
    group.prune()
```

**Source:** [Torch-Pruning GitHub (CVPR 2023)](https://github.com/VainF/Torch-Pruning)

### 3. PyTorch Official Test Suite

The PyTorch repository contains authoritative test cases for verifying pruning correctness.

**Test Reference:** [`pytorch/test/ao/sparsity/test_structured_sparsifier.py`](https://github.com/pytorch/pytorch/blob/main/test/ao/sparsity/test_structured_sparsifier.py)

**Verification Pattern:**
```python
# Set known weights
model.linear1.weight = nn.Parameter(
    torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
)

# Apply pruning (50% sparsity should remove lowest rows)
pruning_config = [{"tensor_fqn": "linear1.weight", "sparsity_level": 0.5}]

# Verify expected outcome
expected = torch.Tensor([[3, 3, 3, 3], [4, 4, 4, 4]])
assert torch.isclose(expected, pruned_model.linear1.weight, rtol=1e-05).all()
```

---

## Expected Results by Sparsity Level

### General Vision Models (ImageNet, CIFAR-10)

Based on peer-reviewed benchmarks from "Is Complexity Required for Neural Network Pruning? A Case Study on Global Magnitude Pruning" (arXiv:2209.14624):

| Sparsity | Model | Dataset | Top-1 Accuracy | Baseline | Degradation |
|----------|-------|---------|----------------|----------|-------------|
| 80% | ResNet-50 | ImageNet | 76.12% | ~76.5% | ~0.4% |
| 90% | ResNet-50 | ImageNet | 74.83% | ~76.5% | ~1.7% |
| 95% | ResNet-50 | ImageNet | 72.14% | ~76.5% | ~4.4% |
| 98% | ResNet-50 | ImageNet | 66.57% | ~76.5% | ~10% |
| 90% | WRN-28-8 | CIFAR-10 | 96.30% | ~96.5% | ~0.2% |
| 95% | WRN-28-8 | CIFAR-10 | 96.16% | ~96.5% | ~0.3% |
| 99.9% | WRN-22-8 | CIFAR-10 | 67.68% | ~96% | ~28% |

**Key Finding:** Global magnitude pruning maintains strong performance up to ~90% sparsity, with graceful degradation until ~95%. Beyond 95%, layer collapse becomes a significant risk.

**Source:** [arXiv:2209.14624v3](https://arxiv.org/html/2209.14624v3)

### Speaker Verification Models

Based on "Hybrid Pruning for Speaker Verification" (arXiv:2508.16232, November 2025):

| Sparsity | Model | Test Set | EER | Baseline EER | Degradation |
|----------|-------|----------|-----|--------------|-------------|
| 0% (baseline) | WavLM-Base | VoxCeleb1-O | 0.70% | 0.70% | - |
| 0% (baseline) | WavLM-Base | VoxCeleb1-H | 1.40% | 1.40% | - |
| 60% | WavLM-Base | VoxCeleb1-O | 0.70% | 0.70% | 0.00% |
| 60% | WavLM-Base | VoxCeleb1-H | 1.50% | 1.40% | 0.10% |
| 70% | WavLM-Base | VoxCeleb1-O | 0.73% | 0.70% | 0.03% |
| 70% | WavLM-Base | VoxCeleb1-H | 1.61% | 1.40% | 0.21% |

**Key Finding:** Speaker verification models can tolerate 60-70% sparsity with minimal EER degradation (<0.25% absolute). This represents a 2.2x CPU / 2.0x GPU speedup at 60% sparsity.

**Source:** [arXiv:2508.16232](https://arxiv.org/html/2508.16232)

### Expected Sparsity-Accuracy Tradeoff Curve

Based on evolutionary pruning research (arXiv:2601.10765, January 2025):

| Target Sparsity | Expected Accuracy Retention |
|-----------------|----------------------------|
| 35% | ~95-96% of baseline |
| 50% | ~88-90% of baseline |
| 70% | ~85-88% of baseline |
| 90% | ~80-85% of baseline |
| 95% | ~70-80% of baseline |

**Source:** [arXiv:2601.10765](https://arxiv.org/abs/2601.10765)

---

## Verification Approach

### Checklist: Is Pruning Working Correctly?

#### 1. Sparsity Calculation Verification

```python
def compute_sparsity(model, parameters_to_prune):
    """Verify sparsity matches expected target."""
    total_zeros = 0
    total_params = 0

    for module, name in parameters_to_prune:
        param = getattr(module, name)
        total_zeros += (param == 0).sum().item()
        total_params += param.numel()

    return total_zeros / total_params

# After pruning to 50%:
actual_sparsity = compute_sparsity(model, params)
assert abs(actual_sparsity - 0.50) < 0.01, f"Expected 50%, got {actual_sparsity:.2%}"
```

#### 2. Mask Verification

```python
import torch.nn.utils.prune as prune

def verify_masks(model, parameters_to_prune):
    """Verify pruning masks are correctly applied."""
    for module, name in parameters_to_prune:
        # Check pruning is applied
        assert prune.is_pruned(module), f"Module {module} not pruned"

        # Check mask exists
        mask_name = f"{name}_mask"
        assert hasattr(module, mask_name), f"No mask buffer {mask_name}"

        mask = getattr(module, mask_name)
        orig = getattr(module, f"{name}_orig")
        weight = getattr(module, name)

        # Verify weight = orig * mask
        expected = orig * mask
        assert torch.allclose(weight, expected), "Weight != orig * mask"

        # Verify mask is binary
        assert ((mask == 0) | (mask == 1)).all(), "Mask not binary"
```

#### 3. Monotonicity Check

```python
def verify_monotonic_sparsity(old_sparsity, new_sparsity, target_sparsity, tol=0.01):
    """Verify sparsity only increases (no un-pruning)."""
    # New sparsity should be >= old (pruning is one-way)
    if new_sparsity < old_sparsity - tol:
        raise RuntimeError(
            f"Sparsity decreased from {old_sparsity:.2%} to {new_sparsity:.2%}"
        )

    # New sparsity should be close to target
    if abs(new_sparsity - target_sparsity) > tol:
        raise RuntimeError(
            f"Sparsity {new_sparsity:.2%} != target {target_sparsity:.2%}"
        )
```

#### 4. Layer Collapse Check

```python
def check_layer_collapse(model, parameters_to_prune, min_active_ratio=0.01):
    """Verify no layer is completely pruned (layer collapse)."""
    for module, name in parameters_to_prune:
        param = getattr(module, name)
        active_ratio = (param != 0).sum().item() / param.numel()

        if active_ratio < min_active_ratio:
            raise RuntimeError(
                f"Layer collapse detected: {module.__class__.__name__}.{name} "
                f"has only {active_ratio:.2%} active weights"
            )
```

#### 5. Gradient Flow Check

```python
def verify_gradient_flow(model, parameters_to_prune, sample_input):
    """Verify gradients flow correctly through pruned network."""
    model.zero_grad()
    output = model(sample_input)
    loss = output.sum()
    loss.backward()

    for module, name in parameters_to_prune:
        # Check gradients exist for original weights
        orig_name = f"{name}_orig"
        if hasattr(module, orig_name):
            orig = getattr(module, orig_name)
            assert orig.grad is not None, f"No gradient for {orig_name}"
```

#### 6. Forward Pass Consistency

```python
def verify_forward_consistency(model, sample_input, parameters_to_prune):
    """Verify pruned model produces consistent outputs."""
    model.eval()
    with torch.no_grad():
        # Run twice - should be identical
        out1 = model(sample_input)
        out2 = model(sample_input)
        assert torch.allclose(out1, out2), "Non-deterministic forward pass"

        # Verify zeros are maintained
        for module, name in parameters_to_prune:
            param = getattr(module, name)
            mask = getattr(module, f"{name}_mask")
            assert (param[mask == 0] == 0).all(), "Pruned weights not zero"
```

### Complete Test Suite Template

```python
import pytest
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class TestMagnitudePruning:
    """Verification tests for magnitude pruning implementation."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model with known weights."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        # Initialize with known values for deterministic testing
        with torch.no_grad():
            model[0].weight.fill_(1.0)
            model[2].weight.fill_(1.0)
        return model

    def test_sparsity_matches_target(self, simple_model):
        """Verify achieved sparsity matches requested target."""
        params = [(simple_model[0], 'weight'), (simple_model[2], 'weight')]

        prune.global_unstructured(
            params,
            pruning_method=prune.L1Unstructured,
            amount=0.5
        )

        total = sum(p.numel() for m, n in params for p in [getattr(m, n)])
        zeros = sum((getattr(m, n) == 0).sum().item() for m, n in params)
        sparsity = zeros / total

        assert abs(sparsity - 0.5) < 0.01

    def test_l1_ordering(self, simple_model):
        """Verify lowest magnitude weights are pruned first."""
        # Set specific weights
        with torch.no_grad():
            simple_model[0].weight = nn.Parameter(
                torch.arange(200, dtype=torch.float).view(20, 10)
            )

        prune.l1_unstructured(simple_model[0], name='weight', amount=0.5)

        # Lower indices (smaller values) should be pruned
        weight = simple_model[0].weight
        mask = simple_model[0].weight_mask

        # Bottom half should be mostly zeros
        bottom_half = mask[:10, :].float().mean()
        top_half = mask[10:, :].float().mean()
        assert bottom_half < top_half

    def test_masks_are_binary(self, simple_model):
        """Verify pruning masks contain only 0s and 1s."""
        prune.l1_unstructured(simple_model[0], name='weight', amount=0.3)

        mask = simple_model[0].weight_mask
        assert ((mask == 0) | (mask == 1)).all()

    def test_remove_makes_permanent(self, simple_model):
        """Verify prune.remove correctly fuses mask into weights."""
        prune.l1_unstructured(simple_model[0], name='weight', amount=0.5)

        weight_before = simple_model[0].weight.clone()
        prune.remove(simple_model[0], 'weight')
        weight_after = simple_model[0].weight

        # Values should match
        assert torch.allclose(weight_before, weight_after)

        # Should no longer be pruned
        assert not prune.is_pruned(simple_model[0])

        # Mask buffer should be gone
        assert not hasattr(simple_model[0], 'weight_mask')

    def test_iterative_pruning_compounds(self, simple_model):
        """Verify iterative pruning compounds correctly."""
        # Prune 20% twice: 1 - (1-0.2)^2 = 0.36 total
        prune.l1_unstructured(simple_model[0], name='weight', amount=0.2)
        prune.l1_unstructured(simple_model[0], name='weight', amount=0.2)

        weight = simple_model[0].weight
        sparsity = (weight == 0).float().mean().item()

        expected = 1 - (1 - 0.2) ** 2  # 0.36
        assert abs(sparsity - expected) < 0.05

    def test_no_layer_collapse(self, simple_model):
        """Verify pruning doesn't completely zero any layer."""
        params = [(simple_model[0], 'weight'), (simple_model[2], 'weight')]

        prune.global_unstructured(
            params,
            pruning_method=prune.L1Unstructured,
            amount=0.9  # High sparsity
        )

        for module, name in params:
            weight = getattr(module, name)
            active = (weight != 0).sum().item()
            assert active > 0, f"Layer collapse: {module}"
```

---

## Common Implementation Bugs

### Bug 1: Incorrect Sparsity Calculation with Masks

**Problem:** Calculating sparsity from `weight` instead of considering the pruning mask structure.

**Wrong:**
```python
# This counts zeros in the computed weight, which is correct
# BUT may not account for the mask correctly during training
sparsity = (model.weight == 0).float().mean()
```

**Correct:**
```python
# Explicitly check mask if it exists
if hasattr(model, 'weight_mask'):
    sparsity = (model.weight_mask == 0).float().mean()
else:
    sparsity = (model.weight == 0).float().mean()
```

**Detection:** Run `prune.is_pruned(module)` to verify pruning state.

### Bug 2: Iterative Pruning Without Compounding Awareness

**Problem:** Applying 20% pruning twice doesn't give 40% sparsity.

**Math:**
- First pruning: removes 20% of weights
- Second pruning: removes 20% of REMAINING weights
- Total sparsity: `1 - (1-0.2) * (1-0.2) = 1 - 0.64 = 0.36` (36%, not 40%)

**Solution:** To achieve exact target sparsity, compute the correct amount:
```python
def get_pruning_amount_for_target(current_sparsity, target_sparsity):
    """Calculate amount to prune to reach target from current state."""
    remaining_current = 1 - current_sparsity
    remaining_target = 1 - target_sparsity

    if remaining_current <= 0:
        return 0.0

    # amount = 1 - (remaining_target / remaining_current)
    return max(0.0, 1 - (remaining_target / remaining_current))
```

### Bug 3: Forgetting to Remove Masks Before Re-applying

**Problem:** Applying pruning to already-pruned modules without removing old masks causes issues.

**Detection:** Check for `PruningContainer` warning in PyTorch logs.

**Solution:**
```python
def apply_pruning_safely(module, name, amount):
    if prune.is_pruned(module):
        prune.remove(module, name)  # Remove old pruning first
    prune.l1_unstructured(module, name=name, amount=amount)
```

### Bug 4: Layer Collapse at High Sparsity

**Problem:** Global pruning can completely zero out small layers.

**Detection:**
```python
for module, name in parameters_to_prune:
    weight = getattr(module, name)
    if (weight == 0).all():
        logger.error(f"Layer collapse: {module}")
```

**Solution:** Use Minimum Threshold technique:
```python
def apply_pruning_with_min_threshold(params, amount, min_weights_per_layer=10):
    """Apply global pruning while preserving minimum weights per layer."""
    # 1. Compute global threshold
    all_weights = torch.cat([getattr(m, n).abs().flatten() for m, n in params])
    k = int(all_weights.numel() * amount)
    threshold = torch.kthvalue(all_weights, k).values

    # 2. Apply per-layer with minimum preservation
    for module, name in params:
        weight = getattr(module, name)
        layer_mask = weight.abs() > threshold

        # Ensure minimum weights survive
        if layer_mask.sum() < min_weights_per_layer:
            topk_indices = weight.abs().flatten().topk(min_weights_per_layer).indices
            layer_mask.flatten()[topk_indices] = True

        # Apply custom mask
        prune.custom_from_mask(module, name, layer_mask)
```

### Bug 5: Checkpoint Incompatibility

**Problem:** Loading pruned checkpoint into unpruned model (or vice versa) fails.

**Detection:** Look for `weight_orig` and `weight_mask` in state_dict keys.

**Solution:** Implement checkpoint handler (already in your codebase):
```python
def is_pruned_checkpoint(state_dict):
    return any(k.endswith('_orig') for k in state_dict.keys())

def load_checkpoint_safely(model, state_dict):
    if is_pruned_checkpoint(state_dict):
        # Reconstruct pruning structure before loading
        for key in list(state_dict.keys()):
            if key.endswith('_orig'):
                base = key[:-5]
                module_path, param = base.rsplit('.', 1)
                module = get_module_by_path(model, module_path)
                # Apply identity mask to create structure
                prune.identity(module, param)
    model.load_state_dict(state_dict)
```

### Bug 6: Not Handling Bias Correctly

**Problem:** Pruning weights but leaving bias untouched leads to inconsistent behavior.

**Recommendation:** Generally DO NOT prune biases:
- Biases have far fewer parameters than weights
- Pruning biases provides minimal compression benefit
- Can significantly hurt model quality

Your implementation correctly defaults to `prune_bias=False`.

### Bug 7: Gradients on Wrong Parameter

**Problem:** After pruning, gradients should flow to `weight_orig`, not `weight`.

**Detection:**
```python
model.zero_grad()
loss.backward()
assert module.weight_orig.grad is not None  # Should have gradient
assert module.weight.grad is None  # Should NOT have gradient (it's a property)
```

**Note:** PyTorch handles this correctly, but custom implementations may not.

### Bug 8: Sparsity Drift During Training

**Problem:** Zeros can become non-zero during gradient updates if masks aren't properly enforced.

**Detection:** Monitor sparsity at epoch boundaries:
```python
def on_epoch_end(model, params, expected_sparsity):
    actual = compute_sparsity(model, params)
    if abs(actual - expected_sparsity) > 0.01:
        logger.warning(f"Sparsity drift: expected {expected_sparsity}, got {actual}")
```

**Solution:** PyTorch's forward_pre_hook handles this, but verify hooks are registered:
```python
assert len(module._forward_pre_hooks) > 0, "Pruning hooks not registered"
```

---

## Comparison: Your Implementation vs PyTorch Reference

Based on review of `/home/ahmad/adversarial-robustness-for-sr/src/callbacks/pruning/prune.py`:

| Aspect | Your Implementation | PyTorch Reference | Assessment |
|--------|---------------------|-------------------|------------|
| Pruning Backend | Uses `torch.nn.utils.prune` | N/A | CORRECT |
| Global Pruning | `pytorch_prune.global_unstructured` | Same | CORRECT |
| Sparsity Calculation | Counts zeros in weight tensor | Same | CORRECT |
| Mask Removal | `pytorch_prune.remove()` | Same | CORRECT |
| Iterative Handling | Removes masks before re-applying | Same | CORRECT |
| Monotonicity Check | `_verify_sparsity_jump()` | Not in PyTorch | ENHANCEMENT |
| Layer Collapse Prevention | Not explicitly implemented | Not in PyTorch | POTENTIAL RISK |

**Recommendations:**
1. Add Minimum Threshold technique to prevent layer collapse at high sparsity
2. Add explicit test comparing against PyTorch's reference behavior
3. Monitor per-layer sparsity distribution, not just global

---

## Sanity Check Experiments

### Experiment 1: Verify Against PyTorch Reference

```python
def test_matches_pytorch_reference():
    """Verify implementation matches PyTorch behavior exactly."""
    import torch.nn.utils.prune as prune

    # Create identical models
    model1 = create_model()  # Your pruning
    model2 = create_model()  # PyTorch reference

    # Same initial weights
    model2.load_state_dict(model1.state_dict())

    # Apply pruning
    your_pruner.prune(model1, amount=0.5)

    params2 = [(m, 'weight') for m in model2.modules() if hasattr(m, 'weight')]
    prune.global_unstructured(params2, prune.L1Unstructured, amount=0.5)

    # Compare sparsity
    sparsity1 = compute_sparsity(model1)
    sparsity2 = compute_sparsity(model2)
    assert abs(sparsity1 - sparsity2) < 0.001
```

### Experiment 2: Verify Gradual Pruning Schedule

```python
def test_linear_schedule():
    """Verify linear schedule reaches correct targets."""
    scheduler = PruningScheduler(
        schedule_type="linear",
        final_sparsity=0.9,
        epochs_to_ramp=10,
        initial_sparsity=0.0
    )

    expected = [0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90]

    for epoch in range(10):
        target = scheduler.get_target_sparsity(epoch)
        assert abs(target - expected[epoch]) < 0.01, f"Epoch {epoch}: {target} != {expected[epoch]}"
```

### Experiment 3: Verify Performance at Different Sparsity Levels

Based on literature, your speaker verification system should see:

| Sparsity | Expected EER Impact |
|----------|---------------------|
| 30% | Negligible (<0.05% absolute) |
| 50% | Minimal (~0.1% absolute) |
| 70% | Small (~0.2-0.3% absolute) |
| 90% | Moderate (~0.5-1.0% absolute) |

Run these experiments and compare against these benchmarks.

---

## Sources

### Primary References (HIGH confidence)
- [PyTorch Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html) - Official documentation
- [PyTorch Test Suite](https://github.com/pytorch/pytorch/blob/main/test/ao/sparsity/test_structured_sparsifier.py) - Official tests

### Benchmark Papers (HIGH confidence)
- [Global Magnitude Pruning Case Study (arXiv:2209.14624)](https://arxiv.org/html/2209.14624v3) - Comprehensive benchmarks
- [Hybrid Pruning for Speaker Verification (arXiv:2508.16232)](https://arxiv.org/html/2508.16232) - Speaker verification specific

### Related Work (MEDIUM confidence)
- [Lottery Ticket Hypothesis (arXiv:1803.03635)](https://arxiv.org/abs/1803.03635) - Theoretical foundation
- [Torch-Pruning (CVPR 2023)](https://github.com/VainF/Torch-Pruning) - Structural pruning reference
- [State of Neural Network Pruning (arXiv:2003.03033)](https://arxiv.org/pdf/2003.03033) - Survey and issues
