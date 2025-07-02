"""
Example usage of LambdaScheduler with sparsity smoothing.

This example demonstrates how to use the LambdaScheduler to handle spurious
zero sparsity readings that can occur during Bregman pruning.
"""

import sys
from pathlib import Path

# Add the src directory to the path for importing
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.callbacks.pruning.bregman.lambda_scheduler import LambdaScheduler


def example_usage():
    """Example of typical LambdaScheduler usage."""
    
    # Initialize scheduler with custom parameters
    scheduler = LambdaScheduler(
        initial_lambda=1e-3,
        target_sparsity=0.9,  # Aiming for 90% sparsity
        adjustment_factor=1.1,  # 10% adjustments
        min_lambda=1e-6,
        max_lambda=1e2
    )
    
    print("LambdaScheduler Example")
    print("=" * 50)
    print(f"Initial lambda: {scheduler.get_lambda():.6f}")
    print(f"Target sparsity: {scheduler.target_sparsity}")
    print()
    
    # Simulate training steps with various sparsity readings
    sparsity_readings = [
        0.7,   # Below target - lambda should increase
        0.0,   # Spurious zero - should be ignored (use last valid)
        0.85,  # Below target - lambda should increase  
        0.0,   # Another spurious zero - should use 0.85
        0.95,  # Above target - lambda should decrease
        0.92,  # Slightly above target - lambda should decrease
        0.0,   # Spurious zero - should use 0.92
    ]
    
    print("Step | Sparsity | Effective | Lambda     | Action")
    print("-" * 55)
    
    for step, current_sparsity in enumerate(sparsity_readings, 1):
        # Get effective sparsity (after smoothing)
        effective_sparsity = scheduler._get_effective_sparsity(current_sparsity)
        
        # Update lambda
        new_lambda = scheduler.step(current_sparsity)
        
        # Determine action
        if effective_sparsity < scheduler.target_sparsity:
            action = "Increase λ"
        elif effective_sparsity > scheduler.target_sparsity:
            action = "Decrease λ"
        else:
            action = "No change"
            
        # Add note if spurious zero was filtered
        if current_sparsity == 0.0 and effective_sparsity != 0.0:
            action += " (filtered)"
            
        print(f"{step:4d} | {current_sparsity:8.2f} | {effective_sparsity:9.2f} | "
              f"{new_lambda:.6f} | {action}")
    
    print()
    print("Final state:")
    state = scheduler.get_state()
    print(f"Lambda: {state['lambda_value']:.6f}")
    print(f"Last valid sparsity: {state['last_sparsity']}")


def spurious_zero_demonstration():
    """Demonstrate the spurious zero filtering mechanism."""
    
    print("\nSpurious Zero Filtering Demonstration")
    print("=" * 50)
    
    print("Scenario: Model has achieved 92% sparsity (above target of 90%)")
    print("Spurious zeros occur, making it seem like sparsity dropped to 0%")
    print()
    
    # Create comparison
    with_smoothing = LambdaScheduler(initial_lambda=1.0, target_sparsity=0.9)
    
    print("Step | Raw Reading | Without Smoothing    | With Smoothing       | Notes")
    print("-" * 85)
    
    # Realistic scenario: good reading above target, then spurious zeros
    readings = [0.92, 0.0, 0.0, 0.0, 0.94]
    
    # Manual simulation without smoothing
    no_smoothing_lambda = 1.0
    target = 0.9
    factor = 1.1
    
    for step, reading in enumerate(readings, 1):
        # WITHOUT smoothing: always use raw reading
        if reading < target:
            no_smoothing_lambda *= factor
            no_smooth_action = f"↑ {no_smoothing_lambda:.4f} (increase)"
        elif reading > target:
            no_smoothing_lambda /= factor 
            no_smooth_action = f"↓ {no_smoothing_lambda:.4f} (decrease)"
        else:
            no_smooth_action = f"= {no_smoothing_lambda:.4f} (no change)"
            
        # WITH smoothing: use effective sparsity
        effective = with_smoothing._get_effective_sparsity(reading)
        smoothed_lambda = with_smoothing.step(reading)
        
        if effective < target:
            smooth_action = f"↑ {smoothed_lambda:.4f} (increase)"
        elif effective > target:
            smooth_action = f"↓ {smoothed_lambda:.4f} (decrease)"
        else:
            smooth_action = f"= {smoothed_lambda:.4f} (no change)"
            
        # Notes
        if reading == 0.0 and effective != 0.0:
            note = f"Zero ignored, used {effective:.2f}"
        elif reading == 0.0:
            note = "Zero used (no prior reading)"
        else:
            note = "Normal reading"
            
        print(f"{step:4d} | {reading:11.2f} | {no_smooth_action:20s} | {smooth_action:20s} | {note}")
    
    print()
    print("Analysis:")
    print(f"• Without smoothing: λ = {no_smoothing_lambda:.4f}")
    print(f"• With smoothing:    λ = {with_smoothing.get_lambda():.4f}")
    
    spurious_impact = abs(no_smoothing_lambda - with_smoothing.get_lambda())
    print(f"• Difference:        {spurious_impact:.4f}")
    
    if no_smoothing_lambda > with_smoothing.get_lambda():
        print("→ Smoothing prevented excessive lambda increases from spurious zeros!")
    elif with_smoothing.get_lambda() > no_smoothing_lambda:
        print("→ Smoothing allowed proper lambda decreases despite spurious zeros!")
    else:
        print("→ Both approaches gave similar results in this case.")
        
    print(f"\nKey benefit: Smoothing maintains stable lambda adjustments by ignoring")
    print(f"spurious zeros that would otherwise cause incorrect lambda oscillations.")


if __name__ == "__main__":
    example_usage()
    spurious_zero_demonstration()