#!/usr/bin/env python3
"""
Script to make pruning permanent by removing mask keys and other pruning artifacts
from PyTorch Lightning checkpoints and standalone .pth files.

This creates a reduced-size checkpoint/model file with pruned weights permanently removed.
"""

import argparse
import torch
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Get the best available device for loading checkpoints.
    
    Returns:
        Device string ('cuda' if available, otherwise 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def identify_pruning_artifacts(state_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Identify pruning artifact keys in the state dict.
    
    Returns:
        Tuple of (mask_keys, orig_keys)
    """
    mask_keys = [key for key in state_dict.keys() if key.endswith('_mask')]
    orig_keys = [key for key in state_dict.keys() if key.endswith('_orig')]
    
    return mask_keys, orig_keys


def make_pruning_permanent_state_dict(state_dict: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Remove pruning artifacts and create permanent pruned weights.
    
    Args:
        state_dict: Original state dict with pruning artifacts
        verbose: Whether to print detailed information
        
    Returns:
        New state dict with permanent pruning applied
    """
    mask_keys, orig_keys = identify_pruning_artifacts(state_dict)
    
    if verbose:
        logger.info(f"Found {len(mask_keys)} mask parameters and {len(orig_keys)} original parameters")
    
    if not mask_keys and not orig_keys:
        logger.info("No pruning artifacts found - model is not pruned or already permanent")
        return state_dict
    
    # Create new state dict without pruning artifacts
    new_state_dict = {}
    
    # Track processed parameters to avoid duplicates
    processed_params = set()
    
    # Process each key in the original state dict
    for key, value in state_dict.items():
        
        # Skip mask keys - we don't want these in the final model
        if key.endswith('_mask'):
            continue
            
        # Handle original parameter keys
        if key.endswith('_orig'):
            # Get the base parameter name (remove '_orig' suffix)
            base_key = key[:-5]  # Remove '_orig'
            mask_key = key[:-5] + '_mask'  # Replace '_orig' with '_mask'
            
            if mask_key in state_dict:
                # Apply the mask to make pruning permanent
                mask = state_dict[mask_key]
                pruned_param = value * mask  # Apply mask to original parameter
                new_state_dict[base_key] = pruned_param
                processed_params.add(base_key)
                
                if verbose:
                    sparsity = (mask == 0).float().mean().item()
                    logger.info(f"Made pruning permanent for {base_key}: {sparsity:.2%} sparsity")
            else:
                # No corresponding mask found, just remove '_orig' suffix
                new_state_dict[base_key] = value
                processed_params.add(base_key)
                logger.warning(f"Found {key} but no corresponding mask - treating as unpruned")
                
        # Handle regular parameters that are not pruned
        else:
            # Only add if we haven't already processed this parameter
            if key not in processed_params:
                new_state_dict[key] = value
    
    logger.info(f"Processed {len(processed_params)} pruned parameters")
    logger.info(f"Final state dict has {len(new_state_dict)} parameters (reduced from {len(state_dict)})")
    
    return new_state_dict


def make_checkpoint_permanent(input_path: str, output_path: str = None, verbose: bool = False):
    """
    Make pruning permanent in a PyTorch Lightning checkpoint.
    
    Args:
        input_path: Path to input checkpoint
        output_path: Path for output checkpoint (optional)
        verbose: Whether to print detailed information
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    
    # Default output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_permanent{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    logger.info(f"Loading checkpoint from: {input_path}")
    
    # Load checkpoint
    device = get_device()
    checkpoint = torch.load(input_path, map_location=device, weights_only=False)
    
    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'state_dict' key - not a valid PyTorch Lightning checkpoint")
    
    # Get original size
    original_size = input_path.stat().st_size / (1024 * 1024)  # MB
    
    # Make state dict permanent
    original_state_dict = checkpoint['state_dict']
    permanent_state_dict = make_pruning_permanent_state_dict(original_state_dict, verbose=verbose)
    
    # Update checkpoint
    checkpoint['state_dict'] = permanent_state_dict
    
    # Remove any pruning-related callback states
    if 'magnitude_pruner_state' in checkpoint:
        if verbose:
            logger.info("Removing magnitude_pruner_state from checkpoint")
        del checkpoint['magnitude_pruner_state']
    
    # Add metadata about the permanent pruning
    if 'permanent_pruning_metadata' not in checkpoint:
        checkpoint['permanent_pruning_metadata'] = {
            'made_permanent': True,
            'original_file': str(input_path),
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
        }
    
    # Save permanent checkpoint
    logger.info(f"Saving permanent checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    
    # Get new size and report savings
    new_size = output_path.stat().st_size / (1024 * 1024)  # MB
    size_reduction = original_size - new_size
    reduction_percentage = (size_reduction / original_size) * 100
    
    logger.info(f"Size reduction: {original_size:.1f} MB -> {new_size:.1f} MB (saved {size_reduction:.1f} MB, {reduction_percentage:.1f}%)")


def make_pth_permanent(input_path: str, output_path: str = None, verbose: bool = False):
    """
    Make pruning permanent in a standalone .pth model file.
    
    Args:
        input_path: Path to input .pth file
        output_path: Path for output .pth file (optional)
        verbose: Whether to print detailed information
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input .pth file not found: {input_path}")
    
    # Default output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_permanent{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    logger.info(f"Loading .pth file from: {input_path}")
    
    # Load model state dict
    device = get_device()
    state_dict = torch.load(input_path, map_location=device, weights_only=False)
    
    # Get original size
    original_size = input_path.stat().st_size / (1024 * 1024)  # MB
    
    # Make state dict permanent
    permanent_state_dict = make_pruning_permanent_state_dict(state_dict, verbose=verbose)
    
    # Save permanent .pth file
    logger.info(f"Saving permanent .pth file to: {output_path}")
    torch.save(permanent_state_dict, output_path)
    
    # Get new size and report savings
    new_size = output_path.stat().st_size / (1024 * 1024)  # MB
    size_reduction = original_size - new_size
    reduction_percentage = (size_reduction / original_size) * 100
    
    logger.info(f"Size reduction: {original_size:.1f} MB -> {new_size:.1f} MB (saved {size_reduction:.1f} MB, {reduction_percentage:.1f}%)")


def analyze_pruning(input_path: str):
    """
    Analyze the pruning artifacts in a checkpoint or .pth file without modifying it.
    
    Args:
        input_path: Path to input file
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Analyzing: {input_path}")
    
    # Load the file
    device = get_device()
    data = torch.load(input_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if isinstance(data, dict) and 'state_dict' in data:
        # PyTorch Lightning checkpoint
        state_dict = data['state_dict']
        file_type = "PyTorch Lightning checkpoint"
    else:
        # Assume it's a raw state dict
        state_dict = data
        file_type = ".pth model file"
    
    logger.info(f"File type: {file_type}")
    
    # Analyze pruning artifacts
    mask_keys, orig_keys = identify_pruning_artifacts(state_dict)
    
    logger.info(f"Total parameters: {len(state_dict)}")
    logger.info(f"Mask parameters: {len(mask_keys)}")
    logger.info(f"Original parameters: {len(orig_keys)}")
    
    if mask_keys:
        logger.info("Example mask parameters:")
        for key in mask_keys[:5]:  # Show first 5
            logger.info(f"  - {key}")
        if len(mask_keys) > 5:
            logger.info(f"  ... and {len(mask_keys) - 5} more")
    
    if orig_keys:
        logger.info("Example original parameters:")
        for key in orig_keys[:5]:  # Show first 5
            logger.info(f"  - {key}")
        if len(orig_keys) > 5:
            logger.info(f"  ... and {len(orig_keys) - 5} more")
    
    # Calculate potential size savings
    if mask_keys or orig_keys:
        pruned_params = len(mask_keys) + len(orig_keys)
        total_params = len(state_dict)
        overhead_percentage = (pruned_params / total_params) * 100
        logger.info(f"Pruning overhead: {pruned_params}/{total_params} parameters ({overhead_percentage:.1f}%)")
        logger.info("This file contains pruning artifacts and can be made permanent to reduce size.")
    else:
        logger.info("This file contains no pruning artifacts.")


def main():
    parser = argparse.ArgumentParser(
        description="Make pruning permanent in PyTorch Lightning checkpoints and .pth files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a checkpoint for pruning artifacts
  python make_pruning_permanent.py --analyze checkpoint.ckpt
  
  # Make pruning permanent in a checkpoint
  python make_pruning_permanent.py --input checkpoint.ckpt --output permanent_checkpoint.ckpt
  
  # Make pruning permanent in a .pth file  
  python make_pruning_permanent.py --input model.pth --output permanent_model.pth
  
  # Auto-generate output filename
  python make_pruning_permanent.py --input checkpoint.ckpt
        """
    )
    
    parser.add_argument('--input', '-i', type=str, help='Input checkpoint (.ckpt) or model (.pth) file')
    parser.add_argument('--output', '-o', type=str, help='Output file path (optional, auto-generated if not provided)')
    parser.add_argument('--analyze', '-a', type=str, help='Analyze pruning artifacts in file without modifying it')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_pruning(args.analyze)
    elif args.input:
        input_path = Path(args.input)
        
        if input_path.suffix.lower() in ['.ckpt']:
            make_checkpoint_permanent(args.input, args.output, args.verbose)
        elif input_path.suffix.lower() in ['.pth', '.pt']:
            make_pth_permanent(args.input, args.output, args.verbose)
        else:
            logger.error(f"Unsupported file type: {input_path.suffix}")
            logger.error("Supported types: .ckpt (checkpoints), .pth/.pt (model files)")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
