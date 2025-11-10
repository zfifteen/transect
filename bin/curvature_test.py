#!/usr/bin/env python3
"""
Curvature Test: Validate κ(n) curvature reduction with prime mapping.

This script empirically validates the claim that prime-mapped slot indices
achieve 25-88% curvature reduction compared to raw slot indices.

Usage:
    python bin/curvature_test.py --slots 1000 --prime nearest --output results/curvature.csv

Mathematical Foundation:
- Discrete curvature: κ(n) = d(n) · ln(n+1) / e²
- For prime n: d(n) = 2 (only divisors are 1 and n)
- Lower κ indicates more stable synchronization paths
- Prime mapping: slot_index → nearest_prime(slot_index)

Validation Criteria:
- Measure curvature reduction across 1000+ slot indices
- Compute 95% confidence interval via bootstrapping (1000 resamples)
- Verify reduction ≥ 25% (target: 25-88% range)
"""

import sys
import os
import argparse
import random
import csv
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available, using basic statistics")

from transec.prime_optimization import (
    normalize_slot_to_prime,
    compute_curvature,
    is_prime,
    compute_curvature_reduction
)


def compute_curvature_statistics(
    num_slots: int,
    prime_strategy: str = "nearest",
    seed: int = 42
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute curvature values for raw and normalized slots.
    
    Args:
        num_slots: Number of slot indices to test
        prime_strategy: Prime normalization strategy ("nearest" or "next")
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (raw_curvatures, normalized_curvatures, reductions)
    """
    random.seed(seed)
    
    raw_curvatures = []
    normalized_curvatures = []
    reductions = []
    
    # Sample slot indices (using a mix of sequential and random for realism)
    slot_indices = list(range(2, num_slots // 2 + 2))  # Sequential from 2
    slot_indices += [random.randint(100, 10000) for _ in range(num_slots // 2)]
    
    for slot in slot_indices:
        # Compute raw curvature
        kappa_raw = compute_curvature(slot)
        raw_curvatures.append(kappa_raw)
        
        # Normalize to prime
        normalized_slot = normalize_slot_to_prime(slot, strategy=prime_strategy)
        kappa_normalized = compute_curvature(normalized_slot)
        normalized_curvatures.append(kappa_normalized)
        
        # Compute reduction percentage
        if kappa_raw > 0:
            reduction = (kappa_raw - kappa_normalized) / kappa_raw * 100
            reductions.append(reduction)
        else:
            reductions.append(0.0)
    
    return raw_curvatures, normalized_curvatures, reductions


def bootstrap_confidence_interval(
    data: List[float],
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval via bootstrapping.
    
    Args:
        data: List of values to resample
        num_resamples: Number of bootstrap resamples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    random.seed(seed)
    
    if NUMPY_AVAILABLE:
        data_array = np.array(data)
        bootstrap_means = []
        
        for _ in range(num_resamples):
            # Resample with replacement
            resample = np.random.choice(data_array, size=len(data_array), replace=True)
            bootstrap_means.append(np.mean(resample))
        
        bootstrap_means = np.array(bootstrap_means)
        mean = np.mean(data_array)
        
        # Compute percentile-based confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return mean, lower_bound, upper_bound
    else:
        # Fallback: simple mean and standard error approximation
        mean = sum(data) / len(data)
        # Approximate 95% CI as mean ± 2*SE
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        se = (variance / len(data)) ** 0.5
        margin = 1.96 * se  # 95% CI
        
        return mean, mean - margin, mean + margin


def save_results_csv(
    output_path: str,
    slot_indices: List[int],
    raw_curvatures: List[float],
    normalized_curvatures: List[float],
    reductions: List[float]
):
    """
    Save curvature test results to CSV file.
    
    Args:
        output_path: Path to output CSV file
        slot_indices: List of slot indices tested
        raw_curvatures: Raw curvature values
        normalized_curvatures: Normalized curvature values
        reductions: Reduction percentages
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['slot_index', 'raw_curvature', 'normalized_curvature', 'reduction_percent'])
        
        for i in range(len(slot_indices)):
            writer.writerow([
                slot_indices[i],
                f"{raw_curvatures[i]:.6f}",
                f"{normalized_curvatures[i]:.6f}",
                f"{reductions[i]:.2f}"
            ])
    
    print(f"Results saved to: {output_path}")


def main():
    """Main entry point for curvature test."""
    parser = argparse.ArgumentParser(
        description="Validate curvature reduction with prime mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with 1000 slots
  python bin/curvature_test.py --slots 1000
  
  # Test with CSV output
  python bin/curvature_test.py --slots 1000 --output results/curvature.csv
  
  # Test with different prime strategy
  python bin/curvature_test.py --slots 500 --prime next
  
  # Test with custom bootstrap resamples
  python bin/curvature_test.py --slots 1000 --bootstraps 2000
        """
    )
    
    parser.add_argument(
        '--slots',
        type=int,
        default=1000,
        help='Number of slot indices to test (default: 1000)'
    )
    parser.add_argument(
        '--prime',
        type=str,
        choices=['nearest', 'next'],
        default='nearest',
        help='Prime normalization strategy (default: nearest)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (optional)'
    )
    parser.add_argument(
        '--bootstraps',
        type=int,
        default=1000,
        help='Number of bootstrap resamples for CI (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRANSEC Curvature Reduction Test")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Slots:           {args.slots}")
    print(f"  Prime strategy:  {args.prime}")
    print(f"  Bootstrap:       {args.bootstraps} resamples")
    print(f"  Random seed:     {args.seed}")
    print()
    
    # Compute curvature statistics
    print("Computing curvature values...")
    raw_curvatures, normalized_curvatures, reductions = compute_curvature_statistics(
        num_slots=args.slots,
        prime_strategy=args.prime,
        seed=args.seed
    )
    
    # Filter out zero reductions (slots that are already prime)
    non_zero_reductions = [r for r in reductions if r > 0]
    
    if len(non_zero_reductions) == 0:
        print("Error: No non-prime slots found in sample")
        return 1
    
    # Compute statistics
    print("\nCurvature Statistics:")
    print(f"  Total slots tested:        {len(reductions)}")
    print(f"  Slots needing normalization: {len(non_zero_reductions)}")
    print(f"  Already prime:             {len(reductions) - len(non_zero_reductions)}")
    print()
    
    # Basic statistics
    if NUMPY_AVAILABLE:
        mean_reduction = np.mean(non_zero_reductions)
        median_reduction = np.median(non_zero_reductions)
        min_reduction = np.min(non_zero_reductions)
        max_reduction = np.max(non_zero_reductions)
    else:
        mean_reduction = sum(non_zero_reductions) / len(non_zero_reductions)
        sorted_reductions = sorted(non_zero_reductions)
        median_reduction = sorted_reductions[len(sorted_reductions) // 2]
        min_reduction = min(non_zero_reductions)
        max_reduction = max(non_zero_reductions)
    
    print(f"Curvature Reduction (κ):")
    print(f"  Mean:    {mean_reduction:.2f}%")
    print(f"  Median:  {median_reduction:.2f}%")
    print(f"  Min:     {min_reduction:.2f}%")
    print(f"  Max:     {max_reduction:.2f}%")
    print()
    
    # Bootstrap confidence interval
    print(f"Computing {args.bootstraps}-resample bootstrap CI...")
    mean_ci, lower_ci, upper_ci = bootstrap_confidence_interval(
        non_zero_reductions,
        num_resamples=args.bootstraps,
        confidence_level=0.95,
        seed=args.seed
    )
    
    print(f"\n95% Confidence Interval:")
    print(f"  Mean:  {mean_ci:.2f}%")
    print(f"  CI:    [{lower_ci:.2f}%, {upper_ci:.2f}%]")
    print()
    
    # Validation against claims
    print("Validation against claims (25-88% reduction):")
    if lower_ci >= 25.0:
        print(f"  ✓ Lower bound {lower_ci:.2f}% meets minimum threshold (25%)")
    else:
        print(f"  ✗ Lower bound {lower_ci:.2f}% below threshold (25%)")
    
    if mean_ci >= 25.0 and mean_ci <= 88.0:
        print(f"  ✓ Mean {mean_ci:.2f}% within target range (25-88%)")
    else:
        print(f"  ⚠ Mean {mean_ci:.2f}% outside target range (25-88%)")
    
    if upper_ci <= 88.0:
        print(f"  ✓ Upper bound {upper_ci:.2f}% within range (≤88%)")
    else:
        print(f"  ⚠ Upper bound {upper_ci:.2f}% exceeds maximum (88%)")
    
    # Save results if requested
    if args.output:
        # Reconstruct slot indices for CSV
        random.seed(args.seed)
        slot_indices = list(range(2, args.slots // 2 + 2))
        slot_indices += [random.randint(100, 10000) for _ in range(args.slots // 2)]
        
        save_results_csv(
            args.output,
            slot_indices,
            raw_curvatures,
            normalized_curvatures,
            reductions
        )
    
    print()
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
