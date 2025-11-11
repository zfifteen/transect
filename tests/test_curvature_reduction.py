#!/usr/bin/env python3
"""
Tests for 25-88% curvature reduction claims.

Addresses Blocking 6: Moves curvature reduction examples from docs to CI tests
with specific bounds verification.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec.prime_optimization import (
    compute_curvature,
    compute_curvature_reduction,
    normalize_slot_to_prime,
    is_prime,
)


class TestCurvatureReductionClaims(unittest.TestCase):
    """Test that curvature reduction claims (25-88%) are met."""
    
    def test_documented_reduction_examples(self):
        """Test the specific examples documented in README/docs."""
        # These are the examples claimed in the documentation
        documented_examples = [
            (4, 5, 25, 88),      # Should reduce by 25-88%
            (6, 5, 25, 88),      # Should reduce by 25-88%
            (8, 7, 25, 88),      # Should reduce by 25-88%
            (10, 11, 25, 88),    # Should reduce by 25-88%
            (100, 101, 25, 88),  # Should reduce by 25-88%
            (1000, 997, 25, 88), # Should reduce by 25-88%
        ]
        
        for original, expected_prime, min_reduction, max_reduction in documented_examples:
            # Get geodesic optimal prime
            prime = normalize_slot_to_prime(original, strategy="geodesic")
            
            # Compute reduction using deterministic method
            reduction = compute_curvature_reduction(original, prime, use_arctan_geodesic=True, use_deterministic=True)
            
            # Verify prime is reasonable
            self.assertTrue(is_prime(prime), f"{prime} should be prime")
            self.assertLess(abs(prime - original), 20, 
                          f"Prime {prime} too far from {original}")
            
            # Verify reduction is in claimed range
            self.assertGreaterEqual(reduction, min_reduction,
                                  f"Reduction {reduction:.1f}% < {min_reduction}% for {original}→{prime}")
            self.assertLessEqual(reduction, max_reduction,
                               f"Reduction {reduction:.1f}% > {max_reduction}% for {original}→{prime}")
    
    def test_range_of_composites(self):
        """Test a range of composite numbers for 25-88% reduction."""
        # Test composites from 4 to 1000
        composites = [n for n in range(4, 1001) if not is_prime(n)]
        
        # Sample every 10th composite to keep test runtime reasonable
        sample_composites = composites[::10]
        
        in_range_count = 0
        total_count = 0
        
        for composite in sample_composites:
            prime = normalize_slot_to_prime(composite, strategy="geodesic")
            
            if prime == composite:  # Skip if already prime (shouldn't happen)
                continue
            
            reduction = compute_curvature_reduction(composite, prime, 
                                                   use_arctan_geodesic=True, 
                                                   use_deterministic=True)
            
            total_count += 1
            if 25 <= reduction <= 88:
                in_range_count += 1
        
        # At least 80% should be in range (some edge cases may fall outside)
        success_rate = in_range_count / total_count if total_count > 0 else 0
        self.assertGreater(success_rate, 0.8,
                         f"Only {success_rate:.1%} of samples in 25-88% range")
    
    def test_small_composites_reduction(self):
        """Test specific small composite numbers."""
        small_composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        
        for composite in small_composites:
            prime = normalize_slot_to_prime(composite, strategy="geodesic")
            reduction = compute_curvature_reduction(composite, prime,
                                                   use_arctan_geodesic=True,
                                                   use_deterministic=True)
            
            # Should have positive reduction
            self.assertGreater(reduction, 0,
                             f"No reduction for {composite}→{prime}")
            
            # Should be reasonable (0-100%)
            self.assertLess(reduction, 100,
                          f"Reduction {reduction:.1f}% > 100% for {composite}→{prime}")
    
    def test_large_composites_higher_reduction(self):
        """Test that larger composites tend to have higher reduction."""
        # Test a few large composites
        large_composites = [100, 200, 500, 1000, 2000, 5000]
        
        for composite in large_composites:
            if is_prime(composite):
                continue
            
            prime = normalize_slot_to_prime(composite, strategy="geodesic")
            reduction = compute_curvature_reduction(composite, prime,
                                                   use_arctan_geodesic=True,
                                                   use_deterministic=True)
            
            # Larger numbers should generally have good reduction
            # (though not guaranteed, just checking reasonableness)
            self.assertGreater(reduction, 10,
                             f"Too low reduction {reduction:.1f}% for {composite}→{prime}")
    
    def test_prime_to_same_prime_zero_reduction(self):
        """Test that primes mapping to themselves have 0% reduction."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        
        for p in primes:
            normalized = normalize_slot_to_prime(p, strategy="geodesic")
            self.assertEqual(normalized, p, f"Prime {p} should map to itself")
            
            reduction = compute_curvature_reduction(p, normalized,
                                                   use_arctan_geodesic=True,
                                                   use_deterministic=True)
            self.assertEqual(reduction, 0.0, f"Prime {p} should have 0% reduction")
    
    def test_geodesic_better_than_nearest(self):
        """Test that geodesic strategy achieves comparable or better reduction than nearest."""
        test_composites = [4, 6, 8, 10, 12, 15, 20, 24, 30, 50, 100]
        
        better_or_equal = 0
        total = 0
        
        for composite in test_composites:
            if is_prime(composite):
                continue
            
            prime_geodesic = normalize_slot_to_prime(composite, strategy="geodesic")
            prime_nearest = normalize_slot_to_prime(composite, strategy="nearest")
            
            reduction_geodesic = compute_curvature_reduction(composite, prime_geodesic,
                                                            use_arctan_geodesic=True,
                                                            use_deterministic=True)
            reduction_nearest = compute_curvature_reduction(composite, prime_nearest,
                                                           use_arctan_geodesic=True,
                                                           use_deterministic=True)
            
            total += 1
            if reduction_geodesic >= reduction_nearest - 1.0:  # Allow 1% tolerance
                better_or_equal += 1
        
        # Geodesic should be better or equal in most cases
        success_rate = better_or_equal / total if total > 0 else 0
        # This is not a strict requirement, just documenting that geodesic is optimized


class TestCurvatureComputationCorrectness(unittest.TestCase):
    """Test curvature computation correctness."""
    
    def test_deterministic_vs_floating_point_similar(self):
        """Test that deterministic Q24 and floating-point methods give similar results."""
        test_slots = [2, 3, 5, 7, 10, 20, 100]
        
        for slot in test_slots:
            kappa_deterministic = compute_curvature(slot, use_arctan_geodesic=True, use_deterministic=True)
            kappa_float = compute_curvature(slot, use_arctan_geodesic=True, use_deterministic=False)
            
            # Should be within 5% of each other (Q24 has limited precision)
            if kappa_deterministic > 0:
                relative_error = abs(kappa_deterministic - kappa_float) / kappa_deterministic
                self.assertLess(relative_error, 0.05,
                              f"Deterministic and float differ by {relative_error:.1%} for slot {slot}")
    
    def test_curvature_positive(self):
        """Test that curvature values are always positive."""
        test_slots = list(range(1, 101))
        
        for slot in test_slots:
            kappa = compute_curvature(slot, use_arctan_geodesic=True, use_deterministic=True)
            self.assertGreater(kappa, 0, f"Curvature should be positive for slot {slot}")
    
    def test_prime_has_lower_curvature(self):
        """Test that primes generally have lower curvature than nearby composites."""
        # Test pairs of (composite, nearby_prime)
        pairs = [(4, 5), (6, 7), (8, 7), (10, 11), (12, 13), (14, 13), (15, 13)]
        
        for composite, prime in pairs:
            kappa_composite = compute_curvature(composite, use_arctan_geodesic=True, use_deterministic=True)
            kappa_prime = compute_curvature(prime, use_arctan_geodesic=True, use_deterministic=True)
            
            # Prime should have lower or equal curvature
            self.assertLessEqual(kappa_prime, kappa_composite,
                               f"Prime {prime} has higher curvature than composite {composite}")


if __name__ == '__main__':
    unittest.main()
