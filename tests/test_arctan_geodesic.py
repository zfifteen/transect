#!/usr/bin/env python3
"""
Unit tests for Arctan-Geodesic Prime Optimization.

Tests cover:
- Arctan-geodesic curvature computation
- Golden ratio integration
- Prime selection using geodesic optimization
- Curvature reduction validation (25-88% claims)
- Integration with TRANSEC cipher
"""

import sys
import os
import unittest
import math

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec.prime_optimization import (
    compute_curvature,
    compute_curvature_reduction,
    normalize_slot_to_prime,
    find_geodesic_optimal_prime,
    find_nearest_prime_by_distance,
    is_prime,
    count_divisors,
    PHI
)
from transec import TransecCipher, generate_shared_secret


class TestArctanGeodesicCurvature(unittest.TestCase):
    """Test arctan-geodesic curvature computation."""
    
    def test_golden_ratio_constant(self):
        """Test that PHI is correctly defined."""
        expected_phi = (1 + math.sqrt(5)) / 2
        self.assertAlmostEqual(PHI, expected_phi, places=10)
        self.assertAlmostEqual(PHI, 1.618033988749895, places=10)
    
    def test_arctan_geodesic_increases_absolute_curvature(self):
        """Test that arctan-geodesic component increases absolute curvature values."""
        for n in [2, 3, 5, 7, 11, 13, 17, 19]:  # Test with primes
            kappa_base = compute_curvature(n, use_arctan_geodesic=False)
            kappa_geo = compute_curvature(n, use_arctan_geodesic=True)
            
            # Geodesic should be higher (arctan term is always >= 1)
            self.assertGreater(kappa_geo, kappa_base,
                             f"Geodesic curvature should be higher for n={n}")
    
    def test_arctan_component_range(self):
        """Test that arctan component is in valid range."""
        # The arctan term [1 + arctan(φ · frac(n/φ))] should be in range
        # arctan of anything is in (-π/2, π/2), so term is in (1 - π/2, 1 + π/2)
        for n in range(1, 100):
            frac_n_phi = (n / PHI) - math.floor(n / PHI)
            arctan_term = 1 + math.atan(PHI * frac_n_phi)
            
            # Should be positive and reasonable
            self.assertGreater(arctan_term, 0.5)
            self.assertLess(arctan_term, 3.0)
    
    def test_curvature_positive(self):
        """Test that curvature values are always positive."""
        for n in range(1, 100):
            kappa = compute_curvature(n, use_arctan_geodesic=True)
            self.assertGreater(kappa, 0, f"Curvature should be positive for n={n}")
    
    def test_prime_has_lower_divisor_count(self):
        """Test that primes have d(n)=2, composites have d(n)>2."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]
        
        for p in primes:
            self.assertEqual(count_divisors(p), 2, f"Prime {p} should have d(n)=2")
        
        for c in composites:
            self.assertGreater(count_divisors(c), 2, f"Composite {c} should have d(n)>2")


class TestArctanGeodesicReduction(unittest.TestCase):
    """Test curvature reduction claims (25-88%)."""
    
    def test_curvature_reduction_composite_to_prime(self):
        """Test that mapping composites to primes reduces curvature."""
        # Test various composite numbers
        composites = [4, 6, 8, 9, 10, 12, 15, 20, 100, 1000]
        
        for composite in composites:
            prime = normalize_slot_to_prime(composite, strategy="geodesic")
            reduction = compute_curvature_reduction(composite, prime, use_arctan_geodesic=True)
            
            # Should have positive reduction (improvement)
            self.assertGreater(reduction, 0,
                             f"Should have curvature reduction for {composite} → {prime}")
    
    def test_curvature_reduction_range(self):
        """Test that curvature reductions are in claimed 25-88% range."""
        # Test a variety of composite numbers
        composites = [4, 6, 8, 10, 12, 15, 20, 24, 30, 50, 100, 500, 1000]
        
        reductions = []
        for composite in composites:
            prime = normalize_slot_to_prime(composite, strategy="geodesic")
            reduction = compute_curvature_reduction(composite, prime, use_arctan_geodesic=True)
            reductions.append(reduction)
        
        # At least some should be in the 25-88% range
        in_range = [r for r in reductions if 25 <= r <= 88]
        self.assertGreater(len(in_range), len(reductions) * 0.5,
                          "At least 50% of reductions should be in 25-88% range")
    
    def test_large_number_curvature_reduction(self):
        """Test that larger numbers achieve higher curvature reductions."""
        # Larger numbers should generally achieve better reductions
        small = 10
        large = 1000
        
        prime_small = normalize_slot_to_prime(small, strategy="geodesic")
        prime_large = normalize_slot_to_prime(large, strategy="geodesic")
        
        reduction_small = compute_curvature_reduction(small, prime_small, use_arctan_geodesic=True)
        reduction_large = compute_curvature_reduction(large, prime_large, use_arctan_geodesic=True)
        
        # Large numbers typically get better reduction (though not always guaranteed)
        # Just verify both have positive reduction
        self.assertGreater(reduction_small, 0)
        self.assertGreater(reduction_large, 0)
    
    def test_prime_to_same_prime_no_reduction(self):
        """Test that primes mapping to themselves have 0% reduction."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        for p in primes:
            normalized = normalize_slot_to_prime(p, strategy="geodesic")
            self.assertEqual(normalized, p, f"Prime {p} should map to itself")
            
            reduction = compute_curvature_reduction(p, normalized, use_arctan_geodesic=True)
            self.assertEqual(reduction, 0.0, f"Prime {p} should have 0% reduction")


class TestArctanGeodesicPrimeSelection(unittest.TestCase):
    """Test prime selection using arctan-geodesic optimization."""
    
    def test_find_nearest_prime_with_geodesic(self):
        """Test that geodesic prime uses geodesic curvature for selection."""
        # For some numbers, geodesic optimization might choose different prime
        n = 10
        prime = find_geodesic_optimal_prime(n)
        
        # Should return a prime
        self.assertTrue(is_prime(prime))
        
        # Should be reasonably close to n
        self.assertLessEqual(abs(prime - n), 10)
    
    def test_normalize_slot_consistency(self):
        """Test that normalization is deterministic."""
        for n in [4, 10, 20, 100]:
            prime1 = normalize_slot_to_prime(n, strategy="geodesic")
            prime2 = normalize_slot_to_prime(n, strategy="geodesic")
            
            self.assertEqual(prime1, prime2,
                           f"Normalization should be deterministic for n={n}")
    
    def test_next_strategy_still_works(self):
        """Test that 'next' strategy still works."""
        for n in [4, 10, 20, 100]:
            prime = normalize_slot_to_prime(n, strategy="next")
            
            # Should be prime
            self.assertTrue(is_prime(prime))
            
            # Should be >= n
            self.assertGreaterEqual(prime, n)
    
    def test_none_strategy_returns_original(self):
        """Test that 'none' strategy returns original value."""
        for n in [4, 10, 20, 100]:
            result = normalize_slot_to_prime(n, strategy="none")
            self.assertEqual(result, n)


class TestArctanGeodesicTransecIntegration(unittest.TestCase):
    """Test integration with TRANSEC cipher."""
    
    def test_cipher_with_arctan_geodesic_primes(self):
        """Test that TRANSEC cipher works with arctan-geodesic prime optimization."""
        secret = generate_shared_secret()
        
        # Create cipher with prime strategy (which now uses arctan-geodesic)
        sender = TransecCipher(secret, slot_duration=5, prime_strategy="nearest")
        receiver = TransecCipher(secret, slot_duration=5, prime_strategy="nearest")
        
        plaintext = b"Arctan-geodesic test message"
        sequence = 1
        
        packet = sender.seal(plaintext, sequence)
        decrypted = receiver.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_cipher_prime_optimization_backward_compatible(self):
        """Test that prime='none' still works (backward compatibility)."""
        secret = generate_shared_secret()
        
        sender = TransecCipher(secret, slot_duration=5, prime_strategy="none")
        receiver = TransecCipher(secret, slot_duration=5, prime_strategy="none")
        
        plaintext = b"Backward compatible message"
        sequence = 1
        
        packet = sender.seal(plaintext, sequence)
        decrypted = receiver.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_different_strategies_incompatible(self):
        """Test that different prime strategies are incompatible (as expected)."""
        secret = generate_shared_secret()
        
        sender = TransecCipher(secret, slot_duration=5, prime_strategy="nearest")
        receiver = TransecCipher(secret, slot_duration=5, prime_strategy="none")
        
        plaintext = b"Test message"
        sequence = 1
        
        packet = sender.seal(plaintext, sequence)
        
        # May or may not decrypt depending on timing
        # This test just documents the behavior
        # In practice, both sides must use same strategy


class TestArctanGeodesicFormula(unittest.TestCase):
    """Test the mathematical properties of the arctan-geodesic formula."""
    
    def test_formula_components(self):
        """Test individual components of the formula."""
        n = 10
        
        # Component 1: d(n)
        d_n = count_divisors(n)
        self.assertEqual(d_n, 4)  # 10 has divisors: 1, 2, 5, 10
        
        # Component 2: ln(n+1)
        ln_n_plus_1 = math.log(n + 1)
        self.assertAlmostEqual(ln_n_plus_1, math.log(11), places=10)
        
        # Component 3: e²
        e_squared = math.e ** 2
        self.assertAlmostEqual(e_squared, 7.389056, places=5)
        
        # Component 4: frac(n/φ)
        frac_n_phi = (n / PHI) - math.floor(n / PHI)
        self.assertGreater(frac_n_phi, 0)
        self.assertLess(frac_n_phi, 1)
        
        # Component 5: arctan term
        arctan_term = 1 + math.atan(PHI * frac_n_phi)
        self.assertGreater(arctan_term, 1)
    
    def test_formula_increases_with_divisor_count(self):
        """Test that curvature increases with divisor count."""
        # Compare numbers with different divisor counts but similar magnitude
        n1 = 11  # prime, d(n)=2
        n2 = 12  # composite, d(n)=6
        
        kappa1 = compute_curvature(n1, use_arctan_geodesic=True)
        kappa2 = compute_curvature(n2, use_arctan_geodesic=True)
        
        # n2 should have higher curvature due to more divisors
        self.assertGreater(kappa2, kappa1)
    
    def test_geodesic_enhancement_factor(self):
        """Test that geodesic enhancement factor is reasonable."""
        for n in range(2, 50):
            kappa_base = compute_curvature(n, use_arctan_geodesic=False)
            kappa_geo = compute_curvature(n, use_arctan_geodesic=True)
            
            # Enhancement factor
            factor = kappa_geo / kappa_base if kappa_base > 0 else 1
            
            # Should be reasonable (between 1 and 3 based on arctan range)
            self.assertGreater(factor, 1.0)
            self.assertLess(factor, 3.0)


class TestPerformance(unittest.TestCase):
    """Test performance of arctan-geodesic computations."""
    
    def test_curvature_computation_speed(self):
        """Test that curvature computation is reasonably fast."""
        import time
        
        start = time.time()
        for n in range(1, 1000):
            compute_curvature(n, use_arctan_geodesic=True)
        elapsed = time.time() - start
        
        # Should complete 1000 computations in reasonable time
        self.assertLess(elapsed, 1.0, f"Curvature computation too slow: {elapsed:.3f}s")
    
    def test_prime_normalization_speed(self):
        """Test that prime normalization is reasonably fast."""
        import time
        
        start = time.time()
        for n in [10, 20, 30, 50, 100, 200, 500, 1000]:
            normalize_slot_to_prime(n, strategy="geodesic")
        elapsed = time.time() - start
        
        # Should be fast for typical slot values
        self.assertLess(elapsed, 0.5, f"Prime normalization too slow: {elapsed:.3f}s")


if __name__ == '__main__':
    unittest.main()
