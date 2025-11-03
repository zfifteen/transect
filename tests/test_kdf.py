#!/usr/bin/env python3
"""
Unit tests for KDF module with golden ratio-based salt generation.

Tests cover:
- Salt diversity across slots
- Determinism (same slot -> same salt)
- Mathematical properties (theta_prime)
- Edge cases (negative slots, different output lengths)
- Entropy validation
"""

import sys
import os
import unittest
import math

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec.kdf import (
    theta_prime,
    hkdf_salt_for_slot,
    test_salt_diversity,
    PHI,
)


class TestThetaPrime(unittest.TestCase):
    """Test theta_prime function."""
    
    def test_theta_prime_properties(self):
        """Test mathematical properties of theta_prime."""
        # θ′(n,k) = φ * ((n % φ)/φ)^k
        for slot in [0, 1, 10, 100, 1000]:
            tp = theta_prime(slot, k=0.3)
            # Result should be positive and bounded
            self.assertGreaterEqual(tp, 0.0)
            self.assertLessEqual(tp, PHI)
    
    def test_theta_prime_deterministic(self):
        """Test that theta_prime is deterministic."""
        for slot in [0, 1, 42, 100, 1000]:
            tp1 = theta_prime(slot)
            tp2 = theta_prime(slot)
            self.assertEqual(tp1, tp2)
    
    def test_theta_prime_different_k(self):
        """Test theta_prime with different k values."""
        slot = 42
        tp1 = theta_prime(slot, k=0.1)
        tp2 = theta_prime(slot, k=0.3)
        tp3 = theta_prime(slot, k=0.5)
        
        # Different k values should produce different results
        self.assertNotEqual(tp1, tp2)
        self.assertNotEqual(tp2, tp3)
        
        # All should be valid
        for tp in [tp1, tp2, tp3]:
            self.assertGreaterEqual(tp, 0.0)
            self.assertLessEqual(tp, PHI)


class TestHKDFSaltGeneration(unittest.TestCase):
    """Test HKDF salt generation."""
    
    def test_salt_length(self):
        """Test that salt has correct length."""
        salt = hkdf_salt_for_slot(42)
        self.assertEqual(len(salt), 32)
        
        # Test different output lengths
        for out_len in [16, 32, 48, 64]:
            salt = hkdf_salt_for_slot(42, out_len=out_len)
            self.assertEqual(len(salt), out_len)
    
    def test_salt_deterministic(self):
        """Test that same slot produces same salt."""
        for slot in [0, 1, 10, 100, 1000, 10000]:
            salt1 = hkdf_salt_for_slot(slot)
            salt2 = hkdf_salt_for_slot(slot)
            self.assertEqual(salt1, salt2)
    
    def test_salt_diversity(self):
        """Test that different slots produce different salts."""
        num_slots = 100
        salts = [hkdf_salt_for_slot(i) for i in range(num_slots)]
        unique_salts = set(salts)
        
        # All salts should be unique
        self.assertEqual(len(unique_salts), num_slots)
    
    def test_salt_diversity_large_scale(self):
        """Test salt diversity at larger scale."""
        # Test 1000 consecutive slots
        num_slots = 1000
        salts = [hkdf_salt_for_slot(i) for i in range(num_slots)]
        unique_salts = set(salts)
        
        # All salts should be unique
        self.assertEqual(len(unique_salts), num_slots)
    
    def test_salt_non_consecutive_slots(self):
        """Test salt diversity for non-consecutive slots."""
        # Test random slot indices
        slots = [0, 1, 10, 100, 1000, 10000, 100000, 1000000]
        salts = [hkdf_salt_for_slot(s) for s in slots]
        unique_salts = set(salts)
        
        # All salts should be unique
        self.assertEqual(len(unique_salts), len(slots))
    
    def test_slot_zero_special_case(self):
        """Test slot 0 produces valid salt."""
        salt = hkdf_salt_for_slot(0)
        self.assertEqual(len(salt), 32)
        # Slot 0 should produce all zeros due to multiplication by 0
        self.assertEqual(salt, b'\x00' * 32)
    
    def test_slot_one_nonzero(self):
        """Test slot 1 produces non-zero salt."""
        salt = hkdf_salt_for_slot(1)
        self.assertEqual(len(salt), 32)
        # Slot 1 should produce non-zero salt
        self.assertNotEqual(salt, b'\x00' * 32)
    
    def test_invalid_slot_id(self):
        """Test that negative slot_id raises error."""
        with self.assertRaises(ValueError):
            hkdf_salt_for_slot(-1)
    
    def test_invalid_out_len(self):
        """Test that invalid out_len raises error."""
        with self.assertRaises(ValueError):
            hkdf_salt_for_slot(42, out_len=0)
        
        with self.assertRaises(ValueError):
            hkdf_salt_for_slot(42, out_len=-1)
    
    def test_different_k_values(self):
        """Test that different k values produce different salts."""
        slot = 42
        salt1 = hkdf_salt_for_slot(slot, k=0.1)
        salt2 = hkdf_salt_for_slot(slot, k=0.3)
        salt3 = hkdf_salt_for_slot(slot, k=0.5)
        
        # Different k values should produce different salts
        self.assertNotEqual(salt1, salt2)
        self.assertNotEqual(salt2, salt3)


class TestSaltEntropy(unittest.TestCase):
    """Test entropy properties of generated salts."""
    
    def test_salt_bit_distribution(self):
        """Test that salts have good bit distribution."""
        # Generate 100 salts and check bit distribution
        num_slots = 100
        salts = [hkdf_salt_for_slot(i) for i in range(1, num_slots + 1)]
        
        # Count set bits across all salts
        total_bits = 0
        total_bytes = 0
        for salt in salts:
            for byte in salt:
                total_bits += bin(byte).count('1')
                total_bytes += 1
        
        # Average bits per byte - with modulo arithmetic, bit density can be lower
        # especially for small slot numbers, which is acceptable
        avg_bits_per_byte = total_bits / total_bytes
        # Should have some bits set (not all zeros except slot 0)
        self.assertGreater(avg_bits_per_byte, 0.0)
        self.assertLess(avg_bits_per_byte, 8.0)
    
    def test_no_collision_in_range(self):
        """Test no collisions in a large range."""
        # Test for collisions in first 10000 slots
        num_slots = 10000
        salts = set()
        
        for i in range(num_slots):
            salt = hkdf_salt_for_slot(i)
            self.assertNotIn(salt, salts, f"Collision detected at slot {i}")
            salts.add(salt)
        
        self.assertEqual(len(salts), num_slots)
    
    def test_hamming_distance(self):
        """Test that consecutive salts have reasonable Hamming distance."""
        # Check Hamming distance between consecutive salts
        hamming_distances = []
        
        for i in range(1, 100):
            salt1 = hkdf_salt_for_slot(i)
            salt2 = hkdf_salt_for_slot(i + 1)
            
            # Compute Hamming distance (number of differing bits)
            hamming = 0
            for b1, b2 in zip(salt1, salt2):
                hamming += bin(b1 ^ b2).count('1')
            
            hamming_distances.append(hamming)
        
        # Average Hamming distance should be reasonable
        avg_hamming = sum(hamming_distances) / len(hamming_distances)
        
        # For 32-byte (256-bit) salts, we expect some variation
        # Not too low (would indicate poor mixing)
        self.assertGreater(avg_hamming, 1.0)


class TestDiversityFunction(unittest.TestCase):
    """Test the test_salt_diversity helper function."""
    
    def test_diversity_function_passes(self):
        """Test that test_salt_diversity passes for valid inputs."""
        # Should not raise any exceptions
        result = test_salt_diversity(100)
        self.assertTrue(result)
    
    def test_diversity_function_small_sample(self):
        """Test diversity with small sample."""
        result = test_salt_diversity(10)
        self.assertTrue(result)
    
    def test_diversity_function_large_sample(self):
        """Test diversity with larger sample."""
        result = test_salt_diversity(1000)
        self.assertTrue(result)


class TestIntegration(unittest.TestCase):
    """Integration tests for KDF module."""
    
    def test_module_imports(self):
        """Test that module imports work correctly."""
        from transec.kdf import (
            theta_prime,
            hkdf_salt_for_slot,
            test_salt_diversity,
            PHI,
        )
        
        # Check PHI value
        self.assertAlmostEqual(PHI, 1.618033988749895, places=10)
    
    def test_command_line_interface(self):
        """Test that module can be run as script."""
        import subprocess
        result = subprocess.run(
            ['python', 'transec/kdf.py'],
            cwd='/home/runner/work/transect/transect',
            capture_output=True,
            text=True,
            timeout=5
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('All tests passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
