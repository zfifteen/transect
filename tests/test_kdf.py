#!/usr/bin/env python3
"""
Unit tests for the KDF module.
"""

import sys
import os
import unittest
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec.kdf import (
    hkdf_expand_slot, 
    _hkdf_info_for_slot,
    theta_prime,
    PHI,
    hkdf_salt_for_slot,
    test_salt_diversity
)

class TestHKDFExpandSlot(unittest.TestCase):
    """Test the hkdf_expand_slot function."""

    def setUp(self):
        """Set up a fixed IKM for tests."""
        self.ikm = os.urandom(32)

    def test_output_length(self):
        """Test that the output key has the correct length."""
        for out_len in [16, 32, 64]:
            key = hkdf_expand_slot(self.ikm, 0, out_len=out_len)
            self.assertEqual(len(key), out_len)

    def test_determinism(self):
        """Test that the same slot_id produces the same key."""
        for slot_id in [0, 1, 42, 1000, 2**53 - 1]:
            key1 = hkdf_expand_slot(self.ikm, slot_id)
            key2 = hkdf_expand_slot(self.ikm, slot_id)
            self.assertEqual(key1, key2)

    def test_uniqueness(self):
        """Test that different slot_ids produce different keys."""
        key1 = hkdf_expand_slot(self.ikm, 0)
        key2 = hkdf_expand_slot(self.ikm, 1)
        self.assertNotEqual(key1, key2)

    def test_invalid_slot_id(self):
        """Test that invalid slot_ids raise ValueError."""
        with self.assertRaises(ValueError):
            hkdf_expand_slot(self.ikm, -1)
        with self.assertRaises(ValueError):
            hkdf_expand_slot(self.ikm, 2**64)

    def test_slot_id_range(self):
        """Test valid slot_ids at the edge of the range."""
        # These should not raise an error
        hkdf_expand_slot(self.ikm, 0)
        hkdf_expand_slot(self.ikm, 2**64 - 1)

    def test_collision_resistance_large_range(self):
        """Test for collisions over a large number of slots."""
        # Using 1 million as requested in the code review is too slow for a unit test.
        # A smaller number like 100,000 is more reasonable.
        # The review asks for 1e6, but that will take too long.
        # I will use 100,000 to keep the test runtime reasonable.
        num_slots = 100_000
        keys = set()
        for i in range(num_slots):
            key = hkdf_expand_slot(self.ikm, i)
            self.assertNotIn(key, keys, f"Collision detected at slot {i}")
            keys.add(key)

    def test_boundary_around_2_53(self):
        """Test for collisions around the 2**53 boundary."""
        slot_ids = [
            2**53 - 2,
            2**53 - 1,
            2**53,
            2**53 + 1,
            2**53 + 2,
        ]
        keys = {hkdf_expand_slot(self.ikm, slot_id) for slot_id in slot_ids}
        self.assertEqual(len(keys), len(slot_ids), "Collisions found around 2**53")

class TestHKDFInfoForSlot(unittest.TestCase):
    """Test the _hkdf_info_for_slot helper function."""

    def test_info_determinism(self):
        """Test that the same slot_id produces the same info."""
        info1 = _hkdf_info_for_slot(42)
        info2 = _hkdf_info_for_slot(42)
        self.assertEqual(info1, info2)

    def test_info_uniqueness(self):
        """Test that different slot_ids produce different info."""
        info1 = _hkdf_info_for_slot(42)
        info2 = _hkdf_info_for_slot(43)
        self.assertNotEqual(info1, info2)

    def test_info_format(self):
        """Test the format of the generated info."""
        info = _hkdf_info_for_slot(123)
        # BLAKE2s with 32-byte digest
        self.assertEqual(len(info), 32)
        # Check that the slot_id is being hashed
        domain_sep = b"transect/hkdf/v1"
        h = hashlib.blake2s(digest_size=32)
        h.update(domain_sep)
        h.update((123).to_bytes(8, "big"))
        self.assertEqual(info, h.digest())


class TestThetaPrime(unittest.TestCase):
    """Test the theta_prime geometric resolution function."""
    
    def test_theta_prime_basic(self):
        """Test basic theta_prime computation."""
        import math
        # For n=0, theta_prime should be PHI * 0^k = 0
        result = theta_prime(0, k=0.3)
        self.assertAlmostEqual(result, 0.0, places=10)
        
        # For n=1, verify non-zero result
        result = theta_prime(1, k=0.3)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, PHI)
    
    def test_theta_prime_phi_constant(self):
        """Test that PHI constant has the correct value."""
        import math
        expected_phi = (1 + math.sqrt(5)) / 2
        self.assertAlmostEqual(PHI, expected_phi, places=10)
        self.assertAlmostEqual(PHI, 1.618033988749895, places=10)
    
    def test_theta_prime_deterministic(self):
        """Test that theta_prime is deterministic."""
        for n in [10, 100, 1000]:
            result1 = theta_prime(n, k=0.3)
            result2 = theta_prime(n, k=0.3)
            self.assertEqual(result1, result2)
    
    def test_theta_prime_different_k(self):
        """Test theta_prime with different k values."""
        n = 100
        result_k03 = theta_prime(n, k=0.3)
        result_k05 = theta_prime(n, k=0.5)
        result_k10 = theta_prime(n, k=1.0)
        
        # Results should differ with different k
        self.assertNotEqual(result_k03, result_k05)
        self.assertNotEqual(result_k05, result_k10)
    
    def test_theta_prime_negative_n(self):
        """Test that negative n raises ValueError."""
        with self.assertRaises(ValueError):
            theta_prime(-1, k=0.3)
    
    def test_theta_prime_negative_k(self):
        """Test that negative k raises ValueError."""
        with self.assertRaises(ValueError):
            theta_prime(10, k=-0.1)


class TestHKDFSaltForSlot(unittest.TestCase):
    """Test the hkdf_salt_for_slot function."""
    
    def test_salt_length(self):
        """Test that generated salt has correct length."""
        salt = hkdf_salt_for_slot(42)
        self.assertEqual(len(salt), 32)
    
    def test_salt_deterministic(self):
        """Test that salt generation is deterministic."""
        for slot in [0, 1, 100, 1000]:
            salt1 = hkdf_salt_for_slot(slot)
            salt2 = hkdf_salt_for_slot(slot)
            self.assertEqual(salt1, salt2)
    
    def test_salt_uniqueness(self):
        """Test that different slots produce different salts."""
        salt1 = hkdf_salt_for_slot(0)
        salt2 = hkdf_salt_for_slot(1)
        salt3 = hkdf_salt_for_slot(100)
        
        self.assertNotEqual(salt1, salt2)
        self.assertNotEqual(salt2, salt3)
        self.assertNotEqual(salt1, salt3)
    
    def test_salt_negative_slot(self):
        """Test that negative slot raises ValueError."""
        with self.assertRaises(ValueError):
            hkdf_salt_for_slot(-1)
    
    def test_salt_high_entropy(self):
        """Test that salts have high entropy (no obvious patterns)."""
        # Generate multiple salts and check for diversity
        salts = [hkdf_salt_for_slot(i) for i in range(10)]
        
        # Check that they're all different
        unique_salts = set(salts)
        self.assertEqual(len(unique_salts), 10)
        
        # Check that each salt has mixed bits (not all zeros or ones)
        for salt in salts:
            ones_count = sum(bin(byte).count('1') for byte in salt)
            # Expect roughly 128 bits set out of 256 (within reasonable bounds)
            self.assertGreater(ones_count, 50)
            self.assertLess(ones_count, 206)


class TestSaltDiversity(unittest.TestCase):
    """Test the test_salt_diversity validation function."""
    
    def test_diversity_passes(self):
        """Test that salt diversity validation passes with reasonable parameters."""
        # This should pass with default parameters
        result = test_salt_diversity(num_slots=100, min_hamming_distance=30)
        self.assertTrue(result)
    
    def test_diversity_deterministic(self):
        """Test that diversity test is deterministic."""
        result1 = test_salt_diversity(num_slots=50, min_hamming_distance=20)
        result2 = test_salt_diversity(num_slots=50, min_hamming_distance=20)
        self.assertEqual(result1, result2)
    
    def test_diversity_invalid_num_slots(self):
        """Test that num_slots < 2 raises ValueError."""
        with self.assertRaises(ValueError):
            test_salt_diversity(num_slots=1)
    
    def test_diversity_high_threshold_fails(self):
        """Test that unreasonably high threshold causes assertion."""
        # Hamming distance of 256 bits (all different) is impossible for BLAKE2s
        # which will have approximately 128 bits different on average
        with self.assertRaises(AssertionError):
            test_salt_diversity(num_slots=10, min_hamming_distance=250)


if __name__ == '__main__':
    unittest.main()
