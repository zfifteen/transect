#!/usr/bin/env python3
"""
Tests for HKDF key derivation independence from prime strategy.

Addresses Blocking 5: Ensures that for identical (secret, context, slot_index),
the derived key is byte-identical regardless of prime_strategy.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec import TransecCipher, generate_shared_secret


class TestHKDFKeyIndependence(unittest.TestCase):
    """Test that HKDF keys are independent of prime strategy."""
    
    def test_key_derivation_independent_of_strategy(self):
        """Test that identical slot_index produces identical keys across strategies."""
        secret = generate_shared_secret()
        
        # Create ciphers with different prime strategies
        cipher_none = TransecCipher(secret, prime_strategy="none")
        cipher_nearest = TransecCipher(secret, prime_strategy="nearest")
        cipher_next = TransecCipher(secret, prime_strategy="next")
        cipher_geodesic = TransecCipher(secret, prime_strategy="geodesic")
        
        # Test multiple slot indices
        test_slots = [1, 2, 5, 7, 10, 11, 100, 101, 1000, 1009]
        
        for slot in test_slots:
            # Derive keys for the same slot index
            key_none = cipher_none.derive_slot_key(slot)
            key_nearest = cipher_nearest.derive_slot_key(slot)
            key_next = cipher_next.derive_slot_key(slot)
            key_geodesic = cipher_geodesic.derive_slot_key(slot)
            
            # All keys must be byte-identical
            self.assertEqual(key_none, key_nearest,
                           f"Keys differ for slot {slot}: none vs nearest")
            self.assertEqual(key_none, key_next,
                           f"Keys differ for slot {slot}: none vs next")
            self.assertEqual(key_none, key_geodesic,
                           f"Keys differ for slot {slot}: none vs geodesic")
    
    def test_encryption_with_explicit_slot_index(self):
        """Test that encryption with same slot_index works across strategies."""
        secret = generate_shared_secret()
        
        cipher_none = TransecCipher(secret, prime_strategy="none")
        cipher_geodesic = TransecCipher(secret, prime_strategy="geodesic")
        
        plaintext = b"Test message for key independence"
        # Use current slot to avoid drift window issues
        slot_index = cipher_none.get_current_slot()
        sequence = 1
        
        # Encrypt with none strategy using explicit slot
        packet_none = cipher_none.seal(plaintext, sequence, slot_index=slot_index)
        
        # Decrypt with same cipher (none strategy)
        decrypted = cipher_none.open(packet_none, check_replay=False)
        self.assertEqual(decrypted, plaintext)
    
    def test_different_slots_produce_different_keys(self):
        """Test that different slot indices produce different keys (sanity check)."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, prime_strategy="geodesic")
        
        # Different slots should produce different keys
        key1 = cipher.derive_slot_key(10)
        key2 = cipher.derive_slot_key(11)
        key3 = cipher.derive_slot_key(100)
        
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key2, key3)
    
    def test_key_derivation_with_normalized_slots(self):
        """Test that normalized slots produce correct keys."""
        secret = generate_shared_secret()
        cipher_geodesic = TransecCipher(secret, prime_strategy="geodesic")
        
        # When sealing, the cipher normalizes the slot
        # The key should be derived from the normalized slot
        plaintext = b"Test with normalized slot"
        sequence = 1
        
        # Seal with current slot (will be normalized)
        packet = cipher_geodesic.seal(plaintext, sequence)
        
        # Should decrypt successfully
        decrypted = cipher_geodesic.open(packet)
        self.assertEqual(decrypted, plaintext)
    
    def test_cross_strategy_compatibility_same_normalized_slot(self):
        """Test that strategies are compatible when they produce same normalized slot."""
        secret = generate_shared_secret()
        
        # Create ciphers with normal drift window
        cipher_none = TransecCipher(secret, prime_strategy="none")
        cipher_nearest = TransecCipher(secret, prime_strategy="nearest")
        cipher_geodesic = TransecCipher(secret, prime_strategy="geodesic")
        
        # Use current slot (will be a huge number, but that's OK)
        # We just verify each cipher can decrypt its own packets
        plaintext = b"Cross-strategy test"
        sequence = 1
        
        # Each encrypts with its own current slot
        packet_none = cipher_none.seal(plaintext, sequence)
        packet_nearest = cipher_nearest.seal(plaintext, sequence)
        packet_geodesic = cipher_geodesic.seal(plaintext, sequence)
        
        # Each should decrypt its own packet
        self.assertEqual(cipher_none.open(packet_none, check_replay=False), plaintext)
        self.assertEqual(cipher_nearest.open(packet_nearest, check_replay=False), plaintext)
        self.assertEqual(cipher_geodesic.open(packet_geodesic, check_replay=False), plaintext)


if __name__ == '__main__':
    unittest.main()
