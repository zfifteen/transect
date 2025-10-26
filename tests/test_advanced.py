#!/usr/bin/env python3
"""
Unit tests for TRANSEC advanced features (Adaptive and OTAR).
"""

import sys
import os
import time
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec import generate_shared_secret, ADAPTIVE_AVAILABLE, OTAR_AVAILABLE

if ADAPTIVE_AVAILABLE:
    from transec import AdaptiveTransecCipher

if OTAR_AVAILABLE:
    from transec import OTARTransecCipher


@unittest.skipIf(not ADAPTIVE_AVAILABLE, "Adaptive features not available")
class TestAdaptiveTransec(unittest.TestCase):
    """Test adaptive slot duration features."""
    
    def setUp(self):
        """Create shared secret for tests."""
        self.secret = generate_shared_secret()
    
    def test_adaptive_cipher_creation(self):
        """Test creating adaptive cipher with various parameters."""
        # Basic creation
        cipher = AdaptiveTransecCipher(self.secret)
        self.assertIsNotNone(cipher)
        
        # With jitter range
        cipher = AdaptiveTransecCipher(
            self.secret,
            jitter_range=(2, 10)
        )
        self.assertEqual(cipher.jitter_range, (2, 10))
    
    def test_invalid_jitter_range(self):
        """Test that invalid jitter ranges are rejected."""
        with self.assertRaises(ValueError):
            AdaptiveTransecCipher(self.secret, jitter_range=(10, 2))
        
        with self.assertRaises(ValueError):
            AdaptiveTransecCipher(self.secret, jitter_range=(0, 5))
    
    def test_adaptive_slot_duration(self):
        """Test that slot durations vary deterministically."""
        cipher = AdaptiveTransecCipher(
            self.secret,
            jitter_range=(2, 10)
        )
        
        # Get durations for different epochs
        durations = [cipher.get_adaptive_slot_duration(i) for i in range(100)]
        
        # Should have variation
        unique_durations = set(durations)
        self.assertGreater(len(unique_durations), 1, "Should have multiple different durations")
        
        # All durations should be in range
        for dur in durations:
            self.assertGreaterEqual(dur, 2)
            self.assertLessEqual(dur, 10)
    
    def test_deterministic_jitter(self):
        """Test that jitter is deterministic (same epoch = same duration)."""
        cipher = AdaptiveTransecCipher(self.secret, jitter_range=(2, 10))
        
        # Same epoch should give same duration
        dur1 = cipher.get_adaptive_slot_duration(42)
        dur2 = cipher.get_adaptive_slot_duration(42)
        self.assertEqual(dur1, dur2)
        
        # Different epochs may give different durations
        dur3 = cipher.get_adaptive_slot_duration(43)
        # May or may not be different, but should be valid
        self.assertGreaterEqual(dur3, 2)
        self.assertLessEqual(dur3, 10)
    
    def test_adaptive_encrypt_decrypt(self):
        """Test encryption/decryption with adaptive cipher."""
        sender = AdaptiveTransecCipher(self.secret, jitter_range=(2, 10))
        receiver = AdaptiveTransecCipher(self.secret, jitter_range=(2, 10))
        
        plaintext = b"Test message with adaptive timing"
        sequence = 1
        
        packet = sender.seal(plaintext, sequence)
        decrypted = receiver.open(packet)
        
        self.assertEqual(decrypted, plaintext)


@unittest.skipIf(not OTAR_AVAILABLE, "OTAR features not available")
class TestOTARTransec(unittest.TestCase):
    """Test OTAR (over-the-air rekeying) features."""
    
    def setUp(self):
        """Create shared secret for tests."""
        self.secret = generate_shared_secret()
    
    def test_otar_cipher_creation(self):
        """Test creating OTAR cipher with various parameters."""
        # Basic creation
        cipher = OTARTransecCipher(self.secret, auto_refresh=False)
        self.assertIsNotNone(cipher)
        self.assertEqual(cipher.generation, 0)
        
        # With auto-refresh
        cipher = OTARTransecCipher(
            self.secret,
            refresh_interval=60,
            auto_refresh=True
        )
        self.assertTrue(cipher.auto_refresh)
        self.assertIsNotNone(cipher.next_secret)
    
    def test_invalid_refresh_interval(self):
        """Test that invalid refresh intervals are rejected."""
        with self.assertRaises(ValueError):
            OTARTransecCipher(self.secret, refresh_interval=30)
    
    def test_manual_refresh(self):
        """Test manual key refresh."""
        cipher = OTARTransecCipher(self.secret, auto_refresh=False)
        
        # Initial state
        initial_secret = cipher.current_secret
        self.assertEqual(cipher.generation, 0)
        
        # Perform manual refresh
        cipher.manual_refresh()
        
        # Verify state changed
        self.assertEqual(cipher.generation, 1)
        self.assertNotEqual(cipher.current_secret, initial_secret)
    
    def test_generation_tracking(self):
        """Test that generation numbers increment correctly."""
        cipher = OTARTransecCipher(self.secret, auto_refresh=False)
        
        self.assertEqual(cipher.get_generation(), 0)
        
        cipher.manual_refresh()
        self.assertEqual(cipher.get_generation(), 1)
        
        cipher.manual_refresh()
        self.assertEqual(cipher.get_generation(), 2)
    
    def test_otar_encrypt_decrypt_same_generation(self):
        """Test encryption/decryption within same generation."""
        sender = OTARTransecCipher(self.secret, auto_refresh=False)
        receiver = OTARTransecCipher(self.secret, auto_refresh=False)
        
        plaintext = b"Test message with OTAR"
        sequence = 1
        
        packet = sender.seal(plaintext, sequence)
        decrypted = receiver.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_otar_encrypt_decrypt_after_refresh(self):
        """Test encryption/decryption after manual refresh."""
        sender = OTARTransecCipher(self.secret, auto_refresh=False)
        receiver = OTARTransecCipher(self.secret, auto_refresh=False)
        
        # Refresh both to generation 1
        sender.manual_refresh()
        receiver.manual_refresh()
        
        plaintext = b"Message after refresh"
        sequence = 1
        
        packet = sender.seal(plaintext, sequence)
        decrypted = receiver.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_generation_mismatch_handling(self):
        """Test handling of generation mismatch between sender/receiver."""
        sender = OTARTransecCipher(self.secret, auto_refresh=False)
        receiver = OTARTransecCipher(self.secret, auto_refresh=False)
        
        # Sender refreshes, receiver doesn't
        sender.manual_refresh()
        
        plaintext = b"Message from new generation"
        sequence = 1
        
        packet = sender.seal(plaintext, sequence)
        
        # Receiver at old generation should fail to decrypt
        # (current implementation doesn't maintain previous secrets)
        decrypted = receiver.open(packet)
        self.assertIsNone(decrypted, "Should fail to decrypt generation mismatch")
    
    def test_time_until_refresh(self):
        """Test refresh timing calculations."""
        cipher = OTARTransecCipher(
            self.secret,
            refresh_interval=3600,
            auto_refresh=True
        )
        
        # Should have nearly full interval remaining
        remaining = cipher.time_until_refresh()
        self.assertGreater(remaining, 3500)
        self.assertLessEqual(remaining, 3600)
        
        # Disabled auto-refresh should return infinity
        cipher_no_auto = OTARTransecCipher(
            self.secret,
            auto_refresh=False
        )
        self.assertEqual(cipher_no_auto.time_until_refresh(), float('inf'))


class TestFeatureAvailability(unittest.TestCase):
    """Test feature availability flags."""
    
    def test_adaptive_availability(self):
        """Test adaptive feature availability flag."""
        # Should be a boolean
        self.assertIsInstance(ADAPTIVE_AVAILABLE, bool)
    
    def test_otar_availability(self):
        """Test OTAR feature availability flag."""
        # Should be a boolean
        self.assertIsInstance(OTAR_AVAILABLE, bool)


if __name__ == '__main__':
    unittest.main()
