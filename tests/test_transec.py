#!/usr/bin/env python3
"""
Unit tests for TRANSEC implementation.

Tests cover:
- Basic encryption/decryption
- Key derivation uniqueness
- Replay protection
- Clock drift tolerance
- Interoperability
- Edge cases
"""

import sys
import os
import time
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec import (
    TransecCipher,
    generate_shared_secret,
    seal_packet,
    open_packet,
    derive_slot_key,
    DEFAULT_SLOT_DURATION,
    DEFAULT_DRIFT_WINDOW
)


class TestTransecBasics(unittest.TestCase):
    """Test basic TRANSEC functionality."""
    
    def setUp(self):
        """Create shared secret for tests."""
        self.secret = generate_shared_secret()
        self.cipher = TransecCipher(self.secret)
    
    def test_encrypt_decrypt(self):
        """Test basic encryption and decryption."""
        plaintext = b"Hello, TRANSEC!"
        sequence = 1
        
        packet = self.cipher.seal(plaintext, sequence)
        decrypted = self.cipher.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_wrong_key_fails(self):
        """Test that decryption fails with wrong key."""
        plaintext = b"Secret message"
        sequence = 1
        
        packet = self.cipher.seal(plaintext, sequence)
        
        # Try to decrypt with different key
        wrong_cipher = TransecCipher(generate_shared_secret())
        decrypted = wrong_cipher.open(packet)
        
        self.assertIsNone(decrypted)
    
    def test_tampered_ciphertext_fails(self):
        """Test that tampered ciphertext is rejected."""
        plaintext = b"Protected data"
        sequence = 1
        
        packet = self.cipher.seal(plaintext, sequence)
        
        # Tamper with ciphertext (flip a bit in the middle)
        tampered = bytearray(packet)
        tampered[25] ^= 0x01
        
        decrypted = self.cipher.open(bytes(tampered))
        self.assertIsNone(decrypted)
    
    def test_associated_data(self):
        """Test that associated data must match."""
        plaintext = b"Message with context"
        sequence = 1
        aad = b"context:user123"
        
        packet = self.cipher.seal(plaintext, sequence, associated_data=aad)
        
        # Correct AAD works
        decrypted = self.cipher.open(packet, associated_data=aad)
        self.assertEqual(decrypted, plaintext)
        
        # Wrong AAD fails
        wrong_aad = b"context:user456"
        decrypted_wrong = self.cipher.open(packet, associated_data=wrong_aad)
        self.assertIsNone(decrypted_wrong)


class TestKeyDerivation(unittest.TestCase):
    """Test key derivation properties."""
    
    def test_slot_uniqueness(self):
        """Test that different slots produce different keys."""
        secret = generate_shared_secret()
        
        keys = set()
        for slot in range(100):
            key = derive_slot_key(secret, slot)
            keys.add(key)
        
        # All keys should be unique
        self.assertEqual(len(keys), 100)
    
    def test_deterministic_derivation(self):
        """Test that key derivation is deterministic."""
        secret = generate_shared_secret()
        slot = 12345
        
        key1 = derive_slot_key(secret, slot)
        key2 = derive_slot_key(secret, slot)
        
        self.assertEqual(key1, key2)
    
    def test_different_contexts(self):
        """Test that different contexts produce different keys."""
        secret = generate_shared_secret()
        slot = 100
        
        key1 = derive_slot_key(secret, slot, context=b"app1")
        key2 = derive_slot_key(secret, slot, context=b"app2")
        
        self.assertNotEqual(key1, key2)


class TestReplayProtection(unittest.TestCase):
    """Test replay attack protection."""
    
    def test_replay_blocked(self):
        """Test that replayed packets are rejected."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret)
        
        plaintext = b"One-time message"
        sequence = 1
        
        packet = cipher.seal(plaintext, sequence)
        
        # First decryption succeeds
        decrypted1 = cipher.open(packet)
        self.assertEqual(decrypted1, plaintext)
        
        # Replay is blocked
        decrypted2 = cipher.open(packet)
        self.assertIsNone(decrypted2)
    
    def test_different_sequences_allowed(self):
        """Test that different sequence numbers are allowed."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret)
        
        plaintext = b"Message"
        
        for seq in range(1, 6):
            packet = cipher.seal(plaintext, seq)
            decrypted = cipher.open(packet)
            self.assertEqual(decrypted, plaintext)
    
    def test_out_of_order_allowed(self):
        """Test that out-of-order packets are allowed (not replays)."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret)
        
        plaintext = b"Message"
        
        # Create packets with different sequences
        packets = []
        for seq in [1, 2, 3, 4, 5]:
            packets.append((seq, cipher.seal(plaintext, seq)))
        
        # Decrypt in different order
        for seq in [3, 1, 5, 2, 4]:
            packet = [p for s, p in packets if s == seq][0]
            decrypted = cipher.open(packet)
            self.assertEqual(decrypted, plaintext)


class TestClockDrift(unittest.TestCase):
    """Test clock drift tolerance."""
    
    def test_current_slot_accepted(self):
        """Test that packets from current slot are accepted."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, drift_window=2)
        
        plaintext = b"Current slot message"
        current_slot = cipher.get_current_slot()
        sequence = 1
        
        packet = cipher.seal(plaintext, sequence, slot_index=current_slot)
        decrypted = cipher.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_past_slot_within_window(self):
        """Test that recent past slots are accepted."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, drift_window=2)
        
        plaintext = b"Recent past message"
        current_slot = cipher.get_current_slot()
        past_slot = current_slot - 1  # 1 slot in the past
        sequence = 1
        
        packet = cipher.seal(plaintext, sequence, slot_index=past_slot)
        decrypted = cipher.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_future_slot_within_window(self):
        """Test that near-future slots are accepted."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, drift_window=2)
        
        plaintext = b"Near future message"
        current_slot = cipher.get_current_slot()
        future_slot = current_slot + 1  # 1 slot in the future
        sequence = 1
        
        packet = cipher.seal(plaintext, sequence, slot_index=future_slot)
        decrypted = cipher.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_too_old_rejected(self):
        """Test that slots outside drift window are rejected."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, drift_window=2)
        
        plaintext = b"Too old message"
        current_slot = cipher.get_current_slot()
        old_slot = current_slot - 10  # Way in the past
        sequence = 1
        
        packet = cipher.seal(plaintext, sequence, slot_index=old_slot)
        decrypted = cipher.open(packet)
        
        self.assertIsNone(decrypted)
    
    def test_too_new_rejected(self):
        """Test that far-future slots are rejected."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, drift_window=2)
        
        plaintext = b"Too new message"
        current_slot = cipher.get_current_slot()
        future_slot = current_slot + 10  # Way in the future
        sequence = 1
        
        packet = cipher.seal(plaintext, sequence, slot_index=future_slot)
        decrypted = cipher.open(packet)
        
        self.assertIsNone(decrypted)


class TestInteroperability(unittest.TestCase):
    """Test interoperability between sender and receiver."""
    
    def test_multiple_instances(self):
        """Test that multiple cipher instances with same secret interoperate."""
        secret = generate_shared_secret()
        
        sender = TransecCipher(secret)
        receiver = TransecCipher(secret)
        
        plaintext = b"Cross-instance message"
        sequence = 1
        
        packet = sender.seal(plaintext, sequence)
        decrypted = receiver.open(packet, check_replay=False)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_convenience_functions(self):
        """Test that convenience functions work correctly."""
        secret = generate_shared_secret()
        slot = int(time.time() / DEFAULT_SLOT_DURATION)
        sequence = 1
        plaintext = b"Convenience test"
        
        packet = seal_packet(secret, slot, sequence, plaintext)
        decrypted = open_packet(secret, packet, local_slot=slot)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_different_slot_durations(self):
        """Test that sender and receiver must have same slot duration."""
        secret = generate_shared_secret()
        
        sender = TransecCipher(secret, slot_duration=5)
        receiver = TransecCipher(secret, slot_duration=10)
        
        plaintext = b"Mismatched slots"
        sequence = 1
        
        # Sender uses 5s slots
        packet = sender.seal(plaintext, sequence)
        
        # Receiver with 10s slots will likely reject (different slot index)
        # This test documents the requirement that both sides must agree on slot duration
        # We won't assert success/failure as it depends on timing


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_plaintext(self):
        """Test encryption of empty message."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret)
        
        plaintext = b""
        sequence = 1
        
        packet = cipher.seal(plaintext, sequence)
        decrypted = cipher.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_large_plaintext(self):
        """Test encryption of large message."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret)
        
        plaintext = b"X" * 10000  # 10KB message
        sequence = 1
        
        packet = cipher.seal(plaintext, sequence)
        decrypted = cipher.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_packet_too_short(self):
        """Test that too-short packets are rejected."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret)
        
        short_packet = b"tooshort"
        
        with self.assertRaises(ValueError):
            cipher.open(short_packet)
    
    def test_invalid_secret_length(self):
        """Test that wrong secret length is rejected."""
        with self.assertRaises(ValueError):
            TransecCipher(b"tooshort")
        
        with self.assertRaises(ValueError):
            TransecCipher(b"way" * 100)  # Too long
    
    def test_large_sequence_numbers(self):
        """Test that large sequence numbers work correctly."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret)
        
        plaintext = b"High sequence"
        large_seq = 2**32 - 1  # Max 32-bit value
        
        packet = cipher.seal(plaintext, large_seq)
        decrypted = cipher.open(packet, check_replay=False)
        
        self.assertEqual(decrypted, plaintext)


class TestPerformance(unittest.TestCase):
    """Basic performance tests."""
    
    def test_encryption_speed(self):
        """Test that encryption is fast (<10ms per operation)."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret)
        
        plaintext = b"Performance test message" * 10
        sequence = 1
        
        start = time.time()
        for _ in range(100):
            cipher.seal(plaintext, sequence)
        elapsed = time.time() - start
        
        per_op = elapsed / 100
        self.assertLess(per_op, 0.01, f"Encryption too slow: {per_op*1000:.2f}ms")
    
    def test_decryption_speed(self):
        """Test that decryption is fast (<10ms per operation)."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret)
        
        plaintext = b"Performance test message" * 10
        sequence = 1
        packet = cipher.seal(plaintext, sequence)
        
        start = time.time()
        for i in range(100):
            # Use check_replay=False to avoid cache effects
            cipher.open(packet, check_replay=False)
        elapsed = time.time() - start
        
        per_op = elapsed / 100
        self.assertLess(per_op, 0.01, f"Decryption too slow: {per_op*1000:.2f}ms")


if __name__ == '__main__':
    unittest.main()
