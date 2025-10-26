#!/usr/bin/env python3
"""
TRANSEC: Time-Synchronized Encryption Reference Implementation

Inspired by military frequency-hopping COMSEC, this module provides
zero-handshake encrypted messaging using deterministic time-sliced key rotation.

Key features:
- Zero-RTT encryption after bootstrap
- HKDF-SHA256 key derivation from shared secret + time slot
- ChaCha20-Poly1305 AEAD for authenticated encryption
- Replay protection via sequence tracking
- Configurable slot duration and drift tolerance
- Prime-based slot normalization for enhanced synchronization stability
"""

import os
import struct
import time
from typing import Tuple, Optional, Set
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# Import prime optimization utilities
try:
    from .prime_optimization import normalize_slot_to_prime
    PRIME_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PRIME_OPTIMIZATION_AVAILABLE = False
    def normalize_slot_to_prime(slot: int, strategy: str = "none") -> int:
        return slot


DEFAULT_CONTEXT = b"z-sandbox:transec:v1"
DEFAULT_SLOT_DURATION = 5  # seconds
DEFAULT_DRIFT_WINDOW = 2   # ±2 slots
DEFAULT_PRIME_STRATEGY = "none"  # Options: "none", "nearest", "next"


class TransecCipher:
    """
    Time-synchronized cipher implementing TRANSEC protocol.
    
    This cipher derives per-slot encryption keys from a shared secret
    and current time epoch, enabling zero-handshake communication.
    
    Supports prime-based slot normalization for enhanced synchronization
    stability through reduced discrete curvature.
    """
    
    def __init__(
        self,
        shared_secret: bytes,
        context: bytes = DEFAULT_CONTEXT,
        slot_duration: int = DEFAULT_SLOT_DURATION,
        drift_window: int = DEFAULT_DRIFT_WINDOW,
        prime_strategy: str = DEFAULT_PRIME_STRATEGY
    ):
        """
        Initialize TRANSEC cipher.
        
        Args:
            shared_secret: Pre-shared 256-bit (32 bytes) secret key
            context: Application-specific context string for key derivation
            slot_duration: Duration of each time slot in seconds
            drift_window: Number of slots to accept (±) for clock drift tolerance
            prime_strategy: Slot normalization strategy - "none" (default), "nearest", or "next"
                           - "none": Use raw slot indices (backward compatible)
                           - "nearest": Map to nearest prime for lower curvature
                           - "next": Map to next prime >= slot_index
        
        Raises:
            ValueError: If shared_secret is not 32 bytes
        """
        if len(shared_secret) != 32:
            raise ValueError("Shared secret must be exactly 32 bytes (256 bits)")
        
        if prime_strategy not in ["none", "nearest", "next"]:
            raise ValueError(f"Invalid prime_strategy: {prime_strategy}")
        
        self.shared_secret = shared_secret
        self.context = context
        self.slot_duration = slot_duration
        self.drift_window = drift_window
        self.prime_strategy = prime_strategy
        
        # Replay protection: track seen (slot_index, sequence) pairs
        self._seen_messages: Set[Tuple[int, int]] = set()
        self._cleanup_interval = 100  # Clean old entries every N messages
        self._message_count = 0
    
    def get_current_slot(self) -> int:
        """Get current time slot index based on system time (normalized)."""
        raw_slot = int(time.time() / self.slot_duration)
        return self._normalize_slot(raw_slot)
    
    def get_raw_current_slot(self) -> int:
        """Get current raw time slot index (before normalization)."""
        return int(time.time() / self.slot_duration)
    
    def _normalize_slot(self, slot_index: int) -> int:
        """
        Normalize slot index according to prime strategy.
        
        Args:
            slot_index: Raw slot index
        
        Returns:
            Normalized slot index (prime or original)
        """
        if self.prime_strategy == "none" or not PRIME_OPTIMIZATION_AVAILABLE:
            return slot_index
        
        return normalize_slot_to_prime(slot_index, strategy=self.prime_strategy)
    
    def derive_slot_key(self, slot_index: int) -> bytes:
        """
        Derive encryption key for a specific time slot.
        
        Args:
            slot_index: Time slot index (epoch time / slot_duration)
        
        Returns:
            32-byte key for ChaCha20-Poly1305
        """
        # Construct info parameter: context || slot_index
        info = self.context + struct.pack(">Q", slot_index)
        
        # Derive key using HKDF-SHA256
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=info,
        )
        return hkdf.derive(self.shared_secret)
    
    def seal(
        self,
        plaintext: bytes,
        sequence: int,
        associated_data: bytes = b"",
        slot_index: Optional[int] = None
    ) -> bytes:
        """
        Encrypt and authenticate a message.
        
        Args:
            plaintext: Data to encrypt
            sequence: Monotonically increasing sequence number
            associated_data: Additional data to authenticate (not encrypted)
            slot_index: Time slot to use (defaults to current slot, will be normalized)
        
        Returns:
            Encrypted packet: slot_index (8) || sequence (8) || random (4) || ciphertext + tag
        """
        if slot_index is None:
            slot_index = self.get_current_slot()
        else:
            # Normalize explicitly provided slot index
            slot_index = self._normalize_slot(slot_index)
        
        # Derive key for this slot
        key = self.derive_slot_key(slot_index)
        cipher = ChaCha20Poly1305(key)
        
        # Generate random component for nonce
        random_bytes = os.urandom(4)
        
        # Construct nonce: slot_index (4, lower bits) || sequence (4, lower bits) || random (4)
        # ChaCha20-Poly1305 requires 12-byte nonces
        nonce = (
            struct.pack(">I", slot_index & 0xFFFFFFFF) +
            struct.pack(">I", sequence & 0xFFFFFFFF) +
            random_bytes
        )
        
        # Construct AAD: slot_index (full 64-bit) || sequence (full 64-bit) || associated_data
        aad = (
            struct.pack(">Q", slot_index) +
            struct.pack(">Q", sequence) +
            associated_data
        )
        
        # Encrypt and authenticate
        ciphertext = cipher.encrypt(nonce, plaintext, aad)
        
        # Return: slot_index || sequence || random || ciphertext
        return struct.pack(">Q", slot_index) + struct.pack(">Q", sequence) + random_bytes + ciphertext
    
    def open(
        self,
        packet: bytes,
        associated_data: bytes = b"",
        check_replay: bool = True
    ) -> Optional[bytes]:
        """
        Decrypt and verify an authenticated message.
        
        Args:
            packet: Encrypted packet (slot_index || sequence || random || ciphertext + tag)
            associated_data: Additional authenticated data (must match seal)
            check_replay: Whether to perform replay protection check
        
        Returns:
            Decrypted plaintext, or None if authentication fails or replay detected
        
        Raises:
            ValueError: If packet format is invalid
        """
        if len(packet) < 20:
            raise ValueError("Packet too short (minimum 20 bytes for header)")
        
        # Extract slot_index, sequence, random, and ciphertext
        slot_index = struct.unpack(">Q", packet[:8])[0]
        sequence = struct.unpack(">Q", packet[8:16])[0]
        random_bytes = packet[16:20]
        ciphertext = packet[20:]
        
        # Validate slot is within acceptable window
        # The slot_index in the packet is already normalized by sender
        current_slot = self.get_current_slot()
        
        # For prime normalization, we need to check drift properly
        # by comparing normalized slots to normalized slots
        if self.prime_strategy != "none" and PRIME_OPTIMIZATION_AVAILABLE:
            # Get raw current slot (before normalization)
            raw_current = self.get_raw_current_slot()
            
            # Normalize the current slot using the same function as the sender
            normalized_current = self._normalize_slot(raw_current)
            
            # Compare normalized slots for drift
            slot_diff = abs(normalized_current - slot_index)
            
            # The drift window applies to the semantic drift between normalized slots
            # We use the standard drift_window since we're comparing like-to-like now
            if slot_diff > self.drift_window:
                return None
        else:
            # Standard drift check for non-normalized slots
            slot_diff = abs(current_slot - slot_index)
            
            if slot_diff > self.drift_window:
                # Slot outside acceptable window - reject
                return None
        
        # Check for replay attack
        if check_replay:
            msg_id = (slot_index, sequence)
            if msg_id in self._seen_messages:
                # Replay detected
                return None
            
            # Mark as seen
            self._seen_messages.add(msg_id)
            self._message_count += 1
            
            # Periodically clean up old entries
            if self._message_count >= self._cleanup_interval:
                self._cleanup_replay_cache(current_slot)
                self._message_count = 0
        
        # Derive key for the message's slot
        key = self.derive_slot_key(slot_index)
        cipher = ChaCha20Poly1305(key)
        
        # Reconstruct nonce (same format as seal)
        nonce = (
            struct.pack(">I", slot_index & 0xFFFFFFFF) +
            struct.pack(">I", sequence & 0xFFFFFFFF) +
            random_bytes
        )
        
        # Reconstruct AAD
        aad = (
            struct.pack(">Q", slot_index) +
            struct.pack(">Q", sequence) +
            associated_data
        )
        
        # Decrypt and verify
        try:
            plaintext = cipher.decrypt(nonce, ciphertext, aad)
            return plaintext
        except Exception:
            # Authentication failed
            return None
    
    def _cleanup_replay_cache(self, current_slot: int):
        """Remove old entries from replay protection cache."""
        # Remove entries older than drift_window + 1 slots
        min_valid_slot = current_slot - self.drift_window - 1
        self._seen_messages = {
            (slot, seq) for slot, seq in self._seen_messages
            if slot >= min_valid_slot
        }


def generate_shared_secret() -> bytes:
    """Generate a cryptographically secure 256-bit shared secret."""
    return os.urandom(32)


# Convenience functions for simple use cases

def seal_packet(
    shared_secret: bytes,
    slot_index: int,
    sequence: int,
    plaintext: bytes,
    associated_data: bytes = b"",
    context: bytes = DEFAULT_CONTEXT
) -> bytes:
    """
    Convenience function to encrypt a single packet.
    
    Args:
        shared_secret: 32-byte pre-shared secret
        slot_index: Time slot index
        sequence: Message sequence number
        plaintext: Data to encrypt
        associated_data: Additional authenticated data
        context: Application context for key derivation
    
    Returns:
        Encrypted packet
    """
    key = derive_slot_key(shared_secret, slot_index, context)
    cipher = ChaCha20Poly1305(key)
    
    random_bytes = os.urandom(4)
    
    nonce = (
        struct.pack(">I", slot_index & 0xFFFFFFFF) +
        struct.pack(">I", sequence & 0xFFFFFFFF) +
        random_bytes
    )
    
    aad = (
        struct.pack(">Q", slot_index) +
        struct.pack(">Q", sequence) +
        associated_data
    )
    
    ciphertext = cipher.encrypt(nonce, plaintext, aad)
    return struct.pack(">Q", slot_index) + struct.pack(">Q", sequence) + random_bytes + ciphertext


def open_packet(
    shared_secret: bytes,
    packet: bytes,
    associated_data: bytes = b"",
    drift_window: int = DEFAULT_DRIFT_WINDOW,
    local_slot: Optional[int] = None,
    context: bytes = DEFAULT_CONTEXT
) -> Optional[bytes]:
    """
    Convenience function to decrypt a single packet.
    
    Args:
        shared_secret: 32-byte pre-shared secret
        packet: Encrypted packet (slot_index || sequence || random || ciphertext + tag)
        associated_data: Additional authenticated data
        drift_window: Number of slots to accept for clock drift
        local_slot: Current local slot (defaults to system time)
        context: Application context for key derivation
    
    Returns:
        Decrypted plaintext, or None if verification fails
    """
    if len(packet) < 20:
        return None
    
    slot_index = struct.unpack(">Q", packet[:8])[0]
    sequence = struct.unpack(">Q", packet[8:16])[0]
    random_bytes = packet[16:20]
    ciphertext = packet[20:]
    
    # Validate slot window
    if local_slot is None:
        local_slot = int(time.time() / DEFAULT_SLOT_DURATION)
    
    if abs(local_slot - slot_index) > drift_window:
        return None
    
    # Derive key and decrypt
    key = derive_slot_key(shared_secret, slot_index, context)
    cipher = ChaCha20Poly1305(key)
    
    nonce = (
        struct.pack(">I", slot_index & 0xFFFFFFFF) +
        struct.pack(">I", sequence & 0xFFFFFFFF) +
        random_bytes
    )
    
    aad = (
        struct.pack(">Q", slot_index) +
        struct.pack(">Q", sequence) +
        associated_data
    )
    
    try:
        return cipher.decrypt(nonce, ciphertext, aad)
    except Exception:
        return None


def derive_slot_key(
    shared_secret: bytes,
    slot_index: int,
    context: bytes = DEFAULT_CONTEXT
) -> bytes:
    """
    Derive encryption key for a specific time slot.
    
    Args:
        shared_secret: 32-byte pre-shared secret
        slot_index: Time slot index
        context: Application context string
    
    Returns:
        32-byte key for ChaCha20-Poly1305
    """
    info = context + struct.pack(">Q", slot_index)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info,
    )
    return hkdf.derive(shared_secret)


if __name__ == "__main__":
    # Demo usage
    print("TRANSEC Reference Implementation Demo")
    print("=" * 50)
    
    # Generate shared secret
    secret = generate_shared_secret()
    print(f"Shared secret: {secret.hex()}")
    
    # Create cipher instances for sender and receiver
    sender = TransecCipher(secret, slot_duration=5, drift_window=2)
    receiver = TransecCipher(secret, slot_duration=5, drift_window=2)
    
    # Encrypt message
    plaintext = b"Hello from TRANSEC!"
    sequence = 1
    packet = sender.seal(plaintext, sequence)
    print(f"\nPlaintext: {plaintext.decode()}")
    print(f"Encrypted packet size: {len(packet)} bytes")
    print(f"Packet (hex): {packet[:32].hex()}...")
    
    # Decrypt message
    decrypted = receiver.open(packet)
    if decrypted:
        print(f"Decrypted: {decrypted.decode()}")
        print("✓ Encryption and decryption successful!")
    else:
        print("✗ Decryption failed")
    
    # Test replay protection
    print("\nTesting replay protection...")
    replay_result = receiver.open(packet)
    if replay_result is None:
        print("✓ Replay attack detected and blocked")
    else:
        print("✗ Replay attack not detected!")
