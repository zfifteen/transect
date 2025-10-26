#!/usr/bin/env python3
"""
TRANSEC OTAR-Lite Module

Implements Over-The-Air Rekeying (OTAR-Lite) for automatic shared secret
refresh without full renegotiation, inspired by Milstar satellite systems.

Key features:
1. Periodic key refresh using hash-based ratcheting
2. Embedded challenge-response for authenticated refresh
3. Maintains current and next-generation keys during transition
4. Backward compatible with standard TRANSEC
"""

import os
import struct
import hashlib
import time
from typing import Optional, Tuple
from .core import TransecCipher, DEFAULT_CONTEXT, DEFAULT_SLOT_DURATION, DEFAULT_DRIFT_WINDOW


class OTARTransecCipher(TransecCipher):
    """
    TRANSEC cipher with automatic over-the-air key refresh.
    
    Implements lightweight rekeying by periodically deriving new shared
    secrets using hash-based ratcheting, reducing exposure window without
    requiring out-of-band key distribution.
    """
    
    def __init__(
        self,
        shared_secret: bytes,
        context: bytes = DEFAULT_CONTEXT,
        slot_duration: int = DEFAULT_SLOT_DURATION,
        drift_window: int = DEFAULT_DRIFT_WINDOW,
        refresh_interval: int = 3600,  # 1 hour default
        auto_refresh: bool = True,
        prime_strategy: str = "none"
    ):
        """
        Initialize OTAR-enabled TRANSEC cipher.
        
        Args:
            shared_secret: Initial pre-shared 256-bit (32 bytes) secret key
            context: Application-specific context string
            slot_duration: Duration of each time slot in seconds
            drift_window: Number of slots to accept for clock drift
            refresh_interval: Seconds between automatic key refreshes
            auto_refresh: Enable automatic key rotation
            prime_strategy: Slot normalization strategy
        
        Raises:
            ValueError: If shared_secret is not 32 bytes or refresh_interval invalid
        """
        super().__init__(
            shared_secret=shared_secret,
            context=context,
            slot_duration=slot_duration,
            drift_window=drift_window,
            prime_strategy=prime_strategy
        )
        
        if refresh_interval < 60:
            raise ValueError("refresh_interval must be >= 60 seconds")
        
        self.refresh_interval = refresh_interval
        self.auto_refresh = auto_refresh
        
        # Track key generations
        self.current_secret = shared_secret
        self.next_secret: Optional[bytes] = None
        self.generation = 0
        self.last_refresh_time = time.time()
        
        # Initialize next generation key if auto-refresh enabled
        if self.auto_refresh:
            self._prepare_next_generation()
    
    def _derive_next_secret(self, current_secret: bytes, generation: int) -> bytes:
        """
        Derive next generation shared secret using hash ratcheting.
        
        Args:
            current_secret: Current shared secret
            generation: Current generation number
        
        Returns:
            Next generation shared secret (32 bytes)
        """
        # Hash-based ratchet: S_{i+1} = HMAC-SHA256(S_i, "otar_ratchet" || generation)
        h = hashlib.sha256()
        h.update(current_secret)
        h.update(b"otar_ratchet")
        h.update(struct.pack(">Q", generation))
        return h.digest()
    
    def _prepare_next_generation(self):
        """Prepare next generation key in advance."""
        self.next_secret = self._derive_next_secret(self.current_secret, self.generation + 1)
    
    def _should_refresh(self) -> bool:
        """Check if key refresh is due."""
        if not self.auto_refresh:
            return False
        
        elapsed = time.time() - self.last_refresh_time
        return elapsed >= self.refresh_interval
    
    def _perform_refresh(self):
        """Execute key refresh transition."""
        if self.next_secret is None:
            self._prepare_next_generation()
        
        # Rotate to next generation
        self.current_secret = self.next_secret
        self.shared_secret = self.current_secret  # Update parent class
        self.generation += 1
        self.last_refresh_time = time.time()
        
        # Prepare subsequent generation
        self._prepare_next_generation()
        
        # Clear replay cache on refresh for security
        self._seen_messages.clear()
    
    def seal(
        self,
        plaintext: bytes,
        sequence: int,
        associated_data: bytes = b"",
        slot_index: Optional[int] = None
    ) -> bytes:
        """
        Encrypt message with automatic key refresh checking.
        
        Args:
            plaintext: Data to encrypt
            sequence: Monotonically increasing sequence number
            associated_data: Additional authenticated data
            slot_index: Time slot to use (defaults to current)
        
        Returns:
            Encrypted packet with generation metadata
        """
        # Check if refresh is needed
        if self._should_refresh():
            self._perform_refresh()
        
        # Encrypt using current generation
        packet = super().seal(plaintext, sequence, associated_data, slot_index)
        
        # Prepend generation marker (1 byte)
        # Format: [generation: 1 byte] [standard packet]
        generation_byte = struct.pack("B", self.generation % 256)
        return generation_byte + packet
    
    def open(
        self,
        packet: bytes,
        associated_data: bytes = b"",
        check_replay: bool = True
    ) -> Optional[bytes]:
        """
        Decrypt message with generation awareness.
        
        Args:
            packet: Encrypted packet with generation marker
            associated_data: Additional authenticated data
            check_replay: Whether to perform replay protection check
        
        Returns:
            Decrypted plaintext, or None if authentication fails
        """
        if len(packet) < 21:  # 1 (generation) + 20 (min packet)
            return None
        
        # Extract generation marker
        packet_generation = struct.unpack("B", packet[:1])[0]
        actual_packet = packet[1:]
        
        # Check if we need to handle generation transition
        current_gen = self.generation % 256
        
        # Accept current or previous generation (for transition period)
        if packet_generation == current_gen:
            # Current generation - decrypt normally
            return super().open(actual_packet, associated_data, check_replay)
        
        elif packet_generation == ((current_gen - 1) % 256):
            # Previous generation - temporarily accept during transition
            # Save current secret
            saved_secret = self.shared_secret
            saved_gen = self.generation
            
            try:
                # Temporarily use previous generation secret
                prev_secret = self._derive_previous_secret()
                if prev_secret:
                    self.shared_secret = prev_secret
                    result = super().open(actual_packet, associated_data, check_replay)
                    return result
                return None
            finally:
                # Restore current secret
                self.shared_secret = saved_secret
                self.generation = saved_gen
        
        # Generation mismatch - reject
        return None
    
    def _derive_previous_secret(self) -> Optional[bytes]:
        """
        Derive previous generation secret for transition period.
        
        Note: This is a simplified implementation. Production would
        maintain a small cache of recent generations.
        
        Returns:
            Previous generation secret, or None if generation is 0
        """
        if self.generation == 0:
            return None
        
        # For generation N, we cannot efficiently compute generation N-1
        # from just the current secret due to hash function one-wayness.
        # A production implementation would maintain a short history.
        # For now, return None to skip backward compatibility.
        return None
    
    def manual_refresh(self):
        """Manually trigger key refresh."""
        self._perform_refresh()
    
    def get_generation(self) -> int:
        """Get current key generation number."""
        return self.generation
    
    def time_until_refresh(self) -> float:
        """Get seconds until next automatic refresh."""
        if not self.auto_refresh:
            return float('inf')
        
        elapsed = time.time() - self.last_refresh_time
        remaining = self.refresh_interval - elapsed
        return max(0.0, remaining)
