#!/usr/bin/env python3
"""
KDF: Key Derivation Function utilities for slot-indexed key derivation.

This module implements a secure key derivation scheme based on HKDF (HMAC-based
Key Derivation Function, RFC 5869). It derives per-slot keys in a deterministic
manner by incorporating the slot index into the HKDF 'info' parameter, which is
the standard-compliant way to introduce context-specific data.

Security Model:
- IKM (Input Keying Material): The initial entropy is provided by the IKM.
  This material should be a high-entropy secret, ideally from a cryptographic
  random number generator.
- Salt: A fixed, zero-filled salt is used. Per RFC 5869, a constant salt is
  acceptable when the IKM is high-entropy. If the IKM might be weak, a
  random salt is recommended, but for deterministic derivation, a fixed
  salt is used.
- Info: To derive unique keys for different contexts (in this case, time
  slots), the slot ID is hashed with a domain-separation tag and used as the
  'info' parameter in the HKDF-Expand step. This ensures that each slot
  produces a unique, cryptographically separate key.
- Versioning: The domain-separation tag is versioned (e.g., "transect/hkdf/v1")
  to allow for future cryptographic agility. If the algorithm is ever
  updated, incrementing the version number will ensure that old and new keys
  do not collide.
"""

import hmac
import hashlib

def hkdf_expand_slot(ikm: bytes, slot_id: int, out_len: int = 32) -> bytes:
    """
    Derives a key for a specific slot using HKDF-Expand.

    This function uses a fixed salt and a deterministic 'info' parameter
    derived from the slot_id.

    Args:
        ikm: Input Keying Material (high-entropy secret).
        slot_id: The slot index (0 <= slot_id < 2**64).
        out_len: The desired output length in bytes.

    Returns:
        The derived key for the given slot.

    Raises:
        ValueError: If slot_id is out of the valid range.
    """
    if not (0 <= slot_id < 2**64):
        raise ValueError(f"slot_id must be in the range [0, 2**64), got {slot_id}")

    # Use a fixed, zero-filled salt as per recommendation for deterministic derivation.
    salt = b'\x00' * 32

    # HKDF-Extract: Create a pseudorandom key (PRK)
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()

    # HKDF-Expand: Derive the output key
    info = _hkdf_info_for_slot(slot_id)
    t = b""
    okm = b""
    i = 1
    while len(okm) < out_len:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1

    return okm[:out_len]

def _hkdf_info_for_slot(slot_id: int) -> bytes:
    """
    Generates a deterministic 'info' parameter for HKDF based on the slot ID.

    This uses BLAKE2s for fast, secure hashing with a domain separation tag.

    Args:
        slot_id: The slot index.

    Returns:
        A byte string to be used as the 'info' parameter in HKDF.
    """
    # Domain separation tag with versioning
    domain_sep = b"transect/hkdf/v1"
    h = hashlib.blake2s(digest_size=32)
    h.update(domain_sep)
    h.update(slot_id.to_bytes(8, "big"))  # Use 8 bytes for 64-bit integer
    return h.digest()
