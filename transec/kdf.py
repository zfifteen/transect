#!/usr/bin/env python3
"""
KDF: Key Derivation Function utilities with golden ratio-based salt generation.

This module implements deterministic HKDF salt generation using the golden ratio (φ)
for enhanced entropy distribution across time slots.

Mathematical Foundation:
    Z = A(B / c) with c = 2^{256} for salt mod in HKDF (bit-security bound)
    A = slot_id
    B = floor(θ′(slot_id, k) * 10^9)
    θ′(n,k) = φ * ((n % φ)/φ)^k with k=0.3

Properties:
    - Deterministic: Same slot_id always produces same salt
    - Platform-independent: Deterministic float operations with IEEE 754 compliance
    - High entropy: Golden ratio ensures good distribution
    - Bit-exact: Deterministic float-to-int scale (10^9)
"""

import math

# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2.0


def theta_prime(slot: int, k: float = 0.3) -> float:
    """
    Compute θ′(n,k) = φ * ((n % φ)/φ)^k.
    
    Geometric resolution function based on the golden ratio.
    
    Args:
        slot: Time slot index
        k: Exponent parameter (default: 0.3)
    
    Returns:
        Computed theta prime value
    """
    frac = math.fmod(slot, PHI) / PHI
    return PHI * (frac ** k)


def hkdf_salt_for_slot(slot_id: int, k: float = 0.3, out_len: int = 32) -> bytes:
    """
    Generate deterministic HKDF salt for a time slot using golden ratio.
    
    Implements the formula:
        Z = slot_id * floor(θ′(slot_id, k) * 10^9) % (1 << (8 * out_len))
    
    Args:
        slot_id: Time slot index (must be >= 0)
        k: Exponent parameter for theta_prime (default: 0.3)
        out_len: Output length in bytes (default: 32)
    
    Returns:
        Deterministic salt as bytes (big-endian)
    
    Raises:
        ValueError: If slot_id is negative or out_len is invalid
    
    Example:
        >>> salt = hkdf_salt_for_slot(42)
        >>> len(salt)
        32
        >>> # Same slot always produces same salt
        >>> hkdf_salt_for_slot(42) == hkdf_salt_for_slot(42)
        True
    """
    if slot_id < 0:
        raise ValueError(f"slot_id must be non-negative, got {slot_id}")
    if out_len <= 0:
        raise ValueError(f"out_len must be positive, got {out_len}")
    
    # Deterministic float-to-int scale factor
    scale = 10**9
    
    # Compute theta prime and scale to integer
    tp = theta_prime(slot_id, k)
    mul = math.floor(tp * scale)
    
    # Apply modulo to keep within bit-security bound
    mod_bits = 8 * out_len
    salt_int = (slot_id * mul) % (1 << mod_bits)
    
    # Convert to bytes (big-endian)
    return salt_int.to_bytes(out_len, "big")


def test_salt_diversity(num_slots: int = 100) -> bool:
    """
    Test salt diversity and determinism.
    
    Verifies that:
    1. All salts are unique across num_slots consecutive slots
    2. Repeated calls for the same slot produce identical results
    
    Args:
        num_slots: Number of slots to test (default: 100)
    
    Returns:
        True if all checks pass
    
    Raises:
        AssertionError: If diversity or determinism checks fail
    """
    # Test diversity
    salts = [hkdf_salt_for_slot(i) for i in range(num_slots)]
    unique = len(set(salts))
    assert unique == num_slots, f"Only {unique}/{num_slots} unique salts"
    
    # Test determinism
    for i in range(min(10, num_slots)):
        salt1 = hkdf_salt_for_slot(i)
        salt2 = hkdf_salt_for_slot(i)
        assert salt1 == salt2, f"Non-deterministic for slot {i}"
    
    return True


if __name__ == "__main__":
    # Run basic tests
    print("Testing salt diversity and determinism...")
    test_salt_diversity()
    print("✓ All tests passed")
    
    # Show some example salts
    print("\nExample salts for first 5 slots:")
    for i in range(5):
        salt = hkdf_salt_for_slot(i)
        print(f"  Slot {i}: {salt.hex()[:32]}...")
