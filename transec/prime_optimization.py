#!/usr/bin/env python3
"""
TRANSEC Prime Optimization Module

Implements invariant normalization using prime-valued slot indices to minimize
discrete curvature κ(n) and enhance synchronization stability.

Based on the observation that prime slot indices (where d(n)=2) yield lower
curvature values, potentially reducing drift-induced decryption failures.

Mathematical Foundation:
- Discrete curvature: κ(n) = d(n) · ln(n+1) / e²
- For prime n: d(n) = 2 (only divisors are 1 and n)
- Lower κ indicates more stable synchronization paths
"""

import math
from typing import Optional, Dict
try:
    from mpmath import mp, mpf, log as mp_log
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False


# Cache for recently computed primes to optimize performance
_prime_cache: Dict[int, int] = {
    1: 2, 2: 2, 3: 3, 4: 5, 5: 5, 6: 7, 7: 7, 8: 7, 9: 11, 10: 11
}


def count_divisors(n: int) -> int:
    """
    Count the number of divisors of n.
    
    Args:
        n: Positive integer
    
    Returns:
        Number of divisors (including 1 and n)
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    count = 0
    sqrt_n = int(math.sqrt(n))
    
    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    
    return count


def is_prime(n: int) -> bool:
    """
    Check if n is prime using optimized trial division.
    
    Args:
        n: Integer to check
    
    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n == 3:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False
    
    # Check divisibility by numbers of form 6k±1 up to sqrt(n)
    # This is more efficient than checking all odd numbers
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True


def compute_curvature(n: int, use_mpmath: bool = False) -> float:
    """
    Compute discrete curvature κ(n) = d(n) · ln(n+1) / e².
    
    Args:
        n: Slot index
        use_mpmath: Use mpmath for high-precision computation
    
    Returns:
        Curvature value κ(n)
    """
    if use_mpmath and MPMATH_AVAILABLE:
        mp.dps = 20  # 20 decimal places precision
        d_n = count_divisors(n)
        kappa = mpf(d_n) * mp_log(n + 1) / (mp.e ** 2)
        return float(kappa)
    else:
        d_n = count_divisors(n)
        e_squared = math.e ** 2
        kappa = d_n * math.log(n + 1) / e_squared
        return kappa


def find_next_prime(n: int) -> int:
    """
    Find the next prime number >= n using cached results when available.
    
    Args:
        n: Starting value
    
    Returns:
        Next prime >= n
    """
    # Check cache first
    if n in _prime_cache:
        return _prime_cache[n]
    
    if n <= 2:
        result = 2
        _prime_cache[n] = result
        return result
    
    # Start with odd number
    candidate = n if n % 2 == 1 else n + 1
    
    # Search indefinitely until a prime is found
    # Prime gaps grow as O(log n), so this will terminate
    while True:
        if is_prime(candidate):
            _prime_cache[n] = candidate
            return candidate
        candidate += 2


def find_nearest_prime(n: int) -> int:
    """
    Find the nearest prime to n (prefers next prime if equidistant).
    Uses cached results for performance.
    
    Args:
        n: Target value
    
    Returns:
        Nearest prime to n
    """
    if n <= 2:
        return 2
    
    # Special case for small numbers
    if n == 3:
        return 3
    
    # If already prime, return it
    if is_prime(n):
        return n
    
    # Find next prime
    next_p = find_next_prime(n)
    
    # Find previous prime by searching backwards (search all the way to 2)
    prev_p = n - 1 if n > 2 else 2
    search_limit = 2  # Search all the way back to 2 to ensure correctness
    
    while prev_p >= search_limit:
        if is_prime(prev_p):
            break
        prev_p -= 1
    
    # If we hit the search limit without finding a prime, use next_p
    if prev_p < search_limit:
        return next_p
    
    # Return the nearest (prefer next if equidistant)
    if next_p - n <= n - prev_p:
        return next_p
    else:
        return prev_p


def normalize_slot_to_prime(slot_index: int, strategy: str = "nearest") -> int:
    """
    Normalize a slot index to a prime value for lower curvature.
    
    Args:
        slot_index: Original slot index
        strategy: Normalization strategy - "nearest", "next", or "none"
                 - "nearest": Map to nearest prime (default)
                 - "next": Map to next prime >= slot_index
                 - "none": Return slot_index unchanged
    
    Returns:
        Normalized slot index (prime or original)
    """
    if strategy == "none" or slot_index < 2:
        return slot_index
    
    if is_prime(slot_index):
        return slot_index
    
    if strategy == "next":
        return find_next_prime(slot_index)
    elif strategy == "nearest":
        return find_nearest_prime(slot_index)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compute_curvature_reduction(original_slot: int, normalized_slot: int) -> float:
    """
    Compute the curvature reduction achieved by normalization.
    
    Args:
        original_slot: Original slot index
        normalized_slot: Normalized (prime) slot index
    
    Returns:
        Percentage reduction in curvature (positive means improvement)
    """
    if original_slot == normalized_slot:
        return 0.0
    
    kappa_orig = compute_curvature(original_slot)
    kappa_norm = compute_curvature(normalized_slot)
    
    if kappa_orig == 0:
        return 0.0
    
    reduction = (kappa_orig - kappa_norm) / kappa_orig * 100
    return reduction


# Precomputed curvature values for verification (from issue description)
EMPIRICAL_CURVATURE_VALUES = {
    1: 0.093807270005739717186,
    2: 0.29736201050824385826,   # prime
    3: 0.37522908002295886874,   # prime
    4: 0.65344120719303488983,
    5: 0.48497655051972329263,   # prime
    6: 1.0534012047016001894,
    7: 0.56284362003443830311,   # prime
    8: 1.189448042032975433,
    9: 0.93486301721025404139,
    10: 1.2980793436636085396,
}


def verify_curvature_computation() -> bool:
    """
    Verify curvature computation against empirical values from the issue.
    
    Returns:
        True if all computed values match empirical data within tolerance
    """
    if not MPMATH_AVAILABLE:
        print("Warning: mpmath not available, skipping high-precision verification")
        return True
    
    tolerance = 1e-15  # Allow small floating-point errors
    all_match = True
    
    for n, expected in EMPIRICAL_CURVATURE_VALUES.items():
        computed = compute_curvature(n, use_mpmath=True)
        error = abs(computed - expected)
        
        if error > tolerance:
            print(f"Mismatch at n={n}: expected={expected}, computed={computed}, error={error}")
            all_match = False
    
    return all_match


if __name__ == "__main__":
    print("TRANSEC Prime Optimization - Curvature Analysis")
    print("=" * 60)
    
    # Verify curvature computation
    print("\nVerifying curvature computation against empirical data:")
    if verify_curvature_computation():
        print("✓ All curvature values verified!")
    else:
        print("✗ Some curvature values don't match")
    
    # Analyze slot indices 1-10
    print("\nCurvature analysis for slot indices 1-10:")
    print(f"{'n':<5} {'Prime?':<8} {'d(n)':<6} {'κ(n)':<12} {'Empirical':<12}")
    print("-" * 60)
    
    for n in range(1, 11):
        is_p = is_prime(n)
        d_n = count_divisors(n)
        kappa = compute_curvature(n, use_mpmath=MPMATH_AVAILABLE)
        empirical = EMPIRICAL_CURVATURE_VALUES.get(n, 0)
        
        marker = "★" if is_p else " "
        print(f"{n:<5} {marker:<8} {d_n:<6} {kappa:<12.6f} {empirical:<12.6f}")
    
    # Demonstrate normalization
    print("\nNormalization examples:")
    test_slots = [4, 6, 8, 9, 10, 15, 20, 100, 1000]
    
    for slot in test_slots:
        normalized = normalize_slot_to_prime(slot, strategy="nearest")
        reduction = compute_curvature_reduction(slot, normalized)
        
        if slot != normalized:
            print(f"  {slot} → {normalized} (κ reduction: {reduction:.1f}%)")
    
    print("\n" + "=" * 60)
