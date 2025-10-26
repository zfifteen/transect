"""
TRANSEC: Time-Synchronized Encryption

Zero-handshake encrypted messaging inspired by military frequency-hopping COMSEC.

Basic usage:
    from transec import TransecCipher, generate_shared_secret
    
    secret = generate_shared_secret()
    cipher = TransecCipher(secret)
    
    packet = cipher.seal(b"Hello!", sequence=1)
    plaintext = cipher.open(packet)

Advanced features:
    from transec import AdaptiveTransecCipher, OTARTransecCipher
    
    # Adaptive slot duration with jitter
    adaptive = AdaptiveTransecCipher(secret, jitter_range=(2, 10))
    
    # Automatic key refresh
    otar = OTARTransecCipher(secret, refresh_interval=3600)
"""

__version__ = '0.1.0'
__author__ = 'Z-Sandbox Project'
__license__ = 'MIT'

# Core functionality
from .core import (
    TransecCipher,
    generate_shared_secret,
    seal_packet,
    open_packet,
    derive_slot_key,
    DEFAULT_CONTEXT,
    DEFAULT_SLOT_DURATION,
    DEFAULT_DRIFT_WINDOW,
    DEFAULT_PRIME_STRATEGY,
)

# Advanced features
try:
    from .adaptive import AdaptiveTransecCipher
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    AdaptiveTransecCipher = None

try:
    from .otar import OTARTransecCipher
    OTAR_AVAILABLE = True
except ImportError:
    OTAR_AVAILABLE = False
    OTARTransecCipher = None

# Prime optimization (optional dependency)
try:
    from .prime_optimization import (
        normalize_slot_to_prime,
        compute_curvature,
        is_prime,
        count_divisors,
    )
    PRIME_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PRIME_OPTIMIZATION_AVAILABLE = False
    normalize_slot_to_prime = None
    compute_curvature = None
    is_prime = None
    count_divisors = None

__all__ = [
    # Core
    'TransecCipher',
    'generate_shared_secret',
    'seal_packet',
    'open_packet',
    'derive_slot_key',
    
    # Advanced
    'AdaptiveTransecCipher',
    'OTARTransecCipher',
    
    # Prime optimization
    'normalize_slot_to_prime',
    'compute_curvature',
    'is_prime',
    'count_divisors',
    
    # Constants
    'DEFAULT_CONTEXT',
    'DEFAULT_SLOT_DURATION',
    'DEFAULT_DRIFT_WINDOW',
    'DEFAULT_PRIME_STRATEGY',
    
    # Feature flags
    'ADAPTIVE_AVAILABLE',
    'OTAR_AVAILABLE',
    'PRIME_OPTIMIZATION_AVAILABLE',
    
    # Metadata
    '__version__',
]
