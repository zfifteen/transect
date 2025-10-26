# TRANSEC: Time-Synchronized Encryption for Zero-Handshake Messaging

## Overview

**Inspired by military frequency-hopping radio COMSEC (SINCGARS, HAVE QUICK)**, this system adapts time-synchronized key rotation to software-defined networking, eliminating handshake latency in tactical/industrial scenarios where zero-RTT encryption is required.

TRANSEC implements a property set **absent from widely-deployed Internet standards** (TLS 1.3, QUIC, IKEv2, Signal): true **zero-handshake, first-contact, replay-bounded AEAD** using only pre-shared key + synchronized time. See [Protocol Comparison](TRANSEC_PROTOCOL_COMPARISON.md) for detailed analysis.

## Core Concept

TRANSEC (Transmission Security) implements deterministic key material derived from a shared secret combined with precise time epochs. This enables instant encrypted communication without per-message handshakes, accepting the COMSEC tradeoff that seed compromise requires network-wide rekeying.

**Military Parallel:** Just as HAVE QUICK/SINCGARS radios load a Word-of-Day (WOD) and Time-of-Day (TOD) and then communicate immediately without over-the-air key exchange, TRANSEC endpoints load a pre-shared secret and synchronized time, then send authenticated-encrypted messages with zero handshake overhead.

## Design Principles

1. **Zero-RTT Communication**: No handshake overhead after initial bootstrap
2. **Time-Sliced Keying**: Deterministic key rotation based on time epochs
3. **Minimal Latency**: Sub-millisecond encryption/decryption overhead
4. **COMSEC Model**: Pre-shared secrets with periodic out-of-band rotation

## Target Applications

TRANSEC is designed for domains where **sub-100ms end-to-end latencies** are required and traditional handshake protocols introduce unacceptable overhead:

### Vehicular Communications (V2X/CAM)
- **Standard:** [ETSI TS 102 637-2](https://www.etsi.org/deliver/etsi_ts/102600_102699/10263702/01.02.01_60/ts_10263702v010201p.pdf) - Cooperative Awareness Messages
- **Requirements:** 1-10 Hz beacons with tight delay budgets for collision avoidance
- **TRANSEC Benefit:** Zero-handshake eliminates initial RTT, preserving budget for safety processing

### Critical Infrastructure (SCADA/DER Telemetry)
- **Standard:** IEC 61850-90-5 - Power utility automation
- **Requirements:** ≤6-100ms for teleprotection classes (trip commands, protection relaying)
- **TRANSEC Benefit:** Deterministic timing meets protection relay requirements

### Tactical Communications
- **Drone swarms**: Real-time formation control and coordination
- **Battlefield mesh networks**: Low-latency command and control

### Autonomous Systems
- **V2V messaging**: Vehicle platoons and collision avoidance
- **Swarm coordination**: Multi-robot systems requiring instant encrypted communication
- **Edge Computing**: Low-latency IoT mesh networks
- **Satellite Ground Stations**: Predictable pass windows requiring instant lock-on

## Architecture

### Core Components

1. **Shared Secret (S)**: High-entropy pre-shared key (256 bits minimum)
2. **Time Epoch (T)**: Synchronized time slot (configurable: 1s, 5s, 30s)
3. **Key Derivation**: HKDF-SHA256 to derive per-slot AEAD keys
4. **AEAD Cipher**: ChaCha20-Poly1305 for authenticated encryption
5. **Replay Protection**: Slot index + sequence number tracking

### Key Derivation

```
slot_index = floor(current_time / slot_duration)
info = context || slot_index (8 bytes, big-endian)
K_t = HKDF-SHA256(S, salt=None, info=info, length=32)
```

### Packet Structure

```
[nonce: 24 bytes][ciphertext + tag: variable]

Nonce composition:
  - slot_index: 8 bytes (big-endian)
  - sequence: 8 bytes (big-endian)
  - random: 8 bytes

Associated Data (AAD):
  - slot_index: 8 bytes
  - sequence: 8 bytes
  - application_context: variable
```

## Security Model

### Assumptions

1. **Pre-Shared Secret**: S is securely provisioned out-of-band
2. **Time Synchronization**: Nodes maintain synchronized clocks (±W slots drift tolerance)
3. **Physical Security**: S is protected in hardware or secure storage
4. **Periodic Rotation**: S is rotated via secure channel (daily/weekly/per-mission)

### Threat Model

**Protected Against:**
- Passive eavesdropping (strong encryption)
- Replay attacks (sequence number tracking)
- Packet injection (AEAD authentication)
- Out-of-order delivery (slot window acceptance)

**NOT Protected Against:**
- Shared secret compromise (requires network-wide rekey)
- Active time desynchronization attacks (requires authenticated time source)
- Traffic analysis (packet timing/size metadata visible)
- Forward secrecy without additional ratcheting

### Known Limitations

1. **No Forward Secrecy**: Compromise of S reveals all past and future keys
2. **Time Dependency**: Requires reliable time synchronization
3. **DoS via Desync**: Attacker can attempt clock skewing (mitigate with authenticated NTP/GPS)
4. **Seed Compromise = Total Break**: Entire network must rekey if S leaks

## Implementation Parameters

### Slot Duration Selection

- **1 second**: Very aggressive, maximum security (key changes every second)
- **5 seconds**: Practical balance of security and robustness
- **30 seconds**: Conservative, tolerates larger clock drift

### Drift Window

- **±1 slot**: Tight synchronization, minimal attack surface
- **±3 slots**: Practical tolerance for network conditions
- **±5 slots**: Loose tolerance for degraded environments

### Time Sources

**Priority Order:**
1. GPS/GNSS with authentication
2. Hardware-secured time (TPM, secure element)
3. Authenticated NTP (NTS protocol)
4. Monotonic counter with periodic resync

## Comparison with Traditional Approaches

**Note:** See [TRANSEC Protocol Comparison](TRANSEC_PROTOCOL_COMPARISON.md) for comprehensive analysis with RFC references, security model details, and performance benchmarks.

| Property | TRANSEC | TLS 1.3 | IPsec | Signal Protocol |
|----------|---------|---------|-------|-----------------|
| Handshake RTT | 0 | 1-RTT | 1-4 RTT | Multiple RTT |
| First Contact | ✅ Yes | ❌ No (requires PSK) | ❌ No | ❌ No (X3DH) |
| Replay Protection | ✅ Inherent | ❌ App-level | ✅ Via IKE | ✅ Via ratchet |
| Forward Secrecy | No (without ratchet) | Yes | Yes | Yes |
| Key Compromise Impact | Network rekey | Session only | SA rekey | Conversation only |
| Time Dependency | Critical | None | Optional | None |
| Latency | <1ms | 10-100ms | 10-50ms | 100-500ms |
| Use Case | Tactical/Real-time | Web/General | VPN | Messaging |

## Operational Considerations

### Key Rotation Schedule

- **Per-Mission**: Rotate S for each operational deployment
- **Daily**: Automated rotation for continuous operations
- **Emergency**: Immediate rotation on suspected compromise

### Deployment Checklist

1. Provision shared secret S via secure channel
2. Verify time synchronization across all nodes
3. Configure slot duration based on operational requirements
4. Test drift tolerance under expected network conditions
5. Establish out-of-band rekey procedures
6. Monitor clock drift and packet loss metrics

### Performance Characteristics

- **Encryption Overhead**: <1ms per packet
- **Key Derivation**: <0.1ms per slot change
- **Memory**: O(drift_window) for replay protection
- **Bandwidth**: 20-byte overhead per packet (8 + 8 + 4 bytes header)

## References

### Military COMSEC Systems

- SINCGARS: Single Channel Ground and Airborne Radio System
- HAVE QUICK: Frequency-hopping air-to-air/air-to-ground radio
- KY-57: Secure voice terminal using time-sliced keys

### Cryptographic Standards

- NIST SP 800-108: Key Derivation Functions
- RFC 7539: ChaCha20-Poly1305 AEAD
- RFC 5869: HMAC-based Key Derivation Function (HKDF)

### Related Work

- NSA Cryptographic Interoperability Strategy
- DTLS Connection ID for zero-RTT scenarios
- QUIC 0-RTT modes and their tradeoffs

## Future Enhancements

1. **Forward Secrecy Layer**: Combine with ephemeral ECDH for hybrid PFS
2. **Ratcheting**: Implement hash-based ratchet: S_{i+1} = H(S_i)
3. **Multi-Channel**: Support multiple independent key streams
4. **Hardware Acceleration**: Leverage AES-NI or crypto accelerators
5. **Post-Quantum**: Upgrade to CRYSTALS-Kyber for quantum resistance

## License

This specification is provided for research and implementation purposes. Use in production systems requires careful security review and appropriate threat modeling for your specific use case.
