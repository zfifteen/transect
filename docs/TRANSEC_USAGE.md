# TRANSEC Usage Guide

## Quick Start

### Installation

```bash
# Install dependencies
pip install cryptography>=42.0.0

# Or install all project dependencies
pip install -r python/requirements.txt
```

### Basic Usage

```python
from transec import TransecCipher, generate_shared_secret

# Generate or provision shared secret (32 bytes)
secret = generate_shared_secret()

# Create cipher instances for sender and receiver
sender = TransecCipher(secret, slot_duration=5, drift_window=2)
receiver = TransecCipher(secret, slot_duration=5, drift_window=2)

# Encrypt message (no handshake needed!)
plaintext = b"Hello, TRANSEC!"
sequence = 1
packet = sender.seal(plaintext, sequence)

# Decrypt message
decrypted = receiver.open(packet)
print(decrypted)  # b"Hello, TRANSEC!"
```

## UDP Demo

### Running the Server

```bash
# Start server on default port (9999)
python3 python/transec_udp_demo.py server

# Start on custom port
python3 python/transec_udp_demo.py server --port 8888
```

### Running the Client

```bash
# Interactive client
python3 python/transec_udp_demo.py client

# Benchmark mode (100 messages)
python3 python/transec_udp_demo.py benchmark --count 100
```

### Example Session

```bash
# Terminal 1: Start server
$ python3 python/transec_udp_demo.py server
============================================================
TRANSEC UDP Demo
Zero-Handshake Encrypted Messaging
============================================================

🔐 TRANSEC UDP Server listening on 127.0.0.1:9999
Waiting for encrypted messages...

# Terminal 2: Run benchmark
$ python3 python/transec_udp_demo.py benchmark --count 20
============================================================
TRANSEC UDP Demo
Zero-Handshake Encrypted Messaging
============================================================

🔐 TRANSEC UDP Client - Benchmarking 20 messages
Connected to 127.0.0.1:9999

Progress: 10/20 messages
Progress: 20/20 messages

============================================================
Benchmark Results:
  Success rate: 20/20 (100.0%)
  Average RTT: 0.34ms
  Min RTT: 0.24ms
  Max RTT: 1.40ms
  Throughput: 2941.7 msg/sec
============================================================
```

## Configuration Options

### Slot Duration

Choose based on your security/robustness tradeoff:

```python
# Aggressive: Keys change every second (high security)
cipher = TransecCipher(secret, slot_duration=1)

# Balanced: Keys change every 5 seconds (recommended)
cipher = TransecCipher(secret, slot_duration=5)

# Conservative: Keys change every 30 seconds (robust)
cipher = TransecCipher(secret, slot_duration=30)
```

### Drift Window

Control clock synchronization tolerance:

```python
# Tight: Accept ±1 slot (minimal attack surface)
cipher = TransecCipher(secret, drift_window=1)

# Balanced: Accept ±2 slots (recommended)
cipher = TransecCipher(secret, drift_window=2)

# Loose: Accept ±5 slots (degraded environments)
cipher = TransecCipher(secret, drift_window=5)
```

### Application Context

Isolate key streams for different applications:

```python
# Different contexts produce different keys
app1_cipher = TransecCipher(secret, context=b"app1:production")
app2_cipher = TransecCipher(secret, context=b"app2:development")
```

## API Reference

### TransecCipher

Main cipher class for TRANSEC encryption.

#### Constructor

```python
TransecCipher(
    shared_secret: bytes,      # 32-byte pre-shared secret
    context: bytes = b"z-sandbox:transec:v1",
    slot_duration: int = 5,    # Seconds per time slot
    drift_window: int = 2      # ±N slots tolerance
)
```

#### Methods

**seal(plaintext, sequence, associated_data=b"", slot_index=None)**

Encrypt and authenticate a message.

```python
packet = cipher.seal(
    plaintext=b"Secret message",
    sequence=1,
    associated_data=b"user_id:123"
)
# Returns: encrypted packet (bytes)
```

**open(packet, associated_data=b"", check_replay=True)**

Decrypt and verify an authenticated message.

```python
plaintext = cipher.open(
    packet=encrypted_packet,
    associated_data=b"user_id:123"
)
# Returns: decrypted plaintext (bytes) or None if failed
```

**get_current_slot()**

Get current time slot index.

```python
slot = cipher.get_current_slot()
# Returns: int (current slot index)
```

**derive_slot_key(slot_index)**

Derive encryption key for a specific time slot.

```python
key = cipher.derive_slot_key(12345)
# Returns: 32-byte key (bytes)
```

### Convenience Functions

**generate_shared_secret()**

Generate a cryptographically secure 256-bit shared secret.

```python
secret = generate_shared_secret()
# Returns: 32-byte secret (bytes)
```

**seal_packet(shared_secret, slot_index, sequence, plaintext, ...)**

Convenience function to encrypt a single packet.

```python
packet = seal_packet(
    shared_secret=secret,
    slot_index=100,
    sequence=1,
    plaintext=b"Message"
)
```

**open_packet(shared_secret, packet, ...)**

Convenience function to decrypt a single packet.

```python
plaintext = open_packet(
    shared_secret=secret,
    packet=encrypted_packet
)
```

## Advanced Usage

### Time Synchronization

TRANSEC requires synchronized clocks. Recommended approaches:

1. **GPS/GNSS** - Best accuracy (nanoseconds)
2. **NTP with Authentication** - Good for most cases
3. **Hardware Time Source** - TPM, secure element
4. **Monotonic Counter** - With periodic resync

### Key Rotation

Rotate shared secrets periodically:

```python
# Daily rotation example
def rotate_key_daily():
    # Fetch new key from secure key management system
    new_secret = fetch_from_kms()
    
    # Create new cipher with updated secret
    cipher = TransecCipher(new_secret)
    return cipher

# Per-mission rotation (tactical use case)
def new_mission_key():
    secret = generate_shared_secret()
    # Distribute via secure out-of-band channel
    distribute_to_nodes(secret)
    return secret
```

### Replay Protection Cache Management

The cipher automatically manages replay protection:

```python
# Cache is cleaned periodically (every 100 messages by default)
# Old entries outside drift window are removed

# For high-volume scenarios, consider:
cipher = TransecCipher(secret)
# Increase cleanup interval if needed
cipher._cleanup_interval = 1000
```

### Associated Data (AAD)

Use AAD to bind context to messages:

```python
# Include user identity
aad = b"user_id:alice"
packet = cipher.seal(plaintext, sequence, associated_data=aad)

# Include message type
aad = b"type:telemetry"
packet = cipher.seal(data, sequence, associated_data=aad)

# Receiver must use same AAD
plaintext = cipher.open(packet, associated_data=aad)
```

## Performance Characteristics

Based on benchmarks on commodity hardware:

- **Encryption**: ~0.01-0.1ms per packet
- **Decryption**: ~0.01-0.1ms per packet
- **RTT (UDP loopback)**: ~0.3-1.5ms
- **Throughput**: ~3,000-10,000 msg/sec
- **Packet Overhead**: 20 bytes (8 + 8 + 4)
- **Key Derivation**: ~0.1ms per slot change

## Testing

Run the test suite:

```bash
# Run all tests
python3 tests/test_transec.py

# Run with verbose output
python3 tests/test_transec.py -v

# Run specific test
python3 tests/test_transec.py TestTransecBasics.test_encrypt_decrypt
```

## Security Considerations

### Threat Model

**Protected Against:**
- ✓ Passive eavesdropping (strong encryption)
- ✓ Replay attacks (sequence tracking)
- ✓ Packet injection (AEAD authentication)
- ✓ Tampered messages (AEAD integrity)

**NOT Protected Against:**
- ✗ Shared secret compromise (requires network rekey)
- ✗ Active time desynchronization (needs authenticated time)
- ✗ Traffic analysis (packet metadata visible)
- ✗ Forward secrecy without ratcheting

### Best Practices

1. **Secure Secret Storage**
   - Store shared secrets in hardware security modules (HSM)
   - Use secure enclaves or TPMs when available
   - Never log or transmit secrets in plaintext

2. **Time Synchronization**
   - Use authenticated time sources (NTS, GPS)
   - Monitor clock drift and alert on anomalies
   - Implement fallback to monotonic counters

3. **Key Rotation**
   - Rotate shared secrets regularly (daily/weekly)
   - Implement emergency rotation procedures
   - Use separate secrets per deployment/mission

4. **Network Security**
   - Deploy TRANSEC over trusted networks when possible
   - Consider outer-layer protection (IPsec, WireGuard)
   - Monitor for DoS attempts (time desync attacks)

5. **Monitoring**
   - Track authentication failures
   - Alert on excessive replay detections
   - Monitor clock drift across nodes

## Troubleshooting

### Decryption Failures

**Problem**: `cipher.open()` returns `None`

**Possible causes:**
1. Clock drift exceeds window → Increase `drift_window` or sync clocks
2. Wrong shared secret → Verify secret provisioning
3. Mismatched `associated_data` → Check AAD matches seal/open
4. Replay detection → Message already processed
5. Packet corruption → Check network quality

### High Latency

**Problem**: RTT higher than expected

**Solutions:**
1. Check network conditions (packet loss, congestion)
2. Verify CPU load isn't excessive
3. Consider using larger `slot_duration` for slower systems
4. Profile encryption overhead with simpler messages

### Authentication Failures

**Problem**: Legitimate packets rejected

**Solutions:**
1. Verify both sides use same `context` parameter
2. Check `slot_duration` matches on sender/receiver
3. Ensure clocks are synchronized within `drift_window`
4. Verify shared secret matches exactly

## Use Cases

### Tactical Communications

```python
# Drone swarm coordination
context = b"swarm:alpha:mission:2024"
cipher = TransecCipher(mission_secret, context=context, slot_duration=1)

# Send command with zero latency
command = b"formation:delta"
packet = cipher.seal(command, sequence_num)
send_to_swarm(packet)
```

### SCADA/Industrial Control

```python
# Power grid telemetry
context = b"scada:grid:west"
cipher = TransecCipher(grid_secret, context=context, slot_duration=5)

# Send critical telemetry without handshake delay
telemetry = encode_sensor_data(voltage, current, frequency)
packet = cipher.seal(telemetry, sequence_num)
send_to_control_center(packet)
```

### Autonomous Vehicles

```python
# Vehicle-to-vehicle messaging
context = b"v2v:highway:eastbound"
cipher = TransecCipher(platoon_secret, context=context, slot_duration=1)

# Send position update with minimal latency
position = encode_gps(lat, lon, speed, heading)
packet = cipher.seal(position, sequence_num)
broadcast_to_platoon(packet)
```

## References

- [TRANSEC Specification](TRANSEC.md) - Full protocol specification
- [Test Suite](../tests/test_transec.py) - Comprehensive test coverage
- [UDP Demo](../python/transec_udp_demo.py) - Working example

## License

This implementation is provided for research and development purposes. Review security considerations carefully before production use.
