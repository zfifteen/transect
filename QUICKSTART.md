# TRANSEC Quick Start Guide

Get started with TRANSEC (Time-Synchronized Encryption) in 5 minutes!

## Installation

### From PyPI (when published)

```bash
pip install transec
```

### From Source

```bash
git clone https://github.com/zfifteen/transec.git
cd transec
pip install -e .
```

## Basic Usage

### 1. Generate Shared Secret

```python
from transec import generate_shared_secret

# Generate a cryptographically secure 256-bit secret
secret = generate_shared_secret()
print(f"Secret (hex): {secret.hex()}")
```

**Important**: Share this secret securely with your communication partner out-of-band (QR code, USB drive, secure channel, etc.)

### 2. Create Cipher Instances

```python
from transec import TransecCipher

# Both sender and receiver use the same secret and parameters
sender = TransecCipher(secret, slot_duration=5, drift_window=2)
receiver = TransecCipher(secret, slot_duration=5, drift_window=2)
```

Parameters:
- `slot_duration`: Time window in seconds (5s = high security, 30s = more robust)
- `drift_window`: ±N slots tolerance for clock differences

### 3. Send Encrypted Message

```python
# Sender encrypts a message
plaintext = b"Hello, TRANSEC!"
sequence = 1  # Monotonically increasing counter

packet = sender.seal(plaintext, sequence)
# packet is ready to send over UDP, TCP, or any transport!
```

### 4. Receive and Decrypt

```python
# Receiver decrypts the packet
decrypted = receiver.open(packet)

if decrypted:
    print(f"Message: {decrypted.decode()}")
else:
    print("Decryption failed (wrong key, replay, or time drift)")
```

## Complete Example

```python
from transec import TransecCipher, generate_shared_secret
import time

# Setup (do once, share secret securely)
secret = generate_shared_secret()

# Create sender and receiver
sender = TransecCipher(secret)
receiver = TransecCipher(secret)

# Communication
for i in range(5):
    # Send
    message = f"Message {i+1}"
    packet = sender.seal(message.encode(), sequence=i+1)
    print(f"Sent: {message} ({len(packet)} bytes)")
    
    # Receive
    decrypted = receiver.open(packet)
    print(f"Received: {decrypted.decode()}")
    print()
    
    time.sleep(1)
```

## Advanced Features

### Adaptive Slot Duration

Add unpredictability with dynamic timing:

```python
from transec import AdaptiveTransecCipher

cipher = AdaptiveTransecCipher(
    secret,
    base_duration=5,
    jitter_range=(2, 10)  # Varies 2-10 seconds per slot
)
```

### Automatic Key Refresh (OTAR)

Long-running sessions with periodic rekeying:

```python
from transec import OTARTransecCipher

cipher = OTARTransecCipher(
    secret,
    refresh_interval=3600,  # Rekey every hour
    auto_refresh=True
)

print(f"Generation: {cipher.get_generation()}")
print(f"Time until refresh: {cipher.time_until_refresh():.0f}s")
```

### Prime Optimization

Enhanced synchronization stability:

```python
cipher = TransecCipher(
    secret,
    prime_strategy="nearest"  # Use prime-valued slot indices
)
```

## UDP Client/Server Example

### Server

```python
import socket
from transec import TransecCipher, generate_shared_secret

secret = generate_shared_secret()  # Share this with client!
cipher = TransecCipher(secret)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5000))

print(f"Secret: {secret.hex()}")
print("Listening on UDP port 5000...")

while True:
    data, addr = sock.recvfrom(4096)
    plaintext = cipher.open(data)
    
    if plaintext:
        print(f"From {addr}: {plaintext.decode()}")
```

### Client

```python
import socket
from transec import TransecCipher

# Use same secret as server
secret = bytes.fromhex("YOUR_SECRET_HEX_HERE")
cipher = TransecCipher(secret)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send encrypted message
packet = cipher.seal(b"Hello from client!", sequence=1)
sock.sendto(packet, ('localhost', 5000))
```

## CLI Tools

### Generate Secret

```bash
python examples/transec_messenger.py generate
```

### Listen for Messages

```bash
python examples/transec_messenger.py listen \
  --port 5000 \
  --secret <base64-secret>
```

### Send Message

```bash
python examples/transec_messenger.py send \
  --host 192.168.1.100 \
  --port 5000 \
  --message "Hello!" \
  --secret <base64-secret>
```

## Run Benchmarks

```bash
python examples/transec_udp_demo.py server &
python examples/transec_udp_demo.py benchmark --count 1000
```

Expected results:
- **Throughput**: ~3,000 msg/sec
- **Latency**: <1ms RTT
- **Success Rate**: 100%

## Time Synchronization

TRANSEC requires synchronized clocks between peers:

### Good Time Sources (in order of preference):
1. **GPS/GNSS** with authentication
2. **Hardware RTC** (TPM, secure element)
3. **NTP with NTS** (authenticated time)
4. **System clock** with periodic sync

### Check Your Clock Drift

```bash
# Linux
timedatectl status

# macOS
sntp -K /dev/null pool.ntp.org
```

### Adjust Drift Window

If you experience frequent decryption failures:

```python
# Increase drift tolerance (±5 slots instead of ±2)
cipher = TransecCipher(secret, drift_window=5)
```

## Security Best Practices

1. **Secret Distribution**: Never send secrets over the network. Use:
   - QR codes for in-person sharing
   - USB drives or secure physical media
   - Pre-provisioned device credentials
   - Secure key management systems (HSM, TPM)

2. **Secret Rotation**: Rotate shared secrets regularly:
   - Per-mission: New secret for each operation
   - Daily: Automated rotation schedule
   - On-demand: Manual rotation if compromise suspected

3. **Time Security**: Protect time synchronization:
   - Use authenticated time sources (NTS, GPS with SAASM)
   - Monitor for time-skewing attacks
   - Deploy intrusion detection for clock manipulation

4. **Monitoring**: Track operational metrics:
   - Decryption failure rate
   - Clock drift measurements
   - Packet loss and retransmissions
   - Key generation counters (for OTAR)

## Troubleshooting

### "Decryption failed"

**Causes**:
- Different shared secrets (check hex values match!)
- Clock drift exceeds drift_window
- Replay attack detected (sending same packet twice)
- Wrong slot_duration settings

**Solutions**:
- Verify secrets match exactly
- Sync clocks with NTP
- Increase drift_window
- Ensure sequence numbers increment

### "ValueError: Shared secret must be 32 bytes"

**Cause**: Secret is wrong length

**Solution**:
```python
# Correct: use generate_shared_secret()
secret = generate_shared_secret()

# Wrong: arbitrary bytes
secret = b"my password"  # Too short!
```

### High Latency or Packet Loss

**Causes**:
- Network congestion
- Insufficient drift_window for network delays
- Cryptographic overhead on constrained devices

**Solutions**:
- Use UDP instead of TCP for lower latency
- Increase slot_duration for more time budget
- Deploy hardware crypto acceleration
- Optimize packet sizes

## Next Steps

- Read [Full Documentation](docs/TRANSEC.md)
- See [Protocol Comparison](docs/TRANSEC_PROTOCOL_COMPARISON.md)
- Explore [Usage Guide](docs/TRANSEC_USAGE.md)
- Check [Examples](examples/)
- Run [Tests](tests/)

## Getting Help

- GitHub Issues: https://github.com/zfifteen/transec/issues
- Documentation: https://github.com/zfifteen/transec/blob/main/docs/
- Security Issues: security@example.com

## License

MIT License - see LICENSE file
