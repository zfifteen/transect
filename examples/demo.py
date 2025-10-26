#!/usr/bin/env python3
"""
TRANSEC Feature Demonstration

Showcases core TRANSEC features and advanced capabilities.
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec import (
    TransecCipher,
    generate_shared_secret,
    ADAPTIVE_AVAILABLE,
    OTAR_AVAILABLE
)

if ADAPTIVE_AVAILABLE:
    from transec import AdaptiveTransecCipher

if OTAR_AVAILABLE:
    from transec import OTARTransecCipher


def demo_basic():
    """Demonstrate basic TRANSEC encryption."""
    print("=" * 70)
    print("DEMO 1: Basic TRANSEC Encryption")
    print("=" * 70)
    
    # Generate shared secret
    secret = generate_shared_secret()
    print(f"\n[*] Generated shared secret: {secret[:8].hex()}... (32 bytes)")
    
    # Create cipher instances
    sender = TransecCipher(secret, slot_duration=5, drift_window=2)
    receiver = TransecCipher(secret, slot_duration=5, drift_window=2)
    print("[*] Created sender and receiver with 5s slots, ±2 drift window")
    
    # Send messages
    print("\n[*] Sending encrypted messages:")
    for i in range(3):
        plaintext = f"Message {i+1}: Hello from TRANSEC!".encode()
        packet = sender.seal(plaintext, sequence=i+1)
        
        print(f"\n    Sent #{i+1}:")
        print(f"      Plaintext: {len(plaintext)} bytes")
        print(f"      Encrypted: {len(packet)} bytes")
        print(f"      Overhead: {len(packet) - len(plaintext)} bytes")
        
        # Decrypt
        decrypted = receiver.open(packet)
        if decrypted:
            print(f"      ✓ Decrypted: {decrypted.decode()}")
        else:
            print(f"      ✗ Decryption failed!")
        
        time.sleep(0.5)
    
    # Test replay protection
    print("\n[*] Testing replay protection:")
    old_packet = sender.seal(b"Replay test", sequence=100)
    
    # First attempt - should succeed
    result1 = receiver.open(old_packet)
    print(f"    First delivery: {'✓ Success' if result1 else '✗ Failed'}")
    
    # Second attempt - should fail (replay)
    result2 = receiver.open(old_packet)
    print(f"    Replay attempt: {'✗ Blocked (good!)' if not result2 else '✓ Accepted (bad!)'}")
    
    print()


def demo_adaptive():
    """Demonstrate adaptive slot duration."""
    if not ADAPTIVE_AVAILABLE:
        print("\n[!] Adaptive features not available (requires cryptography module)")
        return
    
    print("=" * 70)
    print("DEMO 2: Adaptive Slot Duration (Dynamic Timing)")
    print("=" * 70)
    
    secret = generate_shared_secret()
    print(f"\n[*] Using adaptive timing with 2-10s jitter range")
    
    sender = AdaptiveTransecCipher(secret, jitter_range=(2, 10))
    receiver = AdaptiveTransecCipher(secret, jitter_range=(2, 10))
    
    # Show slot duration variation
    print("\n[*] Slot durations for different epochs:")
    for epoch in range(10):
        duration = sender.get_adaptive_slot_duration(epoch)
        print(f"    Epoch {epoch:2d}: {duration}s")
    
    # Send message
    print("\n[*] Sending message with adaptive timing:")
    packet = sender.seal(b"Adaptive message test", sequence=1)
    decrypted = receiver.open(packet)
    
    if decrypted:
        print(f"    ✓ Message encrypted and decrypted successfully")
        print(f"    ✓ Timing unpredictability adds anti-analysis protection")
    else:
        print(f"    ✗ Decryption failed")
    
    print()


def demo_otar():
    """Demonstrate OTAR key refresh."""
    if not OTAR_AVAILABLE:
        print("\n[!] OTAR features not available")
        return
    
    print("=" * 70)
    print("DEMO 3: OTAR-Lite Key Refresh")
    print("=" * 70)
    
    secret = generate_shared_secret()
    print(f"\n[*] Creating OTAR cipher with 60s refresh interval")
    
    cipher = OTARTransecCipher(
        secret,
        refresh_interval=60,
        auto_refresh=False  # Manual for demo
    )
    
    print(f"\n[*] Initial state:")
    print(f"    Generation: {cipher.get_generation()}")
    print(f"    Current secret: {cipher.current_secret[:8].hex()}...")
    
    # Send message in generation 0
    msg1 = cipher.seal(b"Generation 0 message", sequence=1)
    print(f"\n[*] Sent message in generation 0")
    print(f"    Packet size: {len(msg1)} bytes")
    
    # Manual refresh
    print(f"\n[*] Performing manual key refresh...")
    cipher.manual_refresh()
    
    print(f"\n[*] After refresh:")
    print(f"    Generation: {cipher.get_generation()}")
    print(f"    New secret: {cipher.current_secret[:8].hex()}...")
    print(f"    ✓ Secret rotated automatically")
    
    # Send message in generation 1
    msg2 = cipher.seal(b"Generation 1 message", sequence=2)
    print(f"\n[*] Sent message in generation 1")
    print(f"    Packet size: {len(msg2)} bytes")
    
    # Show refresh timing
    print(f"\n[*] Key refresh reduces exposure window:")
    print(f"    Without OTAR: Entire session uses same key")
    print(f"    With OTAR: Key rotates every {cipher.refresh_interval}s")
    print(f"    ✓ Compromised key affects limited time window only")
    
    print()


def demo_performance():
    """Demonstrate performance characteristics."""
    print("=" * 70)
    print("DEMO 4: Performance Characteristics")
    print("=" * 70)
    
    secret = generate_shared_secret()
    cipher = TransecCipher(secret)
    
    # Measure encryption performance
    print("\n[*] Measuring encryption performance:")
    
    message = b"Performance test message" * 10  # ~240 bytes
    iterations = 1000
    
    start = time.time()
    for i in range(iterations):
        packet = cipher.seal(message, sequence=i)
    elapsed = time.time() - start
    
    throughput = iterations / elapsed
    avg_time = (elapsed / iterations) * 1000  # ms
    
    print(f"    Messages encrypted: {iterations}")
    print(f"    Total time: {elapsed:.3f}s")
    print(f"    Average time: {avg_time:.3f}ms per message")
    print(f"    Throughput: {throughput:.0f} msg/sec")
    
    # Measure decryption performance
    print("\n[*] Measuring decryption performance:")
    
    packets = [cipher.seal(message, sequence=i, check_replay=False) for i in range(iterations)]
    
    start = time.time()
    for packet in packets:
        plaintext = cipher.open(packet, check_replay=False)
    elapsed = time.time() - start
    
    throughput = iterations / elapsed
    avg_time = (elapsed / iterations) * 1000  # ms
    
    print(f"    Messages decrypted: {iterations}")
    print(f"    Total time: {elapsed:.3f}s")
    print(f"    Average time: {avg_time:.3f}ms per message")
    print(f"    Throughput: {throughput:.0f} msg/sec")
    
    print(f"\n[*] Result: Sub-millisecond latency, ~{throughput:.0f} msg/sec throughput")
    print()


def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║                    TRANSEC Feature Demonstration                   ║")
    print("║            Time-Synchronized Encryption for Zero-RTT               ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    try:
        demo_basic()
        input("Press Enter to continue to Adaptive Demo...")
        
        demo_adaptive()
        input("Press Enter to continue to OTAR Demo...")
        
        demo_otar()
        input("Press Enter to continue to Performance Demo...")
        
        demo_performance()
        
        print("=" * 70)
        print("Demonstration Complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  - Read QUICKSTART.md for getting started guide")
        print("  - See docs/ for full documentation")
        print("  - Try examples/transec_messenger.py for CLI tool")
        print("  - Run examples/transec_udp_demo.py for UDP benchmark")
        print()
    
    except KeyboardInterrupt:
        print("\n\n[*] Demo interrupted by user")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
