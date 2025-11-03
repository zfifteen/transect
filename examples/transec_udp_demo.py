#!/usr/bin/env python3
"""
TRANSEC UDP Demo: Zero-Handshake Encrypted Messaging

This demo shows TRANSEC in action with UDP client/server communication.
No handshake is required - messages are encrypted and sent immediately.

Usage:
    # Start server
    python3 transec_udp_demo.py server

    # Run client in another terminal
    python3 transec_udp_demo.py client
"""

import sys
import os
import socket
import time
import threading
import argparse
import json
import random

# Add parent directory to path if running from examples/
if os.path.exists('../python/transec.py'):
    sys.path.insert(0, '../python')
else:
    sys.path.insert(0, './python')

from transec import TransecCipher, generate_shared_secret


# For demo purposes, use a fixed shared secret
# In production, this would be provisioned via secure channel
DEMO_SECRET = bytes.fromhex(
    "deadbeefdeadbeefdeadbeefdeadbeef"
    "deadbeefdeadbeefdeadbeefdeadbeef"
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9999


class TransecUDPServer:
    """UDP server with TRANSEC encryption."""
    
    def __init__(self, host: str, port: int, shared_secret: bytes, 
                 slot_duration: float = 5, drift_window: int = 2, 
                 prime_strategy: str = "none"):
        self.host = host
        self.port = port
        self.cipher = TransecCipher(shared_secret, slot_duration=slot_duration, 
                                     drift_window=drift_window, prime_strategy=prime_strategy)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.running = False
        self.sequence = 0
    
    def start(self):
        """Start the server."""
        self.running = True
        print(f"🔐 TRANSEC UDP Server listening on {self.host}:{self.port}")
        print("Waiting for encrypted messages...")
        print()
        
        try:
            while self.running:
                try:
                    # Receive encrypted packet
                    packet, client_addr = self.socket.recvfrom(65536)
                    
                    # Decrypt message
                    plaintext = self.cipher.open(packet)
                    
                    if plaintext:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"[{timestamp}] From {client_addr}: {plaintext.decode('utf-8', errors='replace')}")
                        
                        # Send encrypted response
                        self.sequence += 1
                        response = f"Echo: {plaintext.decode('utf-8', errors='replace')}"
                        response_packet = self.cipher.seal(response.encode(), self.sequence)
                        self.socket.sendto(response_packet, client_addr)
                    else:
                        print(f"⚠️  Rejected packet from {client_addr} (auth failed or replay)")
                
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error: {e}")
        
        except KeyboardInterrupt:
            print("\nServer shutting down...")
        finally:
            self.socket.close()
    
    def stop(self):
        """Stop the server."""
        self.running = False


class TransecUDPClient:
    """UDP client with TRANSEC encryption."""
    
    def __init__(self, host: str, port: int, shared_secret: bytes,
                 slot_duration: float = 5, drift_window: int = 2,
                 prime_strategy: str = "none", skew_slots: int = 0):
        self.host = host
        self.port = port
        self.slot_duration = slot_duration
        self.skew_slots = skew_slots
        self.cipher = TransecCipher(shared_secret, slot_duration=slot_duration,
                                     drift_window=drift_window, prime_strategy=prime_strategy)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(2.0)
        self.sequence = 0
    
    def send_message(self, message: str) -> bool:
        """Send encrypted message and wait for response."""
        try:
            # Encrypt and send (ZERO HANDSHAKE!)
            self.sequence += 1
            packet = self.cipher.seal(message.encode(), self.sequence)
            
            start_time = time.time()
            self.socket.sendto(packet, (self.host, self.port))
            
            # Wait for encrypted response
            response_packet, _ = self.socket.recvfrom(65536)
            rtt = (time.time() - start_time) * 1000
            
            # Decrypt response
            response = self.cipher.open(response_packet)
            
            if response:
                print(f"✓ Response ({rtt:.1f}ms): {response.decode('utf-8', errors='replace')}")
                return True
            else:
                print("✗ Response authentication failed")
                return False
        
        except socket.timeout:
            print("✗ Timeout waiting for response")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def run_interactive(self):
        """Run interactive client session."""
        print(f"🔐 TRANSEC UDP Client connected to {self.host}:{self.port}")
        print("Type messages to send (Ctrl+C to quit)")
        print("=" * 60)
        print()
        
        try:
            while True:
                message = input("Message: ").strip()
                if message:
                    self.send_message(message)
                    print()
        
        except KeyboardInterrupt:
            print("\nClient shutting down...")
        finally:
            self.socket.close()
    
    def _inject_clock_skew(self):
        """Inject synthetic clock skew by manipulating the cipher's time."""
        if self.skew_slots == 0:
            return
        
        # Apply random skew within ±skew_slots range
        skew = random.randint(-abs(self.skew_slots), abs(self.skew_slots))
        skew_seconds = skew * self.slot_duration
        
        # Monkey-patch the cipher's get_current_slot method
        def skewed_get_current_slot():
            # Calculate skewed time
            skewed_time = time.time() + skew_seconds
            raw_slot = int(skewed_time / self.slot_duration)
            return self.cipher._normalize_slot(raw_slot)
        
        self.cipher.get_current_slot = skewed_get_current_slot
        return skew
    
    def run_benchmark(self, count: int = 100, log_file: str = None):
        """Run performance benchmark with optional LDJSON logging."""
        print(f"🔐 TRANSEC UDP Client - Benchmarking {count} messages")
        print(f"Connected to {self.host}:{self.port}")
        if self.skew_slots > 0:
            print(f"Clock skew: ±{self.skew_slots} slots")
        print()
        
        log_fh = None
        if log_file:
            log_fh = open(log_file, 'w')
            # Write benchmark metadata
            metadata = {
                'event': 'benchmark_start',
                'timestamp': time.time(),
                'count': count,
                'slot_duration': self.slot_duration,
                'skew_slots': self.skew_slots,
                'prime_strategy': self.cipher.prime_strategy,
                'drift_window': self.cipher.drift_window,
            }
            log_fh.write(json.dumps(metadata) + '\n')
            log_fh.flush()
        
        successes = 0
        rejections = 0
        total_time = 0
        rtts = []
        
        for i in range(count):
            # Apply clock skew randomly per message
            skew = self._inject_clock_skew() if self.skew_slots > 0 else 0
            
            message = f"Benchmark message {i+1}"
            self.sequence += 1
            
            success = False
            rtt_ms = 0.0
            current_slot = 0
            try:
                # Get current slot for logging
                current_slot = self.cipher.get_current_slot()
                
                packet = self.cipher.seal(message.encode(), self.sequence)
                
                start = time.time()
                self.socket.sendto(packet, (self.host, self.port))
                response_packet, _ = self.socket.recvfrom(65536)
                rtt = time.time() - start
                rtt_ms = rtt * 1000
                
                response = self.cipher.open(response_packet)
                if response:
                    successes += 1
                    # Clamp RTT to 0 to handle any timing precision issues
                    rtt_ms = max(0.0, rtt_ms)
                    rtts.append(rtt_ms)
                    total_time += rtt
                    success = True
                else:
                    rejections += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{count} messages")
            
            except socket.timeout:
                rejections += 1
                rtt_ms = -1.0  # Indicate timeout
            except Exception as e:
                rejections += 1
                print(f"Error at message {i+1}: {e}")
                rtt_ms = -1.0
            
            # Log event in LDJSON format
            if log_fh:
                event = {
                    'event': 'message_sent',
                    'timestamp': time.time(),
                    'sequence': self.sequence,
                    'success': success,
                    'rtt_ms': rtt_ms,
                    'slot_index': current_slot if success else 0,
                    'skew_applied': skew if self.skew_slots > 0 else 0,
                }
                log_fh.write(json.dumps(event) + '\n')
                log_fh.flush()
        
        print()
        print("=" * 60)
        print("Benchmark Results:")
        total_messages = successes + rejections
        print(f"  Success rate: {successes}/{total_messages} ({successes/total_messages*100:.1f}%)")
        print(f"  Rejections: {rejections} ({rejections/total_messages*100:.1f}%)")
        if rtts:
            print(f"  Average RTT: {sum(rtts)/len(rtts):.2f}ms")
            print(f"  Min RTT: {min(rtts):.2f}ms")
            print(f"  Max RTT: {max(rtts):.2f}ms")
            print(f"  Throughput: {successes/total_time:.1f} msg/sec")
        print("=" * 60)
        
        if log_fh:
            # Write benchmark end event
            end_event = {
                'event': 'benchmark_end',
                'timestamp': time.time(),
                'total_messages': total_messages,
                'successes': successes,
                'rejections': rejections,
            }
            log_fh.write(json.dumps(end_event) + '\n')
            log_fh.close()
            print(f"\nLog written to {log_file}")
        
        self.socket.close()


def main():
    parser = argparse.ArgumentParser(
        description="TRANSEC UDP Demo - Zero-Handshake Encrypted Messaging"
    )
    parser.add_argument(
        "mode",
        choices=["server", "client", "benchmark"],
        help="Run as server, interactive client, or benchmark client"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host address (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port number (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of messages for benchmark (default: 100)"
    )
    parser.add_argument(
        "--prime_strategy",
        choices=["none", "nearest", "next"],
        default="none",
        help="Prime slot mapping strategy (default: none)"
    )
    parser.add_argument(
        "--drift_window",
        type=int,
        default=2,
        help="Clock drift tolerance in slots (default: 2)"
    )
    parser.add_argument(
        "--slot_duration",
        type=float,
        default=5.0,
        help="Slot duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--skew_slots",
        type=int,
        default=0,
        help="Synthetic clock skew in ±slots (default: 0, disabled)"
    )
    parser.add_argument(
        "--out",
        help="Output log file for benchmark (LDJSON format)"
    )
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("TRANSEC UDP Demo")
    print("Zero-Handshake Encrypted Messaging")
    print("=" * 60)
    print()
    
    if args.mode == "server":
        server = TransecUDPServer(
            args.host, args.port, DEMO_SECRET,
            slot_duration=args.slot_duration,
            drift_window=args.drift_window,
            prime_strategy=args.prime_strategy
        )
        server.start()
    
    elif args.mode == "client":
        client = TransecUDPClient(
            args.host, args.port, DEMO_SECRET,
            slot_duration=args.slot_duration,
            drift_window=args.drift_window,
            prime_strategy=args.prime_strategy
        )
        client.run_interactive()
    
    elif args.mode == "benchmark":
        client = TransecUDPClient(
            args.host, args.port, DEMO_SECRET,
            slot_duration=args.slot_duration,
            drift_window=args.drift_window,
            prime_strategy=args.prime_strategy,
            skew_slots=args.skew_slots
        )
        client.run_benchmark(args.count, log_file=args.out)


if __name__ == "__main__":
    main()
