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
    
    def __init__(self, host: str, port: int, shared_secret: bytes):
        self.host = host
        self.port = port
        self.cipher = TransecCipher(shared_secret, slot_duration=5, drift_window=2)
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
    
    def __init__(self, host: str, port: int, shared_secret: bytes):
        self.host = host
        self.port = port
        self.cipher = TransecCipher(shared_secret, slot_duration=5, drift_window=2)
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
    
    def run_benchmark(self, count: int = 100):
        """Run performance benchmark."""
        print(f"🔐 TRANSEC UDP Client - Benchmarking {count} messages")
        print(f"Connected to {self.host}:{self.port}")
        print()
        
        successes = 0
        total_time = 0
        rtts = []
        
        for i in range(count):
            message = f"Benchmark message {i+1}"
            self.sequence += 1
            packet = self.cipher.seal(message.encode(), self.sequence)
            
            try:
                start = time.time()
                self.socket.sendto(packet, (self.host, self.port))
                response_packet, _ = self.socket.recvfrom(65536)
                rtt = time.time() - start
                
                response = self.cipher.open(response_packet)
                if response:
                    successes += 1
                    rtts.append(rtt * 1000)
                    total_time += rtt
                
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{count} messages")
            
            except socket.timeout:
                pass
            except Exception as e:
                print(f"Error at message {i+1}: {e}")
        
        print()
        print("=" * 60)
        print("Benchmark Results:")
        print(f"  Success rate: {successes}/{count} ({successes/count*100:.1f}%)")
        if rtts:
            print(f"  Average RTT: {sum(rtts)/len(rtts):.2f}ms")
            print(f"  Min RTT: {min(rtts):.2f}ms")
            print(f"  Max RTT: {max(rtts):.2f}ms")
            print(f"  Throughput: {successes/total_time:.1f} msg/sec")
        print("=" * 60)
        
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
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("TRANSEC UDP Demo")
    print("Zero-Handshake Encrypted Messaging")
    print("=" * 60)
    print()
    
    if args.mode == "server":
        server = TransecUDPServer(args.host, args.port, DEMO_SECRET)
        server.start()
    
    elif args.mode == "client":
        client = TransecUDPClient(args.host, args.port, DEMO_SECRET)
        client.run_interactive()
    
    elif args.mode == "benchmark":
        client = TransecUDPClient(args.host, args.port, DEMO_SECRET)
        client.run_benchmark(args.count)


if __name__ == "__main__":
    main()
