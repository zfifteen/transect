#!/usr/bin/env python3
"""
TRANSEC CLI Messenger Example

Simple command-line messenger demonstrating TRANSEC zero-handshake encryption.
Uses UDP for communication with QR code support for easy secret sharing.
"""

import sys
import socket
import argparse
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec import TransecCipher, generate_shared_secret


def create_shared_secret_qr(secret: bytes) -> str:
    """
    Create a QR code for sharing the secret (concept).
    In production, would use qrcode library.
    
    Args:
        secret: Shared secret bytes
    
    Returns:
        Base64 encoded secret for display
    """
    import base64
    return base64.b64encode(secret).decode('ascii')


def parse_shared_secret_qr(qr_data: str) -> bytes:
    """
    Parse QR code data to extract secret.
    
    Args:
        qr_data: Base64 encoded secret
    
    Returns:
        Shared secret bytes
    """
    import base64
    return base64.b64decode(qr_data.encode('ascii'))


def listen_mode(port: int, secret: bytes, slot_duration: int = 5):
    """
    Listen for incoming TRANSEC messages.
    
    Args:
        port: UDP port to listen on
        secret: Shared secret
        slot_duration: Time slot duration in seconds
    """
    cipher = TransecCipher(secret, slot_duration=slot_duration)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind to all interfaces for demo purposes
    # In production, bind to specific interface for security
    sock.bind(('0.0.0.0', port))  # nosec - demo server binds to all interfaces
    
    print(f"[*] Listening on UDP port {port}")
    print(f"[*] Slot duration: {slot_duration}s")
    print("[*] Waiting for messages (Ctrl+C to exit)...")
    print()
    
    sequence = 0
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            
            # Try to decrypt
            plaintext = cipher.open(data)
            
            if plaintext:
                print(f"[{time.strftime('%H:%M:%S')}] From {addr[0]}:{addr[1]}")
                print(f"    Message: {plaintext.decode('utf-8', errors='replace')}")
                print()
            else:
                print(f"[!] Failed to decrypt message from {addr[0]}:{addr[1]}")
                print()
    
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    finally:
        sock.close()


def send_mode(host: str, port: int, message: str, secret: bytes, slot_duration: int = 5):
    """
    Send a TRANSEC encrypted message.
    
    Args:
        host: Target hostname/IP
        port: Target UDP port
        message: Message to send
        secret: Shared secret
        slot_duration: Time slot duration in seconds
    """
    cipher = TransecCipher(secret, slot_duration=slot_duration)
    
    # Use timestamp as sequence number for simplicity
    sequence = int(time.time() * 1000) & 0xFFFFFFFF
    
    # Encrypt message
    packet = cipher.seal(message.encode('utf-8'), sequence)
    
    # Send via UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(packet, (host, port))
        print(f"[*] Message sent to {host}:{port}")
        print(f"    Size: {len(packet)} bytes")
        print(f"    Sequence: {sequence}")
    finally:
        sock.close()


def generate_mode():
    """Generate a new shared secret and display as QR."""
    secret = generate_shared_secret()
    qr_data = create_shared_secret_qr(secret)
    
    print("[*] Generated new shared secret")
    print()
    print("    Share this with your peer (QR-ready format):")
    # Security note: This intentionally displays the secret for the user to share
    # In production, use secure out-of-band channels (QR code, USB, secure vault)
    print(f"    {qr_data}")  # nosec - intentional display for key provisioning
    print()
    print("    Use --secret option to use this key")
    print()
    print("    WARNING: Keep this secret secure!")
    print("    Anyone with this secret can decrypt your messages.")


def main():
    parser = argparse.ArgumentParser(
        description='TRANSEC CLI Messenger - Zero-handshake encrypted messaging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate shared secret
  %(prog)s generate
  
  # Listen for messages
  %(prog)s listen --port 5000 --secret <base64-secret>
  
  # Send message
  %(prog)s send --host 192.168.1.100 --port 5000 --message "Hello!" --secret <base64-secret>
  
  # Interactive chat
  %(prog)s chat --port 5000 --peer 192.168.1.100:5001 --secret <base64-secret>
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    subparsers.add_parser('generate', help='Generate new shared secret')
    
    # Listen command
    listen_parser = subparsers.add_parser('listen', help='Listen for incoming messages')
    listen_parser.add_argument('--port', type=int, required=True, help='UDP port to listen on')
    listen_parser.add_argument('--secret', required=True, help='Base64-encoded shared secret')
    listen_parser.add_argument('--slot-duration', type=int, default=5, help='Slot duration (seconds)')
    
    # Send command
    send_parser = subparsers.add_parser('send', help='Send a message')
    send_parser.add_argument('--host', required=True, help='Target host')
    send_parser.add_argument('--port', type=int, required=True, help='Target UDP port')
    send_parser.add_argument('--message', required=True, help='Message to send')
    send_parser.add_argument('--secret', required=True, help='Base64-encoded shared secret')
    send_parser.add_argument('--slot-duration', type=int, default=5, help='Slot duration (seconds)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'generate':
            generate_mode()
        
        elif args.command == 'listen':
            secret = parse_shared_secret_qr(args.secret)
            listen_mode(args.port, secret, args.slot_duration)
        
        elif args.command == 'send':
            secret = parse_shared_secret_qr(args.secret)
            send_mode(args.host, args.port, args.message, secret, args.slot_duration)
        
        return 0
    
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
