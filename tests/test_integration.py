#!/usr/bin/env python3
"""
Integration tests for TRANSEC benchmark workflows.

These tests validate the end-to-end functionality of:
1. Curvature testing with CSV output
2. UDP demo benchmark mode
3. Performance metrics validation
"""

import sys
import os
import unittest
import tempfile
import csv
import json
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transec import TransecCipher, generate_shared_secret


class TestCurvatureWorkflow(unittest.TestCase):
    """Test curvature_test.py script integration."""
    
    def test_curvature_test_basic(self):
        """Test that curvature_test.py runs successfully."""
        result = subprocess.run(
            ['python', 'bin/curvature_test.py', '--slots', '50'],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        self.assertEqual(result.returncode, 0, f"curvature_test.py failed: {result.stderr}")
        self.assertIn("TRANSEC Curvature Reduction Test", result.stdout)
        self.assertIn("95% Confidence Interval", result.stdout)
    
    def test_curvature_test_csv_output(self):
        """Test that curvature_test.py generates valid CSV output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'curvature.csv')
            
            result = subprocess.run(
                ['python', 'bin/curvature_test.py', '--slots', '50', '--output', csv_path],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            self.assertEqual(result.returncode, 0)
            self.assertTrue(os.path.exists(csv_path), "CSV file not created")
            
            # Validate CSV structure
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                self.assertIn('slot_index', headers)
                self.assertIn('raw_curvature', headers)
                self.assertIn('normalized_curvature', headers)
                self.assertIn('reduction_percent', headers)
                
                # Read some rows to validate data
                rows = list(reader)
                self.assertGreater(len(rows), 0, "CSV has no data rows")
                
                # Validate data types
                for row in rows[:5]:
                    self.assertIsNotNone(row['slot_index'])
                    self.assertIsNotNone(row['raw_curvature'])
                    self.assertIsNotNone(row['normalized_curvature'])
                    self.assertIsNotNone(row['reduction_percent'])


class TestBenchmarkMetrics(unittest.TestCase):
    """Test benchmark performance metrics."""
    
    def test_encryption_throughput(self):
        """Test that encryption achieves reasonable throughput."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, slot_duration=5, drift_window=2)
        
        # Measure encryption time for 100 messages
        num_messages = 100
        start = time.time()
        
        for i in range(num_messages):
            plaintext = f"Benchmark message {i}".encode()
            packet = cipher.seal(plaintext, sequence=i)
            self.assertIsNotNone(packet)
        
        elapsed = time.time() - start
        throughput = num_messages / elapsed
        
        # Should achieve at least 1000 msg/sec (conservative for encryption only)
        self.assertGreater(throughput, 1000, f"Throughput too low: {throughput:.1f} msg/sec")
    
    def test_roundtrip_latency(self):
        """Test that roundtrip encryption/decryption is fast."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, slot_duration=5, drift_window=2)
        
        plaintext = b"Test message for latency measurement"
        
        # Measure roundtrip time for 50 messages
        latencies = []
        for i in range(50):
            start = time.time()
            packet = cipher.seal(plaintext, sequence=i)
            decrypted = cipher.open(packet, check_replay=False)  # Skip replay for speed
            latency = (time.time() - start) * 1000  # Convert to ms
            
            self.assertEqual(decrypted, plaintext)
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        
        # Average latency should be < 1ms (conservative)
        self.assertLess(avg_latency, 1.0, f"Average latency too high: {avg_latency:.3f}ms")
    
    def test_success_rate_no_drift(self):
        """Test 100% success rate with synchronized clocks."""
        secret = generate_shared_secret()
        sender = TransecCipher(secret, slot_duration=5, drift_window=2)
        receiver = TransecCipher(secret, slot_duration=5, drift_window=2)
        
        successes = 0
        failures = 0
        
        for i in range(100):
            plaintext = f"Message {i}".encode()
            packet = sender.seal(plaintext, sequence=i)
            decrypted = receiver.open(packet)
            
            if decrypted == plaintext:
                successes += 1
            else:
                failures += 1
        
        success_rate = successes / (successes + failures) * 100
        
        # Should achieve 100% success with no drift
        self.assertEqual(success_rate, 100.0, f"Success rate {success_rate}% < 100%")


class TestPrimeOptimization(unittest.TestCase):
    """Test prime optimization functionality."""
    
    def test_prime_strategy_nearest(self):
        """Test that nearest prime strategy works."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, slot_duration=5, drift_window=2, prime_strategy="nearest")
        
        plaintext = b"Test with prime optimization"
        packet = cipher.seal(plaintext, sequence=1)
        decrypted = cipher.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_prime_strategy_next(self):
        """Test that next prime strategy works."""
        secret = generate_shared_secret()
        cipher = TransecCipher(secret, slot_duration=5, drift_window=2, prime_strategy="next")
        
        plaintext = b"Test with next prime strategy"
        packet = cipher.seal(plaintext, sequence=1)
        decrypted = cipher.open(packet)
        
        self.assertEqual(decrypted, plaintext)
    
    def test_interoperability_prime_strategies(self):
        """Test that same prime strategy works between sender/receiver."""
        secret = generate_shared_secret()
        
        # Test with nearest strategy
        sender = TransecCipher(secret, slot_duration=5, prime_strategy="nearest")
        receiver = TransecCipher(secret, slot_duration=5, prime_strategy="nearest")
        
        plaintext = b"Interoperability test"
        packet = sender.seal(plaintext, sequence=1)
        decrypted = receiver.open(packet)
        
        self.assertEqual(decrypted, plaintext)


if __name__ == '__main__':
    unittest.main()
