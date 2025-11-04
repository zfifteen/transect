def run_benchmark(self, count: int = 100, log_file: str = None):
    """Run performance benchmark with optional LDJSON logging."""
    print(f"🔐 TRANSEC UDP Client - Benchmarking {count} messages")
    print(f"Connected to {self.host}:{self.port}")
    if self.skew_slots > 0:
        print(f"Clock skew: ±{self.skew_slots} slots")
    print()
    
    successes = 0
    rejections = 0
    total_time = 0
    rtts = []
    
    log_fh = None
    if log_file:
        with open(log_file, 'w') as log_fh:
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
            
            for i in range(count):
                # Apply clock skew randomly per message
                skew = self._inject_clock_skew() if self.skew_slots > 0 else 0
                
                message = f"Benchmark message {i+1}"
                self.sequence += 1
                
                success = False
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
            
            # Write benchmark end event
            total_messages = successes + rejections
            end_event = {
                'event': 'benchmark_end',
                'timestamp': time.time(),
                'total_messages': total_messages,
                'successes': successes,
                'rejections': rejections,
            }
            log_fh.write(json.dumps(end_event) + '\n')
            print(f"\nLog written to {log_file}")
    else:
        for i in range(count):
            # Apply clock skew randomly per message
            skew = self._inject_clock_skew() if self.skew_slots > 0 else 0
            
            message = f"Benchmark message {i+1}"
            self.sequence += 1
            
            success = False
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
    
    self.socket.close()