#!/usr/bin/env python3
"""
Parse TRANSEC benchmark logs and generate CSV analysis with bootstrap confidence intervals.

Reads LDJSON logs from benchmark runs, calculates metrics, and outputs:
- CSV file with aggregated statistics
- Optional histogram plots comparing drift scenarios
- Bootstrap 95% confidence intervals for rejection rates

Usage:
    python scripts/parse_transect_logs.py baseline.log prime.log --out results.csv
    python scripts/parse_transect_logs.py baseline*.log prime*.log --out results.csv --plot plots/drift-hist.png
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import statistics
import random


def parse_log_file(log_path: str) -> List[Dict]:
    """Parse LDJSON log file and extract benchmark events."""
    events = []
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {log_path}: {e}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        return []
    
    return events


def calculate_metrics(events: List[Dict]) -> Dict:
    """Calculate benchmark metrics from events."""
    if not events:
        return {
            'total_messages': 0,
            'successful': 0,
            'rejected': 0,
            'rejection_rate': 0.0,
            'rtts': [],
            'p50_rtt': 0.0,
            'p95_rtt': 0.0,
            'mean_rtt': 0.0,
            'messages_per_sec': 0.0,
            'total_time': 0.0,
        }
    
    # Extract RTTs for successful messages
    rtts = []
    rejected_count = 0
    successful_count = 0
    
    for event in events:
        if event.get('event') == 'message_sent':
            if event.get('success', False):
                successful_count += 1
                rtt = event.get('rtt_ms', 0)
                if rtt >= 0:  # Filter out negative RTTs
                    rtts.append(rtt)
            else:
                rejected_count += 1
    
    total_messages = successful_count + rejected_count
    rejection_rate = (rejected_count / total_messages * 100) if total_messages > 0 else 0.0
    
    # Calculate RTT percentiles
    p50_rtt = statistics.median(rtts) if rtts else 0.0
    # Use more robust percentile calculation
    if len(rtts) >= 2:
        sorted_rtts = sorted(rtts)
        p95_idx = int(0.95 * len(sorted_rtts))
        p95_rtt = sorted_rtts[min(p95_idx, len(sorted_rtts) - 1)]
    else:
        p95_rtt = max(rtts) if rtts else 0.0
    mean_rtt = statistics.mean(rtts) if rtts else 0.0
    
    # Calculate throughput
    total_time = 0.0
    if events:
        # Find start and end times
        timestamps = [e.get('timestamp', 0) for e in events if 'timestamp' in e]
        if len(timestamps) >= 2:
            total_time = max(timestamps) - min(timestamps)
    
    messages_per_sec = successful_count / total_time if total_time > 0 else 0.0
    
    return {
        'total_messages': total_messages,
        'successful': successful_count,
        'rejected': rejected_count,
        'rejection_rate': rejection_rate,
        'rtts': rtts,
        'p50_rtt': p50_rtt,
        'p95_rtt': p95_rtt,
        'mean_rtt': mean_rtt,
        'messages_per_sec': messages_per_sec,
        'total_time': total_time,
    }


def bootstrap_ci(data: List[float], metric_fn, n_iterations: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        data: List of data points
        metric_fn: Function to calculate metric from data (e.g., statistics.mean)
        n_iterations: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not data:
        return (0.0, 0.0)
    
    bootstrap_metrics = []
    n = len(data)
    
    for _ in range(n_iterations):
        # Resample with replacement
        sample = [random.choice(data) for _ in range(n)]
        metric_value = metric_fn(sample)
        bootstrap_metrics.append(metric_value)
    
    # Calculate percentiles for CI
    bootstrap_metrics.sort()
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_iterations)
    upper_idx = int((1 - alpha / 2) * n_iterations)
    
    return (bootstrap_metrics[lower_idx], bootstrap_metrics[upper_idx])


def analyze_logs(log_paths: List[str]) -> Dict[str, Dict]:
    """Analyze multiple log files and return metrics for each."""
    results = {}
    
    for log_path in log_paths:
        # Extract experiment name from filename
        filename = Path(log_path).stem
        
        print(f"Processing {log_path}...")
        events = parse_log_file(log_path)
        
        if not events:
            print(f"Warning: No events found in {log_path}", file=sys.stderr)
            continue
        
        metrics = calculate_metrics(events)
        
        # Calculate bootstrap CI for rejection rate if we have enough data
        if metrics['total_messages'] >= 20:
            rejection_data = [1 if i < metrics['rejected'] else 0 
                            for i in range(metrics['total_messages'])]
            ci_lower, ci_upper = bootstrap_ci(
                rejection_data, 
                lambda x: sum(x) / len(x) * 100 if x else 0,
                n_iterations=1000
            )
            metrics['rejection_rate_ci_lower'] = ci_lower
            metrics['rejection_rate_ci_upper'] = ci_upper
        else:
            metrics['rejection_rate_ci_lower'] = metrics['rejection_rate']
            metrics['rejection_rate_ci_upper'] = metrics['rejection_rate']
        
        results[filename] = metrics
    
    return results


def write_csv(results: Dict[str, Dict], output_path: str):
    """Write results to CSV file."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Experiment',
            'Total Messages',
            'Successful',
            'Rejected',
            'Rejection Rate (%)',
            'Rejection CI Lower (%)',
            'Rejection CI Upper (%)',
            'P50 RTT (ms)',
            'P95 RTT (ms)',
            'Mean RTT (ms)',
            'Messages/sec',
            'Total Time (s)'
        ])
        
        # Write data rows
        for experiment, metrics in sorted(results.items()):
            writer.writerow([
                experiment,
                metrics['total_messages'],
                metrics['successful'],
                metrics['rejected'],
                f"{metrics['rejection_rate']:.2f}",
                f"{metrics.get('rejection_rate_ci_lower', 0):.2f}",
                f"{metrics.get('rejection_rate_ci_upper', 0):.2f}",
                f"{metrics['p50_rtt']:.2f}",
                f"{metrics['p95_rtt']:.2f}",
                f"{metrics['mean_rtt']:.2f}",
                f"{metrics['messages_per_sec']:.2f}",
                f"{metrics['total_time']:.2f}"
            ])
    
    print(f"\nResults written to {output_path}")


def plot_histogram(results: Dict[str, Dict], output_path: str):
    """Generate histogram plot comparing RTT distributions."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation", file=sys.stderr)
        return
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot RTT histograms
    for experiment, metrics in results.items():
        if metrics['rtts']:
            ax1.hist(metrics['rtts'], bins=30, alpha=0.5, label=experiment)
    
    ax1.set_xlabel('RTT (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('RTT Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot rejection rates with error bars
    experiments = list(results.keys())
    rejection_rates = [results[exp]['rejection_rate'] for exp in experiments]
    ci_lower = [results[exp].get('rejection_rate_ci_lower', results[exp]['rejection_rate']) for exp in experiments]
    ci_upper = [results[exp].get('rejection_rate_ci_upper', results[exp]['rejection_rate']) for exp in experiments]
    
    # Calculate error bars
    yerr_lower = [rejection_rates[i] - ci_lower[i] for i in range(len(experiments))]
    yerr_upper = [ci_upper[i] - rejection_rates[i] for i in range(len(experiments))]
    
    x_pos = range(len(experiments))
    ax2.bar(x_pos, rejection_rates, alpha=0.7)
    ax2.errorbar(x_pos, rejection_rates, yerr=[yerr_lower, yerr_upper], 
                 fmt='none', ecolor='black', capsize=5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(experiments, rotation=45, ha='right')
    ax2.set_ylabel('Rejection Rate (%)')
    ax2.set_title('Rejection Rates with 95% CI')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse TRANSEC benchmark logs and generate analysis'
    )
    parser.add_argument(
        'logs',
        nargs='+',
        help='Log files to parse (supports wildcards)'
    )
    parser.add_argument(
        '--out',
        default='results.csv',
        help='Output CSV file (default: results.csv)'
    )
    parser.add_argument(
        '--plot',
        help='Generate histogram plot (e.g., plots/drift-hist.png)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRANSEC Benchmark Log Analysis")
    print("=" * 60)
    print()
    
    # Analyze logs
    results = analyze_logs(args.logs)
    
    if not results:
        print("Error: No valid log files found", file=sys.stderr)
        return 1
    
    # Print summary to stdout
    print("\nSummary:")
    print("-" * 60)
    for experiment, metrics in sorted(results.items()):
        print(f"\n{experiment}:")
        print(f"  Messages: {metrics['total_messages']} (successful: {metrics['successful']}, rejected: {metrics['rejected']})")
        print(f"  Rejection rate: {metrics['rejection_rate']:.2f}% (95% CI: [{metrics.get('rejection_rate_ci_lower', 0):.2f}%, {metrics.get('rejection_rate_ci_upper', 0):.2f}%])")
        print(f"  P50 RTT: {metrics['p50_rtt']:.2f}ms")
        print(f"  Throughput: {metrics['messages_per_sec']:.2f} msg/sec")
    
    # Write CSV
    write_csv(results, args.out)
    
    # Generate plot if requested
    if args.plot:
        plot_histogram(results, args.plot)
    
    print()
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
