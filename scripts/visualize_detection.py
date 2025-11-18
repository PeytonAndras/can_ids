#!/usr/bin/env python3
"""Visualize detection performance from debug output or live data"""

import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def parse_debug_output(text: str) -> Dict:
    """Parse debug output to extract scores and alerts"""
    windows = []
    alerts = []
    
    # Pattern: [time] Window num [ALERT] | IF=score | PCA=score | ...
    pattern = r'\[\s*(\d+\.\d+)s\] Window\s+(\d+)\s+(🚨 ALERT|OK)\s+\|\s+IF=([\d.]+)\s+.*?\|\s+PCA=([\d.]+)'
    
    for line in text.split('\n'):
        match = re.search(pattern, line)
        if match:
            time = float(match.group(1))
            window = int(match.group(2))
            is_alert = match.group(3) == '🚨 ALERT'
            if_score = float(match.group(4))
            pca_score = float(match.group(5))
            
            windows.append({
                'time': time,
                'window': window,
                'is_alert': is_alert,
                'if_score': if_score,
                'pca_score': pca_score
            })
            
            if is_alert:
                alerts.append({'time': time, 'window': window})
    
    return {'windows': windows, 'alerts': alerts}

def create_detection_plot(data: Dict, thresholds: Dict, output_path: Path):
    """Create visualization of detection performance"""
    windows = data['windows']
    alerts = data['alerts']
    
    if not windows:
        print("No data to plot")
        return
    
    times = [w['time'] for w in windows]
    if_scores = [w['if_score'] for w in windows]
    pca_scores = [w['pca_score'] for w in windows]
    alert_times = [a['time'] for a in alerts]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Isolation Forest scores
    ax1.plot(times, if_scores, 'b-', linewidth=1.5, alpha=0.7, label='IF Score')
    ax1.axhline(y=thresholds['isolation_forest'], color='r', linestyle='--', 
                linewidth=2, label=f'IF Threshold ({thresholds["isolation_forest"]:.4f})')
    
    # Mark alerts
    alert_if_scores = [w['if_score'] for w in windows if w['is_alert']]
    alert_times_if = [w['time'] for w in windows if w['is_alert']]
    if alert_times_if:
        ax1.scatter(alert_times_if, alert_if_scores, color='red', s=50, 
                   marker='x', zorder=5, label='Alerts', linewidths=2)
    
    ax1.set_ylabel('Isolation Forest Score', fontsize=12, fontweight='bold')
    ax1.set_title('CAN Intrusion Detection System Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0, max(if_scores) * 1.1])
    
    # Plot 2: PCA scores
    ax2.plot(times, pca_scores, 'g-', linewidth=1.5, alpha=0.7, label='PCA Score')
    ax2.axhline(y=thresholds['pca'], color='r', linestyle='--', 
                linewidth=2, label=f'PCA Threshold ({thresholds["pca"]:.1f})')
    
    # Mark alerts
    alert_pca_scores = [w['pca_score'] for w in windows if w['is_alert']]
    alert_times_pca = [w['time'] for w in windows if w['is_alert']]
    if alert_times_pca:
        ax2.scatter(alert_times_pca, alert_pca_scores, color='red', s=50, 
                   marker='x', zorder=5, label='Alerts', linewidths=2)
    
    # Highlight attack periods (where PCA > threshold)
    attack_periods = []
    in_attack = False
    attack_start = None
    for w in windows:
        if w['pca_score'] > thresholds['pca']:
            if not in_attack:
                attack_start = w['time']
                in_attack = True
        else:
            if in_attack:
                attack_periods.append((attack_start, w['time']))
                in_attack = False
    if in_attack:
        attack_periods.append((attack_start, times[-1]))
    
    for start, end in attack_periods:
        ax2.axvspan(start, end, alpha=0.2, color='red', label='Attack Period' if start == attack_periods[0][0] else '')
    
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PCA Reconstruction Error', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, max(pca_scores) * 1.1])
    
    # Add statistics text box
    total_windows = len(windows)
    total_alerts = len(alerts)
    alert_rate = (total_alerts / (times[-1] / 3600)) if times else 0
    
    stats_text = f'Total Windows: {total_windows}\n'
    stats_text += f'Total Alerts: {total_alerts}\n'
    stats_text += f'Alert Rate: {alert_rate:.1f} alerts/hour\n'
    stats_text += f'Detection Rate: {(total_alerts/total_windows*100):.1f}%'
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize detection performance')
    parser.add_argument('--input', type=Path, help='Debug output file (or stdin)')
    parser.add_argument('--output', type=Path, default=Path('results/unsupervised/real_can0/detection_performance.png'),
                       help='Output image path')
    parser.add_argument('--thresholds', type=Path, default=Path('results/unsupervised/real_can0/thresholds.json'),
                       help='Thresholds JSON file')
    args = parser.parse_args()
    
    # Load thresholds
    with open(args.thresholds, 'r') as f:
        thresholds_data = json.load(f)
    thresholds = thresholds_data['values']
    
    # Read input
    if args.input and args.input.exists():
        text = args.input.read_text()
    else:
        print("Reading from stdin... (paste debug output and press Ctrl+D)")
        text = sys.stdin.read()
    
    # Parse data
    data = parse_debug_output(text)
    
    if not data['windows']:
        print("No data found in input")
        return
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Create plot
    create_detection_plot(data, thresholds, args.output)

if __name__ == '__main__':
    main()

