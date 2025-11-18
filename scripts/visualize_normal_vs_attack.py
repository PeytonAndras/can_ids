#!/usr/bin/env python3
"""Create scientific visualization comparing normal vs attack traffic detection"""

import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

def parse_debug_output(text: str) -> Dict:
    """Parse debug output to extract scores, alerts, and traffic type"""
    windows = []
    alerts = []
    
    # Pattern: [time] Window num [ALERT] | IF=score | PCA=score | Frames=X IDs=Y 0x039_rate=Z/s
    pattern = r'\[\s*(\d+\.\d+)s\] Window\s+(\d+)\s+(🚨 ALERT|OK)\s+\|\s+IF=([\d.]+)\s+.*?\|\s+PCA=([\d.]+).*?\|\s+Frames=\s*(\d+).*?0x039_rate=\s*(\d+)/s'
    
    for line in text.split('\n'):
        match = re.search(pattern, line)
        if match:
            time = float(match.group(1))
            window = int(match.group(2))
            is_alert = match.group(3) == '🚨 ALERT'
            if_score = float(match.group(4))
            pca_score = float(match.group(5))
            frames = int(match.group(6))
            rate_039 = int(match.group(7))
            
            # Classify as attack if PCA score exceeds threshold or rate is high
            # Threshold is typically 20, but we'll use rate > 100 as indicator
            is_attack_period = pca_score > 20 or rate_039 > 100
            
            windows.append({
                'time': time,
                'window': window,
                'is_alert': is_alert,
                'if_score': if_score,
                'pca_score': pca_score,
                'frames': frames,
                'rate_039': rate_039,
                'is_attack_period': is_attack_period
            })
            
            if is_alert:
                alerts.append({'time': time, 'window': window})
    
    return {'windows': windows, 'alerts': alerts}

def create_scientific_plot(data: Dict, thresholds: Dict, output_path: Path):
    """Create publication-quality visualization comparing normal vs attack"""
    windows = data['windows']
    alerts = data['alerts']
    
    if not windows:
        print("No data to plot")
        return
    
    # Separate normal and attack windows
    normal_windows = [w for w in windows if not w['is_attack_period']]
    attack_windows = [w for w in windows if w['is_attack_period']]
    
    # Extract data
    times = [w['time'] for w in windows]
    if_scores = [w['if_score'] for w in windows]
    pca_scores = [w['pca_score'] for w in windows]
    rates_039 = [w['rate_039'] for w in windows]
    
    # Normal data
    normal_times = [w['time'] for w in normal_windows]
    normal_if = [w['if_score'] for w in normal_windows]
    normal_pca = [w['pca_score'] for w in normal_windows]
    normal_rates = [w['rate_039'] for w in normal_windows]
    
    # Attack data
    attack_times = [w['time'] for w in attack_windows]
    attack_if = [w['if_score'] for w in attack_windows]
    attack_pca = [w['pca_score'] for w in attack_windows]
    attack_rates = [w['rate_039'] for w in attack_windows]
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.3, height_ratios=[1, 1, 0.8])
    
    # Plot 1: Isolation Forest Scores
    ax1 = fig.add_subplot(gs[0])
    
    # Plot normal and attack separately
    if normal_times:
        ax1.scatter(normal_times, normal_if, c='blue', alpha=0.6, s=30, 
                   label='Normal Traffic', edgecolors='darkblue', linewidths=0.5)
    if attack_times:
        ax1.scatter(attack_times, attack_if, c='red', alpha=0.7, s=50, 
                   label='Attack Traffic', edgecolors='darkred', linewidths=0.5, marker='^')
    
    # Threshold line
    ax1.axhline(y=thresholds['isolation_forest'], color='black', linestyle='--', 
                linewidth=2, label=f'Threshold ({thresholds["isolation_forest"]:.4f})', zorder=10)
    
    # Mark alerts
    alert_times_if = [w['time'] for w in windows if w['is_alert']]
    alert_if_scores = [w['if_score'] for w in windows if w['is_alert']]
    if alert_times_if:
        ax1.scatter(alert_times_if, alert_if_scores, color='orange', s=100, 
                   marker='*', zorder=15, label='Alerts', edgecolors='darkorange', linewidths=1)
    
    ax1.set_ylabel('Isolation Forest\nAnomaly Score', fontsize=13, fontweight='bold')
    ax1.set_title('Real-Time CAN Intrusion Detection: Normal vs Attack Traffic', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.set_ylim([0, max(if_scores) * 1.15])
    
    # Add statistics box for IF
    if normal_if and attack_if:
        normal_mean_if = np.mean(normal_if)
        attack_mean_if = np.mean(attack_if)
        stats_if = f'Normal: μ={normal_mean_if:.3f}\nAttack: μ={attack_mean_if:.3f}'
        ax1.text(0.02, 0.98, stats_if, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Plot 2: PCA Reconstruction Error
    ax2 = fig.add_subplot(gs[1])
    
    if normal_times:
        ax2.scatter(normal_times, normal_pca, c='blue', alpha=0.6, s=30, 
                   label='Normal Traffic', edgecolors='darkblue', linewidths=0.5)
    if attack_times:
        ax2.scatter(attack_times, attack_pca, c='red', alpha=0.7, s=50, 
                   label='Attack Traffic', edgecolors='darkred', linewidths=0.5, marker='^')
    
    # Threshold line
    ax2.axhline(y=thresholds['pca'], color='black', linestyle='--', 
                linewidth=2, label=f'Threshold ({thresholds["pca"]:.1f})', zorder=10)
    
    # Mark alerts
    alert_times_pca = [w['time'] for w in windows if w['is_alert']]
    alert_pca_scores = [w['pca_score'] for w in windows if w['is_alert']]
    if alert_times_pca:
        ax2.scatter(alert_times_pca, alert_pca_scores, color='orange', s=100, 
                   marker='*', zorder=15, label='Alerts', edgecolors='darkorange', linewidths=1)
    
    ax2.set_ylabel('PCA Reconstruction\nError', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax2.set_ylim([0, max(pca_scores) * 1.15])
    
    # Add statistics box for PCA
    if normal_pca and attack_pca:
        normal_mean_pca = np.mean(normal_pca)
        attack_mean_pca = np.mean(attack_pca)
        stats_pca = f'Normal: μ={normal_mean_pca:.2f}\nAttack: μ={attack_mean_pca:.2f}'
        ax2.text(0.02, 0.98, stats_pca, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Plot 3: Frame Rate (0x039 messages per second)
    ax3 = fig.add_subplot(gs[2])
    
    if normal_times:
        ax3.scatter(normal_times, normal_rates, c='blue', alpha=0.6, s=30, 
                   label='Normal Traffic', edgecolors='darkblue', linewidths=0.5)
    if attack_times:
        ax3.scatter(attack_times, attack_rates, c='red', alpha=0.7, s=50, 
                   label='Attack Traffic', edgecolors='darkred', linewidths=0.5, marker='^')
    
    # Highlight attack threshold (100 fps)
    ax3.axhline(y=100, color='orange', linestyle=':', linewidth=2, 
                label='Attack Threshold (100 fps)', alpha=0.7)
    
    ax3.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('0x039 Frame Rate\n(frames/second)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add statistics box for rates
    if normal_rates and attack_rates:
        normal_mean_rate = np.mean(normal_rates)
        attack_mean_rate = np.mean(attack_rates)
        stats_rate = f'Normal: μ={normal_mean_rate:.1f} fps\nAttack: μ={attack_mean_rate:.1f} fps'
        ax3.text(0.02, 0.98, stats_rate, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Add overall statistics
    total_windows = len(windows)
    total_alerts = len(alerts)
    normal_windows_count = len(normal_windows)
    attack_windows_count = len(attack_windows)
    detection_rate = (total_alerts / attack_windows_count * 100) if attack_windows_count > 0 else 0
    false_positive_rate = (total_alerts / normal_windows_count * 100) if normal_windows_count > 0 else 0
    
    # Calculate actual FPR (alerts during normal periods)
    alerts_during_normal = sum(1 for w in windows if w['is_alert'] and not w['is_attack_period'])
    actual_fpr = (alerts_during_normal / normal_windows_count * 100) if normal_windows_count > 0 else 0
    
    stats_text = f'Detection Performance Summary\n'
    stats_text += f'{"="*40}\n'
    stats_text += f'Total Windows: {total_windows}\n'
    stats_text += f'  Normal: {normal_windows_count} ({normal_windows_count/total_windows*100:.1f}%)\n'
    stats_text += f'  Attack: {attack_windows_count} ({attack_windows_count/total_windows*100:.1f}%)\n'
    stats_text += f'\nTotal Alerts: {total_alerts}\n'
    stats_text += f'Detection Rate: {detection_rate:.1f}%\n'
    stats_text += f'False Positive Rate: {actual_fpr:.2f}%\n'
    
    if times:
        duration = times[-1] - times[0]
        stats_text += f'\nDuration: {duration:.1f}s\n'
        stats_text += f'Alert Rate: {total_alerts/(duration/3600):.1f} alerts/hour'
    
    fig.text(0.98, 0.02, stats_text, fontsize=11, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved scientific visualization to {output_path}")
    print(f"\nStatistics:")
    print(f"  Normal windows: {normal_windows_count}")
    print(f"  Attack windows: {attack_windows_count}")
    print(f"  Total alerts: {total_alerts}")
    print(f"  Detection rate: {detection_rate:.1f}%")
    print(f"  False positive rate: {actual_fpr:.2f}%")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create scientific visualization of normal vs attack detection')
    parser.add_argument('--input', type=Path, help='Debug output file (or stdin)')
    parser.add_argument('--output', type=Path, default=Path('results/unsupervised/real_can0/normal_vs_attack_detection.png'),
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
    elif args.input and str(args.input) == '-':
        print("Reading from stdin... (paste debug output and press Ctrl+D)")
        text = sys.stdin.read()
    else:
        print("Error: No input provided. Use --input <file> or pipe input")
        return
    
    # Parse data
    data = parse_debug_output(text)
    
    if not data['windows']:
        print("No data found in input")
        return
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Create plot
    create_scientific_plot(data, thresholds, args.output)

if __name__ == '__main__':
    main()

