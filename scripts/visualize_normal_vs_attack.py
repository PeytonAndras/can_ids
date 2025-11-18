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
    
    # Extract data - sorted by time
    windows_sorted = sorted(windows, key=lambda x: x['time'])
    times = [w['time'] for w in windows_sorted]
    if_scores = [w['if_score'] for w in windows_sorted]
    pca_scores = [w['pca_score'] for w in windows_sorted]
    rates_039 = [w['rate_039'] for w in windows_sorted]
    is_attack = [w['is_attack_period'] for w in windows_sorted]
    
    # Normal data
    normal_windows_sorted = sorted(normal_windows, key=lambda x: x['time'])
    normal_times = [w['time'] for w in normal_windows_sorted]
    normal_if = [w['if_score'] for w in normal_windows_sorted]
    normal_pca = [w['pca_score'] for w in normal_windows_sorted]
    normal_rates = [w['rate_039'] for w in normal_windows_sorted]
    
    # Attack data
    attack_windows_sorted = sorted(attack_windows, key=lambda x: x['time'])
    attack_times = [w['time'] for w in attack_windows_sorted]
    attack_if = [w['if_score'] for w in attack_windows_sorted]
    attack_pca = [w['pca_score'] for w in attack_windows_sorted]
    attack_rates = [w['rate_039'] for w in attack_windows_sorted]
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.25, height_ratios=[1, 1, 0.9])
    
    # Define colors
    color_normal = '#2E86AB'  # Professional blue
    color_attack = '#A23B72'  # Professional red/purple
    color_threshold = '#333333'  # Dark gray
    color_alert = '#F18F01'  # Orange
    color_bg = '#F8F9FA'  # Light gray background
    
    # Plot 1: Isolation Forest Scores
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(color_bg)
    
    # Plot as one consecutive line through all time points
    ax1.plot(times, if_scores, color='#34495E', linewidth=2.5, 
            alpha=0.9, zorder=2, label='Anomaly Score')
    
    # Highlight attack periods with background shading
    attack_periods = []
    in_attack = False
    attack_start = None
    for w in windows_sorted:
        if w['is_attack_period']:
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
        ax1.axvspan(start, end, alpha=0.2, color=color_attack, 
                   label='Attack Period' if start == attack_periods[0][0] else '', zorder=1)
    
    # Threshold line
    ax1.axhline(y=thresholds['isolation_forest'], color=color_threshold, 
                linestyle='--', linewidth=2.5, label=f'Threshold ({thresholds["isolation_forest"]:.4f})', 
                zorder=10, alpha=0.8)
    
    # Mark alerts
    alert_times_if = [w['time'] for w in windows_sorted if w['is_alert']]
    alert_if_scores = [w['if_score'] for w in windows_sorted if w['is_alert']]
    if alert_times_if:
        ax1.scatter(alert_times_if, alert_if_scores, color=color_alert, s=120, 
                   marker='*', zorder=15, label='Alerts', edgecolors='white', linewidths=1.5)
    
    ax1.set_ylabel('Isolation Forest\nAnomaly Score', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_title('Real-Time CAN Intrusion Detection: Normal vs Attack Traffic', 
                 fontsize=16, fontweight='bold', pad=25, color='#2C3E50')
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='white')
    # Combine legend items
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#34495E', linewidth=2.5, label='Anomaly Score'),
        Line2D([0], [0], color=color_threshold, linestyle='--', linewidth=2.5, alpha=0.8, label=f'Threshold ({thresholds["isolation_forest"]:.4f})'),
        mpatches.Patch(facecolor=color_attack, alpha=0.2, label='Attack Period'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=color_alert, 
               markersize=12, markeredgecolor='darkorange', markeredgewidth=1.5, label='Alerts', linestyle='None')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95, shadow=True, 
              fancybox=True, edgecolor='gray')
    ax1.set_ylim([0, max(if_scores) * 1.12])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#BDC3C7')
    ax1.spines['bottom'].set_color('#BDC3C7')
    
    # Add statistics box for IF
    if normal_if and attack_if:
        normal_mean_if = np.mean(normal_if)
        attack_mean_if = np.mean(attack_if)
        stats_if = f'Normal: μ={normal_mean_if:.3f}\nAttack: μ={attack_mean_if:.3f}'
        ax1.text(0.02, 0.98, stats_if, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=color_normal, linewidth=2, alpha=0.9))
    
    # Plot 2: PCA Reconstruction Error
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(color_bg)
    
    # Plot as one consecutive line through all time points
    ax2.plot(times, pca_scores, color='#34495E', linewidth=2.5, 
            alpha=0.9, zorder=2, label='Reconstruction Error')
    
    # Highlight attack periods with background shading
    for start, end in attack_periods:
        ax2.axvspan(start, end, alpha=0.2, color=color_attack, zorder=1)
    
    # Threshold line
    ax2.axhline(y=thresholds['pca'], color=color_threshold, linestyle='--', 
                linewidth=2.5, label=f'Threshold ({thresholds["pca"]:.1f})', 
                zorder=10, alpha=0.8)
    
    # Mark alerts
    alert_times_pca = [w['time'] for w in windows_sorted if w['is_alert']]
    alert_pca_scores = [w['pca_score'] for w in windows_sorted if w['is_alert']]
    if alert_times_pca:
        ax2.scatter(alert_times_pca, alert_pca_scores, color=color_alert, s=120, 
                   marker='*', zorder=15, label='Alerts', edgecolors='white', linewidths=1.5)
    
    ax2.set_ylabel('PCA Reconstruction\nError', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='white')
    legend_elements2 = [
        Line2D([0], [0], color='#34495E', linewidth=2.5, label='Reconstruction Error'),
        Line2D([0], [0], color=color_threshold, linestyle='--', linewidth=2.5, alpha=0.8, label=f'Threshold ({thresholds["pca"]:.1f})'),
        mpatches.Patch(facecolor=color_attack, alpha=0.2, label='Attack Period'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=color_alert, 
               markersize=12, markeredgecolor='darkorange', markeredgewidth=1.5, label='Alerts', linestyle='None')
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=11, framealpha=0.95, shadow=True, 
              fancybox=True, edgecolor='gray')
    ax2.set_ylim([0, max(pca_scores) * 1.12])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#BDC3C7')
    ax2.spines['bottom'].set_color('#BDC3C7')
    
    # Add statistics box for PCA
    if normal_pca and attack_pca:
        normal_mean_pca = np.mean(normal_pca)
        attack_mean_pca = np.mean(attack_pca)
        stats_pca = f'Normal: μ={normal_mean_pca:.2f}\nAttack: μ={attack_mean_pca:.2f}'
        ax2.text(0.02, 0.98, stats_pca, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=color_attack, linewidth=2, alpha=0.9))
    
    # Plot 3: Frame Rate (0x039 messages per second)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(color_bg)
    
    # Plot as one consecutive line through all time points
    ax3.plot(times, rates_039, color='#34495E', linewidth=2.5, 
            alpha=0.9, zorder=2, label='Frame Rate')
    
    # Highlight attack periods with background shading
    for start, end in attack_periods:
        ax3.axvspan(start, end, alpha=0.2, color=color_attack, zorder=1)
    
    # Highlight attack threshold (100 fps)
    ax3.axhline(y=100, color=color_alert, linestyle=':', linewidth=2.5, 
                label='Attack Threshold (100 fps)', alpha=0.8, zorder=5)
    
    ax3.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold', color='#2C3E50')
    ax3.set_ylabel('0x039 Frame Rate\n(frames/second)', fontsize=14, fontweight='bold', color='#2C3E50')
    ax3.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='white')
    legend_elements3 = [
        Line2D([0], [0], color='#34495E', linewidth=2.5, label='Frame Rate'),
        Line2D([0], [0], color=color_alert, linestyle=':', linewidth=2.5, alpha=0.8, label='Attack Threshold (100 fps)'),
        mpatches.Patch(facecolor=color_attack, alpha=0.2, label='Attack Period')
    ]
    ax3.legend(handles=legend_elements3, loc='upper right', fontsize=11, framealpha=0.95, shadow=True, 
              fancybox=True, edgecolor='gray')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#BDC3C7')
    ax3.spines['bottom'].set_color('#BDC3C7')
    
    # Add statistics box for rates
    if normal_rates and attack_rates:
        normal_mean_rate = np.mean(normal_rates)
        attack_mean_rate = np.mean(attack_rates)
        stats_rate = f'Normal: μ={normal_mean_rate:.1f} fps\nAttack: μ={attack_mean_rate:.1f} fps'
        ax3.text(0.02, 0.98, stats_rate, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=color_alert, linewidth=2, alpha=0.9))
    
    # Add overall statistics
    total_windows = len(windows)
    total_alerts = len(alerts)
    normal_windows_count = len(normal_windows)
    attack_windows_count = len(attack_windows)
    detection_rate = (total_alerts / attack_windows_count * 100) if attack_windows_count > 0 else 0
    
    # Calculate actual FPR (alerts during normal periods)
    alerts_during_normal = sum(1 for w in windows if w['is_alert'] and not w['is_attack_period'])
    actual_fpr = (alerts_during_normal / normal_windows_count * 100) if normal_windows_count > 0 else 0
    
    stats_text = f'Detection Performance\n'
    stats_text += f'{"─"*25}\n'
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
    
    fig.text(0.98, 0.02, stats_text, fontsize=11, fontweight='bold',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                      edgecolor='#34495E', linewidth=2, alpha=0.95))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.2)
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

