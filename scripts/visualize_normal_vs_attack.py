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
    
    # Set elegant, publication-quality style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10.5,
        'axes.labelsize': 11.5,
        'axes.titlesize': 13,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.linewidth': 0.9,
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.3,
        'axes.labelpad': 11,
        'xtick.major.pad': 7,
        'ytick.major.pad': 7,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
    })
    
    fig = plt.figure(figsize=(12.5, 8), facecolor='white', dpi=100)
    gs = fig.add_gridspec(3, 1, hspace=0.42, height_ratios=[1, 1, 1])
    
    # Beautiful, harmonious color palette
    color_line = '#1E3A5F'  # Rich navy blue for main line
    color_attack_bg = '#DC3545'  # Clean red for attack periods  
    color_threshold = '#6C757D'  # Soft gray for thresholds
    color_alert = '#FD7E14'  # Vibrant orange for alerts
    color_bg = '#FFFFFF'  # Pure white
    color_grid = '#F8F9FA'  # Very subtle gray for grid
    color_text = '#212529'  # Near black text
    
    # Identify attack periods
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
    
    # Plot 1: Isolation Forest Scores
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(color_bg)
    
    # Highlight attack periods with subtle, elegant shading - slightly more visible
    for start, end in attack_periods:
        ax1.axvspan(start, end, alpha=0.13, color=color_attack_bg, zorder=0, linewidth=0)
    
    # Plot main line - smooth and elegant with better visual quality
    ax1.plot(times, if_scores, color=color_line, linewidth=2.5, 
            alpha=0.85, zorder=4, label='Anomaly Score', antialiased=True, 
            solid_capstyle='round', solid_joinstyle='round')
    
    # Threshold line - refined dashed style
    ax1.axhline(y=thresholds['isolation_forest'], color=color_threshold, 
                linestyle='--', linewidth=1.9, dashes=(12, 6), 
                zorder=6, alpha=0.55, label=f'Threshold ({thresholds["isolation_forest"]:.4f})')
    
    ax1.set_ylabel('Isolation Forest\nAnomaly Score', fontsize=11.5, fontweight='500', 
                   color=color_text, labelpad=13)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.4, color=color_grid, zorder=1)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.97, shadow=False, 
              fancybox=False, edgecolor='#DDDDDD', facecolor='white', frameon=True,
              borderpad=0.7, handlelength=2.8, columnspacing=1.2)
    ax1.set_ylim([0, max(if_scores) * 1.2])
    
    # Refined spines
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax1.spines[spine].set_color('#CCCCCC')
        ax1.spines[spine].set_linewidth(0.9)
    
    # Elegant statistics box
    if normal_if and attack_if:
        normal_mean_if = np.mean(normal_if)
        attack_mean_if = np.mean(attack_if)
        stats_if = f'Normal: μ = {normal_mean_if:.3f}\nAttack: μ = {attack_mean_if:.3f}'
        ax1.text(0.02, 0.96, stats_if, transform=ax1.transAxes, 
                fontsize=8.5, verticalalignment='top', fontweight='400',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white', 
                         edgecolor='#DDDDDD', linewidth=1.0, alpha=0.9))
    
    # Plot 2: PCA Reconstruction Error
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(color_bg)
    
    # Highlight attack periods
    for start, end in attack_periods:
        ax2.axvspan(start, end, alpha=0.13, color=color_attack_bg, zorder=0, linewidth=0)
    
    # Plot main line - smooth and elegant
    ax2.plot(times, pca_scores, color=color_line, linewidth=2.5, 
            alpha=0.85, zorder=4, label='Reconstruction Error', antialiased=True,
            solid_capstyle='round', solid_joinstyle='round')
    
    # Threshold line
    ax2.axhline(y=thresholds['pca'], color=color_threshold, linestyle='--', 
                linewidth=1.9, dashes=(12, 6), zorder=6, alpha=0.55,
                label=f'Threshold ({thresholds["pca"]:.1f})')
    
    ax2.set_ylabel('PCA Reconstruction\nError', fontsize=11.5, fontweight='500', 
                   color=color_text, labelpad=13)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.4, color=color_grid, zorder=1)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.97, shadow=False, 
              fancybox=False, edgecolor='#DDDDDD', facecolor='white', frameon=True,
              borderpad=0.7, handlelength=2.8, columnspacing=1.2)
    ax2.set_ylim([0, max(pca_scores) * 1.2])
    
    # Refined spines
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax2.spines[spine].set_color('#CCCCCC')
        ax2.spines[spine].set_linewidth(0.9)
    
    # Elegant statistics box
    if normal_pca and attack_pca:
        normal_mean_pca = np.mean(normal_pca)
        attack_mean_pca = np.mean(attack_pca)
        stats_pca = f'Normal: μ = {normal_mean_pca:.2f}\nAttack: μ = {attack_mean_pca:.2f}'
        ax2.text(0.02, 0.96, stats_pca, transform=ax2.transAxes, 
                fontsize=8.5, verticalalignment='top', fontweight='400',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white', 
                         edgecolor='#DDDDDD', linewidth=1.0, alpha=0.9))
    
    # Plot 3: Frame Rate (0x039 messages per second)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(color_bg)
    
    # Highlight attack periods
    for start, end in attack_periods:
        ax3.axvspan(start, end, alpha=0.13, color=color_attack_bg, zorder=0, linewidth=0)
    
    # Plot main line - smooth and elegant
    ax3.plot(times, rates_039, color=color_line, linewidth=2.5, 
            alpha=0.85, zorder=4, label='Frame Rate', antialiased=True,
            solid_capstyle='round', solid_joinstyle='round')
    
    # Highlight attack threshold
    ax3.axhline(y=100, color=color_alert, linestyle=':', linewidth=1.9, 
                dashes=(8, 5), label='Attack Threshold (100 fps)', 
                alpha=0.55, zorder=6)
    
    ax3.set_xlabel('Time (seconds)', fontsize=11.5, fontweight='500', 
                   color=color_text, labelpad=11)
    ax3.set_ylabel('0x039 Frame Rate\n(frames/second)', fontsize=11.5, fontweight='500', 
                   color=color_text, labelpad=13)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.4, color=color_grid, zorder=1)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.97, shadow=False, 
              fancybox=False, edgecolor='#DDDDDD', facecolor='white', frameon=True,
              borderpad=0.7, handlelength=2.8, columnspacing=1.2)
    
    # Refined spines
    for spine in ['top', 'right']:
        ax3.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax3.spines[spine].set_color('#CCCCCC')
        ax3.spines[spine].set_linewidth(0.9)
    
    # Elegant statistics box
    if normal_rates and attack_rates:
        normal_mean_rate = np.mean(normal_rates)
        attack_mean_rate = np.mean(attack_rates)
        stats_rate = f'Normal: μ = {normal_mean_rate:.1f} fps\nAttack: μ = {attack_mean_rate:.1f} fps'
        ax3.text(0.02, 0.96, stats_rate, transform=ax3.transAxes, 
                fontsize=8.5, verticalalignment='top', fontweight='400',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white', 
                         edgecolor='#DDDDDD', linewidth=1.0, alpha=0.9))
    
    # Add overall statistics
    total_windows = len(windows)
    total_alerts = len(alerts)
    normal_windows_count = len(normal_windows)
    attack_windows_count = len(attack_windows)
    detection_rate = (total_alerts / attack_windows_count * 100) if attack_windows_count > 0 else 0
    
    # Calculate actual FPR (alerts during normal periods)
    alerts_during_normal = sum(1 for w in windows if w['is_alert'] and not w['is_attack_period'])
    actual_fpr = (alerts_during_normal / normal_windows_count * 100) if normal_windows_count > 0 else 0
    
    # Add refined main title
    fig.suptitle('Real-Time CAN Intrusion Detection System', 
                fontsize=14, fontweight='500', color=color_text, y=0.992)
    
    # Add refined performance summary
    stats_text = f'Performance Summary\n'
    stats_text += f'{"─"*19}\n'
    stats_text += f'Total Windows: {total_windows}\n'
    stats_text += f'  Normal: {normal_windows_count} ({normal_windows_count/total_windows*100:.1f}%)\n'
    stats_text += f'  Attack: {attack_windows_count} ({attack_windows_count/total_windows*100:.1f}%)\n'
    stats_text += f'\nAlerts: {total_alerts}\n'
    stats_text += f'Detection Rate: {detection_rate:.1f}%\n'
    stats_text += f'False Positive Rate: {actual_fpr:.2f}%\n'
    
    if times:
        duration = times[-1] - times[0]
        stats_text += f'\nDuration: {duration:.1f}s'
    
    fig.text(0.985, 0.02, stats_text, fontsize=9, fontweight='400',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.55', facecolor='#FAFAFA', 
                      edgecolor='#DDDDDD', linewidth=1.0, alpha=0.93),
             color=color_text)
    
    plt.tight_layout(rect=[0, 0.025, 1, 0.985])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.1, format='png')
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

