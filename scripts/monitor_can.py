#!/usr/bin/env python3
"""Monitor CAN traffic to verify attack is happening"""

import sys
import time
from collections import Counter
from datetime import datetime

try:
    import can
except ImportError:
    print("ERROR: python-can not installed. Install with: pip install python-can")
    sys.exit(1)

def monitor_can(channel='can0', duration=10):
    """Monitor CAN traffic and count frames by ID"""
    bus = can.interface.Bus(channel=channel, interface='socketcan')
    
    print(f"Monitoring {channel} for {duration} seconds...")
    print("Press Ctrl+C to stop early\n")
    
    frame_counts = Counter()
    attack_id = 0x039
    attack_frames = 0
    total_frames = 0
    start_time = time.time()
    last_report = start_time
    
    try:
        while time.time() - start_time < duration:
            msg = bus.recv(timeout=0.1)
            if msg is None:
                continue
                
            total_frames += 1
            frame_counts[msg.arbitration_id] += 1
            
            if msg.arbitration_id == attack_id:
                attack_frames += 1
                print(f"[ATTACK FRAME] ID=0x{attack_id:03X} Data={msg.data.hex()} DLC={msg.dlc}")
            
            # Report every second
            if time.time() - last_report >= 1.0:
                elapsed = time.time() - start_time
                fps = total_frames / elapsed if elapsed > 0 else 0
                attack_fps = attack_frames / elapsed if elapsed > 0 else 0
                print(f"\n[{elapsed:.1f}s] Total: {total_frames} frames ({fps:.1f} fps) | "
                      f"0x{attack_id:03X}: {attack_frames} frames ({attack_fps:.1f} fps)")
                last_report = time.time()
                
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        bus.shutdown()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SUMMARY ({elapsed:.1f} seconds)")
    print(f"{'='*60}")
    print(f"Total frames: {total_frames}")
    print(f"Total 0x{attack_id:03X} frames: {attack_frames}")
    print(f"Average rate: {total_frames/elapsed:.1f} fps")
    print(f"Average 0x{attack_id:03X} rate: {attack_frames/elapsed:.1f} fps")
    print(f"\nTop 10 CAN IDs:")
    for can_id, count in frame_counts.most_common(10):
        pct = (count / total_frames * 100) if total_frames > 0 else 0
        print(f"  0x{can_id:03X}: {count:6d} frames ({pct:5.1f}%)")
    
    if attack_frames == 0:
        print(f"\n⚠️  WARNING: No 0x{attack_id:03X} frames detected!")
        print("   This suggests the attack is not running or not on this interface.")
    elif attack_frames < 10:
        print(f"\n⚠️  WARNING: Very few 0x{attack_id:03X} frames detected ({attack_frames})")
        print("   Normal traffic has ~10 fps of 0x039. Attack should flood much faster.")
    else:
        attack_rate = attack_frames / elapsed
        if attack_rate > 100:
            print(f"\n✓ High rate of 0x{attack_id:03X} frames detected ({attack_rate:.1f} fps)")
            print("   This looks like an attack!")
        else:
            print(f"\n⚠️  Moderate rate of 0x{attack_id:03X} frames ({attack_rate:.1f} fps)")
            print("   Attack should be flooding much faster (>100 fps)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Monitor CAN traffic')
    parser.add_argument('--channel', default='can0', help='CAN interface (default: can0)')
    parser.add_argument('--duration', type=int, default=10, help='Duration in seconds (default: 10)')
    args = parser.parse_args()
    
    monitor_can(args.channel, args.duration)

