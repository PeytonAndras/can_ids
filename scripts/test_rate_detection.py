#!/usr/bin/env python3
"""
Test script for rate-based detection.

This script helps test the rate detector by:
1. Simulating your cycling attack pattern
2. Checking if alerts are generated
3. Verifying detection works correctly
"""

import sys
import time
import subprocess
import signal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.rate_detector import RateDetector, RateDetectorConfig
except ImportError:
    print("Error: Could not import rate_detector module")
    sys.exit(1)


def test_rate_detector_directly():
    """Test the rate detector directly without CAN interface"""
    print("=" * 60)
    print("Direct Rate Detector Test")
    print("=" * 60)
    
    # Configure for your attack pattern
    config = RateDetectorConfig(
        history_window_seconds=10.0,
        min_samples=5,
        rate_minimum_threshold=0.1,
        regularity_threshold=0.15,
        monitored_ids=[0x062, 0x024, 0x039]
    )
    
    detector = RateDetector(config)
    
    # Simulate normal traffic first (build up baseline)
    print("\n[Phase 1] Building baseline with normal traffic...")
    base_time = time.time()
    
    # Add normal traffic over 5 seconds to build statistics
    for second in range(5):
        for i in range(80):  # 80 msg/s
            detector.add_frame(0x039, base_time + second + i * 0.0125)
        # Update rates periodically
        detector.check_anomalies(base_time + second + 1.0, 0.1)
    
    # Check normal traffic (should not alert after baseline is built)
    result = detector.check_anomalies(base_time + 5.0, 0.1)
    print(f"  Normal traffic check: Anomaly={result['is_anomaly']}, Score={result['score']:.3f}")
    if result['alerts']:
        print(f"  ⚠️  Unexpected alerts: {result['alerts']}")
    
    # Simulate your cycling attack (0.1s intervals, 10 msg/s)
    print("\n[Phase 2] Simulating cycling attack (0.1s intervals)...")
    attack_start = base_time + 6.0
    
    # Add attack traffic
    for i in range(30):
        can_id = [0x062, 0x024, 0x039, 0x062, 0x024][i % 5]
        detector.add_frame(can_id, attack_start + i * 0.1)
    
    # Check after attack (give it time to detect)
    result = detector.check_anomalies(attack_start + 3.0, 0.1)
    print(f"  Attack check: Anomaly={result['is_anomaly']}, Score={result['score']:.3f}")
    if result['alerts']:
        print("  Alerts:")
        for alert in result['alerts']:
            print(f"    - {alert}")
    else:
        print("  ⚠️  No alerts generated - may need tuning")
    
    print("\n" + "=" * 60)
    return result['is_anomaly']


def test_with_can_interface():
    """Test with actual CAN interface"""
    print("=" * 60)
    print("CAN Interface Test")
    print("=" * 60)
    print("\nThis will:")
    print("1. Start the IDS in the background")
    print("2. Inject your cycling attack pattern")
    print("3. Check for alerts")
    print("\nMake sure:")
    print("- can0 interface is available")
    print("- You have permission to send CAN messages")
    print("- Rate detection is enabled in deployment/config.yaml")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        return
    
    # Start IDS
    print("\n[1] Starting IDS...")
    ids_process = subprocess.Popen(
        ["python3", "scripts/deploy_realtime.py", "--can-channel", "can0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    time.sleep(2)  # Let IDS initialize
    
    # Inject attack pattern
    print("[2] Injecting cycling attack pattern...")
    attack_messages = [
        "062#07B63A0BF6623BCF",
        "024#006C3A0D9C4F913B",
        "039#00003A0DD87D5C7A",
        "062#07B33A0CBE359457",
        "024#00693A0ECDDC53A4",
    ]
    
    try:
        for cycle in range(10):  # 10 cycles = 5 seconds
            for msg in attack_messages:
                subprocess.run(["cansend", "can0", msg], check=False)
                time.sleep(0.1)
        
        time.sleep(2)  # Let IDS process
        
        # Check output
        print("[3] Checking IDS output...")
        ids_process.send_signal(signal.SIGINT)
        stdout, _ = ids_process.communicate(timeout=5)
        
        if "rate_detection" in stdout or "ALERT" in stdout:
            print("\n✓ Alerts detected!")
            print("\nRelevant output:")
            for line in stdout.split('\n'):
                if "rate_detection" in line or "ALERT" in line:
                    print(f"  {line}")
        else:
            print("\n⚠️  No alerts found in output")
            print("Check deployment/config.yaml rate_detection settings")
            
    except subprocess.TimeoutExpired:
        ids_process.kill()
        print("\n⚠️  IDS process timed out")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        ids_process.kill()


def main():
    print("Rate-Based Detection Test Suite")
    print("=" * 60)
    print("\nChoose test mode:")
    print("1. Direct detector test (no CAN interface needed)")
    print("2. Full CAN interface test (requires can0)")
    print()
    
    choice = input("Choice (1-2): ").strip()
    
    if choice == "1":
        test_rate_detector_directly()
    elif choice == "2":
        test_with_can_interface()
    else:
        print("Invalid choice")
        sys.exit(1)


if __name__ == "__main__":
    main()

