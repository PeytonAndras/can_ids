#!/usr/bin/env python3
"""Debug version that shows all window scores, not just alerts"""

import sys
import signal
from pathlib import Path

# Add parent directory to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.deploy_realtime import (
    load_config, ModelEnsemble, WindowAggregator,
    CanListenerWorker, configure_logging
)
import logging
import json
import time

stop_requested = False

def handle_signal(signum, frame):
    global stop_requested
    logging.info("Received signal %s; shutting down gracefully...", signum)
    stop_requested = True

def main():
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    config = load_config(Path("deployment/config.yaml"))
    configure_logging("INFO", None)
    
    ensemble = ModelEnsemble(config)
    aggregator = WindowAggregator(config.window_ms)
    
    # Use live CAN
    can_worker = CanListenerWorker(channel="can0", bustype="socketcan")
    
    window_count = 0
    alert_count = 0
    start_time = time.time()
    
    print("="*80)
    print("DEBUG MODE: Showing all window scores")
    print("="*80)
    print(f"Thresholds: {ensemble.thresholds}")
    print(f"Ensemble strategy: {config.ensemble}")
    print(f"Window size: {config.window_ms}ms")
    print("="*80)
    print()
    
    try:
        for frame in can_worker.frames():
            if stop_requested:
                break
            for features in aggregator.add_frame(frame):
                window_count += 1
                scores = ensemble.score(features)
                is_anomaly = ensemble.is_anomaly(scores)
                
                # Show every window
                elapsed = time.time() - start_time
                status = "🚨 ALERT" if is_anomaly else "  OK"
                
                if_score = scores.get('isolation_forest', 0)
                if_thresh = ensemble.thresholds.get('isolation_forest', 0)
                pca_score = scores.get('pca', 0)
                pca_thresh = ensemble.thresholds.get('pca', 0)
                
                print(f"[{elapsed:6.1f}s] Window {window_count:4d} {status} | "
                      f"IF={if_score:.4f} (thresh={if_thresh:.4f}) {'✓' if if_score >= if_thresh else '✗'} | "
                      f"PCA={pca_score:.4f} (thresh={pca_thresh:.4f}) {'✓' if pca_score >= pca_thresh else '✗'} | "
                      f"Frames={int(features.get('total_frames', 0)):4d} "
                      f"IDs={int(features.get('unique_ids', 0)):2d} "
                      f"0x039_rate={int(features.get('total_frames', 0) * (features.get('unique_ids', 1) / max(features.get('unique_ids', 1), 1))):4d}/s")
                
                if is_anomaly:
                    alert_count += 1
                    print(f"  ⚠️  ALERT #{alert_count}")
                
                # Stop after 100 windows or user interrupt
                if window_count >= 100:
                    print("\n(Stopping after 100 windows - use Ctrl+C to stop earlier)")
                    break
                    
    except KeyboardInterrupt:
        pass
    
    elapsed = time.time() - start_time
    print()
    print("="*80)
    print(f"Summary: {window_count} windows processed in {elapsed:.1f}s, {alert_count} alerts")
    print(f"Alert rate: {alert_count / (elapsed/3600):.1f} alerts/hour")
    print("="*80)

if __name__ == '__main__':
    main()

