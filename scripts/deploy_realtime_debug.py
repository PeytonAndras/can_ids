#!/usr/bin/env python3
"""Debug version that shows all window scores, not just alerts"""

import sys
import signal
from pathlib import Path

# Add parent directory to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.deploy_realtime import (
    load_config, ModelEnsemble, WindowAggregator,
    CanListenerWorker, configure_logging, DeploymentConfig
)
try:
    from scripts.rate_detector import RateDetector, RateDetectorConfig
    RATE_DETECTOR_AVAILABLE = True
except ImportError:
    RateDetector = None
    RateDetectorConfig = None
    RATE_DETECTOR_AVAILABLE = False
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
    
    # Initialize rate detector if configured
    rate_detector = None
    if config.rate_detection and config.rate_detection.get("enabled", False):
        if RATE_DETECTOR_AVAILABLE and RateDetector is not None:
            rate_config_dict = config.rate_detection
            rate_config = RateDetectorConfig(
                history_window_seconds=float(rate_config_dict.get("history_window_seconds", 30.0)),
                min_samples=int(rate_config_dict.get("min_samples", 10)),
                rate_deviation_threshold=float(rate_config_dict.get("rate_deviation_threshold", 3.0)),
                rate_multiplier_threshold=float(rate_config_dict.get("rate_multiplier_threshold", 2.0)),
                rate_minimum_threshold=float(rate_config_dict.get("rate_minimum_threshold", 0.1)),
                regularity_threshold=float(rate_config_dict.get("regularity_threshold", 0.1)),
                irregularity_threshold=float(rate_config_dict.get("irregularity_threshold", 2.0)),
                monitored_ids=[int(x, 16) if isinstance(x, str) else int(x) 
                              for x in rate_config_dict.get("monitored_ids", [])],
                ignored_ids=[int(x, 16) if isinstance(x, str) else int(x) 
                            for x in rate_config_dict.get("ignored_ids", [])],
            )
            rate_detector = RateDetector(rate_config)
            print("✓ Rate-based detection enabled")
        else:
            print("⚠️  Rate detection enabled in config but module not available")
    
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
    if rate_detector:
        print(f"Rate detection: ENABLED")
    else:
        print(f"Rate detection: DISABLED")
    print("="*80)
    print()
    
    try:
        for frame in can_worker.frames():
            if stop_requested:
                break
            
            # Feed frame to rate detector if enabled
            if rate_detector is not None:
                rate_detector.add_frame(frame.arbitration_id, frame.timestamp)
            
            for features in aggregator.add_frame(frame):
                window_count += 1
                scores = ensemble.score(features)
                is_anomaly = ensemble.is_anomaly(scores)
                
                # Check rate-based anomalies
                rate_anomaly = False
                rate_score = 0.0
                rate_alerts = []
                if rate_detector is not None:
                    window_duration = config.window_ms / 1000.0
                    current_time = float(features.get("end_time", time.time()))
                    rate_result = rate_detector.check_anomalies(current_time, window_duration)
                    rate_anomaly = rate_result.get("is_anomaly", False)
                    rate_score = rate_result.get("score", 0.0)
                    rate_alerts = rate_result.get("alerts", [])
                    rate_detector.reset_window(current_time)
                
                # Combine anomalies
                combined_anomaly = is_anomaly or rate_anomaly
                
                # Show every window
                elapsed = time.time() - start_time
                status = "🚨 ALERT" if combined_anomaly else "  OK"
                
                if_score = scores.get('isolation_forest', 0)
                if_thresh = ensemble.thresholds.get('isolation_forest', 0)
                pca_score = scores.get('pca', 0)
                pca_thresh = ensemble.thresholds.get('pca', 0)
                
                # Calculate 0x039 rate from features
                total_frames = int(features.get('total_frames', 0))
                unique_ids = int(features.get('unique_ids', 0))
                
                rate_info = ""
                if rate_detector is not None:
                    rate_info = f" | Rate={rate_score:.2f} {'✓' if rate_anomaly else '✗'}"
                
                print(f"[{elapsed:6.1f}s] Window {window_count:4d} {status} | "
                      f"IF={if_score:.4f} (thresh={if_thresh:.4f}) {'✓' if if_score >= if_thresh else '✗'} | "
                      f"PCA={pca_score:.4f} (thresh={pca_thresh:.4f}) {'✓' if pca_score >= pca_thresh else '✗'}"
                      f"{rate_info} | "
                      f"Frames={total_frames:4d} IDs={unique_ids:2d}")
                
                if rate_alerts:
                    for alert in rate_alerts:
                        print(f"  📊 {alert}")
                
                if combined_anomaly:
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

