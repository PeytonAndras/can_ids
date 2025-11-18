# Rate-Based Detection Guide

## Overview

Rate-based detection complements the existing PCA and Isolation Forest detectors by detecting attacks based on message rate patterns and timing characteristics. This is particularly effective for:

- **Low-volume injection attacks** (like your cycling script)
- **Regular timing patterns** (suspiciously consistent intervals)
- **Rate deviations** (unexpected spikes or drops)
- **Timing anomalies** (too regular or too irregular)

## Configuration

Add to your `deployment/config.yaml`:

```yaml
rate_detection:
  enabled: true
  history_window_seconds: 30.0      # Track rates over 30 seconds
  min_samples: 10                    # Need 10 samples before alerting
  rate_deviation_threshold: 3.0      # Alert if rate deviates by 3 std devs
  rate_multiplier_threshold: 2.0     # Alert if rate > 2x mean
  rate_minimum_threshold: 0.1        # Alert if rate < 10% of mean (for low-volume attacks)
  regularity_threshold: 0.1          # Alert if CV < 0.1 (very regular timing)
  irregularity_threshold: 2.0         # Alert if CV > 2.0 (highly irregular)
  monitored_ids: []                  # Empty = monitor all IDs
  ignored_ids: []                    # IDs to ignore
```

## Detection Capabilities

### 1. Low-Volume Attack Detection
Your cycling script sends only 10 msg/s, which is much lower than normal traffic (80+ msg/s). The rate detector will alert when:
- Rate drops below 10% of normal (configurable via `rate_minimum_threshold`)
- Timing becomes suspiciously regular (CV < 0.1)

### 2. High-Rate Attack Detection
Detects flood attacks when:
- Rate exceeds 2x normal (configurable via `rate_multiplier_threshold`)
- Rate deviates by more than 3 standard deviations

### 3. Timing Pattern Detection
Detects injection patterns:
- **Too regular**: CV < 0.1 (like your 0.1s intervals)
- **Too irregular**: CV > 2.0 (bursty patterns)

## Usage

The rate detector is automatically integrated when enabled in config. It works alongside PCA/IF detectors using logical OR (if any detector flags an anomaly, alert is raised).

### Example Alert Output

```json
{
  "rate_detection": {
    "is_anomaly": true,
    "score": 0.9,
    "alerts": [
      "ID 0x039: suspiciously regular timing (CV=0.05 < 0.1 threshold)",
      "ID 0x062: suspiciously low rate (1.0 msg/s < 8.0 threshold, normal=80.0 msg/s)"
    ],
    "details": {
      "checked_ids": 17,
      "alert_count": 2
    }
  }
}
```

## Tuning for Your Attack

To better detect your cycling script attack, adjust:

```yaml
rate_detection:
  enabled: true
  rate_minimum_threshold: 0.15      # Alert if rate < 15% of normal
  regularity_threshold: 0.15        # Alert if CV < 0.15 (less strict)
  monitored_ids: ["0x062", "0x024", "0x039"]  # Focus on these IDs
```

This will catch:
- Low rates (your attack is 10 msg/s vs 80+ normal)
- Regular timing (your 0.1s intervals create CV ≈ 0.05)

## Testing

Run your attack script and monitor alerts:

```bash
# Terminal 1: Run the IDS
python scripts/deploy_realtime.py --can-channel can0

# Terminal 2: Run your attack script
./your_attack_script.sh
```

You should see alerts like:
```
ALERT {"rate_detection": {"is_anomaly": true, "alerts": ["ID 0x039: suspiciously regular timing..."]}}
```

