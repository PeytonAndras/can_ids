# Live Deployment Guide

## Quick Start

### 1. Start the IDS

```bash
python3 scripts/deploy_realtime.py --can-channel can0
```

**Expected output:**
```
2025-11-16 10:00:00 INFO Listening on CAN interface can0 (socketcan)
2025-11-16 10:00:00 INFO Rate-based detection enabled
```

If you don't see "Rate-based detection enabled", check:
- `deployment/config.yaml` has `rate_detection.enabled: true`
- `scripts/rate_detector.py` exists and is importable

### 2. Verify Rate Detection is Active

Use the debug script to see rate detection in action:

```bash
python3 scripts/deploy_realtime_debug.py
```

Look for:
- `Rate detection: ENABLED` in the header
- Rate scores in each window output
- Rate alerts when anomalies detected

### 3. Run Your Attack

In another terminal:

```bash
# Your cycling attack script
./your_attack_script.sh
```

Or manually inject:

```bash
for i in {1..30}; do
  cansend can0 062#07B63A0BF6623BCF && sleep 0.1
  cansend can0 024#006C3A0D9C4F913B && sleep 0.1
  cansend can0 039#00003A0DD87D5C7A && sleep 0.1
done
```

## What You Should See

### Normal Traffic (No Alerts)

```
[   2.0s] Window   20   OK | IF=0.50 (thresh=0.68) ✗ | PCA=6.5 (thresh=20.0) ✗ | Rate=0.00 ✗ | Frames=  80 IDs=17
```

### Attack Detected

```
[  30.5s] Window  305 🚨 ALERT | IF=0.55 (thresh=0.68) ✗ | PCA=7.2 (thresh=20.0) ✗ | Rate=0.85 ✓ | Frames=  82 IDs=17
  📊 ID 0x039: suspiciously regular timing (CV=0.05 < 0.1 threshold)
  📊 ID 0x062: suspiciously low rate (2.0 msg/s < 8.0 threshold)
  ⚠️  ALERT #1
```

## Troubleshooting

### Rate Detection Not Enabled

**Symptom**: No "Rate-based detection enabled" message

**Fix**:
1. Check config: `grep "enabled: true" deployment/config.yaml | grep -A 1 rate_detection`
2. Verify module: `python3 -c "from scripts.rate_detector import RateDetector; print('OK')"`
3. Restart IDS

### No Alerts During Attack

**Possible causes**:

1. **Not enough baseline data**
   - Wait 30+ seconds before starting attack
   - Rate detector needs time to learn normal patterns

2. **Thresholds too strict**
   ```yaml
   rate_detection:
     regularity_threshold: 0.15    # Less strict
     rate_minimum_threshold: 0.15  # Less strict
     min_samples: 5                # Faster detection
   ```

3. **Attack too subtle**
   - Your cycling attack (10 msg/s) might be drowned out by normal traffic (80+ msg/s)
   - Try monitoring specific IDs:
   ```yaml
   monitored_ids: ["0x062", "0x024", "0x039"]
   ```

### Too Many False Positives

**Fix**:
```yaml
rate_detection:
  regularity_threshold: 0.05      # More strict
  rate_deviation_threshold: 4.0  # More strict
  min_samples: 20                # Require more data
```

## Monitoring in Production

### View Alerts Live

```bash
# Terminal 1: Run IDS
python3 scripts/deploy_realtime.py --can-channel can0 2>&1 | tee deployment.log

# Terminal 2: Watch for rate alerts
tail -f deployment.log | grep -i "rate\|alert"
```

### Check Alert Logs

```bash
# View all alerts
cat logs/alerts.jsonl | jq 'select(.rate_detection.is_anomaly == true)'

# Count rate detection alerts
cat logs/alerts.jsonl | jq 'select(.rate_detection.is_anomaly == true)' | wc -l
```

### Debug Mode

For detailed debugging:

```bash
python3 scripts/deploy_realtime_debug.py
```

Shows:
- All window scores (not just alerts)
- Rate detection scores
- Rate alerts
- Detailed frame statistics

## Configuration Reference

Full rate detection config options:

```yaml
rate_detection:
  enabled: true                    # Enable/disable rate detection
  history_window_seconds: 30.0     # How long to track rates
  min_samples: 10                  # Minimum samples before alerting
  rate_deviation_threshold: 3.0    # Alert if rate deviates by N std devs
  rate_multiplier_threshold: 2.0   # Alert if rate > N× mean
  rate_minimum_threshold: 0.1      # Alert if rate < N× mean (low-volume attacks)
  regularity_threshold: 0.1        # Alert if CV < N (regular timing)
  irregularity_threshold: 2.0      # Alert if CV > N (irregular timing)
  monitored_ids: []                # Empty = monitor all IDs
  ignored_ids: []                  # IDs to ignore
```

## Expected Behavior

### Timeline

- **0-10s**: IDS starts, rate detector initializes
- **10-30s**: Baseline building (normal traffic patterns learned)
- **30s+**: Attack starts
- **30-35s**: Rate detection alerts should appear

### Alert Types

1. **Regular Timing Alert**: `"suspiciously regular timing (CV=0.05 < 0.1 threshold)"`
   - Detects your 0.1s intervals

2. **Low Rate Alert**: `"suspiciously low rate (2.0 msg/s < 8.0 threshold)"`
   - Detects when rate drops significantly

3. **High Rate Alert**: `"high rate (160.0 msg/s > 160.0 threshold)"`
   - Detects flood attacks

4. **Rate Deviation Alert**: `"rate anomaly (current=150.0, mean=80.0±10.0, z=7.0)"`
   - Detects statistical anomalies

## Integration with Existing System

Rate detection works seamlessly:

- ✅ Uses same config file
- ✅ Works with PCA/IF models
- ✅ Outputs to same alert log
- ✅ No changes to existing models needed
- ✅ Logical OR ensemble (any detector can trigger alert)

Just enable it and restart!
