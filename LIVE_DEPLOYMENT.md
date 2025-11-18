# Live Deployment Testing Guide

## Quick Start

### 1. **Live Monitoring (Normal Traffic)**
Run the detector on the live `can0` interface:

```bash
cd /Users/peytonandras/Projects/Research/can_ids
python scripts/deploy_realtime.py --config deployment/config.yaml --can-channel can0
```

**What to expect:**
- The detector will start listening on `can0`
- Alerts will be logged to both stdout and `deployment/logs/alerts.jsonl`
- With current thresholds (99.5th percentile), you should see very few false positives (~2 per hour based on simulation)

**To stop:** Press `Ctrl+C`

### 2. **Monitor Alert Logs in Real-Time**
In a separate terminal, watch the alert log:

```bash
tail -f deployment/logs/alerts.jsonl | jq .
```

Or without `jq`:
```bash
tail -f deployment/logs/alerts.jsonl
```

### 3. **Testing with Attack Traffic**

**Option A: Capture Attack Data First, Then Replay**
1. Start capturing normal traffic:
   ```bash
   candump -L can0 > data/raw/can0/attack_test.log
   ```
2. In another terminal, run your attack script
3. Stop capturing after a few seconds (`Ctrl+C`)
4. Convert to CSV and test:
   ```bash
   # Convert log to CSV (if needed)
   python scripts/feature_extractor_can_ids.py data/raw/can0/attack_test.log data/raw/can0/attack_test.csv
   
   # Replay through detector
   python scripts/deploy_realtime.py --config deployment/config.yaml --replay data/raw/can0/attack_test.csv
   ```

**Option B: Live Detection During Attack**
1. Start the detector in one terminal:
   ```bash
   python scripts/deploy_realtime.py --config deployment/config.yaml --can-channel can0
   ```
2. In another terminal, run your attack script
3. Watch for alerts in real-time

## Current Configuration

- **Models:** Isolation Forest (trained on real `can0` data)
- **Threshold:** 99.5th percentile (0.6764)
- **Window Size:** 100ms
- **Smoothing:** 1 consecutive window (immediate detection)
- **Expected Performance:**
  - False Positives: ~2 per hour on normal traffic
  - Detection: Should detect accelerator flood attacks immediately

## Monitoring Commands

**Check recent alerts:**
```bash
tail -20 deployment/logs/alerts.jsonl | jq .
```

**Count total alerts:**
```bash
wc -l deployment/logs/alerts.jsonl
```

**View alert details:**
```bash
cat deployment/logs/alerts.jsonl | jq '.scores, .thresholds, .metrics'
```

## Troubleshooting

**If you see too many false positives:**
- Increase the threshold percentile (retrain with `--percentile=99.9`)
- Increase smoothing windows in config: `consecutive_windows: 2`

**If attacks aren't detected:**
- Verify attack is actually sending `0x039` frames
- Check that attack traffic is on `can0` (not `vcan0`)
- Lower threshold percentile (retrain with `--percentile=99.0`)

**If no frames are received:**
```bash
# Check CAN interface status
ip link show can0

# Check if interface is up
candump can0 -n 10
```

