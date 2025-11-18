# Testing Rate-Based Detection

## Quick Start

### Option 1: Direct Python Test (No CAN Interface Needed)

Test the rate detector logic directly:

```bash
python3 scripts/test_rate_detection.py
# Choose option 1
```

This simulates your attack pattern and shows if it would be detected.

### Option 2: Full Integration Test (Requires CAN Interface)

#### Step 1: Enable Rate Detection

Make sure `deployment/config.yaml` has:

```yaml
rate_detection:
  enabled: true
  rate_minimum_threshold: 0.1      # Detect low rates
  regularity_threshold: 0.15        # Detect regular timing
  monitored_ids: []                # Monitor all IDs
```

#### Step 2: Start the IDS

```bash
# Terminal 1
python3 scripts/deploy_realtime.py --can-channel can0
```

#### Step 3: Run Your Attack Script

```bash
# Terminal 2
./your_attack_script.sh
```

Or use the test script:

```bash
python3 scripts/test_rate_detection.py
# Choose option 2
```

#### Step 4: Check Alerts

Look for alerts in the IDS output or `logs/alerts.jsonl`:

```json
{
  "rate_detection": {
    "is_anomaly": true,
    "alerts": [
      "ID 0x039: suspiciously regular timing (CV=0.05 < 0.15 threshold)",
      "ID 0x062: suspiciously low rate (1.0 msg/s < 8.0 threshold)"
    ]
  }
}
```

## Testing Your Specific Attack

Your cycling script sends:
- Messages: `062, 024, 039, 062, 024` (cycling)
- Interval: 0.1s (100ms)
- Rate: 10 msg/s total

### Expected Detection

The rate detector should alert on:

1. **Regular Timing**: CV < 0.15 (your 0.1s intervals are very regular)
2. **Low Rate**: Rate < 10% of normal (10 msg/s vs 80+ normal)

### Test Commands

```bash
# Terminal 1: Start IDS
python3 scripts/deploy_realtime.py --can-channel can0 2>&1 | tee test_output.log

# Terminal 2: Run attack (after IDS starts)
for i in {1..50}; do
  cansend can0 062#07B63A0BF6623BCF
  sleep 0.1
  cansend can0 024#006C3A0D9C4F913B
  sleep 0.1
  cansend can0 039#00003A0DD87D5C7A
  sleep 0.1
  cansend can0 062#07B33A0CBE359457
  sleep 0.1
  cansend can0 024#00693A0ECDDC53A4
  sleep 0.1
done

# Check Terminal 1 for alerts
```

## Verification

### What to Look For

1. **Rate Detection Alerts**:
   ```
   ALERT {"rate_detection": {"is_anomaly": true, ...}}
   ```

2. **Alert Messages**:
   - "suspiciously regular timing"
   - "suspiciously low rate"

3. **Detection Score**: Should be > 0.7 for your attack

### Troubleshooting

**No alerts generated?**

1. Check config is enabled:
   ```bash
   grep -A 5 "rate_detection" deployment/config.yaml
   ```

2. Increase sensitivity:
   ```yaml
   rate_detection:
     regularity_threshold: 0.2    # Less strict
     rate_minimum_threshold: 0.2  # Less strict
   ```

3. Check minimum samples:
   ```yaml
   min_samples: 5  # Lower threshold
   ```

**Too many false positives?**

1. Increase thresholds:
   ```yaml
   regularity_threshold: 0.05     # More strict
   rate_deviation_threshold: 4.0  # More strict
   ```

2. Monitor specific IDs:
   ```yaml
   monitored_ids: ["0x062", "0x024", "0x039"]
   ```

## Advanced Testing

### Test Different Attack Patterns

```bash
# High-rate attack
for i in {1..1000}; do
  cansend can0 039#00003A0DD87D5C7A
  sleep 0.001
done

# Irregular timing attack
for i in {1..100}; do
  cansend can0 039#00003A0DD87D5C7A
  sleep $(python3 -c "import random; print(random.uniform(0.01, 0.5))")
done
```

### Monitor Detection in Real-Time

```bash
# Watch alerts live
tail -f logs/alerts.jsonl | jq '.rate_detection'

# Or with grep
python3 scripts/deploy_realtime.py --can-channel can0 2>&1 | grep -i "rate\|alert"
```

## Expected Results

For your cycling attack script, you should see:

```
ALERT {
  "rate_detection": {
    "is_anomaly": true,
    "score": 0.85,
    "alerts": [
      "ID 0x039: suspiciously regular timing (CV=0.05 < 0.15 threshold)",
      "ID 0x062: suspiciously low rate (2.0 msg/s < 8.0 threshold, normal=80.0 msg/s)"
    ]
  }
}
```

This confirms the rate detector is working!

