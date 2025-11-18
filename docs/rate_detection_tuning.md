# Rate Detection Tuning Guide

## Problem: Too Many False Positives

If you're seeing alerts on **all normal traffic**, it's because:

1. **Normal CAN traffic IS regular** - Periodic messages are expected
2. **Regularity threshold too high** - Catching normal periodic patterns
3. **Need baseline comparison** - Should compare to historical patterns, not absolute values

## Solution: Adjust Thresholds

### Option 1: Disable Regularity Detection (Quick Fix)

If normal traffic is always regular, disable regularity detection:

```yaml
rate_detection:
  enabled: true
  regularity_threshold: 0.001  # Very strict - only near-perfect regularity
  # Or focus on other detection methods:
  rate_minimum_threshold: 0.1  # Low rate detection
  rate_multiplier_threshold: 2.0  # High rate detection
```

### Option 2: Use Baseline Comparison (Recommended)

The updated code now compares current CV to baseline CV. This means:
- Normal regular traffic won't alert (it matches baseline)
- Only NEW regular patterns will alert (different from baseline)

### Option 3: Focus on Rate Deviations

Instead of regularity, focus on rate anomalies:

```yaml
rate_detection:
  enabled: true
  regularity_threshold: 0.001  # Disable regularity alerts
  rate_minimum_threshold: 0.15  # Detect low rates
  rate_multiplier_threshold: 2.0  # Detect high rates
  rate_deviation_threshold: 3.0  # Detect statistical anomalies
  monitored_ids: ["0x062", "0x024", "0x039"]  # Focus on specific IDs
```

## Current Configuration

After the fix, the default is:

```yaml
regularity_threshold: 0.005  # Very strict - only near-perfect regularity
```

This will:
- ✅ Still detect injection attacks with perfect timing (CV ≈ 0.000)
- ✅ Ignore normal periodic traffic (CV ≈ 0.03-0.05)
- ✅ Compare to baseline when available

## Testing

After adjusting thresholds:

1. **Restart IDS**: `python3 scripts/deploy_realtime.py --can-channel can0`
2. **Let baseline build**: Wait 30+ seconds
3. **Check normal traffic**: Should see NO alerts
4. **Run attack**: Should detect your cycling script

## Expected Behavior

### Normal Traffic (No Alerts)
```
[  30.0s] Window  300   OK | IF=0.50 ✗ | PCA=6.5 ✗ | Rate=0.00 ✗ | Frames=  80 IDs=17
```

### Attack Detected
```
[  35.0s] Window  350 🚨 ALERT | IF=0.55 ✗ | PCA=7.2 ✗ | Rate=0.85 ✓ | Frames=  82 IDs=17
  📊 ID 0x039: suspiciously regular timing (CV=0.000 < baseline=0.030±0.005)
```

## Fine-Tuning

If still getting false positives:

1. **Increase regularity threshold**: `0.005 → 0.001` (more strict)
2. **Require more samples**: `min_samples: 20` (more conservative)
3. **Monitor specific IDs**: Only watch IDs you know are attack targets
4. **Disable regularity**: Set `regularity_threshold: 0.0001` to effectively disable

If missing attacks:

1. **Decrease regularity threshold**: `0.005 → 0.01` (less strict)
2. **Lower min_samples**: `min_samples: 5` (faster detection)
3. **Focus on rate changes**: Use `rate_minimum_threshold` and `rate_multiplier_threshold`

