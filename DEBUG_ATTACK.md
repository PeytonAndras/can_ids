# Debugging Attack Detection

## Issue
The detector is alerting immediately but not detecting the accelerator flood attack.

## Step 1: Verify Attack is Actually Running

Run this in a separate terminal to monitor CAN traffic:

```bash
python scripts/monitor_can.py --channel can0 --duration 10
```

**What to look for:**
- Normal: ~100 fps of `0x039` frames
- Attack: Should see 1000+ fps of `0x039` frames flooding

**If you see 0 or very few `0x039` frames:**
- The attack script isn't running
- The attack is on a different interface (check `vcan0` vs `can0`)
- The attack script has an error

## Step 2: Check Current Alert

The alert you saw had:
- Isolation Forest score: **0.6788** (threshold: 0.6764) - just barely above
- This happened in **window 0** (first 100ms)

This suggests:
1. **False positive** on normal traffic variation
2. The threshold might be too sensitive

## Step 3: Solutions

### Option A: Increase Threshold (Reduce False Positives)

Retrain with a higher percentile:

```bash
python scripts/train_unsupervised.py \
  --dataset=real_can0=data/processed/real_can0 \
  --percentile=99.9
```

Then update the config to use the new thresholds.

### Option B: Add Smoothing (Require Multiple Windows)

Edit `deployment/config.yaml`:
```yaml
smoothing:
  consecutive_windows: 3  # Require 3 consecutive anomalous windows
```

This will reduce false positives but may delay detection slightly.

### Option C: Verify Attack is Actually Happening

1. **Check your attack script** - make sure it's:
   - Sending to `can0` (not `vcan0`)
   - Flooding `0x039` frames at high rate
   - Actually running

2. **Monitor during attack:**
   ```bash
   # Terminal 1: Monitor CAN
   python scripts/monitor_can.py --channel can0 --duration 30
   
   # Terminal 2: Run your attack script
   # (your attack command here)
   ```

3. **Check if frames are reaching can0:**
   ```bash
   candump can0 | grep "039#"
   ```
   You should see frames flooding rapidly during attack.

## Step 4: Test Detection During Attack

Once you verify the attack is flooding frames:

1. **Start detector:**
   ```bash
   python scripts/deploy_realtime.py --config deployment/config.yaml --can-channel can0
   ```

2. **In another terminal, run attack**

3. **Watch for alerts** - you should see alerts within 100-300ms of attack starting

## Expected Behavior

- **Normal traffic:** 0-2 alerts per hour
- **During attack:** Immediate alerts (within 100-300ms)
- **Attack frames:** 1000+ fps of `0x039` frames

