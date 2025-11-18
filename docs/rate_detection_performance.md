# Rate Detection Performance Optimization

## Faster Detection & Normalization

The rate detector has been optimized for faster adaptation:

### Changes Made

1. **Short-term window**: Added `short_term_window_seconds: 10.0` for faster baseline updates
2. **Reduced min_samples**: Changed from 10 to 5 for faster initial detection
3. **Exponential weighting**: Recent rates weighted more heavily than old rates
4. **Limited history**: Rate history automatically pruned to prevent stale data
5. **Faster baseline building**: Reduced requirements for baseline comparison

### Configuration

```yaml
rate_detection:
  enabled: true
  history_window_seconds: 30.0      # Long-term baseline
  short_term_window_seconds: 10.0  # Fast adaptation window
  min_samples: 5                     # Faster initial detection
```

### How It Works

**Detection Speed:**
- Uses 10-second window for recent patterns (vs 30-second)
- Requires only 5 samples before alerting (vs 10)
- Exponential weighting favors recent data
- Baseline comparison uses fewer windows (3 vs 5)

**Normalization Speed:**
- Short-term window adapts quickly to new patterns
- Rate history automatically pruned
- Recent rates weighted more heavily
- Baseline updates faster when attack stops

### Expected Performance

**Before:**
- Detection delay: ~20-30 seconds
- Normalization delay: ~30-40 seconds

**After:**
- Detection delay: ~5-10 seconds
- Normalization delay: ~10-15 seconds

### Fine-Tuning

If still too slow:

```yaml
short_term_window_seconds: 5.0   # Even faster
min_samples: 3                   # Faster detection
```

If too fast (false positives):

```yaml
short_term_window_seconds: 15.0  # More stable
min_samples: 8                    # More conservative
```

### Trade-offs

- **Faster adaptation** = More responsive but potentially more false positives
- **Slower adaptation** = More stable but slower to detect/normalize

Choose based on your needs!

