# Real-Time Deployment Guide

This guide explains how to run the unsupervised CAN intrusion detector on the
RAMN testbed alongside CARLA-generated traffic.

## Prerequisites
- Python 3.10+
- [`python-can`](https://python-can.readthedocs.io/) with SocketCAN access to RAMN (`can0`)
- `numpy`, `pandas`, `scikit-learn`, `joblib`, `PyYAML`
- Trained model artifacts in `models/unsupervised/` (produced by `scripts/train_unsupervised.py`)
- Threshold file in `results/unsupervised/thresholds.json`
- RAMN connected via USB-CAN adapter with `sudo ip link set can0 up type can bitrate 500000`

Install the Python dependencies:

```bash
python -m pip install python-can numpy pandas scikit-learn joblib pyyaml
```

## Configuration
Review `deployment/config.yaml`:

```yaml
window_ms: 100
smoothing:
  consecutive_windows: 2
models:
  isolation_forest: ../models/unsupervised/isolation_forest.joblib
  pca: ../models/unsupervised/pca_reconstruction.joblib
thresholds: ../results/unsupervised/thresholds.json
ensemble: logical_and
logging:
  path: logs/alerts.jsonl
  level: INFO
telemetry:
  http_port: 8080
```

Adjust paths if you move the artifacts. You can also override `window_ms` or
`smoothing` from the CLI without editing the file.

## Running Against Live CAN Traffic

```bash
python scripts/deploy_realtime.py --can-channel can0
```

This command:
1. Connects to the `can0` interface via python-can.
2. Builds 100 ms feature windows.
3. Applies Isolation Forest and PCA detectors using the configured thresholds.
4. Requires 2 consecutive anomalous windows before declaring an alert.
5. Logs all alerts to stdout and `deployment/logs/alerts.jsonl`.

## Replaying Recorded Logs

```bash
python scripts/deploy_realtime.py --replay path/to/raw_log.csv
```

The CSV must contain `timestamp`, `id`, `dlc`, and `data` columns to mirror
the outputs of the ingestion scripts.

## Validation Checklist
- [ ] Collect a 10-minute benign RAMN drive and confirm `fp_per_hour <= 6`.
- [ ] Replay each attack scenario and record detection latency (target < 400 ms).
- [ ] Capture alert logs and attach them to experiment notes.
- [ ] Update `results/unsupervised/thresholds.json` if you retune thresholds for RAMN-specific traffic.
- [ ] Commit deployment configuration changes (artifacts remain ignored by `.gitignore`).

## Troubleshooting
- **No frames received:** verify `ip link show can0` and CAN bitrate configuration.
- **ImportError: python-can:** install dependencies with the pip command above.
- **Missing thresholds:** rerun `scripts/train_unsupervised.py` and ensure the results directory is populated.
- **Frequent false positives:** increase `smoothing.consecutive_windows` or tighten the percentile when training.
