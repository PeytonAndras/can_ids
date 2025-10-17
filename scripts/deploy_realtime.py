#!/usr/bin/env python3
"""deploy_realtime.py

Real-time CAN anomaly detection pipeline for RAMN + CARLA deployment.

This script connects to a live CAN interface (via python-can) or replays logs
from CSV, constructs 100 ms feature windows, applies the trained unsupervised
models produced by ``scripts/train_unsupervised.py``, and emits alerts when
scores exceed the configured thresholds.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import signal
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd
from joblib import load as joblib_load

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    import can  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    can = None

MAX_PAYLOAD_BYTES = 8
IGNORED_COLUMNS = {"file", "label", "win"}
DEFAULT_CONFIG_PATH = Path("deployment/config.yaml")
DEFAULT_WINDOW_MS = 100.0
DEFAULT_SMOOTHING_WINDOWS = 2
SUPPORTED_ENSEMBLES = {"logical_and", "logical_or", "isolation_forest", "pca", "autoencoder"}


@dataclass
class DeploymentConfig:
    window_ms: float = DEFAULT_WINDOW_MS
    consecutive_windows: int = DEFAULT_SMOOTHING_WINDOWS
    models: Dict[str, Path] = field(default_factory=dict)
    thresholds_path: Path | None = None
    ensemble: str = "logical_and"
    logging_path: Path | None = None
    logging_level: str = "INFO"
    telemetry_port: Optional[int] = None
    replay_csv: Optional[Path] = None

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any], config_dir: Path) -> "DeploymentConfig":
        def resolve_path(value: Optional[str | Path]) -> Optional[Path]:
            if value is None:
                return None
            path = Path(value)
            if not path.is_absolute():
                path = (config_dir / path).resolve()
            return path

        models_section = mapping.get("models", {})
        models = {}
        for name, path_str in models_section.items():
            if path_str is None:
                continue
            models[name] = resolve_path(path_str)

        thresholds = resolve_path(mapping.get("thresholds"))
        logging_cfg = mapping.get("logging", {}) or {}

        return cls(
            window_ms=float(mapping.get("window_ms", DEFAULT_WINDOW_MS)),
            consecutive_windows=int(mapping.get("smoothing", {}).get(
                "consecutive_windows", DEFAULT_SMOOTHING_WINDOWS
            )),
            models=models,
            thresholds_path=thresholds,
            ensemble=str(mapping.get("ensemble", "logical_and")),
            logging_path=resolve_path(logging_cfg.get("path")),
            logging_level=str(logging_cfg.get("level", "INFO")).upper(),
            telemetry_port=mapping.get("telemetry", {}).get("http_port"),
            replay_csv=resolve_path(mapping.get("replay_csv")),
        )


@dataclass
class Frame:
    timestamp: float
    arbitration_id: int
    dlc: int
    data: bytes


def load_config(path: Path) -> DeploymentConfig:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} does not exist")

    text = path.read_text()
    config_dir = path.parent

    data: Dict[str, Any]
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configuration files")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)

    return DeploymentConfig.from_mapping(data, config_dir)


def ensure_model_paths(models: Dict[str, Optional[Path]]) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for name, path in models.items():
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Model artifact for {name!r} not found at {path}")
        resolved[name] = path
    if not resolved:
        raise ValueError("No model artifacts were provided in the configuration")
    return resolved


def load_thresholds(path: Path | None) -> Dict[str, float]:
    if path is None:
        raise ValueError("Thresholds path must be provided in the configuration")
    payload = json.loads(path.read_text())
    values = payload.get("values") or payload.get("datasets")
    if values is None:
        raise ValueError(f"Thresholds file at {path} is missing 'values' or 'datasets' section")
    if isinstance(values, dict) and "values" in values:
        values = values["values"]
    flattened: Dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            # Dataset map from aggregate thresholds.json
            flattened.update({f"{key}:{k}": float(v) for k, v in value.items()})
        else:
            flattened[key] = float(value)
    return flattened


def parse_hex_id(value: int | str) -> int:
    if isinstance(value, int):
        return value
    return int(str(value), 16)


def payload_variance(payload: bytes) -> float:
    if not payload:
        return 0.0
    arr = np.frombuffer(payload, dtype=np.uint8)
    return float(np.var(arr))


def extract_payload_bytes(payload: bytes) -> List[float]:
    values = [0.0] * MAX_PAYLOAD_BYTES
    for idx in range(min(len(payload), MAX_PAYLOAD_BYTES)):
        values[idx] = float(payload[idx])
    return values


def compute_entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counter.values()), dtype=float) / float(total)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def compute_window_features(
    frames: List[Frame],
    window_index: int,
    prev_timestamp: Optional[float],
) -> Optional[pd.Series]:
    if not frames:
        return None

    timestamps = np.array([frame.timestamp for frame in frames], dtype=float)
    start_time = float(timestamps.min())
    end_time = float(timestamps.max())
    duration_ms = (end_time - start_time) * 1000.0

    if prev_timestamp is not None:
        ts = np.concatenate(([prev_timestamp], timestamps))
    else:
        ts = timestamps
    if ts.size > 1:
        dts = np.diff(ts) * 1000.0
        mean_dt = float(np.mean(dts))
        std_dt = float(np.std(dts))
    else:
        mean_dt = duration_ms if duration_ms > 0 else 0.0
        std_dt = 0.0

    ids = Counter(frame.arbitration_id for frame in frames)
    entropy_ids = compute_entropy(ids)
    avg_dlc = float(np.mean([frame.dlc for frame in frames]))

    payload_vars = [payload_variance(frame.data) for frame in frames]
    avg_payload_var = float(np.mean(payload_vars)) if payload_vars else 0.0

    byte_stats: Dict[str, float] = {}
    for byte_idx in range(MAX_PAYLOAD_BYTES):
        column_name = f"payload_byte_{byte_idx}"
        values = [frame.data[byte_idx] for frame in frames if len(frame.data) > byte_idx]
        if values:
            arr = np.array(values, dtype=float)
            byte_stats[f"{column_name}_mean"] = float(arr.mean())
            byte_stats[f"{column_name}_std"] = float(arr.std(ddof=0))
            byte_stats[f"{column_name}_min"] = float(arr.min())
            byte_stats[f"{column_name}_max"] = float(arr.max())
        else:
            byte_stats[f"{column_name}_mean"] = 0.0
            byte_stats[f"{column_name}_std"] = 0.0
            byte_stats[f"{column_name}_min"] = 0.0
            byte_stats[f"{column_name}_max"] = 0.0

    series = pd.Series(
        {
            "win": window_index,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "total_frames": len(frames),
            "unique_ids": len(ids),
            "mean_dt_ms": mean_dt,
            "std_dt_ms": std_dt,
            "entropy_ids": entropy_ids,
            "avg_dlc": avg_dlc,
            "avg_payload_var": avg_payload_var,
            **byte_stats,
        }
    )
    return series


class WindowAggregator:
    def __init__(self, window_ms: float) -> None:
        self.window_ms = window_ms
        self.window_sec = window_ms / 1000.0
        self.current_start: Optional[float] = None
        self.frames: List[Frame] = []
        self.prev_timestamp: Optional[float] = None
        self.window_index = 0

    def add_frame(self, frame: Frame) -> Iterator[pd.Series]:
        ts = frame.timestamp
        if self.current_start is None:
            self.current_start = math.floor(ts / self.window_sec) * self.window_sec

        if ts < self.current_start:
            # Out-of-order frame; adjust window start backwards
            self.current_start = math.floor(ts / self.window_sec) * self.window_sec

        flushed = []
        while ts >= self.current_start + self.window_sec:
            if self.frames:
                features = compute_window_features(self.frames, self.window_index, self.prev_timestamp)
                if features is not None:
                    flushed.append(features)
            self.prev_timestamp = self.frames[-1].timestamp if self.frames else self.prev_timestamp
            self.frames = []
            self.window_index += 1
            self.current_start += self.window_sec

        self.frames.append(frame)
        self.prev_timestamp = frame.timestamp
        return iter(flushed)

    def flush(self) -> Optional[pd.Series]:
        if not self.frames:
            return None
        features = compute_window_features(self.frames, self.window_index, self.prev_timestamp)
        self.frames = []
        self.window_index += 1
        return features


class ModelEnsemble:
    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.models: Dict[str, Any] = {}
        self.thresholds = load_thresholds(config.thresholds_path)
        self.feature_names: List[str] = []
        self.scaler = None
        self._load_models()

    def _load_models(self) -> None:
        resolved_models = ensure_model_paths(self.config.models)
        for name, path in resolved_models.items():
            artifact = joblib_load(path)
            if name == "pca" and "pca" in artifact:
                model = artifact["pca"]
                scaler = artifact.get("scaler")
            else:
                model = artifact.get("model") or artifact.get("pca")
                scaler = artifact.get("scaler")
            feature_names = artifact.get("feature_names")

            if model is None or scaler is None or feature_names is None:
                raise ValueError(f"Artifact at {path} is missing model, scaler, or feature names")

            if not self.feature_names:
                self.feature_names = list(feature_names)
                self.scaler = scaler
            self.models[name] = model

        if not self.feature_names:
            raise ValueError("No feature names available from model artifacts")
        if self.config.ensemble not in SUPPORTED_ENSEMBLES:
            raise ValueError(
                f"Unsupported ensemble strategy '{self.config.ensemble}'. Options: {sorted(SUPPORTED_ENSEMBLES)}"
            )

    def _align_features(self, features: pd.Series) -> np.ndarray:
        aligned = []
        for name in self.feature_names:
            aligned.append(float(features.get(name, 0.0)))
        return np.array(aligned, dtype=float).reshape(1, -1)

    def score(self, features: pd.Series) -> Dict[str, float]:
        X = self._align_features(features)
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        scores: Dict[str, float] = {}
        for name, model in self.models.items():
            if name == "isolation_forest":
                scores[name] = float(-model.score_samples(X_scaled)[0])
            elif name == "pca":
                reconstructed = model.inverse_transform(model.transform(X_scaled))
                scores[name] = float(np.mean((X_scaled - reconstructed) ** 2))
            elif name == "autoencoder":
                reconstructed = model.predict(X_scaled)
                scores[name] = float(np.mean((X_scaled - reconstructed) ** 2))
            else:
                raise ValueError(f"Unsupported detector name '{name}' in model config")
        return scores

    def is_anomaly(self, scores: Dict[str, float]) -> bool:
        flags = {}
        for name, score in scores.items():
            threshold = self._lookup_threshold(name)
            if threshold is None:
                continue
            flags[name] = score >= threshold
        if not flags:
            return False
        strat = self.config.ensemble
        if strat == "logical_and":
            return all(flags.values())
        if strat == "logical_or":
            return any(flags.values())
        return bool(flags.get(strat, False))

    def _lookup_threshold(self, name: str) -> Optional[float]:
        if name in self.thresholds:
            return float(self.thresholds[name])
        # Support dataset-prefixed keys from aggregate thresholds
        for key, value in self.thresholds.items():
            if key.endswith(f":{name}"):
                return float(value)
        return None


@dataclass
class MetricsTracker:
    consecutive_anomalies: int = 0
    total_windows: int = 0
    total_alerts: int = 0
    total_start_time: float = field(default_factory=time.time)
    last_alert_time: Optional[float] = None

    def register_window(self, is_anomaly: bool, smoothing_required: int) -> bool:
        self.total_windows += 1
        if is_anomaly:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0
        if self.consecutive_anomalies >= smoothing_required:
            self.total_alerts += 1
            self.last_alert_time = time.time()
            self.consecutive_anomalies = 0
            return True
        return False

    def fp_per_hour(self) -> float:
        elapsed = time.time() - self.total_start_time
        if elapsed <= 0:
            return 0.0
        hours = elapsed / 3600.0
        if hours <= 0:
            return 0.0
        return float(self.total_alerts / hours)


class AlertSink:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("a", encoding="utf-8")
        else:
            self.file = None

    def emit(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True)
        logging.info("ALERT %s", line)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class ReplayReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        raw_df = pd.read_csv(path)
        self.df = self._normalise_columns(raw_df)
        missing = {"timestamp", "id", "dlc", "data"} - set(self.df.columns)
        if missing:
            raise ValueError(f"Replay CSV {path} is missing required columns: {sorted(missing)}")
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        normalised = df.copy()
        normalised.columns = [str(c).strip().lower() for c in normalised.columns]
        rename_map = {
            "can_id": "id",
            "canid": "id",
            "identifier": "id",
            "arbitration_id": "id",
            "data_hex": "data",
            "payload": "data",
            "payload_hex": "data",
            "dlc_len": "dlc",
        }
        normalised = normalised.rename(columns=rename_map)

        if "timestamp" not in normalised.columns and "time" in normalised.columns:
            normalised = normalised.rename(columns={"time": "timestamp"})

        # Canonicalise payload strings
        if "data" in normalised.columns:
            normalised["data"] = normalised["data"].fillna("").astype(str).str.strip()

        # Ensure numeric fields are parsed correctly
        if "timestamp" in normalised.columns:
            normalised["timestamp"] = pd.to_numeric(normalised["timestamp"], errors="coerce")
        if "dlc" in normalised.columns:
            normalised["dlc"] = pd.to_numeric(normalised["dlc"], errors="coerce")

        normalised = normalised.dropna(subset=["timestamp", "dlc", "data", "id"], how="any")
        return normalised

    def frames(self) -> Iterable[Frame]:
        for row in self.df.itertuples(index=False):
            try:
                timestamp = float(row.timestamp)
                arbitration_id = parse_hex_id(row.id)
                dlc = int(row.dlc)
                payload_str = str(row.data).strip()
                payload_str = payload_str.replace(" ", "")
                data_bytes = bytes.fromhex(payload_str) if payload_str else b""
            except Exception as exc:  # pragma: no cover - defensive branch
                logging.warning("Skipping malformed row: %s", exc)
                continue
            yield Frame(timestamp=timestamp, arbitration_id=arbitration_id, dlc=dlc, data=data_bytes)


class CanListenerWorker:
    def __init__(self, channel: str, bustype: str = "socketcan") -> None:
        if can is None:
            raise RuntimeError("python-can is required for live CAN mode. Install it with `pip install python-can`. ")
        self.channel = channel
        self.bustype = bustype
        self.bus = can.ThreadSafeBus(channel=channel, bustype=bustype)

    def frames(self) -> Iterable[Frame]:
        if can is None:  # pragma: no cover - defensive branch
            return []
        for message in self.bus:  # blocking iterator
            timestamp = float(message.timestamp or time.time())
            arbitration_id = int(message.arbitration_id)
            dlc = int(message.dlc)
            data_bytes = bytes(message.data)
            yield Frame(timestamp=timestamp, arbitration_id=arbitration_id, dlc=dlc, data=data_bytes)


def configure_logging(level: str, path: Optional[Path]) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def build_alert_payload(
    features: pd.Series,
    scores: Dict[str, float],
    thresholds: Dict[str, float],
    metrics: MetricsTracker,
    scaler: Any,
) -> Dict[str, Any]:
    z_scores = {}
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        for name, mean, scale in zip(scaler.feature_names_in_, scaler.mean_, scaler.scale_):  # type: ignore[attr-defined]
            value = float(features.get(name, 0.0))
            if scale != 0:
                z_scores[name] = (value - float(mean)) / float(scale)
            else:
                z_scores[name] = 0.0

    payload = {
        "window": int(features.get("win", -1)),
        "start_time": float(features.get("start_time", 0.0)),
        "end_time": float(features.get("end_time", 0.0)),
        "duration_ms": float(features.get("duration_ms", 0.0)),
        "scores": {k: float(v) for k, v in scores.items()},
        "thresholds": {k: float(thresholds.get(k, 0.0)) for k in scores.keys()},
        "metrics": {
            "total_windows": metrics.total_windows,
            "total_alerts": metrics.total_alerts,
            "fp_per_hour": metrics.fp_per_hour(),
            "last_alert_time": metrics.last_alert_time,
        },
        "feature_zscores": z_scores,
    }
    return payload


def run_pipeline(config: DeploymentConfig, live_channel: Optional[str], bustype: str, replay_path: Optional[Path]) -> None:
    configure_logging(config.logging_level, config.logging_path)
    ensemble = ModelEnsemble(config)
    metrics = MetricsTracker()
    alert_sink = AlertSink(config.logging_path)
    aggregator = WindowAggregator(config.window_ms)

    def frame_iter() -> Iterable[Frame]:
        if replay_path is not None:
            logging.info("Replaying CAN log from %s", replay_path)
            replay_reader = ReplayReader(replay_path)
            return replay_reader.frames()
        if live_channel is not None:
            logging.info("Listening on CAN interface %s (%s)", live_channel, bustype)
            can_worker = CanListenerWorker(channel=live_channel, bustype=bustype)
            return can_worker.frames()
        raise ValueError("Either --replay or --can-channel must be provided")

    frame_iterator = frame_iter()

    stop_requested = False

    def handle_signal(signum, frame):  # type: ignore[override]
        nonlocal stop_requested
        logging.info("Received signal %s; shutting down gracefully...", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        for frame in frame_iterator:
            if stop_requested:
                break
            for features in aggregator.add_frame(frame):
                scores = ensemble.score(features)
                is_anomaly = ensemble.is_anomaly(scores)
                should_alert = metrics.register_window(is_anomaly, config.consecutive_windows)
                if should_alert:
                    payload = build_alert_payload(features, scores, ensemble.thresholds, metrics, ensemble.scaler)
                    alert_sink.emit(payload)
        # Flush remainder window on shutdown
        final_features = aggregator.flush()
        if final_features is not None:
            scores = ensemble.score(final_features)
            is_anomaly = ensemble.is_anomaly(scores)
            should_alert = metrics.register_window(is_anomaly, config.consecutive_windows)
            if should_alert:
                payload = build_alert_payload(final_features, scores, ensemble.thresholds, metrics, ensemble.scaler)
                alert_sink.emit(payload)
    finally:
        alert_sink.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time CAN anomaly detection deployment")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to deployment config file (YAML/JSON)")
    parser.add_argument("--can-channel", type=str, help="socketcan interface to listen on (e.g., can0)")
    parser.add_argument("--bustype", type=str, default="socketcan", help="python-can bus type (default: socketcan)")
    parser.add_argument("--replay", type=Path, help="Optional CSV log to replay instead of live CAN")
    parser.add_argument("--window-ms", type=float, help="Override window size from config")
    parser.add_argument("--smoothing", type=int, help="Override consecutive anomaly windows required")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.window_ms is not None:
        config.window_ms = args.window_ms
    if args.smoothing is not None:
        config.consecutive_windows = args.smoothing
    if args.replay is not None:
        config.replay_csv = args.replay
    if args.can_channel is None and config.replay_csv is None:
        raise ValueError("Either --can-channel or --replay must be specified")

    replay_path = config.replay_csv
    live_channel = args.can_channel
    run_pipeline(config, live_channel=live_channel, bustype=args.bustype, replay_path=replay_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
