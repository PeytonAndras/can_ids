#!/usr/bin/env python3
"""
featurize_can.py
Converts raw CAN logs (timestamp,id,dlc,data) into time-windowed statistical features.
"""

import argparse
import os
import sys
import math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# -------- CONFIG --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"       # folder containing CSV logs
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" # where to save features
WINDOW_MS = 100.0                   # size of each time window
SAVE_PARQUET = False                # also save as Parquet
MAX_PAYLOAD_BYTES = 8               # CAN payload length
# -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract windowed CAN features")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing raw CSV files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR, help="Directory to write feature CSV files")
    parser.add_argument("--window-ms", type=float, default=WINDOW_MS, help="Window size in milliseconds")
    parser.add_argument("--save-parquet", action="store_true", help="Also emit Parquet outputs")
    parser.add_argument("--prefix", type=str, default="", help="Optional prefix for generated feature filenames")
    return parser.parse_args()

def compute_entropy(counts):
    """Shannon entropy from a Counter of ID frequencies."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs))

def parse_hex(x):
    try:
        return int(str(x), 16)
    except Exception:
        return np.nan

def payload_variance(hex_payload):
    """Compute variance of payload byte values (0–255)."""
    try:
        b = bytes.fromhex(hex_payload)
        if len(b) == 0:
            return 0
        return np.var(np.frombuffer(b, dtype=np.uint8))
    except Exception:
        return 0


def extract_payload_bytes(hex_payload, max_bytes=MAX_PAYLOAD_BYTES):
    """Return a fixed-length list of byte values (0-255) parsed from a hex payload string."""
    values = [np.nan] * max_bytes
    if not isinstance(hex_payload, str):
        return values

    payload = hex_payload.strip()
    if len(payload) == 0:
        return values

    try:
        raw_bytes = bytes.fromhex(payload)
    except ValueError:
        return values

    limit = min(len(raw_bytes), max_bytes)
    for i in range(limit):
        values[i] = raw_bytes[i]
    return values

def load_raw_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, engine="python")
        except Exception as exc:
            raise ValueError(f"Unable to read {path}: {exc}") from exc


def normalise_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        "can_id": "id",
        "data_hex": "data",
    }
    df = df.rename(columns=rename_map)
    return df


def ensure_required_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"timestamp", "id", "dlc", "data"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "id", "dlc"]).reset_index(drop=True)
    df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(0).astype(int)
    df["data"] = df["data"].fillna("").astype(str)
    return df


def featurize_file(path, window_ms=WINDOW_MS):
    print(f"[+] Processing {path.name}")
    df = load_raw_csv(path)
    df = normalise_raw_frame(df)
    df = ensure_required_columns(df, path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["id_num"] = df["id"].apply(parse_hex)
    df["payload_var"] = df["data"].apply(payload_variance)
    byte_cols = [f"payload_byte_{i}" for i in range(MAX_PAYLOAD_BYTES)]
    byte_df = pd.DataFrame(
        df["data"].apply(lambda x: extract_payload_bytes(x, MAX_PAYLOAD_BYTES)).tolist(),
        columns=byte_cols,
        index=df.index
    )
    df = pd.concat([df, byte_df], axis=1)
    # compute inter-arrival
    df["dt_ms"] = df["timestamp"].diff() * 1000.0
    df["dt_ms"] = df["dt_ms"].fillna(df["dt_ms"].mean())

    # window index
    start = df["timestamp"].iloc[0]
    df["win"] = ((df["timestamp"] - start) * 1000.0 // window_ms).astype(int)

    attack_flag = 1 if "attack" in path.stem.lower() else 0
    if "is_attack" in df.columns:
        attack_flag = int(pd.to_numeric(df["is_attack"], errors="coerce").fillna(0).max() > 0)

    feats = []
    for win, g in df.groupby("win"):
        ids = Counter(g["id_num"])
        entropy = compute_entropy(ids)
        stats = {
            "win": win,
            "start_time": g["timestamp"].min(),
            "end_time": g["timestamp"].max(),
            "duration_ms": (g["timestamp"].max() - g["timestamp"].min()) * 1000.0,
            "total_frames": len(g),
            "unique_ids": len(ids),
            "mean_dt_ms": g["dt_ms"].mean(),
            "std_dt_ms": g["dt_ms"].std(),
            "entropy_ids": entropy,
            "avg_dlc": g["dlc"].mean(),
            "avg_payload_var": g["payload_var"].mean()
        }

        for col in byte_cols:
            series = g[col].dropna()
            if series.empty:
                stats[f"{col}_mean"] = 0.0
                stats[f"{col}_std"] = 0.0
                stats[f"{col}_min"] = 0.0
                stats[f"{col}_max"] = 0.0
            else:
                stats[f"{col}_mean"] = series.mean()
                stats[f"{col}_std"] = series.std(ddof=0)
                stats[f"{col}_min"] = series.min()
                stats[f"{col}_max"] = series.max()
        feats.append(stats)
    feat_df = pd.DataFrame(feats)
    feat_df["file"] = path.name
    feat_df["label"] = attack_flag
    return feat_df

def main():
    args = parse_args()
    raw_dir = args.input
    out_dir = args.output
    window_ms = args.window_ms
    save_parquet = args.save_parquet or SAVE_PARQUET
    prefix = args.prefix

    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {raw_dir}")
        sys.exit(0)
    all_feats = []
    for f in files:
        try:
            feat_df = featurize_file(f, window_ms=window_ms)
            suffix = f"{f.stem}_features.csv"
            out_name = f"{prefix}_{suffix}" if prefix else suffix
            out_csv = out_dir / out_name
            feat_df.to_csv(out_csv, index=False)
            if save_parquet:
                feat_df.to_parquet(out_csv.with_suffix(".parquet"), index=False)
            all_feats.append(feat_df)
            print(f"  → saved {out_csv.name}")
        except Exception as e:
            print(f"[!] Error with {f}: {e}")

    if all_feats:
        merged = pd.concat(all_feats, ignore_index=True)
        merged_name = "all_features.csv" if not prefix else f"{prefix}_all_features.csv"
        merged_path = out_dir / merged_name
        merged.to_csv(merged_path, index=False)
        if save_parquet:
            merged.to_parquet(merged_path.with_suffix(".parquet"), index=False)
        print(f"[✓] Merged features saved in {merged_path}")

if __name__ == "__main__":
    main()
