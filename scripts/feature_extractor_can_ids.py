#!/usr/bin/env python3
"""
featurize_can.py
Converts raw CAN logs (timestamp,id,dlc,data) into time-windowed statistical features.
"""

import os
import sys
import math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# -------- CONFIG --------
RAW_DIR = Path("../data/raw")       # folder containing CSV logs
OUT_DIR = Path("../data/processed") # where to save features
WINDOW_MS = 100.0                   # size of each time window
SAVE_PARQUET = False                # also save as Parquet
# -------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def featurize_file(path, window_ms=WINDOW_MS):
    print(f"[+] Processing {path.name}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        "can_id": "id",
        "data_hex": "data",
    }
    df = df.rename(columns=rename_map)
    required = {"timestamp", "id", "dlc", "data"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {df.columns}")
    df["timestamp"] = df["timestamp"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["id_num"] = df["id"].apply(parse_hex)
    df["payload_var"] = df["data"].apply(payload_variance)
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
        feats.append({
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
        })
    feat_df = pd.DataFrame(feats)
    feat_df["file"] = path.name
    feat_df["label"] = attack_flag
    return feat_df

def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {RAW_DIR}")
        sys.exit(0)
    all_feats = []
    for f in files:
        try:
            feat_df = featurize_file(f)
            out_csv = OUT_DIR / f"{f.stem}_features.csv"
            feat_df.to_csv(out_csv, index=False)
            if SAVE_PARQUET:
                feat_df.to_parquet(out_csv.with_suffix(".parquet"), index=False)
            all_feats.append(feat_df)
            print(f"  → saved {out_csv.name}")
        except Exception as e:
            print(f"[!] Error with {f}: {e}")

    if all_feats:
        merged = pd.concat(all_feats, ignore_index=True)
        merged.to_csv(OUT_DIR / "all_features.csv", index=False)
        if SAVE_PARQUET:
            merged.to_parquet(OUT_DIR / "all_features.parquet", index=False)
        print(f"[✓] Merged features saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
