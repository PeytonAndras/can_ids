#!/usr/bin/env python3
"""
train_unsupervised.py
Train unsupervised anomaly detectors for CAN feature windows.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "unsupervised"
RESULTS_DIR = PROJECT_ROOT / "results" / "unsupervised"
TRAIN_DEFAULT = DATA_DIR / "train.parquet"
VAL_DEFAULT = DATA_DIR / "val.parquet"
ALL_FEATURES_NAME = "all_features.csv"
RANDOM_STATE = 42
DEFAULT_PERCENTILE = 99.0
IGNORE_COLUMNS = {"file", "label"}


@dataclass
class DatasetSpec:
    name: str
    data_dir: Path
    train_path: Path
    val_path: Path


@dataclass
class DetectorResult:
    name: str
    threshold: float
    train_scores: np.ndarray
    val_scores: np.ndarray
    model: Any


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def parse_dataset_arguments(entries: Iterable[str] | None) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    if not entries:
        return specs

    for raw in entries:
        raw = raw.strip()
        if not raw:
            continue
        if "=" in raw:
            name_part, path_part = raw.split("=", 1)
            name = name_part.strip()
            path_str = path_part.strip()
        else:
            path_str = raw
            name = ""
        data_dir = resolve_path(path_str)
        if not name:
            name = data_dir.name or "dataset"
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory {data_dir} does not exist for entry '{raw}'")
        train_path = data_dir / "train.parquet"
        val_path = data_dir / "val.parquet"
        specs.append(DatasetSpec(name=name, data_dir=data_dir, train_path=train_path, val_path=val_path))
    return specs


def isolation_forest_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(X)


def pca_reconstruction_scores(pca: PCA, X: np.ndarray) -> np.ndarray:
    reconstructed = pca.inverse_transform(pca.transform(X))
    return np.mean((X - reconstructed) ** 2, axis=1)


def autoencoder_scores(ae: MLPRegressor, X: np.ndarray) -> np.ndarray:
    reconstructed = ae.predict(X)
    return np.mean((X - reconstructed) ** 2, axis=1)


def compute_metrics(scores: np.ndarray, labels: np.ndarray | None, threshold: float,
                    durations_ms: pd.Series | None = None) -> dict[str, Any]:
    if labels is None or labels.size == 0:
        return {}

    labels_int = labels.astype(int)
    positive_mask = labels_int != 0
    predictions = scores >= threshold

    tp = int(np.sum(predictions & positive_mask))
    fp = int(np.sum(predictions & ~positive_mask))
    tn = int(np.sum(~predictions & ~positive_mask))
    fn = int(np.sum(~predictions & positive_mask))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = None
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else None
    tpr = recall

    fp_per_hour = None
    if durations_ms is not None and not durations_ms.empty:
        total_ms = float(durations_ms.sum())
        if total_ms > 0:
            total_hours = total_ms / 3_600_000.0
            if total_hours > 0:
                fp_per_hour = float(fp / total_hours)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
        "support_pos": int(np.sum(positive_mask)),
        "support_neg": int(np.sum(~positive_mask)),
        "fp_per_hour": fp_per_hour,
    }


def load_additional_split(data_dir: Path) -> pd.DataFrame | None:
    candidates = [data_dir / "test.parquet", data_dir / "test.csv"]
    for candidate in candidates:
        if candidate.exists():
            return read_table(candidate)
    fallback = data_dir / ALL_FEATURES_NAME
    if fallback.exists():
        try:
            return pd.read_csv(fallback)
        except Exception:
            return None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unsupervised detectors for CAN data")
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT, help="Path to the training set (parquet or csv)")
    parser.add_argument("--val", type=Path, default=VAL_DEFAULT, help="Path to the validation set (parquet or csv)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory with processed feature files")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Where to store trained models")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Where to store score summaries")
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE, help="Percentile for threshold selection")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fallback validation size when splitting all_features.csv")
    parser.add_argument("--ae", action="store_true", help="Train the MLP autoencoder in addition to IF and PCA")
    parser.add_argument("--ae-hidden", type=str, default="128,64,128", help="Comma separated hidden sizes for the autoencoder")
    parser.add_argument("--ae-max-iter", type=int, default=200, help="Maximum iterations for the autoencoder")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset at {path} does not exist")
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except ImportError as exc:
        if path.suffix == ".parquet":
            fallback = path.with_suffix(".csv")
            if fallback.exists():
                print(f"[!] Unable to read {path} ({exc}); using {fallback} instead")
                return pd.read_csv(fallback)
        raise


def maybe_create_split(train_path: Path, val_path: Path, data_dir: Path, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_path.exists() and val_path.exists():
        return read_table(train_path), read_table(val_path)

    csv_train = train_path.with_suffix(".csv")
    csv_val = val_path.with_suffix(".csv")
    if csv_train.exists() and csv_val.exists():
        return pd.read_csv(csv_train), pd.read_csv(csv_val)

    source_csv = data_dir / ALL_FEATURES_NAME
    if not source_csv.exists():
        raise FileNotFoundError(
            "Training and validation files are missing and data/processed/all_features.csv was not found"
        )

    print(f"[!] {train_path.name} or {val_path.name} not found. Creating a split from {source_csv.name} ...")
    full = pd.read_csv(source_csv)
    if "label" in full.columns and (full["label"] == 0).any():
        base = full[full["label"] == 0]
        if base.empty:
            base = full
    else:
        base = full

    train_df, val_df = train_test_split(
        base, test_size=val_ratio, random_state=RANDOM_STATE, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    try:
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        print(f"[+] Saved new parquet splits to {train_path} and {val_path}")
    except Exception as exc:
        csv_train = train_path.with_suffix(".csv")
        csv_val = val_path.with_suffix(".csv")
        train_df.to_csv(csv_train, index=False)
        val_df.to_csv(csv_val, index=False)
        print(f"[!] Could not save parquet splits ({exc}); wrote CSV fallbacks to {csv_train} and {csv_val}")

    return train_df, val_df


def select_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in IGNORE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns available after filtering")
    return cols


def score_percentile(scores: np.ndarray, labels: np.ndarray | None, percentile: float) -> float:
    baseline = scores
    if labels is not None and labels.size:
        normal_mask = labels == 0
        if np.any(normal_mask):
            baseline = scores[normal_mask]
    return float(np.percentile(baseline, percentile))


def save_scores(name: str, split: str, scores: np.ndarray, labels: np.ndarray | None, out_dir: Path) -> None:
    payload = {"score": scores}
    if labels is not None and labels.size:
        payload["label"] = labels.astype(int)
    df = pd.DataFrame(payload)
    df.to_csv(out_dir / f"{name}_{split}_scores.csv", index=False)


def plot_scores(name: str, scores: np.ndarray, labels: np.ndarray | None, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    if labels is not None and labels.size and np.unique(labels).size > 1:
        for category in np.unique(labels):
            mask = labels == category
            plt.hist(scores[mask], bins=40, alpha=0.6, label=f"label={category}")
    else:
        plt.hist(scores, bins=40, alpha=0.8, label="scores")
    plt.title(f"{name} validation score distribution")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_isolation_forest(X_train: np.ndarray, X_val: np.ndarray, val_labels: np.ndarray | None,
                           percentile: float, feature_names: list[str], scaler: StandardScaler,
                           models_dir: Path, results_dir: Path) -> DetectorResult:
    print("[+] Training IsolationForest ...")
    iso = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso.fit(X_train)
    train_scores = isolation_forest_scores(iso, X_train)
    val_scores = isolation_forest_scores(iso, X_val)
    threshold = score_percentile(val_scores, val_labels, percentile)

    dump({"model": iso, "scaler": scaler, "feature_names": feature_names}, models_dir / "isolation_forest.joblib")
    save_scores("isolation_forest", "train", train_scores, None, results_dir)
    save_scores("isolation_forest", "val", val_scores, val_labels, results_dir)
    plot_scores("IsolationForest", val_scores, val_labels, results_dir / "isolation_forest_val_hist.png")
    print(f"[✓] IsolationForest threshold ({percentile:.1f} pct) = {threshold:.6f}")
    return DetectorResult(
        name="isolation_forest",
        threshold=threshold,
        train_scores=train_scores,
        val_scores=val_scores,
        model=iso,
    )


def train_pca_detector(X_train: np.ndarray, X_val: np.ndarray, val_labels: np.ndarray | None,
                       percentile: float, feature_names: list[str], scaler: StandardScaler,
                       models_dir: Path, results_dir: Path) -> DetectorResult:
    print("[+] Training PCA reconstruction detector ...")
    pca = PCA(n_components=0.95, svd_solver="full")
    pca.fit(X_train)

    train_scores = pca_reconstruction_scores(pca, X_train)
    val_scores = pca_reconstruction_scores(pca, X_val)
    threshold = score_percentile(val_scores, val_labels, percentile)

    dump({"pca": pca, "scaler": scaler, "feature_names": feature_names}, models_dir / "pca_reconstruction.joblib")
    save_scores("pca", "train", train_scores, None, results_dir)
    save_scores("pca", "val", val_scores, val_labels, results_dir)
    plot_scores("PCA", val_scores, val_labels, results_dir / "pca_val_hist.png")
    print(f"[✓] PCA threshold ({percentile:.1f} pct) = {threshold:.6f}")
    return DetectorResult(
        name="pca",
        threshold=threshold,
        train_scores=train_scores,
        val_scores=val_scores,
        model=pca,
    )


def parse_hidden_layers(hidden: str) -> tuple[int, ...]:
    values = []
    for chunk in hidden.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("Autoencoder hidden layer configuration is empty")
    return tuple(values)


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray, val_labels: np.ndarray | None,
                      percentile: float, feature_names: list[str], scaler: StandardScaler,
                      models_dir: Path, results_dir: Path, hidden: tuple[int, ...], max_iter: int) -> DetectorResult:
    print("[+] Training Autoencoder (MLPRegressor) ...")
    ae = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        verbose=False
    )
    ae.fit(X_train, X_train)
    train_scores = autoencoder_scores(ae, X_train)
    val_scores = autoencoder_scores(ae, X_val)
    threshold = score_percentile(val_scores, val_labels, percentile)

    dump({"model": ae, "scaler": scaler, "feature_names": feature_names}, models_dir / "autoencoder.joblib")
    save_scores("autoencoder", "train", train_scores, None, results_dir)
    save_scores("autoencoder", "val", val_scores, val_labels, results_dir)
    plot_scores("Autoencoder", val_scores, val_labels, results_dir / "autoencoder_val_hist.png")
    print(f"[✓] Autoencoder threshold ({percentile:.1f} pct) = {threshold:.6f}")
    return DetectorResult(
        name="autoencoder",
        threshold=threshold,
        train_scores=train_scores,
        val_scores=val_scores,
        model=ae,
    )


def main() -> None:
    args = parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = maybe_create_split(args.train, args.val, args.data_dir, args.val_ratio)

    train_df = train_df.fillna(0)
    val_df = val_df.fillna(0)
    feature_names = select_features(train_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_names].values)
    X_val = scaler.transform(val_df[feature_names].values)
    val_labels = val_df["label"].values if "label" in val_df.columns else None

    thresholds: dict[str, dict[str, float]] = {
        "percentile": args.percentile,
        "values": {}
    }

    iso_threshold = train_isolation_forest(X_train, X_val, val_labels, args.percentile, feature_names,
                                           scaler, args.models_dir, args.results_dir)
    thresholds["values"]["isolation_forest"] = iso_threshold

    pca_threshold = train_pca_detector(X_train, X_val, val_labels, args.percentile, feature_names,
                                       scaler, args.models_dir, args.results_dir)
    thresholds["values"]["pca"] = pca_threshold

    if args.ae:
        hidden = parse_hidden_layers(args.ae_hidden)
        ae_threshold = train_autoencoder(X_train, X_val, val_labels, args.percentile, feature_names,
                                         scaler, args.models_dir, args.results_dir, hidden, args.ae_max_iter)
        thresholds["values"]["autoencoder"] = ae_threshold

    thresholds_path = args.results_dir / "thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds, indent=2))
    print(f"[✓] Thresholds saved to {thresholds_path}")


if __name__ == "__main__":
    main()
