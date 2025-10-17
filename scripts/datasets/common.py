#!/usr/bin/env python3
"""Shared utilities for converting public CAN datasets into the repo's raw schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

STANDARD_COLUMNS = ["timestamp", "can_id", "dlc", "data_hex", "scenario", "is_attack"]


@dataclass
class ColumnConfig:
    """Configuration describing how to map dataset-specific columns into the standard schema."""

    timestamp: str
    can_id: str
    dlc: str | None
    data_hex: str | None = None
    data_bytes: Sequence[str] | None = None
    scenario: str | None = None
    label: str | None = None


def _normalise_column_name(name: str) -> str:
    cleaned = name.strip().lower()
    for ch in [" ", "-", "[", "]", "(", ")", ":"]:
        cleaned = cleaned.replace(ch, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [_normalise_column_name(col) for col in renamed.columns]
    return renamed


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of the candidates {candidates} found in columns {df.columns.tolist()}")


def _pack_bytes(row: pd.Series, byte_columns: Sequence[str]) -> str:
    values: list[str] = []
    for col in byte_columns:
        value = row.get(col)
        if pd.isna(value):
            continue
        if isinstance(value, str):
            cleaned = value.strip().lower().replace(" ", "")
            if cleaned.startswith("0x"):
                cleaned = cleaned[2:]
            if len(cleaned) == 0:
                continue
            if len(cleaned) > 2 and all(ch in "0123456789abcdef" for ch in cleaned):
                # Already a concatenated payload string.
                return cleaned
            try:
                values.append(f"{int(cleaned, 16):02x}")
            except ValueError:
                try:
                    values.append(f"{int(float(cleaned)):02x}")
                except (TypeError, ValueError):
                    continue
            continue
        try:
            values.append(f"{int(value):02x}")
        except (TypeError, ValueError):
            # If byte can't be parsed, drop it.
            continue
    return "".join(values)


def _normalise_hex(value: str | int | float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("0x"):
            cleaned = cleaned[2:]
        return cleaned.lower()
    try:
        return f"{int(value):x}"
    except (TypeError, ValueError):
        return ""


def load_dataset_csv(path: Path) -> pd.DataFrame:
    """Load a CSV/Parquet file with basic error handling."""

    if not path.exists():
        raise FileNotFoundError(f"Input file {path} does not exist")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def convert_frame(
    frame: pd.DataFrame,
    config: ColumnConfig,
    default_scenario: str,
    default_label: int,
    extra_constant_columns: Mapping[str, str | int | float] | None = None,
) -> pd.DataFrame:
    """Convert a dataset-specific DataFrame to the standard schema."""

    df = frame.copy()

    df["timestamp"] = pd.to_numeric(df[config.timestamp], errors="coerce")
    df["can_id"] = df[config.can_id].apply(_normalise_hex)

    if config.dlc is not None:
        df["dlc"] = pd.to_numeric(df[config.dlc], errors="coerce").fillna(0).astype(int)
    else:
        df["dlc"] = 8

    if config.data_hex:
        df["data_hex"] = df[config.data_hex].apply(_normalise_hex)
    elif config.data_bytes:
        df["data_hex"] = df.apply(lambda row: _pack_bytes(row, config.data_bytes or []), axis=1)
    else:
        df["data_hex"] = ""

    if config.scenario and config.scenario in df.columns:
        scenario_series = df[config.scenario].fillna(default_scenario).astype(str)
    else:
        scenario_series = default_scenario

    if config.label and config.label in df.columns:
        label_series = pd.to_numeric(df[config.label], errors="coerce").fillna(default_label).astype(int)
    else:
        label_series = default_label

    if extra_constant_columns:
        for key, value in extra_constant_columns.items():
            df[key] = value

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "can_id": df["can_id"],
            "dlc": df["dlc"],
            "data_hex": df["data_hex"],
            "scenario": scenario_series,
            "is_attack": label_series,
        }
    )

    out = out.dropna(subset=["timestamp", "can_id"]).reset_index(drop=True)
    out["is_attack"] = out["is_attack"].astype(int)
    return out


def write_raw_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def convert_multiple(
    files: Iterable[Path],
    converter,
    output_dir: Path,
    suffix: str = "_converted.csv",
) -> list[Path]:
    generated: list[Path] = []
    for file_path in files:
        out_df, scenario, label = converter(file_path)
        base_name = f"{file_path.stem}{suffix}" if suffix else f"{file_path.stem}.csv"
        destination = output_dir / base_name
        write_raw_csv(out_df, destination)
        generated.append(destination)
    return generated
