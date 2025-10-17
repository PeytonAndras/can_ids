#!/usr/bin/env python3
"""Convert the HCRL Car-Hacking dataset into the common raw schema."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

try:  # pragma: no cover - allow running as a script
    from .common import (
        ColumnConfig,
        convert_frame,
        find_column,
        load_dataset_csv,
        normalise_columns,
        write_raw_csv,
    )
except ImportError:  # pragma: no cover
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from scripts.datasets.common import (  # type: ignore
        ColumnConfig,
        convert_frame,
        find_column,
        load_dataset_csv,
        normalise_columns,
        write_raw_csv,
    )

DATA_SUBDIRS_ATTACK = {
    "dos": 1,
    "fuzzy": 1,
    "gear": 1,
    "rpm": 1,
    "spoof": 1,
}
NORMAL_KEYWORDS = {"normal", "baseline", "benign"}


def infer_scenario(path: Path) -> tuple[str, int]:
    lowered = str(path).lower()
    stem = path.stem

    for keyword in NORMAL_KEYWORDS:
        if keyword in lowered:
            return stem, 0
    for keyword, label in DATA_SUBDIRS_ATTACK.items():
        if keyword in lowered:
            return stem, label
    return stem, 0


def convert_with_header(path: Path, scenario: str, label: int) -> pd.DataFrame | None:
    try:
        df = load_dataset_csv(path)
    except Exception:
        return None

    if not all(isinstance(col, str) for col in df.columns):
        return None

    df = normalise_columns(df)
    try:
        timestamp_col = find_column(df, ("timestamp", "time", "t"))
        can_id_col = find_column(df, ("can_id", "id", "canid"))
        dlc_col = find_column(df, ("dlc",))
    except KeyError:
        return None

    byte_columns: Iterable[str] = [
        col
        for col in df.columns
        if col.startswith("data") and col not in {"data", "data_hex", "dataset"}
    ]

    data_hex_col = "data" if "data" in df.columns and not byte_columns else None

    config = ColumnConfig(
        timestamp=timestamp_col,
        can_id=can_id_col,
        dlc=dlc_col,
        data_hex=data_hex_col,
        data_bytes=tuple(byte_columns) if not data_hex_col else None,
    )

    return convert_frame(
        df,
        config,
        default_scenario=scenario,
        default_label=label,
    )


def convert_headerless(path: Path, scenario: str, label: int) -> pd.DataFrame:
    records = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            try:
                timestamp = float(row[0])
            except (ValueError, IndexError):
                continue
            try:
                can_id_raw = row[1].strip()
                if can_id_raw.lower().startswith("0x"):
                    can_id_raw = can_id_raw[2:]
                can_id = can_id_raw.lower()
            except IndexError:
                continue
            try:
                dlc_raw = row[2].strip()
                dlc = int(dlc_raw, 16) if dlc_raw.lower().startswith("0x") else int(dlc_raw)
            except (IndexError, ValueError):
                dlc = 0

            payload_segments: list[str] = []
            for value in row[3 : 3 + dlc]:
                cleaned = value.strip().lower()
                if not cleaned:
                    continue
                if cleaned.startswith("0x"):
                    cleaned = cleaned[2:]
                try:
                    payload_segments.append(f"{int(cleaned, 16):02x}")
                except ValueError:
                    try:
                        payload_segments.append(f"{int(float(cleaned)):02x}")
                    except (TypeError, ValueError):
                        continue

            data_hex = "".join(payload_segments)

            scenario_flag = None
            extra_index = 3 + max(dlc, 0)
            if extra_index < len(row):
                scenario_flag = row[extra_index].strip()

            scenario_value = f"{scenario}_{scenario_flag}" if scenario_flag else scenario

            records.append(
                {
                    "timestamp": timestamp,
                    "can_id": can_id,
                    "dlc": dlc,
                    "data_hex": data_hex,
                    "scenario": scenario_value,
                    "is_attack": label,
                }
            )

    return pd.DataFrame(records)


def transform_file(path: Path, output_dir: Path) -> Path:
    scenario, label = infer_scenario(path)
    converted = convert_with_header(path, scenario, label)
    if converted is None:
        converted = convert_headerless(path, scenario, label)

    destination = output_dir / f"{path.stem}.csv"
    write_raw_csv(converted, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the HCRL Car-Hacking dataset")
    parser.add_argument("root", type=Path, help="Path to the extracted car-hacking dataset root")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/hcrl"),
        help="Directory to write normalised CSV files",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Dataset root {args.root} not found")

    csv_files = sorted(p for p in args.root.rglob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files discovered under {args.root}")

    args.output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for csv_path in csv_files:
        try:
            destination = transform_file(csv_path, args.output)
            written.append(destination)
            print(f"[+] Converted {csv_path.relative_to(args.root)} -> {destination}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[!] Skipping {csv_path}: {exc}")

    if written:
        manifest_path = args.output / "manifest.txt"
        manifest_path.write_text("\n".join(str(path) for path in written))
        print(f"[✓] Wrote {len(written)} files. Manifest: {manifest_path}")
    else:
        print("[!] No files were converted. Check dataset paths and format.")


if __name__ == "__main__":
    main()
