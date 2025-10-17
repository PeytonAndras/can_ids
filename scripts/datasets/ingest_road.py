#!/usr/bin/env python3
"""Convert the ROAD (ORNL) dataset into the common raw schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    ColumnConfig,
    convert_frame,
    find_column,
    load_dataset_csv,
    normalise_columns,
    write_raw_csv,
)

LABEL_CANDIDATES = ("label", "attack", "is_attack", "class")
SCENARIO_CANDIDATES = ("scenario", "attack_type", "category")


def convert_road_file(path: Path, output_dir: Path) -> Path:
    df = load_dataset_csv(path)
    df = normalise_columns(df)

    timestamp_col = find_column(df, ("timestamp", "time"))
    can_id_col = find_column(df, ("can_id", "canid", "id"))
    dlc_col = find_column(df, ("dlc",))

    if "data" in df.columns:
        data_hex_col = "data"
        byte_columns = None
    else:
        byte_columns = tuple(col for col in df.columns if col.startswith("data_"))
        data_hex_col = None

    try:
        label_col = find_column(df, LABEL_CANDIDATES)
    except KeyError:
        label_col = None

    try:
        scenario_col = find_column(df, SCENARIO_CANDIDATES)
    except KeyError:
        scenario_col = None

    config = ColumnConfig(
        timestamp=timestamp_col,
        can_id=can_id_col,
        dlc=dlc_col,
        data_hex=data_hex_col,
        data_bytes=byte_columns,
        scenario=scenario_col,
        label=label_col,
    )

    scenario = path.stem
    converted = convert_frame(
        df,
        config,
        default_scenario=scenario,
        default_label=0,
    )

    destination = output_dir / f"{path.stem}.csv"
    write_raw_csv(converted, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the ROAD dataset")
    parser.add_argument("root", type=Path, help="Path containing ROAD CSV files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/road"),
        help="Directory to store normalised CSV files",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Dataset root {args.root} not found")

    csv_files = sorted(p for p in args.root.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files discovered under {args.root}")

    args.output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for csv_path in csv_files:
        try:
            destination = convert_road_file(csv_path, args.output)
            written.append(destination)
            print(f"[+] Converted {csv_path.name} -> {destination}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[!] Failed to convert {csv_path}: {exc}")

    if written:
        manifest_path = args.output / "manifest.txt"
        manifest_path.write_text("\n".join(str(path) for path in written))
        print(f"[✓] Wrote {len(written)} files. Manifest: {manifest_path}")
    else:
        print("[!] No files were converted. Check dataset paths and format.")


if __name__ == "__main__":
    main()
