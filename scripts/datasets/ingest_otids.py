#!/usr/bin/env python3
"""Convert the OTIDS CAN dataset to the common raw schema."""

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


def convert_otids_file(path: Path, output_dir: Path, scenario_override: str | None = None) -> Path:
    df = load_dataset_csv(path)
    df = normalise_columns(df)

    timestamp_col = find_column(df, ("timestamp", "time"))
    can_id_col = find_column(df, ("can_id", "id", "message_id"))

    dlc_candidates = [col for col in df.columns if col.startswith("dlc") or col == "length"]
    dlc_col = dlc_candidates[0] if dlc_candidates else None

    if "data" in df.columns:
        data_hex_col = "data"
        data_bytes = None
    else:
        data_candidates = [col for col in df.columns if col.startswith("data")]
        data_hex_col = None
        data_bytes = tuple(data_candidates) if data_candidates else None

    try:
        label_col = find_column(df, LABEL_CANDIDATES)
    except KeyError:
        label_col = None

    scenario = scenario_override or path.stem

    config = ColumnConfig(
        timestamp=timestamp_col,
        can_id=can_id_col,
        dlc=dlc_col,
        data_hex=data_hex_col,
        data_bytes=data_bytes,
        scenario=None,
        label=label_col,
    )

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
    parser = argparse.ArgumentParser(description="Ingest the OTIDS dataset")
    parser.add_argument("path", type=Path, help="Path to an OTIDS CSV file or directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/otids"),
        help="Directory to store normalised CSV files",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional scenario label to use for all generated files",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"Provided path {args.path} does not exist")

    if args.path.is_file():
        csv_files = [args.path]
    else:
        csv_files = sorted(p for p in args.path.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files discovered for OTIDS under {args.path}")

    args.output.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for csv_path in csv_files:
        try:
            destination = convert_otids_file(csv_path, args.output, scenario_override=args.scenario)
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
