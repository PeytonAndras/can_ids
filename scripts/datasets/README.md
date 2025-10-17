# Dataset ingestion helpers

Utilities for converting popular CAN intrusion datasets into the repository's canonical
`data/raw` schema (`timestamp, can_id, dlc, data_hex, scenario, is_attack`).

## Available adapters

| Dataset | Script | Notes |
| --- | --- | --- |
| HCRL Car-Hacking | `ingest_hcrl.py` | Handles DoS, Fuzzy, Gear, RPM folders. Scenario is inferred from file name. |
| ROAD (ORNL) | `ingest_road.py` | Works with CSV exports (e.g. `train.csv`, `test.csv`) that include label/attack columns. |
| OTIDS | `ingest_otids.py` | Accepts a single CSV or directory containing OTIDS captures. |

Each script normalises column names, merges byte-wise payload columns, and produces
CSV files under `data/raw/<dataset_name>/`. A `manifest.txt` file is emitted listing the
generated files to simplify bookkeeping.

## Usage examples

```bash
# HCRL (after extracting the dataset archive)
python scripts/datasets/ingest_hcrl.py /path/to/car-hacking-dataset --output data/raw/hcrl

# ROAD dataset (directory containing train/test CSV files)
python scripts/datasets/ingest_road.py /path/to/road_csv --output data/raw/road

# OTIDS dataset (single CSV)
python scripts/datasets/ingest_otids.py /path/to/OTIDS.csv --output data/raw/otids
```

The scripts require the raw dataset archives to be downloaded manually due to licensing.
Consult the paper/hosted download page for each dataset for access instructions.

## Implementation details

All adapters share the utilities in `common.py` which

- normalise column names (lowercase, snake_case)
- combine byte-wise payload columns into a `data_hex` string
- coerce timestamps to floating-point seconds
- annotate each row with a `scenario` string and an `is_attack` label

If a dataset exposes additional metadata, it can be added later by extending the
`ColumnConfig` in the adapter script.
