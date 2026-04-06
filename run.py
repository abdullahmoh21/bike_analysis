#!/usr/bin/env python3
"""Full pipeline runner: ETL → EDA → Modeling.

Usage:
    python run.py              # ETL + EDA + Modeling
    python run.py --load-db    # same, plus load to PostgreSQL (requires DATABASE_URL in .env)
"""

from __future__ import annotations

import argparse
import subprocess
import sys

TRIP_DIR = "raw/2017-capitalbikeshare-tripdata"
YEAR = 2017
OUTPUT_CSV = "data/hourly_bike_weather_2017.csv"
TABLE_NAME = "bike_demand_hourly_2017"


def run(label: str, cmd: list[str]) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: '{label}' failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full bike analysis pipeline.")
    parser.add_argument(
        "--load-db",
        action="store_true",
        help="Load the cleaned dataset into PostgreSQL after ETL (requires DATABASE_URL in .env)",
    )
    args = parser.parse_args()

    python = sys.executable

    etl_cmd = [
        python, "src/pipeline.py",
        "--trip-dir", TRIP_DIR,
        "--year", str(YEAR),
        "--output-csv", OUTPUT_CSV,
    ]
    if args.load_db:
        etl_cmd += ["--load-db", "--table-name", TABLE_NAME, "--if-exists", "replace"]

    run("Step 1/3 — ETL (pipeline.py)", etl_cmd)
    run("Step 2/3 — EDA (eda.py)", [python, "src/eda.py", "--input-csv", OUTPUT_CSV])
    run("Step 3/3 — Modeling (modeling.py)", [python, "src/modeling.py", "--input-csv", OUTPUT_CSV])

    print(f"\n{'─' * 60}")
    print("  All steps complete.")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
