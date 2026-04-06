from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas.tseries.holiday import USFederalHolidayCalendar
from sqlalchemy import create_engine, text


TRIP_START_COLUMN_CANDIDATES = [
    "start_time",
    "start time",
    "startdate",
    "start date",
    "started_at",
    "starttime",
]

MEMBER_TYPE_COLUMN_CANDIDATES = [
    "member_type",
    "member type",
    "usertype",
    "user type",
    "rider_type",
    "rider type",
]

WEATHER_COLUMNS = ["temp", "rhum", "wspd"]
EXPECTED_MEMBER_OTHER_RATIO = 0.05


def expected_hour_count(year: int) -> int:
    return len(
        pd.date_range(
            start=f"{year}-01-01 00:00:00",
            end=f"{year}-12-31 23:00:00",
            freq="h",
        )
    )


def season_code_from_name(name: str) -> int:
    mapping = {
        "spring": 1,
        "summer": 2,
        "fall": 3,
        "winter": 4,
    }
    return mapping[name]


def is_holiday(ts: pd.Timestamp, holidays: pd.DatetimeIndex) -> int:
    return int(ts.normalize() in holidays)


def apparent_temperature(temp_c: pd.Series, rhum: pd.Series, wspd: pd.Series) -> pd.Series:
    # Steadman-style approximation with vapor pressure term.
    vapor_pressure = (rhum / 100.0) * 6.105 * np.exp((17.27 * temp_c) / (237.7 + temp_c))
    return temp_c + (0.33 * vapor_pressure) - (0.70 * wspd) - 4.0


def normalize_column_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def detect_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        match = normalized.get(normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def map_member_bucket(value: object) -> str:
    text_value = str(value).strip().lower()
    if "casual" in text_value:
        return "casual"
    if "member" in text_value or "registered" in text_value or "subscriber" in text_value:
        return "registered"
    return "other"


def aggregate_trip_file(csv_path: Path, chunksize: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
    header = pd.read_csv(csv_path, nrows=0)
    start_col = detect_column(header.columns, TRIP_START_COLUMN_CANDIDATES)
    member_col = detect_column(header.columns, MEMBER_TYPE_COLUMN_CANDIDATES)

    if start_col is None or member_col is None:
        raise ValueError(
            f"Could not detect required columns in {csv_path.name}. "
            f"Detected columns: {list(header.columns)}"
        )

    per_chunk = []
    member_counts = {"casual": 0, "registered": 0, "other": 0}
    usecols = [start_col, member_col]

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk[start_col] = pd.to_datetime(chunk[start_col], errors="coerce")
        chunk = chunk.dropna(subset=[start_col])
        if chunk.empty:
            continue

        chunk["hour"] = chunk[start_col].dt.floor("h")
        chunk["member_class"] = chunk[member_col].map(map_member_bucket)
        counts = chunk["member_class"].value_counts()
        for key in member_counts:
            member_counts[key] += int(counts.get(key, 0))

        grouped = chunk.groupby(["hour", "member_class"], observed=True).size().unstack(fill_value=0)
        grouped = grouped.reindex(columns=["casual", "registered"], fill_value=0)
        per_chunk.append(grouped)

    if not per_chunk:
        return pd.DataFrame(columns=["casual_count", "registered_count"]), {
            "trip_file": csv_path.name,
            "start_column": start_col,
            "member_column": member_col,
            "member_counts": member_counts,
            "other_ratio": 0.0,
        }

    combined = pd.concat(per_chunk).groupby(level=0).sum().sort_index()
    combined = combined.rename(columns={"casual": "casual_count", "registered": "registered_count"})
    combined.index.name = "hour"

    total_classified = sum(member_counts.values())
    other_ratio = member_counts["other"] / total_classified if total_classified else 0.0
    if other_ratio > EXPECTED_MEMBER_OTHER_RATIO:
        raise ValueError(
            f"{csv_path.name} has {other_ratio:.2%} member values mapped to 'other'. "
            "Review member type mapping before continuing."
        )

    audit = {
        "trip_file": csv_path.name,
        "start_column": start_col,
        "member_column": member_col,
        "member_counts": member_counts,
        "other_ratio": round(other_ratio, 6),
    }
    return combined, audit


def load_all_trip_data(trip_dir: Path, chunksize: int, column_audit_path: Optional[Path] = None) -> pd.DataFrame:
    csv_files = sorted(trip_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {trip_dir}")

    file_aggs = []
    audits = []
    for csv_file in csv_files:
        print(f"Aggregating trips from: {csv_file}")
        grouped, audit = aggregate_trip_file(csv_file, chunksize=chunksize)
        file_aggs.append(grouped)
        audits.append(audit)
        print(
            f"Detected columns for {audit['trip_file']}: "
            f"start={audit['start_column']}, member={audit['member_column']}; "
            f"other_ratio={audit['other_ratio']:.2%}"
        )

    if column_audit_path is not None:
        column_audit_path.parent.mkdir(parents=True, exist_ok=True)
        with column_audit_path.open("w", encoding="utf-8") as fp:
            json.dump(audits, fp, indent=2)

    combined = pd.concat(file_aggs).groupby(level=0).sum().sort_index()
    combined = combined.reset_index()
    combined["casual_count"] = combined["casual_count"].astype("int64")
    combined["registered_count"] = combined["registered_count"].astype("int64")
    return combined


def load_noaa_weather(weather_csv: Path, year: int) -> pd.DataFrame:
    """Load NOAA LCD CSV (Reagan National Airport) and return a clean hourly
    weather DataFrame with columns: hour, temp, rhum, wspd,
    weather_quality_flag, weather_source.

    NOAA LCD contains multiple report types per timestamp (FM-15 METAR,
    FM-12 SYNOP, FM-16 SPECI). We keep only FM-15 (routine hourly METAR)
    to get one clean observation per hour, then LEFT JOIN onto the full
    8,760-hour timeline and impute any gaps.
    """
    if not weather_csv.exists():
        raise FileNotFoundError(
            f"NOAA weather CSV not found: {weather_csv}\n"
            "Download hourly LCD data for Reagan National Airport (WBAN 13743) "
            "and place it at the path given by --weather-csv."
        )

    df = pd.read_csv(weather_csv, low_memory=False)

    # Keep only routine hourly METAR reports.
    df = df[df["REPORT_TYPE"].str.strip() == "FM-15"].copy()

    df["hour"] = pd.to_datetime(df["DATE"], errors="coerce").dt.floor("h")
    df = df.dropna(subset=["hour"])

    # Strip trailing flag characters (e.g. "46s" → "46") before numeric conversion.
    def clean_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce",
        )

    df["temp_f"] = clean_numeric(df["HourlyDryBulbTemperature"])
    df["rhum"]   = clean_numeric(df["HourlyRelativeHumidity"])
    df["wspd_mph"] = clean_numeric(df["HourlyWindSpeed"])

    df["temp"] = (df["temp_f"] - 32) * 5 / 9   # °F → °C
    df["wspd"] = df["wspd_mph"] * 1.60934        # mph → km/h
    # One observation per hour — take the first FM-15 within each floored hour.
    df = df.sort_values("hour").drop_duplicates(subset=["hour"], keep="first")

    # LEFT JOIN onto the complete year timeline so every hour is present.
    timeline = pd.DataFrame({
        "hour": pd.date_range(
            start=f"{year}-01-01 00:00:00",
            end=f"{year}-12-31 23:00:00",
            freq="h",
        )
    })
    weather = timeline.merge(df[["hour", "temp", "rhum", "wspd"]], on="hour", how="left")

    weather[WEATHER_COLUMNS] = weather[WEATHER_COLUMNS].apply(pd.to_numeric, errors="coerce")
    raw_missing = weather[WEATHER_COLUMNS].isna()

    interpolated = weather[WEATHER_COLUMNS].interpolate(limit_direction="both")
    interpolated_mask = raw_missing & interpolated.notna()

    filled = interpolated.ffill().bfill()
    forward_backward_mask = raw_missing & ~interpolated_mask & filled.notna()

    before_median_missing = filled.isna()
    for col in WEATHER_COLUMNS:
        if filled[col].isna().all():
            filled[col] = 0.0
        else:
            filled[col] = filled[col].fillna(filled[col].median())
    median_mask = before_median_missing & filled.notna()

    quality_flag = np.zeros(len(filled), dtype=np.int8)
    quality_flag = np.maximum(quality_flag, np.where(interpolated_mask.any(axis=1), 1, 0))
    quality_flag = np.maximum(quality_flag, np.where(forward_backward_mask.any(axis=1), 2, 0))
    quality_flag = np.maximum(quality_flag, np.where(median_mask.any(axis=1), 3, 0))

    weather[WEATHER_COLUMNS] = filled
    weather["weather_quality_flag"] = quality_flag
    weather["weather_source"] = "noaa_lcd"

    return weather[["hour", "temp", "rhum", "wspd", "weather_quality_flag", "weather_source"]]


def build_master_dataset(trips: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    master = weather.merge(trips, on="hour", how="left")
    master["is_imputed_rides"] = master["casual_count"].isna() | master["registered_count"].isna()
    master["casual_count"] = master["casual_count"].fillna(0).astype("int64")
    master["registered_count"] = master["registered_count"].fillna(0).astype("int64")
    master["total_rentals"] = master["casual_count"] + master["registered_count"]

    master["hour_of_day"] = master["hour"].dt.hour.astype("int16")
    master["day_of_week"] = master["hour"].dt.dayofweek.astype("int16")
    master["is_weekend"] = (master["day_of_week"] >= 5).astype("int8")
    master["month"] = master["hour"].dt.month.astype("int16")
    master["season"] = master["month"].map(season_from_month)
    master["season_code"] = master["season"].map(season_code_from_name).astype("int8")
    master["time_of_day"] = master["hour_of_day"].map(time_of_day_bucket)
    master["quarter"] = master["hour"].dt.quarter.map(lambda q: f"Q{q}")

    holiday_calendar = USFederalHolidayCalendar()
    holidays = holiday_calendar.holidays(
        start=master["hour"].min().normalize(),
        end=master["hour"].max().normalize(),
    )
    master["holiday"] = master["hour"].map(lambda ts: is_holiday(ts, holidays)).astype("int8")
    master["workingday"] = ((master["is_weekend"] == 0) & (master["holiday"] == 0)).astype("int8")
    master["atemp"] = apparent_temperature(master["temp"], master["rhum"], master["wspd"])

    if "weather_quality_flag" not in master.columns:
        master["weather_quality_flag"] = 0
    master["weather_quality_flag"] = master["weather_quality_flag"].astype("int8")
    if "weather_source" not in master.columns:
        master["weather_source"] = "unknown"
    master["run_generated_at"] = pd.Timestamp.now("UTC").isoformat()
    master["is_imputed_rides"] = master["is_imputed_rides"].astype("int8")

    return master.sort_values("hour").reset_index(drop=True)


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def time_of_day_bucket(hour: int) -> str:
    if 6 <= hour < 10:
        return "morning_rush"
    if 10 <= hour < 16:
        return "midday"
    if 16 <= hour < 20:
        return "evening_rush"
    return "night"


def create_quality_report(df: pd.DataFrame, year: int) -> Dict[str, Any]:
    expected_rows = expected_hour_count(year)
    duplicates = int(df["hour"].duplicated().sum())
    nulls = {col: int(value) for col, value in df.isna().sum().items()}

    range_checks = {
        "temp": {
            "min": float(df["temp"].min()),
            "max": float(df["temp"].max()),
            "in_range": bool(df["temp"].between(-40, 50).all()),
        },
        "rhum": {
            "min": float(df["rhum"].min()),
            "max": float(df["rhum"].max()),
            "in_range": bool(df["rhum"].between(0, 100).all()),
        },
        "wspd": {
            "min": float(df["wspd"].min()),
            "max": float(df["wspd"].max()),
            "in_range": bool((df["wspd"] >= 0).all()),
        },
    }

    warnings = []
    if int(len(df)) != expected_rows:
        warnings.append(f"Expected {expected_rows} rows for {year}, got {len(df)}")
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate hours")
    for metric, details in range_checks.items():
        if not details["in_range"]:
            warnings.append(f"{metric} has values outside expected range")

    status = "pass" if not warnings else "warn"
    return {
        "status": status,
        "year": year,
        "row_count": int(len(df)),
        "expected_row_count": expected_rows,
        "duplicate_hours": duplicates,
        "missing_by_column": nulls,
        "range_checks": range_checks,
        "weather_quality_flag_distribution": {
            str(k): int(v) for k, v in df["weather_quality_flag"].value_counts(dropna=False).sort_index().items()
        },
        "warnings": warnings,
    }


def save_quality_report(report: Dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)


def load_to_postgresql(df: pd.DataFrame, db_url: str, table_name: str, if_exists: str) -> None:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    upload_df = df.copy()
    upload_df["is_weekend"] = upload_df["is_weekend"].astype(bool)
    upload_df["holiday"] = upload_df["holiday"].astype(bool)
    upload_df["workingday"] = upload_df["workingday"].astype(bool)
    upload_df["is_imputed_rides"] = upload_df["is_imputed_rides"].astype(bool)

    upload_df.to_sql(
        table_name,
        con=engine,
        index=False,
        if_exists=if_exists,
        chunksize=2000,
        method="multi",
    )
    engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bike-sharing ETL + weather join pipeline.")
    parser.add_argument(
        "--trip-dir",
        type=str,
        default="raw/2017-capitalbikeshare-tripdata",
        help="Directory containing quarterly Capital Bikeshare CSV files.",
    )
    parser.add_argument("--year", type=int, default=2017, help="Year to process.")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="CSV chunksize used while aggregating trip files.",
    )
    parser.add_argument(
        "--weather-csv",
        type=str,
        default="raw/2017-DC-Hourly-Weather.csv",
        help="Path to the raw NOAA LCD hourly weather CSV (Reagan National Airport).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/hourly_bike_weather_2017.csv",
        help="Path for final joined hourly dataset CSV.",
    )
    parser.add_argument(
        "--quality-report",
        type=str,
        default="outputs/tables/pipeline/data_quality_report.json",
        help="Path for ETL data quality report JSON.",
    )
    parser.add_argument(
        "--column-audit",
        type=str,
        default="outputs/tables/pipeline/column_detection_log.json",
        help="Path for trip column detection audit JSON.",
    )
    parser.add_argument(
        "--load-db",
        action="store_true",
        help="If set, load final dataset to PostgreSQL.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="SQLAlchemy DB URL. If omitted, DATABASE_URL env var is used.",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default=os.getenv("DB_TABLE_NAME", "bike_demand_hourly_2017"),
        help="PostgreSQL table name for loading final dataset.",
    )
    parser.add_argument(
        "--if-exists",
        type=str,
        default="replace",
        choices=["fail", "replace", "append"],
        help="Behavior when table exists during to_sql.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    trip_dir = Path(args.trip_dir)
    weather_csv = Path(args.weather_csv)
    output_csv = Path(args.output_csv)
    quality_report_path = Path(args.quality_report)
    column_audit_path = Path(args.column_audit)

    cleaned_dir = Path("outputs/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    trips = load_all_trip_data(
        trip_dir=trip_dir,
        chunksize=args.chunksize,
        column_audit_path=column_audit_path,
    )
    trips_out = cleaned_dir / "trips_2017_hourly_cleaned.csv"
    trips.to_csv(trips_out, index=False)
    print(f"Saved cleaned trip counts to: {trips_out}")

    weather = load_noaa_weather(weather_csv=weather_csv, year=args.year)
    weather_out = cleaned_dir / "weather_2017_hourly_cleaned.csv"
    weather.to_csv(weather_out, index=False)
    print(f"Saved cleaned weather to: {weather_out}")

    master = build_master_dataset(trips=trips, weather=weather)
    quality_report = create_quality_report(master, year=args.year)
    save_quality_report(quality_report, quality_report_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_csv, index=False)

    print(f"Final dataset shape: {master.shape}")
    print(f"Saved final dataset to: {output_csv}")
    print(f"Saved quality report to: {quality_report_path}")
    print(f"Saved column detection audit to: {column_audit_path}")

    if args.load_db:
        db_url = args.db_url or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("No database URL provided. Set DATABASE_URL or pass --db-url.")

        load_to_postgresql(
            df=master,
            db_url=db_url,
            table_name=args.table_name,
            if_exists=args.if_exists,
        )
        print(f"Loaded dataset into PostgreSQL table: {args.table_name}")


if __name__ == "__main__":
    main()
