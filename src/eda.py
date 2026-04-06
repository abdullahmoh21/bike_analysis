from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DAY_LABELS = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for hourly bike demand dataset.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/hourly_bike_weather_2017.csv",
        help="Joined hourly dataset produced by pipeline.py.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="outputs/figures",
        help="Directory where figures will be saved.",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="outputs/tables/eda",
        help="Directory where summary tables will be saved.",
    )
    return parser.parse_args()


def save_and_close(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    df = pd.read_csv(input_csv, parse_dates=["hour"])

    numeric_summary = df.select_dtypes(include="number").describe().T
    numeric_summary.to_csv(tables_dir / "numeric_summary.csv")

    missing_summary = df.isna().sum().rename("missing_count")
    missing_summary.to_csv(tables_dir / "missing_summary.csv")

    demand_by_hour = df.groupby("hour_of_day", as_index=False)["total_rentals"].mean()
    demand_by_hour.to_csv(tables_dir / "demand_by_hour.csv", index=False)

    demand_by_day = df.groupby("day_of_week", as_index=False)["total_rentals"].mean()
    demand_by_day["day_label"] = demand_by_day["day_of_week"].map(DAY_LABELS)
    demand_by_day.to_csv(tables_dir / "demand_by_day_of_week.csv", index=False)

    weekday_weekend = df.groupby(["is_weekend", "hour_of_day"], as_index=False)["total_rentals"].mean()
    weekday_weekend["day_type"] = weekday_weekend["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    weekday_weekend.to_csv(tables_dir / "weekday_vs_weekend_hourly.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=weekday_weekend,
        x="hour_of_day",
        y="total_rentals",
        hue="day_type",
        marker="o",
    )
    plt.title("Weekday vs Weekend Rental Pattern")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Rentals")
    save_and_close(figures_dir / "weekday_vs_weekend_pattern.png")

    season_user_mix = (
        df.groupby("season", as_index=False)[["casual_count", "registered_count"]]
        .sum()
        .sort_values("season")
    )
    season_user_mix.to_csv(tables_dir / "seasonal_user_mix.csv", index=False)

    season_plot = season_user_mix.set_index("season")
    season_plot.plot(kind="bar", stacked=True, figsize=(9, 5), color=["#70a1d7", "#374c80"])
    plt.title("Casual vs Registered Rentals by Season")
    plt.xlabel("Season")
    plt.ylabel("Total Rentals")
    save_and_close(figures_dir / "seasonal_user_mix.png")

    corr_cols = [
        "total_rentals",
        "casual_count",
        "registered_count",
        "temp",
        "rhum",
        "wspd",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
    ]
    corr_matrix = df[corr_cols].corr(numeric_only=True)
    corr_matrix.to_csv(tables_dir / "correlation_matrix.csv")

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title("Correlation Matrix")
    save_and_close(figures_dir / "correlation_heatmap.png")

    # ── Distribution plots ────────────────────────────────────────────────
    # Rental demand distribution: reveals strong right skew (most hours are
    # low-demand; a small number of peak hours drive the tail).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["total_rentals"], bins=60, kde=True, ax=axes[0], color="#374c80")
    axes[0].set_title("Distribution of Hourly Total Rentals")
    axes[0].set_xlabel("Total Rentals per Hour")
    axes[0].set_ylabel("Frequency")

    season_order = ["spring", "summer", "fall", "winter"]
    sns.boxplot(
        data=df,
        x="season",
        y="total_rentals",
        order=season_order,
        hue="season",
        legend=False,
        palette="Blues",
        ax=axes[1],
    )
    axes[1].set_title("Hourly Rental Distribution by Season")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Total Rentals per Hour")
    save_and_close(figures_dir / "rental_distributions.png")

    # Weather variable distributions: confirms temp is roughly bimodal
    # (reflecting D.C.'s cold winters and hot summers), rhum is left-skewed,
    # and wind speed is right-skewed — justifying median imputation choices.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(df["temp"].dropna(), bins=40, kde=True, ax=axes[0], color="#5b8db8")
    axes[0].set_title("Temperature Distribution")
    axes[0].set_xlabel("Temperature (C)")
    axes[0].set_ylabel("Hours")

    sns.histplot(df["rhum"].dropna(), bins=40, kde=True, ax=axes[1], color="#5b8db8")
    axes[1].set_title("Relative Humidity Distribution")
    axes[1].set_xlabel("Relative Humidity (%)")
    axes[1].set_ylabel("Hours")

    sns.histplot(df["wspd"].dropna(), bins=40, kde=True, ax=axes[2], color="#5b8db8")
    axes[2].set_title("Wind Speed Distribution")
    axes[2].set_xlabel("Wind Speed (km/h)")
    axes[2].set_ylabel("Hours")
    save_and_close(figures_dir / "weather_variable_distributions.png")

    casual_registered_hourly = (
        df.groupby("hour_of_day", as_index=False)[["casual_count", "registered_count"]]
        .mean()
        .rename(columns={"casual_count": "casual_avg", "registered_count": "registered_avg"})
    )
    total_user = casual_registered_hourly["casual_avg"] + casual_registered_hourly["registered_avg"]
    casual_registered_hourly["casual_pct"] = np.where(total_user > 0, casual_registered_hourly["casual_avg"] / total_user, 0.0)
    casual_registered_hourly.to_csv(tables_dir / "casual_vs_registered_hourly.csv", index=False)

    user_hour_plot = casual_registered_hourly.melt(
        id_vars="hour_of_day",
        value_vars=["casual_avg", "registered_avg"],
        var_name="user_type",
        value_name="avg_rentals",
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=user_hour_plot, x="hour_of_day", y="avg_rentals", hue="user_type", marker="o")
    plt.title("Casual vs Registered Rentals by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Rentals")
    save_and_close(figures_dir / "casual_vs_registered_by_hour.png")

    casual_registered_day = (
        df.groupby("day_of_week", as_index=False)[["casual_count", "registered_count"]]
        .mean()
        .rename(columns={"casual_count": "casual_avg", "registered_count": "registered_avg"})
    )
    casual_registered_day["day_label"] = casual_registered_day["day_of_week"].map(DAY_LABELS)
    casual_registered_day.to_csv(tables_dir / "casual_vs_registered_by_day.csv", index=False)

    day_plot = casual_registered_day.melt(
        id_vars="day_label",
        value_vars=["casual_avg", "registered_avg"],
        var_name="user_type",
        value_name="avg_rentals",
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=day_plot, x="day_label", y="avg_rentals", hue="user_type")
    plt.title("Casual vs Registered Rentals by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average Rentals")
    save_and_close(figures_dir / "casual_vs_registered_by_day.png")

    weather_impact_table = pd.DataFrame(
        {
            "variable": ["temp", "rhum", "wspd"],
            "casual_corr": [
                df["casual_count"].corr(df["temp"]),
                df["casual_count"].corr(df["rhum"]),
                df["casual_count"].corr(df["wspd"]),
            ],
            "registered_corr": [
                df["registered_count"].corr(df["temp"]),
                df["registered_count"].corr(df["rhum"]),
                df["registered_count"].corr(df["wspd"]),
            ],
            "total_corr": [
                df["total_rentals"].corr(df["temp"]),
                df["total_rentals"].corr(df["rhum"]),
                df["total_rentals"].corr(df["wspd"]),
            ],
        }
    )
    weather_impact_table.to_csv(tables_dir / "weather_impact_by_user_type.csv", index=False)

    temp_bins = pd.cut(
        df["temp"],
        bins=[-50, 0, 10, 20, 30, 60],
        labels=["<0C", "0-10C", "10-20C", "20-30C", ">=30C"],
    )
    temp_bin_table = (
        df.assign(temp_band=temp_bins)
        .groupby(["temp_band", "time_of_day"], as_index=False)["total_rentals"]
        .mean()
    )
    temp_bin_table.to_csv(tables_dir / "temperature_band_by_time_of_day.csv", index=False)

    plt.figure(figsize=(11, 5))
    sns.barplot(data=temp_bin_table, x="temp_band", y="total_rentals", hue="time_of_day")
    plt.title("Average Rentals by Temperature Band and Time of Day")
    plt.xlabel("Temperature Band")
    plt.ylabel("Average Rentals")
    save_and_close(figures_dir / "temperature_band_by_time_of_day.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    weather_axes = [("temp", "Temperature (C)"), ("rhum", "Relative Humidity"), ("wspd", "Wind Speed (km/h)")]

    for idx, (feature, label) in enumerate(weather_axes):
        sns.regplot(
            data=df,
            x=feature,
            y="casual_count",
            scatter_kws={"alpha": 0.25, "s": 12},
            ax=axes[0, idx],
        )
        axes[0, idx].set_title(f"Casual vs {label}")
        axes[0, idx].set_xlabel(label)
        axes[0, idx].set_ylabel("Casual Rentals")

        sns.regplot(
            data=df,
            x=feature,
            y="registered_count",
            scatter_kws={"alpha": 0.25, "s": 12},
            ax=axes[1, idx],
        )
        axes[1, idx].set_title(f"Registered vs {label}")
        axes[1, idx].set_xlabel(label)
        axes[1, idx].set_ylabel("Registered Rentals")

    save_and_close(figures_dir / "weather_effect_by_user_type.png")

    print("EDA complete.")
    print(f"Saved tables to: {tables_dir}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()