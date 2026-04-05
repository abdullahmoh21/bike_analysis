# Bike Sharing Demand and Weather Analysis

This project builds a full 2017 hourly bike-demand dataset by combining:

- Capital Bikeshare raw trip history (4 quarterly CSV files)
- NOAA weather data via the Meteostat API

It then performs EDA and trains two predictive models:

- Baseline: Linear Regression
- Improved: Random Forest Regressor

## Project Structure

```
bike_analysis/
  2017-capitalbikeshare-tripdata/
  data/
  outputs/
    figures/
    models/
    tables/
  sql/
    create_hourly_table.sql
  src/
    pipeline.py
    eda.py
    modeling.py
  requirements.txt
  .env.example
```

## 1. Setup

Create and activate a virtual environment, then install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: create your local environment file.

```bash
cp .env.example .env
```

If you want PostgreSQL loading, edit `DATABASE_URL` in `.env`.

## 2. Build the 2017 Hourly Master Dataset (ETL)

Run the ETL pipeline:

```bash
python src/pipeline.py \
  --trip-dir 2017-capitalbikeshare-tripdata \
  --year 2017 \
  --output-csv data/hourly_bike_weather_2017.csv
```

What it does:

- Aggregates raw rides to hourly counts (`casual_count`, `registered_count`)
- Fetches hourly weather for D.C. (`temp`, `rhum`, `wspd`, `coco`)
- LEFT JOINs bike counts onto the continuous 8,760-hour timeline
- Imputes null bike counts with zero (`fillna(0)` behavior)
- Adds engineered time + proposal-parity features (`season`, `season_code`, `is_weekend`, `time_of_day`, `holiday`, `workingday`, `atemp`)
- Writes ETL validation artifacts (`outputs/tables/data_quality_report.json`, `outputs/tables/column_detection_log.json`)

Optional ETL controls:

```bash
python src/pipeline.py --force-refresh-weather
```

This bypasses weather cache reuse and fetches fresh weather observations.

## 3. Load into PostgreSQL (AWS RDS)

If `DATABASE_URL` is set in `.env`, run:

```bash
python src/pipeline.py \
  --trip-dir 2017-capitalbikeshare-tripdata \
  --load-db \
  --table-name bike_demand_hourly_2017 \
  --if-exists replace
```

SQL schema and indexes are provided in:

- `sql/create_hourly_table.sql`

## 4. Run Exploratory Data Analysis

```bash
python src/eda.py --input-csv data/hourly_bike_weather_2017.csv
```

Outputs:

- Tables in `outputs/tables/`
- Charts in `outputs/figures/`

## 5. Train Baseline and Improved Models

```bash
python src/modeling.py --input-csv data/hourly_bike_weather_2017.csv
```

Evaluation metrics reported:

- RMSE
- MAE
- R^2
- Time-series cross-validation summary (`cv_rmse_mean`, `cv_mae_mean`, `cv_r2_mean`)

Evaluation methodology:

- Chronological split (first 80% hours for train, last 20% hours for test)
- TimeSeriesSplit cross-validation on training segment

Artifacts:

- `outputs/models/baseline_linear_regression.joblib`
- `outputs/models/improved_random_forest.joblib`
- `outputs/tables/model_metrics.json`
- `outputs/tables/model_metrics.csv`
- `outputs/tables/test_predictions.csv`
- `outputs/tables/model_residuals.csv`
- `outputs/tables/random_forest_feature_importance.csv`
- `outputs/figures/model_actual_vs_predicted.png`
- `outputs/figures/model_residuals_vs_predicted.png`

Additional EDA artifacts:

- `outputs/tables/casual_vs_registered_hourly.csv`
- `outputs/tables/casual_vs_registered_by_day.csv`
- `outputs/tables/weather_impact_by_user_type.csv`
- `outputs/tables/temperature_band_by_time_of_day.csv`
- `outputs/figures/casual_vs_registered_by_hour.png`
- `outputs/figures/casual_vs_registered_by_day.png`
- `outputs/figures/weather_effect_by_user_type.png`
- `outputs/figures/temperature_band_by_time_of_day.png`

## Notes

- The pipeline is chunked while reading trip CSVs for memory efficiency.
- Weather cache is stored at `data/weather_2017_hourly.csv` after first fetch.
- This implementation focuses on weather/time features only and does not include event metadata.