# Data Dictionary

Master dataset: `data/hourly_bike_weather_2017.csv`  
PostgreSQL table: `bike_demand_hourly_2017`  
Grain: one row per hour of 2017 (8,760 rows total)

---

## Source Columns

These columns come directly from external sources with minimal transformation.

| Column | Type | Source | Description |
|--------|------|---------|-------------|
| `hour` | TIMESTAMP | Capital Bikeshare trip CSVs / timeline | The hour timestamp (tz-naive, America/New_York). Built as a continuous 8,760-hour timeline; trip counts are joined onto it, so every hour of the year is present even if zero rides occurred. |
| `temp` | FLOAT | Meteostat API | Dry-bulb air temperature in degrees Celsius at Reagan National Airport (lat 38.8512, lon -77.0402). |
| `rhum` | FLOAT | Meteostat API | Relative humidity (%). |
| `wspd` | FLOAT | Meteostat API | Wind speed in km/h. |
| `coco` | FLOAT | Meteostat API | Meteostat weather condition code (e.g., 1 = clear, 4 = overcast). Sparse in 2017 data; not used in models. |
| `casual_count` | INTEGER | Capital Bikeshare trip CSVs | Number of rides started in this hour by casual (non-member) riders. Hours with no trip records are filled with 0. |
| `registered_count` | INTEGER | Capital Bikeshare trip CSVs | Number of rides started in this hour by registered (member / subscriber) riders. Hours with no trip records are filled with 0. |

---

## ETL-Derived Columns

These columns are created during cleaning and joining in `src/pipeline.py`.

| Column | Type | Transformation | Description |
|--------|------|----------------|-------------|
| `total_rentals` | INTEGER | `casual_count + registered_count` | Total rides in the hour. This is the prediction target used in modeling. |
| `is_imputed_rides` | INT (0/1) | 1 if hour was absent from trip CSVs | Flag indicating the hour had no trip records. In these hours `casual_count` and `registered_count` are 0 by construction, not observed zeros. |
| `weather_quality_flag` | INTEGER | Set during weather imputation | Tracks how each hour's weather was filled: **0** = directly observed, **1** = linear interpolation between neighbors, **2** = forward/backward fill, **3** = column median (last resort). |
| `weather_source` | STRING | Set during weather fetch | Either `"meteostat_api"` (freshly fetched) or `"weather_cache_legacy"` (loaded from cached CSV). |
| `run_generated_at` | STRING | `pd.Timestamp.now("UTC").isoformat()` | UTC timestamp of when the pipeline was run. Useful for auditing dataset versions. |

---

## Engineered Features

These columns are derived from the timestamp or weather values to support EDA and modeling.

| Column | Type | Logic | Description |
|--------|------|-------|-------------|
| `hour_of_day` | INTEGER | `hour.dt.hour` | Hour of the day (0–23). |
| `day_of_week` | INTEGER | `hour.dt.dayofweek` | Day of week (0 = Monday, 6 = Sunday). |
| `month` | INTEGER | `hour.dt.month` | Calendar month (1–12). |
| `quarter` | STRING | `hour.dt.quarter` | Quarter label: `"Q1"`, `"Q2"`, `"Q3"`, `"Q4"`. |
| `is_weekend` | INT (0/1) | `day_of_week >= 5` | 1 if the hour falls on Saturday or Sunday, else 0. |
| `season` | STRING | Month-based: Dec–Feb = winter, Mar–May = spring, Jun–Aug = summer, Sep–Nov = fall | Named season. Used as a categorical feature in both models. |
| `season_code` | INTEGER | spring=1, summer=2, fall=3, winter=4 | Numeric encoding of season for storage and sorting. Not used directly in models (season string is one-hot encoded instead). |
| `time_of_day` | STRING | hour_of_day 6–9 = `morning_rush`, 10–15 = `midday`, 16–19 = `evening_rush`, 20–5 = `night` | Bucketed time period. Motivated by EDA showing registered riders have sharp commute peaks while casual riders peak midday. Used as a categorical feature in the improved model. |
| `holiday` | INT (0/1) | `USFederalHolidayCalendar` (pandas) | 1 if the date is a U.S. federal holiday, else 0. |
| `workingday` | INT (0/1) | `is_weekend == 0 AND holiday == 0` | 1 if the hour is on a day that is neither a weekend nor a federal holiday. Used in both models. |
| `atemp` | FLOAT | Steadman approximation: `temp + (0.33 × vapor_pressure) − (0.70 × wspd) − 4.0`, where `vapor_pressure = (rhum/100) × 6.105 × exp(17.27 × temp / (237.7 + temp))` | Apparent ("feels like") temperature in °C. Combines temperature, humidity, and wind to reflect perceived conditions. Mirrors the `atemp` feature in the UCI Bike Sharing Dataset for comparability. |

---

## Raw vs Cleaned Data Separation

| Path | Status | Description |
|------|--------|-------------|
| `2017-capitalbikeshare-tripdata/*.csv` | **Raw** (tracked in git) | Original quarterly trip files as downloaded from Capital Bikeshare. Never modified. |
| `data/weather_2017_hourly.csv` | **Intermediate** (gitignored, generated) | Raw weather fetch from Meteostat, cached to avoid repeat API calls. |
| `data/hourly_bike_weather_2017.csv` | **Cleaned** (gitignored, generated) | Final master dataset produced by `src/pipeline.py`. This is the input to EDA and modeling. |
| `outputs/` | **Generated** (gitignored) | All figures, tables, model files, and quality reports. Reproduced by running EDA and modeling scripts. |
