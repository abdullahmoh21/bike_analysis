CREATE TABLE IF NOT EXISTS bike_demand_hourly_2017 (
    hour TIMESTAMP PRIMARY KEY,
    temp DOUBLE PRECISION,
    atemp DOUBLE PRECISION,
    rhum DOUBLE PRECISION,
    wspd DOUBLE PRECISION,
    coco DOUBLE PRECISION,
    casual_count INTEGER NOT NULL,
    registered_count INTEGER NOT NULL,
    total_rentals INTEGER NOT NULL,
    hour_of_day SMALLINT NOT NULL,
    day_of_week SMALLINT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    holiday BOOLEAN NOT NULL,
    workingday BOOLEAN NOT NULL,
    quarter VARCHAR(2) NOT NULL,
    month SMALLINT NOT NULL,
    season VARCHAR(10) NOT NULL,
    season_code SMALLINT NOT NULL,
    time_of_day VARCHAR(20) NOT NULL,
    weather_quality_flag SMALLINT NOT NULL,
    weather_source VARCHAR(32) NOT NULL,
    is_imputed_rides BOOLEAN NOT NULL,
    run_generated_at VARCHAR(40) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bike_hourly_day_hour
ON bike_demand_hourly_2017 (day_of_week, hour_of_day);

CREATE INDEX IF NOT EXISTS idx_bike_hourly_season
ON bike_demand_hourly_2017 (season);