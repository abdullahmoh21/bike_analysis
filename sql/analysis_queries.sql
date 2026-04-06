-- Analysis queries for bike_demand_hourly_2017
-- Run against the PostgreSQL table loaded via: python src/pipeline.py --load-db
-- All queries target the table: bike_demand_hourly_2017

-- ── 1. Average hourly demand by hour of day ────────────────────────────────
-- Shows the bimodal commuter pattern for registered vs casual riders.
SELECT
    hour_of_day,
    ROUND(AVG(total_rentals)::numeric, 1)    AS avg_total,
    ROUND(AVG(casual_count)::numeric, 1)     AS avg_casual,
    ROUND(AVG(registered_count)::numeric, 1) AS avg_registered
FROM bike_demand_hourly_2017
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- ── 2. Seasonal demand split by user type ─────────────────────────────────
-- Confirms that casual riders are more seasonal than registered riders.
SELECT
    season,
    SUM(casual_count)                                                    AS total_casual,
    SUM(registered_count)                                                AS total_registered,
    SUM(total_rentals)                                                   AS total_rides,
    ROUND(100.0 * SUM(casual_count) / NULLIF(SUM(total_rentals), 0), 1) AS casual_pct
FROM bike_demand_hourly_2017
GROUP BY season
ORDER BY MIN(season_code);


-- ── 3. Weather impact: average demand by temperature band ─────────────────
-- Used in EDA to understand the non-linear relationship between temp and demand.
SELECT
    CASE
        WHEN temp <  0  THEN '< 0 C'
        WHEN temp < 10  THEN '0–10 C'
        WHEN temp < 20  THEN '10–20 C'
        WHEN temp < 30  THEN '20–30 C'
        ELSE                 '>= 30 C'
    END                                       AS temp_band,
    COUNT(*)                                  AS hours,
    ROUND(AVG(total_rentals)::numeric, 1)     AS avg_total,
    ROUND(AVG(casual_count)::numeric, 1)      AS avg_casual,
    ROUND(AVG(registered_count)::numeric, 1)  AS avg_registered
FROM bike_demand_hourly_2017
GROUP BY temp_band
ORDER BY MIN(temp);


-- ── 4. Weekday vs weekend demand by time of day ───────────────────────────
-- Key insight driving the use of time_of_day × is_weekend interaction.
SELECT
    time_of_day,
    CASE WHEN is_weekend THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    ROUND(AVG(total_rentals)::numeric, 1)                  AS avg_total,
    COUNT(*)                                               AS hours
FROM bike_demand_hourly_2017
GROUP BY time_of_day, is_weekend
ORDER BY
    CASE time_of_day
        WHEN 'morning_rush'  THEN 1
        WHEN 'midday'        THEN 2
        WHEN 'evening_rush'  THEN 3
        ELSE                      4
    END,
    is_weekend;


-- ── 5. Monthly demand trend ───────────────────────────────────────────────
-- Shows seasonal arc across the year.
SELECT
    month,
    SUM(total_rentals)                        AS total_rides,
    ROUND(AVG(total_rentals)::numeric, 1)     AS avg_hourly_rides,
    SUM(casual_count)                         AS casual_rides,
    SUM(registered_count)                     AS registered_rides
FROM bike_demand_hourly_2017
GROUP BY month
ORDER BY month;


-- ── 6. Top 10 busiest hours on record ────────────────────────────────────
SELECT
    hour,
    total_rentals,
    casual_count,
    registered_count,
    temp,
    time_of_day,
    CASE WHEN is_weekend THEN 'Weekend' ELSE 'Weekday' END AS day_type
FROM bike_demand_hourly_2017
ORDER BY total_rentals DESC
LIMIT 10;


-- ── 7. Data quality: weather imputation summary ───────────────────────────
-- Confirms how many hours required imputation and at what severity.
SELECT
    weather_quality_flag,
    CASE weather_quality_flag
        WHEN 0 THEN 'Observed'
        WHEN 1 THEN 'Interpolated'
        WHEN 2 THEN 'Forward/Backward Fill'
        WHEN 3 THEN 'Median Fill'
    END                  AS description,
    COUNT(*)             AS hours,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_year
FROM bike_demand_hourly_2017
GROUP BY weather_quality_flag
ORDER BY weather_quality_flag;


-- ── 8. Holiday vs working day vs weekend demand ───────────────────────────
SELECT
    CASE
        WHEN holiday    THEN 'Holiday'
        WHEN workingday THEN 'Working Day'
        ELSE                 'Weekend'
    END                                    AS day_category,
    COUNT(*)                               AS hours,
    ROUND(AVG(total_rentals)::numeric, 1)  AS avg_total,
    ROUND(AVG(casual_count)::numeric, 1)   AS avg_casual,
    ROUND(AVG(registered_count)::numeric, 1) AS avg_registered
FROM bike_demand_hourly_2017
GROUP BY day_category
ORDER BY avg_total DESC;
