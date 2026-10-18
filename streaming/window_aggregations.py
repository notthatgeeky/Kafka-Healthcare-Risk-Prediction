"""
Sliding window statistics and trend slope computation using purely built-in Spark SQL functions.
Zero Python UDFs.
"""
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

def compute_windowed_vitals(telemetry_df: DataFrame, window_duration: str = "30 seconds", slide_duration: str = "10 seconds") -> DataFrame:
    """
    Computes statistical aggregates (mean, std, min, max) and linear trend slopes
    over sliding event-time windows for each patient.
    """
    # 1. Filter extreme invalid artifacts (vital range validation)
    valid_df = (telemetry_df
        .filter(
            (F.col("heart_rate").isNull() | ((F.col("heart_rate") >= 25.0) & (F.col("heart_rate") <= 230.0))) &
            (F.col("spo2").isNull() | ((F.col("spo2") >= 50.0) & (F.col("spo2") <= 100.0))) &
            (F.col("respiratory_rate").isNull() | ((F.col("respiratory_rate") >= 4.0) & (F.col("respiratory_rate") <= 55.0)))
        )
    )

    # 2. Add time in seconds as numeric for closed-form OLS slope
    with_time = valid_df.withColumn("t_sec", F.unix_timestamp("event_time").cast("double"))

    t = F.col("t_sec")
    hr = F.col("heart_rate")
    spo2 = F.col("spo2")
    rr = F.col("respiratory_rate")

    # Closed-form OLS slope helper expressions:
    # slope = (N * sum(t * y) - sum(t) * sum(y)) / (N * sum(t * t) - (sum(t))^2 + 1e-6)
    def ols_slope(y_col, alias_name):
        n = F.count(y_col)
        sum_t = F.sum(F.when(y_col.isNotNull(), t))
        sum_y = F.sum(y_col)
        sum_ty = F.sum(t * y_col)
        sum_tt = F.sum(F.when(y_col.isNotNull(), t * t))
        denom = n * sum_tt - F.pow(sum_t, 2)
        numer = n * sum_ty - (sum_t * sum_y)
        return F.when((n >= 3) & (denom > 1e-5), numer / denom).otherwise(0.0).alias(alias_name)

    # 3. Window aggregation
    agg_df = (with_time
        .groupBy(
            F.col("patient_id"),
            F.window(F.col("event_time"), window_duration, slide_duration)
        )
        .agg(
            # First non-null bed_id
            F.first("bed_id", ignorenulls=True).alias("bed_id"),
            # Record counts and data quality hygiene
            F.count("*").alias("record_count"),
            F.sum(F.when(hr.isNull() | spo2.isNull() | rr.isNull(), 1).otherwise(0)).alias("null_count"),
            # Max ground truth risk in the window
            F.max("risk_label").alias("risk_label"),

            # Heart Rate metrics
            F.avg(hr).alias("hr_mean"),
            F.stddev(hr).alias("hr_std"),
            F.min(hr).alias("hr_min"),
            F.max(hr).alias("hr_max"),
            ols_slope(hr, "hr_slope"),

            # SpO2 metrics
            F.avg(spo2).alias("spo2_mean"),
            F.stddev(spo2).alias("spo2_std"),
            F.min(spo2).alias("spo2_min"),
            F.max(spo2).alias("spo2_max"),
            ols_slope(spo2, "spo2_slope"),

            # Respiratory Rate metrics
            F.avg(rr).alias("rr_mean"),
            F.stddev(rr).alias("rr_std"),
            F.min(rr).alias("rr_min"),
            F.max(rr).alias("rr_max"),
            ols_slope(rr, "rr_slope"),
        )
        # Fill missing stddev with 0.0
        .na.fill({
            "hr_std": 0.0,
            "spo2_std": 0.0,
            "rr_std": 0.0,
            "hr_slope": 0.0,
            "spo2_slope": 0.0,
            "rr_slope": 0.0
        })
    )

    return agg_df
