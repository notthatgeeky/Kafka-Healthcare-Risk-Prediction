"""
Clinical alert rules and NEWS (National Early Warning Score) calculation.
Implemented strictly with native Spark Catalyst expressions (Zero Python UDFs).
"""
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

def compute_news_score_and_indices(df: DataFrame) -> DataFrame:
    """
    Computes Shock Index, Pulse Pressure, and NEWS score using built-in Spark functions.
    """
    cols = df.columns
    with_bp = df
    if "systolic_bp" not in cols:
        with_bp = with_bp.withColumn("systolic_bp", F.lit(120.0))
    else:
        with_bp = with_bp.withColumn("systolic_bp", F.coalesce(F.col("systolic_bp"), F.lit(120.0)))

    if "diastolic_bp" not in cols:
        with_bp = with_bp.withColumn("diastolic_bp", F.lit(75.0))
    else:
        with_bp = with_bp.withColumn("diastolic_bp", F.coalesce(F.col("diastolic_bp"), F.lit(75.0)))

    if "mean_arterial_pressure" not in cols:
        with_bp = with_bp.withColumn("mean_arterial_pressure", F.lit(90.0))
    else:
        with_bp = with_bp.withColumn("mean_arterial_pressure", F.coalesce(F.col("mean_arterial_pressure"), F.lit(90.0)))


    # 2. Derived hemodynamics
    with_indices = (with_bp
        .withColumn("pulse_pressure", F.col("systolic_bp") - F.col("diastolic_bp"))
        .withColumn("shock_index", F.round(F.col("hr_mean") / F.greatest(F.col("systolic_bp"), F.lit(40.0)), 3))
        .withColumn("spo2_to_rr_ratio", F.round(F.col("spo2_mean") / F.greatest(F.col("rr_mean"), F.lit(4.0)), 2))
    )

    # 3. NEWS sub-scores
    # Heart Rate subscore
    hr_news = (
        F.when((F.col("hr_mean") <= 40) | (F.col("hr_mean") >= 131), 3)
        .when((F.col("hr_mean") >= 111) & (F.col("hr_mean") <= 130), 2)
        .when(((F.col("hr_mean") >= 41) & (F.col("hr_mean") <= 50)) | ((F.col("hr_mean") >= 91) & (F.col("hr_mean") <= 110)), 1)
        .otherwise(0)
    )

    # SpO2 subscore
    spo2_news = (
        F.when(F.col("spo2_mean") <= 91, 3)
        .when((F.col("spo2_mean") >= 92) & (F.col("spo2_mean") <= 93), 2)
        .when((F.col("spo2_mean") >= 94) & (F.col("spo2_mean") <= 95), 1)
        .otherwise(0)
    )

    # Respiratory Rate subscore
    rr_news = (
        F.when((F.col("rr_mean") <= 8) | (F.col("rr_mean") >= 25), 3)
        .when((F.col("rr_mean") >= 21) & (F.col("rr_mean") <= 24), 2)
        .when((F.col("rr_mean") >= 9) & (F.col("rr_mean") <= 11), 1)
        .otherwise(0)
    )

    # Systolic BP subscore
    sbp_news = (
        F.when((F.col("systolic_bp") <= 90) | (F.col("systolic_bp") >= 220), 3)
        .when((F.col("systolic_bp") >= 91) & (F.col("systolic_bp") <= 100), 2)
        .when((F.col("systolic_bp") >= 101) & (F.col("systolic_bp") <= 110), 1)
        .otherwise(0)
    )

    with_news = (with_indices
        .withColumn("news_score", hr_news + spo2_news + rr_news + sbp_news)
    )

    return with_news


def evaluate_clinical_alerts(df: DataFrame) -> DataFrame:
    """
    Evaluates rule-based alert conditions and returns an alerts DataFrame.
    """
    # Define clinical conditions
    is_hypoxia = F.col("spo2_mean") < 90.0
    is_tachycardia = F.col("hr_mean") > 120.0
    is_bradycardia = F.col("hr_mean") < 50.0
    is_tachypnea = F.col("rr_mean") > 25.0
    is_hypotension = F.col("systolic_bp") < 90.0
    is_shock = F.col("shock_index") >= 1.0

    # Severity and alert type mapping
    alert_condition = (
        is_hypoxia | is_tachycardia | is_bradycardia | is_tachypnea | is_hypotension | is_shock | (F.col("news_score") >= 7)
    )

    alert_type = (
        F.when(is_shock, "SHOCK_RISK")
        .when(is_hypoxia, "SEVERE_HYPOXIA")
        .when(is_hypotension, "HYPOTENSION")
        .when(is_tachycardia, "TACHYCARDIA")
        .when(is_bradycardia, "BRADYCARDIA")
        .when(is_tachypnea, "TACHYPNEA")
        .otherwise("CRITICAL_NEWS")
    )

    severity = (
        F.when(is_shock | is_hypoxia | is_hypotension | (F.col("news_score") >= 7), "CRITICAL")
        .otherwise("WARNING")
    )

    message = F.concat(
        alert_type,
        F.lit(" | HR: "), F.round(F.col("hr_mean"), 1),
        F.lit(", SpO2: "), F.round(F.col("spo2_mean"), 1),
        F.lit("%, SBP: "), F.round(F.col("systolic_bp"), 1),
        F.lit(", NEWS: "), F.col("news_score")
    )

    alerts_df = (df
        .filter(alert_condition)
        .select(
            F.col("patient_id"),
            F.col("bed_id"),
            F.current_timestamp().alias("alert_time"),
            alert_type.alias("alert_type"),
            severity.alias("severity"),
            message.alias("message"),
            F.coalesce(F.col("ml_risk_score"), F.lit(0.0)).alias("ml_risk_score"),
            F.coalesce(F.col("ml_risk_level"), F.lit("RULE_BASED")).alias("ml_risk_level"),
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
        )
    )

    return alerts_df
