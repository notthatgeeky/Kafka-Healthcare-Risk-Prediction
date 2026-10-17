"""
Explicit StructType schemas for Kafka JSON payloads.
"""
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    TimestampType
)

TELEMETRY_SCHEMA = StructType([
    StructField("patient_id", StringType(), False),
    StructField("bed_id", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("event_time", TimestampType(), False),
    StructField("heart_rate", DoubleType(), True),
    StructField("spo2", DoubleType(), True),
    StructField("respiratory_rate", DoubleType(), True),
    StructField("risk_label", IntegerType(), True),
])

BP_SCHEMA = StructType([
    StructField("patient_id", StringType(), False),
    StructField("bed_id", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("event_time", TimestampType(), False),
    StructField("systolic_bp", DoubleType(), True),
    StructField("diastolic_bp", DoubleType(), True),
    StructField("mean_arterial_pressure", DoubleType(), True),
    StructField("risk_label", IntegerType(), True),
])

ALERT_SCHEMA = StructType([
    StructField("patient_id", StringType(), False),
    StructField("bed_id", StringType(), True),
    StructField("alert_time", TimestampType(), False),
    StructField("alert_type", StringType(), False),
    StructField("severity", StringType(), False),
    StructField("message", StringType(), True),
    StructField("ml_risk_score", DoubleType(), True),
    StructField("ml_risk_level", StringType(), True),
    StructField("window_start", TimestampType(), True),
    StructField("window_end", TimestampType(), True),
])
