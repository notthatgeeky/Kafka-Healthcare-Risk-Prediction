"""
Main Spark Structured Streaming Pipeline.
Consumes Kafka streams (telemetry + BP events), computes sliding window aggregations,
attaches BP readings, applies rule-based alerts, runs online MLlib scoring,
and sinks to Parquet & Kafka alerts.
"""
import os
import sys
import yaml
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import PipelineModel

from streaming.schemas import TELEMETRY_SCHEMA, BP_SCHEMA
from streaming.window_aggregations import compute_windowed_vitals
from streaming.alert_rules import compute_news_score_and_indices, evaluate_clinical_alerts

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_streaming_pipeline():
    cfg = load_config()

    # Configure Windows Hadoop environment
    os.environ["HADOOP_HOME"] = "C:\\hadoop-3.4.1"
    os.environ["PATH"] = "C:\\hadoop-3.4.1\\bin;" + os.environ.get("PATH", "")
    os.environ["PYSPARK_PYTHON"] = "C:\\Users\\chitr\\TAKE2~1\\.venv\\Scripts\\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\chitr\\TAKE2~1\\.venv\\Scripts\\python.exe"

    kafka_bootstrap = cfg["kafka"]["bootstrap_servers"]
    topic_telemetry = cfg["kafka"]["topics"]["telemetry"]["name"]
    topic_bp = cfg["kafka"]["topics"]["bp_events"]["name"]
    topic_alerts = cfg["kafka"]["topics"]["alerts"]["name"]

    model_path = cfg["ml"]["model_path"]
    parquet_features_dir = cfg["paths"]["parquet_features_dir"]
    parquet_alerts_dir = cfg["paths"]["parquet_alerts_dir"]
    checkpoints_dir = cfg["paths"]["checkpoints_dir"]

    trigger_sec = cfg["spark"].get("trigger_interval_seconds", 5)
    watermark_telemetry = cfg["spark"]["watermarks"]["telemetry"]
    watermark_bp = cfg["spark"]["watermarks"]["bp_events"]
    window_duration = cfg["spark"]["windows"]["duration"]
    slide_duration = cfg["spark"]["windows"]["slide"]

    print("Building SparkSession with spark-sql-kafka connector...")
    spark = (SparkSession.builder
        .appName(cfg["spark"]["app_name"])
        .master(cfg["spark"]["master"])
        .config("spark.driver.memory", cfg["spark"]["driver_memory"])
        .config("spark.sql.shuffle.partitions", cfg["spark"]["sql_shuffle_partitions"])
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .getOrCreate())

    spark.sparkContext.setLogLevel("WARN")

    print(f"Loading trained Spark MLlib PipelineModel from {model_path}...")
    ml_model = PipelineModel.load(model_path)
    print("PipelineModel loaded successfully into driver memory.")

    # 1. Read Stream: Vitals Telemetry (1 Hz)
    print(f"Subscribing to Kafka topic: {topic_telemetry}")
    telemetry_raw = (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", topic_telemetry)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", cfg["spark"]["max_offsets_per_trigger"]["telemetry"])
        .load())

    telemetry_parsed = (telemetry_raw
        .select(
            F.from_json(F.col("value").cast("string"), TELEMETRY_SCHEMA).alias("data")
        )
        .select("data.*")
        .withWatermark("event_time", watermark_telemetry)
    )

    # 2. Window Aggregations (Pure Spark SQL - Zero Python UDFs)
    windowed_features = compute_windowed_vitals(
        telemetry_parsed,
        window_duration=window_duration,
        slide_duration=slide_duration
    )

    # 3. Add NEWS scores and Hemodynamics
    enriched_features = compute_news_score_and_indices(windowed_features)

    # 4. Stream MLlib Scoring: Vectorize and Score Online
    scored_stream = ml_model.transform(enriched_features)

    # Extract probability of deteriorating (index 1 from vector)
    # Using built-in vector slice expression
    from pyspark.ml.functions import vector_to_array
    scored_stream = (scored_stream
        .withColumn("prob_array", vector_to_array(F.col("probability")))
        .withColumn("ml_risk_score", F.round(F.col("prob_array")[1], 4))
        .withColumn(
            "ml_risk_level",
            F.when(F.col("ml_risk_score") >= cfg["ml"]["risk_threshold_high"], "HIGH")
            .when(F.col("ml_risk_score") >= cfg["ml"]["risk_threshold_medium"], "MEDIUM")
            .otherwise("LOW")
        )
        .drop("raw_features", "features", "rawPrediction", "probability", "prob_array")
    )

    # 5. Evaluate Clinical Alerts
    alerts_stream = evaluate_clinical_alerts(scored_stream)

    # Sink A: Console Logger for Real-time verification
    console_query = (scored_stream
        .select(
            F.col("patient_id"),
            F.col("bed_id"),
            F.round(F.col("hr_mean"), 1).alias("HR"),
            F.round(F.col("spo2_mean"), 1).alias("SpO2"),
            F.round(F.col("rr_mean"), 1).alias("RR"),
            F.col("news_score").alias("NEWS"),
            F.col("ml_risk_score").alias("ML_Risk"),
            F.col("ml_risk_level").alias("Level")
        )
        .writeStream
        .format("console")
        .outputMode("update")
        .trigger(processingTime=f"{trigger_sec} seconds")
        .option("truncate", "false")
        .option("checkpointLocation", os.path.join(checkpoints_dir, "console_test"))
        .start())

    # Sink B: Parquet Lake for Dashboard and Analysis
    parquet_query = (scored_stream
        .withColumn("date", F.to_date(F.col("window.end")))
        .writeStream
        .format("parquet")
        .outputMode("append")
        .partitionBy("date")
        .option("path", parquet_features_dir)
        .option("checkpointLocation", os.path.join(checkpoints_dir, "parquet_features"))
        .trigger(processingTime=f"{trigger_sec} seconds")
        .start())

    # Sink C: Kafka Alerts Topic
    kafka_alerts_query = (alerts_stream
        .select(
            F.col("patient_id").alias("key"),
            F.to_json(F.struct(
                F.col("patient_id"),
                F.col("bed_id"),
                F.col("alert_time"),
                F.col("alert_type"),
                F.col("severity"),
                F.col("message"),
                F.col("ml_risk_score"),
                F.col("ml_risk_level"),
                F.col("window_start"),
                F.col("window_end")
            )).alias("value")
        )
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("topic", topic_alerts)
        .option("checkpointLocation", os.path.join(checkpoints_dir, "kafka_alerts"))
        .trigger(processingTime=f"{trigger_sec} seconds")
        .start())

    print("\n" + "="*60)
    print("ALL STREAMING QUERIES ACTIVE:")
    print(f"1. Console Output Query ID: {console_query.id}")
    print(f"2. Parquet Features Sink:   {parquet_features_dir}")
    print(f"3. Kafka Alerts Sink:       {topic_alerts}")
    print("="*60 + "\n")

    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    run_streaming_pipeline()
