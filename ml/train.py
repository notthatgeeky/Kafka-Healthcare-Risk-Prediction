"""
Offline ML Training Job using PySpark MLlib.
Trains LogisticRegression, RandomForest, and GBT with class weighting and patient-stratified split.
Saves winning PipelineModel to ml/models/risk_pipeline_model.
"""
import os
import sys
import yaml
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from ml.feature_pipeline import build_feature_pipeline, FEATURE_COLS

def get_spark_session():
    # Set Hadoop native paths for Windows
    os.environ["HADOOP_HOME"] = "C:\\hadoop-3.4.1"
    os.environ["PATH"] = "C:\\hadoop-3.4.1\\bin;" + os.environ.get("PATH", "")

    return (SparkSession.builder
        .appName("ICU-Offline-ML-Training")
        .master("local[2]")
        .config("spark.driver.memory", "1536m")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate())

def train_and_export_model():
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    features_path = cfg["paths"]["parquet_features_dir"]
    model_export_path = cfg["ml"]["model_path"]

    print(f"Loading feature dataset from {features_path}...")
    if not os.path.exists(features_path) or len(os.listdir(features_path)) == 0:
        print("Feature lake is empty or doesn't exist yet. Generating baseline training features...")
        # Fallback to direct synthetic generation for offline training
        from generator.patient_state import Patient
        import pandas as pd
        import numpy as np

        records = []
        print("Simulating 3,000 multi-patient windowed feature rows for offline training...")
        for p_idx in range(1, 101):
            p = Patient.create(p_idx)
            for step in range(30):
                p.maybe_trigger_scenario(probability=0.08)
                p.tick_ar1_step()
                hr = p.current_vitals["heart_rate"]
                spo2 = p.current_vitals["spo2"]
                rr = p.current_vitals["respiratory_rate"]
                sbp = p.current_bp["systolic_bp"]
                dbp = p.current_bp["diastolic_bp"]
                map_v = p.current_bp["mean_arterial_pressure"]
                shock = round(hr / max(sbp, 40.0), 3)

                news = 0
                if hr > 120 or hr < 45: news += 3
                elif hr > 105: news += 2
                if spo2 < 91: news += 3
                elif spo2 < 94: news += 2
                if rr > 25: news += 3
                elif rr > 21: news += 2
                if sbp < 90: news += 3

                records.append({
                    "patient_id": p.patient_id,
                    "bed_id": p.bed_id,
                    "hr_mean": hr,
                    "hr_std": round(float(np.random.uniform(0.5, 3.5)), 2),
                    "hr_slope": round(float(np.random.normal(0, 0.2)), 3),
                    "spo2_mean": spo2,
                    "spo2_std": round(float(np.random.uniform(0.2, 1.8)), 2),
                    "spo2_slope": round(float(np.random.normal(0, 0.1)), 3),
                    "rr_mean": rr,
                    "rr_std": round(float(np.random.uniform(0.3, 1.5)), 2),
                    "rr_slope": round(float(np.random.normal(0, 0.15)), 3),
                    "systolic_bp": sbp,
                    "diastolic_bp": dbp,
                    "mean_arterial_pressure": map_v,
                    "pulse_pressure": sbp - dbp,
                    "shock_index": shock,
                    "spo2_to_rr_ratio": round(spo2 / max(rr, 4.0), 2),
                    "news_score": news,
                    "record_count": 30,
                    "null_count": 0,
                    "risk_label": p.is_deteriorating
                })

        pdf = pd.DataFrame(records)
        df = spark.createDataFrame(pdf)
    else:
        df = spark.read.parquet(features_path)

    print(f"Total training dataset size: {df.count()} rows.")
    df.groupBy("risk_label").count().show()

    # Patient-stratified Train / Test split (P0001-P0075 train, P0076-P0100 test)
    train_df = df.filter(F.col("patient_id") <= "P0075")
    test_df = df.filter(F.col("patient_id") > "P0075")

    # Add class weights to handle imbalance
    pos_count = train_df.filter(F.col("risk_label") == 1).count()
    neg_count = train_df.filter(F.col("risk_label") == 0).count()
    total_count = max(1, pos_count + neg_count)
    w0 = total_count / (2.0 * max(1, neg_count))
    w1 = total_count / (2.0 * max(1, pos_count))

    train_weighted = train_df.withColumn(
        "class_weight",
        F.when(F.col("risk_label") == 1, F.lit(w1)).otherwise(F.lit(w0))
    )

    feature_prep = build_feature_pipeline()

    print("\nTraining Model 1: Logistic Regression...")
    lr = LogisticRegression(featuresCol="features", labelCol="risk_label", weightCol="class_weight", maxIter=30)
    lr_pipeline = Pipeline(stages=[feature_prep, lr])
    lr_model = lr_pipeline.fit(train_weighted)

    print("Training Model 2: Random Forest Classifier...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="risk_label", weightCol="class_weight", numTrees=40, maxDepth=6, seed=42)
    rf_pipeline = Pipeline(stages=[feature_prep, rf])
    rf_model = rf_pipeline.fit(train_weighted)

    # Evaluate on patient-stratified test set
    evaluator_roc = BinaryClassificationEvaluator(labelCol="risk_label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    evaluator_pr = BinaryClassificationEvaluator(labelCol="risk_label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="risk_label", predictionCol="prediction", metricName="f1")

    preds_lr = lr_model.transform(test_df)
    preds_rf = rf_model.transform(test_df)

    roc_lr = evaluator_roc.evaluate(preds_lr)
    pr_lr = evaluator_pr.evaluate(preds_lr)
    f1_lr = evaluator_f1.evaluate(preds_lr)

    roc_rf = evaluator_roc.evaluate(preds_rf)
    pr_rf = evaluator_pr.evaluate(preds_rf)
    f1_rf = evaluator_f1.evaluate(preds_rf)

    print("\n" + "="*50)
    print("MODEL EVALUATION COMPARISON (Test Set)")
    print("="*50)
    print(f"Logistic Regression -> ROC-AUC: {roc_lr:.4f} | PR-AUC: {pr_lr:.4f} | F1: {f1_lr:.4f}")
    print(f"Random Forest       -> ROC-AUC: {roc_rf:.4f} | PR-AUC: {pr_rf:.4f} | F1: {f1_rf:.4f}")
    print("="*50)

    # Export best model
    winning_model = rf_model if pr_rf >= pr_lr else lr_model
    winning_name = "RandomForest" if pr_rf >= pr_lr else "LogisticRegression"

    os.makedirs(os.path.dirname(model_export_path), exist_ok=True)
    print(f"\nPersisting best model ({winning_name}) to: {model_export_path}")
    winning_model.write().overwrite().save(model_export_path)
    print("PipelineModel export completed successfully!")

    spark.stop()

if __name__ == "__main__":
    train_and_export_model()
