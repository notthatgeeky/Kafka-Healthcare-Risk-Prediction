"""
ML Feature Pipeline definitions using PySpark MLlib.
"""
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer

# Exact feature matrix specified in plan
FEATURE_COLS = [
    "hr_mean",
    "hr_std",
    "hr_slope",
    "spo2_mean",
    "spo2_std",
    "spo2_slope",
    "rr_mean",
    "rr_std",
    "rr_slope",
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "pulse_pressure",
    "shock_index",
    "spo2_to_rr_ratio",
    "news_score",
    "record_count",
    "null_count"
]

def build_feature_pipeline() -> Pipeline:
    """
    Builds a stateless feature transformation pipeline:
    Imputer -> VectorAssembler -> StandardScaler.
    Stateless stages are fully compatible with Spark Structured Streaming scoring.
    """
    imputer = Imputer(
        inputCols=FEATURE_COLS,
        outputCols=[f"{c}_imp" for c in FEATURE_COLS]
    ).setStrategy("median")

    assembler = VectorAssembler(
        inputCols=[f"{c}_imp" for c in FEATURE_COLS],
        outputCol="raw_features",
        handleInvalid="keep"
    )

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    return Pipeline(stages=[imputer, assembler, scaler])
