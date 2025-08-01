import os
import mlflow
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def main():
    """
    Loads pre-processed data, trains a model via a hyperparameter-tuned pipeline,
    and logs the results to MLflow.
    """
    spark = SparkSession.builder.appName("ChurnModelTraining")\
    .config("spark.driver.memory", "12g") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()
        
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("KKBoxChurnPrediction")

    # --- Load Processed Data ---
    processed_path = "data/processed/features.parquet"
    df_model = spark.read.parquet(processed_path)
    print("--- Processed Data Loaded ---")
    
    # --- Define and Run the ML Pipeline ---
    with mlflow.start_run():
        # STAGE 1: Assemble feature vector
        feature_columns = [
            "transaction_count", "total_plan_days", "total_amount_paid",
            "total_songs_completed", "total_songs_985_completed", "total_unique_songs", "total_secs_played", "listening_day_count",
            "age_cleaned", "is_male", "is_female"
        ]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")

        # STAGE 2: Define the model
        scale_pos_weight_value = df_model.filter("is_churn == 0").count() / df_model.filter("is_churn == 1").count()
        xgb = SparkXGBClassifier(
            features_col="features",
            label_col="label",
            scale_pos_weight=scale_pos_weight_value
        )
        
        pipeline = Pipeline(stages=[assembler, xgb])

        # Define hyperparameter grid
        paramGrid = ParamGridBuilder() \
                .addGrid(xgb.n_estimators, [100, 300, 500]) \
                .addGrid(xgb.max_depth, [5, 10, 15]) \
                .addGrid(xgb.learning_rate, [0.1, 0.05, 0.01]) \
                .addGrid(xgb.subsample, [0.8, 1.0]) \
                .build()

        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=4,
            parallelism=6
        )
        
        # Prepare data for the pipeline
        df_model = df_model.withColumnRenamed("is_churn", "label")
        (training_data, test_data) = df_model.randomSplit([0.8, 0.2], seed=42)

        # Run the training
        print("--- Starting cross-validation ---")
        cv_model = cv.fit(training_data)
        best_pipeline_model = cv_model.bestModel
        
        # Evaluate on the test set
        predictions = best_pipeline_model.transform(test_data)
        auc = evaluator.evaluate(predictions)
        
        # Log results to MLflow
        print(f"Logging results to MLflow...")
        best_xgb_model = best_pipeline_model.stages[-1]
        best_params = {
            "n_estimators": best_xgb_model.getOrDefault('n_estimators'),
            "max_depth": best_xgb_model.getOrDefault('max_depth'),
            "learning_rate": best_xgb_model.getOrDefault('learning_rate')
        }
        
        mlflow.log_params(best_params)
        mlflow.log_metric("auc_on_test_set", auc)
        mlflow.spark.log_model(best_pipeline_model, "spark-xgb-pipeline-model-best")

        print(f"--- Pipeline training complete. Best Model AUC on Test Set: {auc} ---")

if __name__ == "__main__":
    main()
