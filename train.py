import os
import mlflow
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from xgboost.spark import SparkXGBoostClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def main():
    """
    Main function for the training script.
    """
    # --- 1. Setup Spark and MLflow ---
    spark = SparkSession.builder.appName("ChurnPredictionTraining").getOrCreate()
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("KKBoxChurnPrediction")

    # --- 2. Load Data ---
    data_path = "data/raw/"
    df_train = spark.read.csv(os.path.join(data_path, "train.csv"), header=True, inferSchema=True)
    df_members = spark.read.csv(os.path.join(data_path, "members.csv"), header=True, inferSchema=True)
    df_transactions = spark.read.csv(os.path.join(data_path, "transactions.csv"), header=True, inferSchema=True)
    df_user_logs = spark.read.csv(os.path.join(data_path, "user_logs.csv"), header=True, inferSchema=True)

    print("--- Data Loading Complete ---")

    # --- 3. Feature Engineering ---
    print("--- Starting Feature Engineering ---")

    # Process transactions data
    transactions_features = df_transactions.groupBy("msno").agg(
        F.count("transaction_date").alias("transaction_count"),
        F.sum("payment_plan_days").alias("total_plan_days"),
        F.sum("actual_amount_paid").alias("total_amount_paid")
    )

    # Process user_logs data
    user_logs_features = df_user_logs.groupBy("msno").agg(
        F.sum("num_100").alias("total_songs_completed"),
        F.sum("total_secs").alias("total_secs_played"),
        F.count("date").alias("listening_day_count")
    )
    
    # Process members data (clean age and gender)
    members_cleaned = df_members.withColumn(
        "age_cleaned",
        F.when((F.col("bd") >= 5) & (F.col("bd") <= 100), F.col("bd")).otherwise(None)
    ).withColumn(
        "gender_cleaned",
        F.when(F.col("gender").isin("male", "female"), F.col("gender")).otherwise(None)
    )

    # --- 4. Join DataFrames ---
    print("--- Joining DataFrames ---")
    df_model = df_train.join(transactions_features, on="msno", how="left")
    df_model = df_model.join(user_logs_features, on="msno", how="left")
    df_model = df_model.join(members_cleaned.select("msno", "age_cleaned", "gender_cleaned"), on="msno", how="left")
    
    # --- 5. Final Data Preparation ---
    # Convert gender to numeric
    indexer = StringIndexer(inputCol="gender_cleaned", outputCol="gender_indexed")
    df_model = indexer.fit(df_model).transform(df_model)
    
    # One-hot encode the indexed gender
    ohe = OneHotEncoder(inputCols=["gender_indexed"], outputCols=["gender"])
    df_model = ohe.fit(df_model).transform(df_model)

    # Fill missing values
    # df_model = df_model.fillna(0)
    
    # Assemble feature vector
    feature_columns = [
        "transaction_count", "total_plan_days", "total_amount_paid",
        "total_songs_completed", "total_secs_played", "listening_day_count",
        "age_cleaned", "gender"
    ]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    final_data = assembler.transform(df_model).select(F.col("is_churn").alias("label"), "features")

    # --- 6. Define and Run the ML Pipeline ---
    print("--- Defining ML Pipeline ---")
    with mlflow.start_run():
        # --- STAGE 1: Feature Engineering Transformers ---
        gender_indexer = StringIndexer(inputCol="gender_cleaned", outputCol="gender_indexed", handleInvalid="keep")
        ohe = OneHotEncoder(inputCols=["gender_indexed"], outputCols=["gender_ohe"])
        
        feature_columns = [
            "transaction_count", "total_plan_days", "total_amount_paid",
            "total_songs_completed", "total_secs_played", "listening_day_count",
            "age_cleaned", "gender_ohe"
        ]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

        # --- STAGE 2: Model Estimator ---
        scale_pos_weight_value = 929460 / 63471
        xgb = SparkXGBoostClassifier(
            features_col="features",
            label_col="label",
            scale_pos_weight=scale_pos_weight_value
        )

        # --- Define the full pipeline ---
        pipeline = Pipeline(stages=[gender_indexer, ohe, assembler, xgb])

        # --- Define the new parameter grid ---
        paramGrid = ParamGridBuilder() \
            .addGrid(xgb.n_estimators, [100, 300, 500]) \
            .addGrid(xgb.max_depth, [5, 10]) \
            .addGrid(xgb.learning_rate, [0.1, 0.05]) \
            .build()

        # --- Set up the evaluator ---
        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

        # --- Set up the CrossValidator with new settings ---
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=5,
            parallelism=8
        )
        
        # Rename the target column to the standard 'label'
        df_model = df_model.withColumnRenamed("is_churn", "label")

        # Split data
        (training_data, test_data) = df_model.randomSplit([0.8, 0.2], seed=42)

        # --- Run the cross-validation and train the best pipeline ---
        print("--- Starting cross-validation, this will take a long time... ---")
        cv_model = cv.fit(training_data)
        best_pipeline_model = cv_model.bestModel
        
        # --- Evaluate the best model on the test set ---
        predictions = best_pipeline_model.transform(test_data)
        auc = evaluator.evaluate(predictions)
        
        # --- Log results and the best model ---
        best_xgb_model = best_pipeline_model.stages[-1]
        mlflow.log_param("best_n_estimators", best_xgb_model.getOrDefault('n_estimators'))
        mlflow.log_param("best_max_depth", best_xgb_model.getOrDefault('max_depth'))
        mlflow.log_param("best_learning_rate", best_xgb_model.getOrDefault('learning_rate'))
        mlflow.log_metric("auc_on_test_set", auc)
        mlflow.spark.log_model(best_pipeline_model, "spark-xgb-pipeline-model-best")

        print(f"--- Pipeline training complete ---")
        print(f"Best Model AUC on Test Set: {auc}")

if __name__ == "__main__":
    main()
