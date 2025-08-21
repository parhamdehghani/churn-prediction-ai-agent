import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main():
    """
    Loads raw data, performs feature engineering, and saves the final feature set.
    """
    spark = SparkSession.builder.appName("ChurnFeatureEngineering").getOrCreate()

    # --- Load Data ---
    data_path = "gs://churn-prediction-ai-agent/dvc/files/md5/"
    output_path = "gs://churn-prediction-ai-agent/data/processed/"
    
    df_train = spark.read.csv(os.path.join(data_path, "6f/aa8475570da65b17af858fc049cf6a"), header=True, inferSchema=True)
    df_members = spark.read.csv(os.path.join(data_path, "06/c6508e09f6371c4b6259307e2fa429"), header=True, inferSchema=True)
    df_transactions = spark.read.csv(os.path.join(data_path, "19/f32c65e29ef8cf87a5922d4b834b04"), header=True, inferSchema=True)
    df_user_logs = spark.read.csv(os.path.join(data_path, "f8/43c4fbbc84627dfd344fa69322fc5e"), header=True, inferSchema=True)
    print("--- Data Loading Complete ---")
    print(f"Initial df_train row count: {df_train.count()}")

    # --- Aggregate and Clean Features ---
    print("--- Starting Feature Engineering ---")
    transactions_features = df_transactions.groupBy("msno").agg(
        F.count("transaction_date").alias("transaction_count"),
        F.sum("payment_plan_days").alias("total_plan_days"),
        F.sum("actual_amount_paid").alias("total_amount_paid")
    )
    print(f"Aggregated transactions_features row count: {transactions_features.count()}")

    user_logs_features = df_user_logs.groupBy("msno").agg(
        F.sum("num_100").alias("total_songs_completed"),
        F.sum("num_985").alias("total_songs_985_completed"),
        F.sum("total_secs").alias("total_secs_played"),
        F.sum("num_unq").alias("total_unique_songs"),
        F.count("date").alias("listening_day_count")
    )
    print(f"Aggregated user_logs_features row count: {user_logs_features.count()}")

    members_cleaned = df_members.withColumn(
        "age_cleaned",
        F.when((F.col("bd") >= 5) & (F.col("bd") <= 100), F.col("bd")).otherwise(None)
    ).withColumn(
        "gender_cleaned",
        F.when(F.col("gender").isin("male", "female"), F.col("gender")).otherwise(None)
    )

    # --- Join DataFrames ---
    print("--- Joining DataFrames ---")
    df_features = df_train.join(transactions_features, on="msno", how="left")
    print(f"Row count after joining transactions: {df_features.count()}")

    df_features = df_features.join(user_logs_features, on="msno", how="left")
    print(f"Row count after joining user logs: {df_features.count()}")

    df_features = df_features.join(members_cleaned.select("msno", "age_cleaned", "gender_cleaned"), on="msno", how="left")
    print(f"Final row count before processing gender: {df_features.count()}")
    
    # Manually create binary columns for gender for robustness
    df_features = df_features.withColumn("is_male", F.when(F.col("gender_cleaned") == "male", 1).otherwise(0))
    df_features = df_features.withColumn("is_female", F.when(F.col("gender_cleaned") == "female", 1).otherwise(0))
    df_features = df_features.drop("gender_cleaned")
    
    print("--- Feature Engineering Complete ---")

    # --- Save Processed Data ---
    # Parquet is a columnar format, very efficient for Spark
    df_features.write.mode("overwrite").parquet(os.path.join(output_path, "features.parquet"))
    print(f"--- Processed features saved to {output_path} ---")

if __name__ == "__main__":
    main()
