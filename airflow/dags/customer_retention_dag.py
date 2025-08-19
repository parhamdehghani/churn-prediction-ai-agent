from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.dataproc import DataprocSubmitJobOperator

# --- Job Configurations ---
GCP_PROJECT_ID = "churn-prediction-ai-agent"
GCS_BUCKET_NAME = "churn-prediction-ai-agent"
GCP_REGION = "us-central1"

# Dataproc job for feature engineering
FEATURE_ENGINEERING_JOB = {
    "reference": {"project_id": GCP_PROJECT_ID},
    "placement": {"cluster_name": f"feature-eng-cluster-{{{{ ds_nodash }}}}"},
    "pyspark_job": {
        "main_python_file_uri": f"gs://{GCS_BUCKET_NAME}/code/feature_engineering.py",
    },
}

# Dataproc job for model training
MODEL_TRAINING_JOB = {
    "reference": {"project_id": GCP_PROJECT_ID},
    "placement": {"cluster_name": f"model-train-cluster-{{{{ ds_nodash }}}}"},
    "pyspark_job": {
        "main_python_file_uri": f"gs://{GCS_BUCKET_NAME}/code/train_model.py",
    },
}

with DAG(
    dag_id="customer_retention_pipeline",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["mlops", "churn_prediction"],
) as dag:
    feature_engineering_task = DataprocSubmitJobOperator(
        task_id="run_feature_engineering",
        project_id=GCP_PROJECT_ID,
        region=GCP_REGION,
        job=FEATURE_ENGINEERING_JOB,
        gcp_conn_id="google_cloud_default",
    )

    train_model_task = DataprocSubmitJobOperator(
        task_id="run_model_training",
        project_id=GCP_PROJECT_ID,
        region=GCP_REGION,
        job=MODEL_TRAINING_JOB,
        gcp_conn_id="google_cloud_default",
    )

    # Define the dependency
    feature_engineering_task >> train_model_task
