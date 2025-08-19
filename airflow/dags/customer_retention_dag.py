from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.dataproc import DataprocSubmitJobOperator

# --- Constants ---
GCP_PROJECT_ID = "churn-prediction-ai-agent"
GCS_BUCKET_NAME = "churn-prediction-ai-agent"
GCP_REGION = "us-central1"

# --- Dataproc Job Configuration ---
# This defines the configuration for the temporary Dataproc cluster
# that Airflow will create for our Spark jobs.
PYSPARK_JOB_CONFIG = {
    "reference": {"project_id": GCP_PROJECT_ID},
    "placement": {"cluster_name": f"churn-cluster-{{{{ ds_nodash }}}}"},
    "pyspark_job": {}, # Specific script path will be added per task
    "driver_scheduling_config": {
        "initialization_actions": [
            {
                # Installs Git and DVC and pulls the data onto the cluster
                "executable_file": f"gs://{GCS_BUCKET_NAME}/code/bootstrap.sh",
            }
        ]
    }
}

with DAG(
    dag_id="customer_retention_pipeline",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["mlops", "churn_prediction"],
) as dag:
    
    # Task 1: Run Feature Engineering on Dataproc
    # This task creates a Dataproc cluster, runs the bootstrap script, and then runs feature_engineering.py
    feature_engineering_job = PYSPARK_JOB_CONFIG.copy()
    # The script path is the local path INSIDE the cluster after the git clone in the bootstrap script
    feature_engineering_job["pyspark_job"]["main_python_file_uri"] = "file:///churn-repo/src/churn_prediction/feature_engineering.py"
    
    feature_engineering_task = DataprocSubmitJobOperator(
        task_id="run_feature_engineering",
        project_id=GCP_PROJECT_ID,
        region=GCP_REGION,
        job=feature_engineering_job,
        gcp_conn_id="google_cloud_default",
    )

    # Task 2: Run Model Training on Dataproc
    # This task uses the same cluster and runs the cloud-specific training script
    model_training_job = PYSPARK_JOB_CONFIG.copy()
    model_training_job["pyspark_job"]["main_python_file_uri"] = "file:///churn-repo/src/churn_prediction/model_training_cluster.py"

    train_model_task = DataprocSubmitJobOperator(
        task_id="run_model_training",
        project_id=GCP_PROJECT_ID,
        region=GCP_REGION,
        job=model_training_job,
        gcp_conn_id="google_cloud_default",
    )

    # Define the dependency: feature engineering must complete before model training can start
    feature_engineering_task >> train_model_task
