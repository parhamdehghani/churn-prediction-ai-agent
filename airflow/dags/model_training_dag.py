from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocSubmitJobOperator,
    DataprocDeleteClusterOperator,
)
from airflow.utils.trigger_rule import TriggerRule

# --- Constants ---
GCP_PROJECT_ID = "churn-prediction-ai-agent"
GCS_BUCKET_NAME = "churn-prediction-ai-agent"
GCP_REGION = "us-central1"
CLUSTER_NAME = "ephemeral-training-cluster"

# --- Dataproc Cluster Configuration ---
# A separate cluster config for the training job
CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "n1-highmem-4",
    },
    "worker_config": {
        "num_instances": 8,
        "machine_type_uri": "n1-standard-4",
    },
    "initialization_actions": [
        {
            "executable_file": f"gs://{GCS_BUCKET_NAME}/code/dataproc_bootstrap.sh",
        }
    ],
    "software_config": {
        "properties": {
            "spark:spark.dynamicAllocation.enabled": "false"
        }
    }
}

with DAG(
    dag_id="model_training_pipeline",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    schedule=None, # This DAG is manually triggered
    catchup=False,
    tags=["mlops", "model_training"],
) as dag:
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_dataproc_cluster",
        project_id=GCP_PROJECT_ID,
        cluster_config=CLUSTER_CONFIG,
        region=GCP_REGION,
        cluster_name=CLUSTER_NAME,
        gcp_conn_id="google_cloud_default",
    )

    train_model_task = DataprocSubmitJobOperator(
        task_id="run_model_training",
        project_id=GCP_PROJECT_ID,
        region=GCP_REGION,
        job={
            "reference": {"project_id": GCP_PROJECT_ID},
            "placement": {"cluster_name": CLUSTER_NAME},
            "pyspark_job": {
                "main_python_file_uri": f"gs://{GCS_BUCKET_NAME}/code/model_training_cluster.py",
                "properties": {
                "spark.driver.memory": "16g"
                }
            },
        },
        gcp_conn_id="google_cloud_default",
    )

    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_dataproc_cluster",
        project_id=GCP_PROJECT_ID,
        cluster_name=CLUSTER_NAME,
        region=GCP_REGION,
        trigger_rule=TriggerRule.ALL_DONE,
        gcp_conn_id="google_cloud_default",
    )

    create_cluster >> train_model_task >> delete_cluster
