from __future__ import annotations
import pendulum
import subprocess
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# --- Constants ---
GCP_PROJECT_ID = "churn-prediction-ai-agent"
GCP_REGION = "us-central1" 
REPO_NAME = "churn-repo"
IMAGE_NAME = "churn-prediction-api"

def find_latest_image_tag(**kwargs):
    """
    Queries Google Artifact Registry to find the most recent numeric image tag,
    constructs the full image URI, and pushes it to XComs.
    """
    image_path = f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/{REPO_NAME}/{IMAGE_NAME}"

    command = f"""
    gcloud artifacts docker images list {image_path} \
      --sort-by=~CREATE_TIME \
      --limit=1 \
      --format='value(tags)'
    """

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        
        if not result.stdout.strip():
            raise ValueError("No images found in the repository")
            
        # Handle both comma-separated and single tags
        tags_output = result.stdout.strip()
        tags = [tag.strip() for tag in tags_output.split(',') if tag.strip()]
        
        # More robust numeric tag detection
        numeric_tag = None
        for tag in tags:
            if tag.isdigit() and tag != 'latest':
                numeric_tag = tag
                break
        
        if not numeric_tag:
            raise ValueError(f"Could not find a valid numeric image tag. Available tags: {tags}")

        full_image_uri = f"{image_path}:{numeric_tag}"
        print(f"Found latest image URI to deploy: {full_image_uri}")

        kwargs['ti'].xcom_push(key='full_image_uri', value=full_image_uri)
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"gcloud command failed: {e.stderr}")
    except Exception as e:
        raise ValueError(f"Failed to find latest image tag: {str(e)}")

with DAG(
    dag_id="deployment_pipeline",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    schedule=None, # This DAG should be triggered by Jenkins
    catchup=False,
    tags=["mlops", "cd"],
) as dag:
    get_latest_tag_task = PythonOperator(
        task_id="get_latest_image_tag",
        python_callable=find_latest_image_tag,
    )

    # Option 1: Using BashOperator with gcloud command
    deploy_to_cloud_run = BashOperator(
        task_id="deploy_api_to_cloud_run",
        bash_command="""
        IMAGE_URI="{{ ti.xcom_pull(task_ids='get_latest_image_tag', key='full_image_uri') }}"
        
        if [ -z "$IMAGE_URI" ]; then
            echo "Error: No image URI found from previous task"
            exit 1
        fi
        
        echo "Deploying image: $IMAGE_URI"
        
        gcloud run deploy churn-api-service \
            --image="$IMAGE_URI" \
            --region="{{ params.region }}" \
            --project="{{ params.project_id }}" \
            --platform=managed \
            --allow-unauthenticated \
            --port=8080 \
            --memory=1Gi \
            --cpu=1 \
            --min-instances=0 \
            --max-instances=10
        """,
        params={
            'project_id': GCP_PROJECT_ID,
            'region': GCP_REGION
        }
    )

    get_latest_tag_task >> deploy_to_cloud_run