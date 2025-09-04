# Churn Prediction MLOps Pipeline 

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-20.10-blue)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-GCP-orange)
![Kubernetes](https://img.shields.io/badge/Kubernetes-GKE-blue)
![Airflow](https://img.shields.io/badge/Airflow-2.8-lightgrey)
![Jenkins](https://img.shields.io/badge/Jenkins-CI-red)

This project architects, builds, and deploys a scalable, end-to-end MLOps system on Google Cloud. This system uses version-controlled data (DVC), tracked experiments (MLflow), a Jenkins CI/CD pipeline, and an Airflow-orchestrated training and deployment workflow.

## Dataset

This project utilizes the [**KKBox Churn Prediction Challenge**](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) dataset, a large-scale, real-world dataset totaling over 8 GB. Its size and complexity necessitate the use of a distributed data processing engine like Apache Spark to perform feature engineering and model training efficiently. The data is version-controlled using DVC, with the raw files stored in Google Cloud Storage.

### Data Structure and Relevance to Churn

The dataset is composed of several tables that provide a holistic view of user behavior:

* **`train.csv`**: This is the primary training file, containing the user ID (`msno`) and the target variable, `is_churn`. The churn label is defined as a user not renewing their subscription within 30 days of its expiration.

* **`transactions.csv`**: This table contains the full transaction history for users up to February 2017. It is critical for creating features related to a user's financial commitment and subscription patterns. Key features include:
    * `payment_plan_days`, `plan_list_price`, `actual_amount_paid`: These columns describe the value and duration of a user's subscriptions.
    * `is_auto_renew`, `is_cancel`: These boolean flags provide direct insight into user intent and subscription management habits.

* **`user_logs.csv`**: This table contains daily aggregated user listening activity until the end of February 2017. It is the richest source for creating features related to user engagement with the music streaming service. Key features include:
    * `num_100`, `num_unq`, `total_secs`: These columns quantify how deeply a user engages with the product (e.g., number of songs fully listened to, variety of music, and total time spent on the platform). High engagement is often a strong negative predictor of churn.

* **`members.csv`**: This file contains user demographic information.
    * `city`, `bd` (age), `gender`: Standard demographic features that can reveal patterns across different user segments. The age column (`bd`) contains significant outliers and requires careful cleaning.

## Live Demo

For cost-effectiveness in a portfolio setting, the final public endpoint is hosted on **Google Cloud Run**, a serverless platform, while the main deployment has been developed over Kubernetes pods in the test phase. 

**Service URL**: `https://churn-api-service-rdxj3z25yq-uc.a.run.app/` -> It takes time for the first request as the endpoint is serverless

**Sample Inference Request**:
You can test the live endpoint by sending a POST request with the following `curl` command:

```bash
curl -X POST "[https://churn-api-service-rdxj3z25yq-uc.a.run.app/predict](https://churn-api-service-rdxj3z25yq-uc.a.run.app/predict)" \
-H "Content-Type: application/json" \
-d '{
  "transaction_count": 15,
  "total_plan_days": 450,
  "total_amount_paid": 2000,
  "total_songs_completed": 8000,
  "total_songs_985_completed": 500,
  "total_unique_songs": 2500,
  "total_secs_played": 700000,
  "listening_day_count": 150,
  "age_cleaned": 35,
  "is_male": 0,
  "is_female": 1
}'
```

## MLOps Architecture

The project follows a modern, microservice-based MLOps architecture, where each component of the ML lifecycle is handled by a dedicated, industry-standard tool. The entire workflow, from code commit to cloud deployment, is automated.

Git Push -\> Jenkins (CI) -\> Artifact Registry -\> Airflow (Orchestration) -\> Dataproc (Training) -\> MLflow (Tracking) -\> GKE/Cloud Run (Deployment)

## Technology Stack

  * **Data & Model Versioning**: DVC (Data Version Control)
  * **Experiment Tracking & Registry**: MLflow
  * **Data Processing**: Apache Spark (PySpark), Google Cloud Storage (GCS), Google Cloud Dataproc
  * **API & Containerization**: Python, FastAPI, Docker
  * **CI/CD**: Jenkins, Git, GitHub, Google Artifact Registry
  * **Workflow Orchestration**: Apache Airflow
  * **Deployment**: Google Kubernetes Engine (GKE), Google Cloud Run

## Project Structure

```
.
├── airflow/              # Contains the Airflow project, including DAGs and configurations.
├── data/                 # Data directory, managed by DVC. Contains the raw data pointers.
├── docker/               # Holds custom Dockerfiles for services like Jenkins and MLflow.
├── kubernetes/           # Kubernetes manifest files (.yaml) for deploying services to GKE.
├── src/                  # Main source code for the project's Python package.
│   └── churn_prediction/
│       ├── api/          # Source code for the FastAPI application.
│       ├── feature_engineering.py
│       └── train_model.py
├── docker-compose.yml    # Manages the local MLflow and PostgreSQL services.
├── Dockerfile            # Dockerfile for building the main FastAPI application image.
├── Jenkinsfile           # The pipeline-as-code definition for the Jenkins CI pipeline.
└── requirements.txt      # Python dependencies for the main application.
```

## Project Phases

### ✅ Phase 1: Data Engineering & Predictive Modeling

  * **Goal**: Establish a version-controlled data foundation and a reproducible model training process.
  * **Execution**: Processed a 30GB+ dataset from the KKBox Churn Prediction Challenge. Data was versioned with **DVC** and stored in GCS. An extensive hyperparameter tuning job was performed using **PySpark** to train a high-performance XGBoost model (AUC 0.825). All experiments, parameters, metrics, and the final model artifact were logged to a persistent **MLflow** server.

### ✅ Phase 2: API Development & Containerization

  * **Goal**: Encapsulate the model's prediction logic into a lightweight, portable microservice.
  * **Execution**: Developed a **FastAPI** application to serve predictions. The API loads the trained Spark ML Pipeline model directly from the MLflow server. The application was containerized using a production-ready, multi-stage **Dockerfile** that builds a multi-platform image (AMD64 & ARM64).

### ✅ Phase 3: CI Automation with Jenkins

  * **Goal**: Create an automated Continuous Integration pipeline to validate the code and publish the application container.
  * **Execution**: A `Jenkinsfile` was created to define a pipeline-as-code. This pipeline, running on a custom Jenkins Docker image with all necessary tools (Docker, gcloud, DVC), automatically triggers on a `git push`. It builds the multi-platform API image and pushes the versioned image to a private **Google Artifact Registry** repository.

### ✅ Phase 4: Orchestration and Deployment to Kubernetes

  * **Goal**: Use a dedicated orchestrator to manage the training and deployment workflows on a production-grade platform.
  * **Execution**:
      * **Orchestration**: **Apache Airflow** was set up locally using the Astro CLI. Modular DAGs were developed to orchestrate the cloud-based training workflow, using operators like `DataprocCreateClusterOperator` and `DataprocSubmitJobOperator` to create a temporary cluster, run the Spark training job, and tear it down to manage costs.
      * **Infrastructure**: A **Google Kubernetes Engine (GKE)** Autopilot cluster was provisioned to serve as the runtime environment.
      * **Microservice Deployment**: The entire application was deployed to GKE using declarative Kubernetes manifest files (`.yaml`). This included deploying the PostgreSQL database with a `PersistentVolumeClaim` for stateful storage, the MLflow server, and the FastAPI prediction service.
      * **Authentication**: **GKE Workload Identity** was configured to provide a secure, fine-grained authentication mechanism for the API pods to access the MLflow model artifacts stored in GCS.


